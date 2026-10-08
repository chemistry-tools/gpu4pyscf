"""Benchmark ASE energies, forces and optimizer steps from a frozen external input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

from _extensions import load_extensions
from gpu_performance import _checkout_revision, _resources, _source_provenance

_RECIPE_KEYS = {
    'functional',
    'basis',
    'ecp',
    'auxbasis',
    'atom_grid',
    'grid_level',
    'conv_tol',
    'max_cycle',
    'reference',
    'grid_response',
    'auxbasis_response',
    'warm_start',
    'max_memory_mb',
    'verbose',
}


def _input(path, case_id):
    source = json.loads(path.read_text())
    if set(source) - {'recipe', 'cases', 'provenance'} or not {'recipe', 'cases'} <= set(source):
        raise ValueError('Unknown or missing benchmark input keys')
    recipe = source['recipe']
    if set(recipe) != _RECIPE_KEYS:
        raise ValueError('Unknown or missing recipe keys')
    cases = source['cases']
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError('cases must be a list of objects')
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Case IDs must be nonempty and unique')
    for case in cases:
        if set(case) - {'id', 'numbers', 'positions', 'charge', 'multiplicity', 'provenance'}:
            raise ValueError('Unknown case keys')
        if not {'id', 'numbers', 'positions', 'charge', 'multiplicity'} <= set(case):
            raise ValueError('Missing case keys')
        if not isinstance(case['id'], str) or not case['id']:
            raise ValueError('Case IDs must be nonempty strings')
        if not case['numbers'] or len(case['numbers']) != len(case['positions']):
            raise ValueError('Atomic numbers and positions must have matching nonzero lengths')
        if any(type(z) is not int or not 1 <= z <= 118 for z in case['numbers']):
            raise ValueError('Atomic numbers must be integers from one to 118')
        if any(
            len(xyz) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in xyz)
            for xyz in case['positions']
        ):
            raise ValueError('Positions must be finite triples in Angstrom')
        if type(case['charge']) is not int or type(case['multiplicity']) is not int or case['multiplicity'] < 1:
            raise ValueError('Charge and positive multiplicity must be explicit integers')
    case = next((case for case in cases if case['id'] == case_id), None)
    if case is None:
        raise ValueError('Requested case is absent from the frozen input')
    return source, case


def _method(case, recipe):
    from ase.data import chemical_symbols
    from pyscf import gto

    from gpu4pyscf import dft

    memory = {'max_memory': recipe['max_memory_mb']} if recipe['max_memory_mb'] is not None else {}
    mol = gto.M(
        atom=[(chemical_symbols[z], position) for z, position in zip(case['numbers'], case['positions'])],
        basis=recipe['basis'],
        charge=case['charge'],
        spin=case['multiplicity'] - 1,
        ecp={chemical_symbols[z]: recipe['ecp'] for z in set(case['numbers']) if z >= 37}
        if recipe['ecp'] is not None
        else {},
        **memory,
        verbose=recipe['verbose'],
    )
    unrestricted = recipe['reference'] == 'uks' or case['multiplicity'] != 1
    if recipe['reference'] not in ('auto', 'uks'):
        raise ValueError('reference must be auto or uks')
    mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=recipe['functional'])
    if recipe['auxbasis'] is not None:
        mf = mf.density_fit(auxbasis=recipe['auxbasis'])
    if recipe['atom_grid'] is not None:
        mf.grids.atom_grid = tuple(recipe['atom_grid'])
    elif recipe['grid_level'] is not None:
        mf.grids.level = recipe['grid_level']
    mf.conv_tol = recipe['conv_tol']
    mf.max_cycle = recipe['max_cycle']
    mf.chkfile = None
    return mf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--precision', choices=('float64', 'mixed'), required=True)
    parser.add_argument('--cpus', type=lambda value: [int(x) for x in value.split(',')], default=[4, 5, 6, 7])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extension-root', type=Path)
    parser.add_argument('--steps', type=int, default=0, help='BFGS steps after the initial energy/forces')
    parser.add_argument('--fmax', type=float, default=0.005)
    parser.add_argument('--hessian', action='store_true')
    args = parser.parse_args()
    if args.steps < 0 or not 0 < args.fmax < 1:
        parser.error('steps must be nonnegative and fmax positive and below one eV/Angstrom')
    if args.output.suffix != '.json' or args.output.exists():
        parser.error('Use a new .json report path')
    artifact = args.output.with_suffix('')
    artifact.mkdir(parents=True, exist_ok=False)
    _resources(args.cpus)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Only physical GPU 1 is authorized for this benchmark')
    frozen, case = _input(args.input, args.case)

    import cupy
    import numpy as np
    import pyscf
    from ase import Atoms
    from ase.io import write
    from ase.optimize import BFGS
    from pyscf import lib
    from pyscf.data.nist import BOHR, HARTREE2EV
    from threadpoolctl import threadpool_info

    import gpu4pyscf
    from gpu4pyscf.dft import numint

    lib.num_threads(4)
    if lib.num_threads() != 4 or cupy.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Expected four OpenMP threads and one visible GPU')
    physical = (
        subprocess.check_output(
            ['nvidia-smi', '-i', '1', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'],
            text=True,
        )
        .strip()
        .split(',')
    )
    actual_pci = cupy.cuda.Device(0).pci_bus_id

    def normalize(value):
        domain, location = value.strip().split(':', 1)
        return int(domain, 16), location.lower()

    if normalize(actual_pci) != normalize(physical[1]):
        raise RuntimeError('Visible device is not physical GPU 1')
    extensions = load_extensions(args.extension_root, ase=True) if args.extension_root else None
    from gpu4pyscf.tools.ase_interface import PySCF

    recipe = frozen['recipe']
    atoms = Atoms(numbers=case['numbers'], positions=case['positions'])
    mf = _method(case, recipe)
    atoms.calc = PySCF(
        method=mf,
        precision=args.precision,
        grid_response=recipe['grid_response'],
        auxbasis_response=bool(recipe['auxbasis_response']),
    )
    if not recipe['warm_start']:
        atoms.calc.method_scan = None
    if not hasattr(atoms.calc, 'precision_info'):
        raise RuntimeError('The installed ASE interface does not implement the precision option')
    frames = []
    started = time.perf_counter()
    revision = _checkout_revision(Path(__file__).resolve().parents[2])

    def progress(env):
        record = {
            'cycle': int(env['cycle']) + 1,
            'elapsed_seconds': time.perf_counter() - started,
            'energy_hartree': float(env['e_tot']),
            'orbital_gradient_norm': float(env['norm_gorb']),
        }
        with (artifact / 'scf-progress.jsonl').open('a') as output:
            output.write(json.dumps(record, allow_nan=False) + '\n')

    mf.callback = progress
    if atoms.calc.method_scan is not None:
        atoms.calc.method_scan.callback = progress

    def evaluate():
        if frames and np.array_equal(atoms.positions, frames[-1]['positions_angstrom']):
            return
        cupy.cuda.get_current_stream().synchronize()
        frame_started = time.perf_counter()
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()
        cupy.cuda.get_current_stream().synchronize()
        base = atoms.calc.method if atoms.calc.method_scan is None else atoms.calc.method_scan
        entry = {
            'energy_ev': float(energy),
            'energy_hartree': float(base.e_tot),
            'forces_ev_angstrom': forces.tolist(),
            'positions_angstrom': atoms.positions.tolist(),
            'fmax_ev_angstrom': float(np.linalg.norm(forces, axis=1).max()),
            'seconds': time.perf_counter() - frame_started,
            'calculator_timing': atoms.calc.calculation_info,
            'scf_cycles': int(base.cycles),
            'precision_info': atoms.calc.precision_info,
            'effective_orbitals': int(base.mo_coeff.shape[-1]),
            'grid_points': len(base.grids.weights),
            'nlc_grid_points': len(base.nlcgrids.weights),
            'nlc_grid_level': int(base.nlcgrids.level),
        }
        frames.append(entry)
        (artifact / 'frames.json').write_text(json.dumps(frames, indent=2, allow_nan=False) + '\n')

    evaluate()
    optimized = False
    if args.steps:
        optimizer = BFGS(
            atoms,
            maxstep=0.15,
            logfile=str(artifact / 'optimization.log'),
            trajectory=str(artifact / 'optimization.traj'),
        )
        optimizer.attach(evaluate, interval=1)
        optimized = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
    calculation_seconds = time.perf_counter() - started
    write(artifact / 'final.xyz', atoms)
    hessian_seconds = None
    if args.hessian:
        hessian_started = time.perf_counter()
        hessian = atoms.calc.get_hessian(atoms, auxbasis_response=recipe['auxbasis_response'])
        np.save(artifact / 'hessian_ev_angstrom2.npy', hessian)
        n = len(atoms)
        native_units = hessian.reshape(n, 3, n, 3).transpose(0, 2, 1, 3) * (BOHR**2 / HARTREE2EV)
        np.save(artifact / 'hessian_hartree_bohr2.npy', native_units)
        cupy.cuda.get_current_stream().synchronize()
        hessian_seconds = time.perf_counter() - hessian_started
    report = {
        'recipe': recipe,
        'unit_constants': {'hartree_ev': HARTREE2EV, 'bohr_angstrom': BOHR},
        'case': case,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'precision': args.precision,
        'frames': frames,
        'optimization_converged': optimized,
        'requested_steps': args.steps,
        'fmax_target_ev_angstrom': args.fmax,
        'calculation_seconds': calculation_seconds,
        'hessian_seconds': hessian_seconds,
        'hessian_info': atoms.calc.hessian_info,
        'benchmark_revision': revision,
        'gpu4pyscf_source': _source_provenance(gpu4pyscf, numint.libgdft),
        'python_extension_sha256': extensions,
        'versions': {
            'pyscf': pyscf.__version__,
            'gpu4pyscf': gpu4pyscf.__version__,
            'cupy': cupy.__version__,
            'ase': importlib.metadata.version('ase'),
        },
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'openmp_threads': lib.num_threads(),
        'threadpools': threadpool_info(),
        'gpu_pci_bus_id': actual_pci,
        'gpu_uuid': physical[0].strip(),
        'artifact_directory': str(artifact),
        'provenance': frozen.get('provenance'),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(
        json.dumps(
            {
                'precision': args.precision,
                'calculation_seconds': calculation_seconds,
                'frames': len(frames),
                'optimization_converged': optimized,
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
