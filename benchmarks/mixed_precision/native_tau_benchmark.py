"""Measure fresh native ASE energy and forces with or without symmetric tau products."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from gpu_performance import _resources, _source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--tau', choices=('baseline', 'symmetric'), required=True)
    parser.add_argument('--require-gpu-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != '.json':
        parser.error('Use a new .json output path')
    artifact = args.output.with_suffix('')
    artifact.mkdir(parents=True, exist_ok=False)
    _resources([4, 5, 6, 7])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Select physical GPU 1')

    import cupy as cp
    import numpy as np
    from _larger_cases import LARGER_CASES
    from _organic_cases import ORGANIC_CASES
    from ase import Atoms
    from pyscf import gto, lib
    from pyscf.data.nist import BOHR, HARTREE2EV
    from threadpoolctl import threadpool_info

    import gpu4pyscf
    from gpu4pyscf import dft
    from gpu4pyscf.dft import _tau_syrk, numint
    from gpu4pyscf.scf import hf
    from gpu4pyscf.tools.ase_interface import PySCF

    source = Path(gpu4pyscf.__file__).resolve().parent
    if source != args.require_gpu_source.resolve():
        raise RuntimeError(f'Unexpected GPU4PySCF import: {source}')
    if cp.cuda.runtime.getDeviceCount() != 1 or not cp.cuda.Device(0).pci_bus_id.lower().endswith('05:00.0'):
        raise RuntimeError('Visible device must be physical GPU 1 on the designated host')
    lib.num_threads(4)
    if any(pool['num_threads'] > 4 for pool in threadpool_info()):
        raise RuntimeError('Numerical thread pools exceed the resource allowance')
    definitions = LARGER_CASES | ORGANIC_CASES
    if args.case not in definitions:
        parser.error(f'Unknown generic development case: {args.case}')
    hf.overlap_zero_eigenvalue_threshold = 1e-10
    atom, charge, multiplicity = definitions[args.case]
    mol = gto.M(atom=atom, basis='def2-tzvpd', charge=charge, spin=multiplicity - 1, verbose=0)
    mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.chkfile = 1e-8, 1e-4, 200, None
    mf.callback = lambda env: print(
        json.dumps(
            {
                'cycle': int(env['cycle']) + 1,
                'energy': float(env['e_tot']),
                'gradient': float(env['norm_gorb']),
            }
        ),
        flush=True,
    )
    atoms = Atoms(mol.elements, positions=mol.atom_coords(unit='Angstrom'))
    atoms.calc = PySCF(method=mf, precision='auto', grid_response=True, auxbasis_response=True)
    original_min, original_tau = _tau_syrk._MIN_ORBITALS, numint._symmetric_tau
    calls = []

    def counted(ao, weights, out=None):
        calls.append([int(ao.shape[1]), int(ao.shape[2])])
        return original_tau(ao, weights, out=out)

    numint._symmetric_tau = counted
    if args.tau == 'baseline':
        _tau_syrk._MIN_ORBITALS = 10**6
    try:
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        energy_ev = atoms.get_potential_energy()
        cp.cuda.get_current_stream().synchronize()
        energy_seconds = time.perf_counter() - started
        energy_info, precision_info = dict(atoms.calc.calculation_info), atoms.calc.precision_info
        if not precision_info['accepted']:
            raise RuntimeError('SCF must pass full float64 verification')
        started = time.perf_counter()
        forces = atoms.get_forces()
        cp.cuda.get_current_stream().synchronize()
        force_seconds = time.perf_counter() - started
        if not atoms.calc.calculation_info['scf_reused'] or atoms.calc.calculation_info['scf_seconds'] != 0.0:
            raise RuntimeError('Unchanged-geometry forces must reuse the accepted SCF state')
        current = atoms.calc.method_scan
        if current.mo_coeff.shape[-1] != mol.nao:
            raise RuntimeError('The full basis must be retained')
        if (args.tau == 'symmetric') != bool(calls):
            raise RuntimeError('The requested tau implementation was not exercised')
        state = artifact / 'scf.npz'
        np.savez(state, mo_coeff=current.mo_coeff.get(), mo_occ=current.mo_occ.get(), mo_energy=current.mo_energy.get())
        report = {
            'case': args.case,
            'tau': args.tau,
            'geometry_angstrom': atom,
            'charge': charge,
            'multiplicity': multiplicity,
            'functional': mf.xc,
            'basis': mol.basis,
            'auxbasis': mf.with_df.auxbasis,
            'grid_level': 5,
            'nlc_grid_level': 3,
            'grid_response': True,
            'auxbasis_response': True,
            'conv_tol': 1e-8,
            'conv_tol_grad': 1e-4,
            'max_cycle': 200,
            'overlap_cutoff': 1e-10,
            'energy_hartree': energy_ev / HARTREE2EV,
            'energy_seconds': energy_seconds,
            'scf_seconds': energy_info['scf_seconds'],
            'force_seconds': force_seconds,
            'energy_force_seconds': energy_seconds + force_seconds,
            'forces_hartree_bohr': (forces * BOHR / HARTREE2EV).tolist(),
            'precision_info': precision_info,
            'force_calculation_info': atoms.calc.calculation_info,
            'nao': mol.nao,
            'effective_orbitals': int(current.mo_coeff.shape[-1]),
            'grid_points': len(current.grids.weights),
            'nlc_grid_points': len(current.nlcgrids.weights),
            'tau_syrk_calls': calls,
            'state_sha256': hashlib.sha256(state.read_bytes()).hexdigest(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'gpu_pci': cp.cuda.Device(0).pci_bus_id,
            'threadpools': threadpool_info(),
            'sources': _source_provenance(gpu4pyscf, numint.libgdft),
            'native_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('lib/*.so')},
            'tau_sha256': hashlib.sha256(Path(_tau_syrk.__file__).read_bytes()).hexdigest(),
            'numint_sha256': hashlib.sha256(Path(numint.__file__).read_bytes()).hexdigest(),
            'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        'case',
                        'tau',
                        'energy_hartree',
                        'scf_seconds',
                        'force_seconds',
                        'energy_force_seconds',
                    )
                }
            ),
            flush=True,
        )
    finally:
        _tau_syrk._MIN_ORBITALS, numint._symmetric_tau = original_min, original_tau


if __name__ == '__main__':
    main()
