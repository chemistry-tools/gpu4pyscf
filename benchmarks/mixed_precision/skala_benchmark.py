"""Compare Skala ASE calculations and preoptimization with the fixed DFT recipe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

from _extensions import load_extensions
from gpu_performance import CASES, _resources, _settings, _source_provenance, _timed_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('skala', 'native', 'mixed', 'cascade'), required=True)
    parser.add_argument('--case', choices=tuple(CASES), default='benzene')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--scale', type=float, default=1.025)
    parser.add_argument('--fmax', type=float, default=0.005)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--skala-grid-level', type=int, default=5)
    parser.add_argument('--refresh-skala-dispersion', action='store_true')
    parser.add_argument('--check-forces', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix != '.json':
        parser.error('Use a new .json output path')
    if args.repeats < 1 or args.steps < 1 or not 0 < args.fmax < 1 or not 0.9 <= args.scale <= 1.1:
        parser.error('Invalid repeat, step, force or geometry scale setting')
    if args.method == 'cascade' and not args.optimize:
        parser.error('Cascade requires --optimize')
    if not 0 <= args.skala_grid_level <= 9:
        parser.error('Skala grid level must be from zero to nine')
    if args.check_forces and (args.method != 'skala' or args.optimize):
        parser.error('Force check requires a Skala single-point run')
    _resources([4, 5, 6, 7])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Only physical GPU 1 is authorized')
    settings, settings_hash = _settings()
    started = time.perf_counter()
    import cupy as cp
    import gpu4pyscf
    import numpy as np
    from ase import Atoms
    from ase.optimize import BFGS
    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint
    from gpu4pyscf.scf import hf
    from pyscf import gto, lib
    from threadpoolctl import threadpool_info

    lib.num_threads(4)
    if lib.num_threads() != 4 or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Expected four OpenMP threads and one CUDA device')
    physical = (
        subprocess.check_output(
            ['nvidia-smi', '-i', '1', '--query-gpu=uuid,pci.bus_id,name', '--format=csv,noheader,nounits'], text=True
        )
        .strip()
        .split(',')
    )
    actual_pci = cp.cuda.Device(0).pci_bus_id

    def normalize(value):
        domain, location = value.strip().split(':', 1)
        return int(domain, 16), location.lower()

    if normalize(actual_pci) != normalize(physical[1]):
        raise RuntimeError('Visible device is not physical GPU 1')
    extensions = load_extensions(args.extension_root, ase=True)
    from gpu4pyscf.tools.ase_interface import PySCF

    hf.overlap_zero_eigenvalue_threshold = 1e-10
    functional = None
    functional_info = None
    if args.method in ('skala', 'cascade'):
        import torch
        from skala.ase import Skala
        from skala.functional import resolve_functional_artifact
        from skala.gpu4pyscf import dft as skala_dft
        from skala.gpu4pyscf import gradients as skala_gradients
        from skala.gpu4pyscf.grids import SkalaGrids

        class _GeometrySkala(Skala):
            def calculate(self, atoms=None, properties=None, system_changes=None):
                # The published GPU gradient reset leaves the D3 backend on the old geometry.
                if args.refresh_skala_dispersion and self._ks is not None and atoms is not None:
                    dispersion = self._ks.base.with_dftd3
                    if dispersion is not None:
                        mol = self._mol.set_geom_([(atom.symbol, atom.position) for atom in atoms], inplace=False)
                        dispersion.reset(mol)
                return super().calculate(atoms, properties, system_changes)

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        model_started = time.perf_counter()
        artifact = resolve_functional_artifact('skala-1.1', device=torch.device('cuda:0'))
        functional = artifact.load(device=torch.device('cuda:0'))
        functional_info = {
            'name': 'skala-1.1',
            'path': str(artifact.path),
            'sha256': hashlib.sha256(artifact.path.read_bytes()).hexdigest(),
            'expected_sha256': artifact.expected_hash,
            'load_seconds': time.perf_counter() - model_started,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'torch_cuda': torch.version.cuda,
            'd3_settings': str(functional.get_d3_settings()),
        }
    imports_seconds = time.perf_counter() - started
    atom, charge, multiplicity = CASES[args.case]
    template = gto.M(atom=atom, basis=settings['basis'], charge=charge, spin=multiplicity - 1, verbose=0)
    symbols = template.elements
    positions = template.atom_coords(unit='Angstrom')
    if args.optimize:
        positions *= args.scale
    samples = []

    def synchronize():
        cp.cuda.get_current_stream().synchronize()
        if functional is not None:
            torch.cuda.synchronize()

    def calculator(atoms, method, history):
        mol = gto.M(
            atom=list(zip(symbols, atoms.positions)),
            basis=settings['basis'],
            charge=charge,
            spin=multiplicity - 1,
            verbose=0,
        )

        def progress(env):
            history.append(
                {
                    'cycle': int(env['cycle']) + 1,
                    'energy_hartree': float(env['e_tot']),
                    'orbital_gradient_norm': float(env['norm_gorb']),
                }
            )

        if method == 'skala':
            # Skala requires its atom-major grids; equal levels do not imply identical quadrature.
            grids = SkalaGrids(mol)
            grids.level = args.skala_grid_level
            return _GeometrySkala(
                xc=functional,
                basis=settings['basis'],
                charge=charge,
                multiplicity=multiplicity,
                device='cuda',
                with_density_fit=True,
                auxbasis='def2-universal-jkfit',
                with_dftd3=True,
                with_retry=False,
                verbose=0,
                ks_config={
                    'conv_tol': settings['conv_tol'],
                    'conv_tol_grad': settings['conv_tol'] ** 0.5,
                    'max_cycle': settings['max_cycle'],
                    'chkfile': None,
                    'grids': grids,
                    'callback': progress,
                },
            )
        mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc=settings['xc'])
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
        mf.grids.level = settings['grid_level']
        mf.nlcgrids.level = settings['nlc_grid_level']
        mf.conv_tol = settings['conv_tol']
        mf.conv_tol_grad = settings['conv_tol'] ** 0.5
        mf.max_cycle = settings['max_cycle']
        mf.chkfile = None
        mf.callback = progress
        return PySCF(
            method=mf, precision='mixed' if method == 'mixed' else 'float64', grid_response=True, auxbasis_response=True
        )

    def stage(atoms, method, optimize):
        history, frames = [], []
        timings = {'scf': [], 'gradient': []}
        synchronize()
        stage_started = time.perf_counter()
        atoms.calc = calculator(atoms, method, history)

        def evaluate():
            if frames and np.array_equal(atoms.positions, frames[-1]['positions_angstrom']):
                return
            synchronize()
            evaluation_started = time.perf_counter()
            forces = atoms.get_forces()
            energy = atoms.get_potential_energy()
            synchronize()
            mf = atoms.calc._ks.base if method == 'skala' else atoms.calc.method_scan
            if not mf.converged or not np.isfinite(energy) or not np.isfinite(forces).all():
                raise RuntimeError('Unconverged or nonfinite ASE evaluation')
            frames.append(
                {
                    'seconds': time.perf_counter() - evaluation_started,
                    'energy_ev': float(energy),
                    'forces_ev_angstrom': forces.tolist(),
                    'fmax_ev_angstrom': float(np.linalg.norm(forces, axis=1).max()),
                    'positions_angstrom': atoms.positions.tolist(),
                    'scf_cycles_total': len(history),
                    'grid_points': len(mf.grids.weights),
                    'effective_orbitals': int(mf.mo_coeff.shape[-1]),
                    'precision_info': getattr(atoms.calc, 'precision_info', None),
                    'calculation_info': getattr(atoms.calc, 'calculation_info', None),
                }
            )
            print(
                json.dumps(
                    {
                        'method': method,
                        'frame': len(frames),
                        'seconds': frames[-1]['seconds'],
                        'fmax': frames[-1]['fmax_ev_angstrom'],
                    }
                ),
                flush=True,
            )

        with ExitStack() as stack:
            if method == 'skala':
                for cls in (skala_dft.SkalaRKS, skala_dft.SkalaUKS):
                    stack.enter_context(_timed_method(cls, 'kernel', {'kernel': timings['scf']}, synchronize))
                for cls in (skala_gradients.SkalaRKSGradient, skala_gradients.SkalaUKSGradient):
                    stack.enter_context(_timed_method(cls, 'kernel', {'kernel': timings['gradient']}, synchronize))
            if optimize:
                opt = BFGS(atoms, logfile=None, maxstep=0.1)
                opt.attach(evaluate)
                converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
                steps = opt.nsteps
            else:
                evaluate()
                converged, steps = True, 0
        synchronize()
        report = {
            'method': method,
            'seconds': time.perf_counter() - stage_started,
            'optimizer_converged': converged,
            'optimizer_steps': steps,
            'scf_cycles': len(history),
            'history': history,
            'frames': frames,
            'scf_call_seconds': timings['scf'],
            'gradient_call_seconds': timings['gradient'],
        }
        if optimize and not converged:
            raise RuntimeError('Optimizer did not converge within the step budget')
        return report

    # First evaluation is retained separately; later fresh mean-fields use warm process caches.
    count = 1 if args.optimize else args.repeats + 1
    for index in range(count):
        gc.collect()
        atoms = Atoms(symbols=symbols, positions=positions.copy())
        stages = []
        if args.method == 'cascade':
            stages.append(stage(atoms, 'skala', True))
            stages.append(stage(atoms, 'mixed', True))
        else:
            stages.append(stage(atoms, args.method, args.optimize))
        samples.append(
            {
                'index': index,
                'first_evaluation': index == 0,
                'seconds': sum(item['seconds'] for item in stages),
                'stages': stages,
            }
        )
        atoms.calc = None
    geometry_check = None
    if args.check_forces:
        atoms = Atoms(symbols=symbols, positions=positions.copy())
        atoms.calc = calculator(atoms, 'skala', [])
        atoms.get_forces()
        atoms.positions[-1, 0] += 0.02
        reused_forces = atoms.get_forces()
        reused_energy = atoms.get_potential_energy()
        fresh = atoms.copy()
        fresh.calc = calculator(fresh, 'skala', [])
        fresh_forces = fresh.get_forces()
        fresh_energy = fresh.get_potential_energy()
        center = atoms.positions.copy()
        delta = 0.001
        energies = []
        converged = [bool(atoms.calc._ks.base.converged), bool(fresh.calc._ks.base.converged)]
        for sign in (1, -1):
            displaced = center.copy()
            displaced[-1, 0] += sign * delta
            atoms.positions = displaced
            energies.append(atoms.get_potential_energy())
            converged.append(bool(atoms.calc._ks.base.converged))
        finite_difference = -(energies[0] - energies[1]) / (2 * delta)
        energy_error = abs(reused_energy - fresh_energy)
        force_error = float(np.max(np.abs(reused_forces - fresh_forces)))
        fd_error = abs(finite_difference - reused_forces[-1, 0])
        geometry_check = {
            'reused_vs_fresh_energy_error_ev': energy_error,
            'reused_vs_fresh_max_force_error_ev_angstrom': force_error,
            'finite_difference_force_ev_angstrom': float(finite_difference),
            'analytic_force_ev_angstrom': float(reused_forces[-1, 0]),
            'finite_difference_error_ev_angstrom': float(fd_error),
            'finite_difference_step_angstrom': delta,
            'all_scf_converged': all(converged),
            'passed': bool(all(converged) and energy_error < 1e-4 and force_error < 5e-4 and fd_error < 1e-3),
        }
        atoms.calc = fresh.calc = None
    packages = ['pyscf', 'gpu4pyscf-cuda13x', 'cupy-cuda13x', 'numpy', 'scipy', 'ase', 'cuda-toolkit']
    if functional is not None:
        packages += ['skala', 'torch']
    report = {
        'method': args.method,
        'case': args.case,
        'synthetic_geometry_angstrom': positions.tolist(),
        'symbols': symbols,
        'charge': charge,
        'multiplicity': multiplicity,
        'settings': settings,
        'settings_sha256': settings_hash,
        'auxbasis': 'def2-universal-jkfit',
        'skala_model': functional_info,
        'skala_grid_level': args.skala_grid_level,
        'skala_explicit_dispersion_reset': args.refresh_skala_dispersion,
        'geometry_and_force_check': geometry_check,
        'optimize': args.optimize,
        'fmax': args.fmax,
        'steps': args.steps,
        'gpu_uuid': physical[0].strip(),
        'gpu_name': physical[2].strip(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'openmp_threads': lib.num_threads(),
        'threadpools': threadpool_info(),
        'versions': {name: version(name) for name in packages},
        'gpu4pyscf_source': _source_provenance(gpu4pyscf, numint.libgdft),
        'extensions_sha256': extensions,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'imports_and_model_seconds': imports_seconds,
        'samples': samples,
        'warm_median_seconds': float(np.median([item['seconds'] for item in samples[1:]]))
        if len(samples) > 1
        else None,
        'notes': 'Different XC functionals and native grid representations. Skala includes its published D3; '
        'wb97m-v includes VV10. No energy agreement claim across methods. Full grid and auxiliary response '
        'for wb97m-v forces; Skala includes its explicit grid response. Single-point samples have fresh '
        'calculators; optimizations reuse density within each stage. No Hessian/TS validation.',
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(
        json.dumps(
            {
                'output': str(args.output),
                'warm_median_seconds': report['warm_median_seconds'],
                'sample_seconds': [item['seconds'] for item in samples],
            }
        ),
        flush=True,
    )
    # Release calculator cycles and allocator blocks while both CUDA runtimes are still alive.
    functional = None
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if args.method in ('skala', 'cascade'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
