"""Benchmark fresh fixed-method SCFs under a four-CPU, one-GPU limit."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

BENZENE = """
C 0 1.396792 0; C 1.209657 .698396 0; C 1.209657 -.698396 0;
C 0 -1.396792 0; C -1.209657 -.698396 0; C -1.209657 .698396 0;
H 0 2.484212 0; H 2.151390 1.242106 0; H 2.151390 -1.242106 0;
H 0 -2.484212 0; H -2.151390 -1.242106 0; H -2.151390 1.242106 0
"""
CASES = {
    'benzene': (BENZENE, 0, 1),
    'water': ('O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043', 0, 1),
    'water-cation': ('O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043', 1, 2),
}
THREAD_VARIABLES = (
    'OMP_NUM_THREADS',
    'OMP_THREAD_LIMIT',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)


def _settings(path=None):
    path = path or Path(__file__).with_name('protocol.json')
    source = path.read_bytes()
    settings = json.loads(source)
    keys = {
        'protocol_version',
        'xc',
        'basis',
        'ecp',
        'auxiliary_basis',
        'density_fitting',
        'grid_level',
        'nlc_grid_level',
        'conv_tol',
        'max_cycle',
    }
    if not isinstance(settings, dict) or settings.keys() != keys:
        raise ValueError('The benchmark protocol must contain exactly the supported settings')
    if type(settings['protocol_version']) is not int or settings['protocol_version'] != 1:
        raise ValueError('Unsupported benchmark protocol version')
    if settings['ecp'] != 'auto-def2':
        raise ValueError('Review the configured ECP before running these light-element cases')
    for name in ('xc', 'basis'):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f'{name} must be a nonempty name')
    if type(settings['density_fitting']) is not bool or not settings['density_fitting']:
        raise ValueError('This benchmark requires density fitting')
    if settings['auxiliary_basis'] is not None and (
        not isinstance(settings['auxiliary_basis'], str) or not settings['auxiliary_basis'].strip()
    ):
        raise ValueError('auxiliary_basis must be a basis name or null')
    for name in ('grid_level', 'nlc_grid_level'):
        if type(settings[name]) is not int or not 0 <= settings[name] <= 9:
            raise ValueError(f'{name} must be an integer from zero to nine')
    if type(settings['conv_tol']) not in (int, float) or not 0 < settings['conv_tol'] <= 1e-3:
        raise ValueError('conv_tol must be positive and at most 1e-3 Hartree')
    if type(settings['max_cycle']) is not int or not 1 <= settings['max_cycle'] <= 1000:
        raise ValueError('max_cycle must be an integer from one to 1000')
    return settings, hashlib.sha256(source).hexdigest()


def _checkout_revision(root):
    if not (root / '.git').exists():
        return None
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def _source_provenance(module, native_library):
    root = Path(module.__file__).resolve().parents[1]
    sources = (
        'gpu4pyscf/dft/numint.py',
        'gpu4pyscf/dft/libxc.py',
        'gpu4pyscf/dft/gen_grid.py',
        'gpu4pyscf/scf/hf.py',
        'gpu4pyscf/df/df_jk.py',
    )
    native_path = Path(native_library._name).resolve()
    return {
        'module': str(Path(module.__file__).resolve()),
        'module_version': module.__version__,
        'checkout_revision': _checkout_revision(root),
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        'native_library': str(native_path),
        'native_library_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
    }


def _resources(cpus):
    if len(cpus) != 4 or len(set(cpus)) != 4:
        raise ValueError('Exactly four distinct logical CPUs are required')
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('This benchmark requires Linux CPU affinity')
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('Requested CPUs are outside the available affinity')
    os.sched_setaffinity(0, cpus)
    for name in THREAD_VARIABLES:
        os.environ[name] = '4'
    os.environ.update(OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1')


@contextmanager
def _timed_method(obj, name, timings, synchronize):
    original = getattr(obj, name)
    previous = obj.__dict__.get(name)
    had_attribute = name in obj.__dict__

    def call(*args, **kwargs):
        synchronize()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            synchronize()
            timings[name].append(time.perf_counter() - started)

    setattr(obj, name, call)
    try:
        yield
    finally:
        if had_attribute:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def _calculate(args):
    process_started = time.perf_counter()
    _resources(args.cpus)
    settings, settings_hash = _settings()
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib
    from threadpoolctl import threadpool_info

    lib.num_threads(4)
    if lib.num_threads() != 4:
        raise RuntimeError('PySCF must provide four OpenMP threads')
    gpu = None
    synchronize = lambda: None
    if args.backend == 'gpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise RuntimeError('Set CUDA_VISIBLE_DEVICES=1 to select physical GPU 1')
        import cupy

        import gpu4pyscf

        if args.require_gpu_source is not None and not Path(gpu4pyscf.__file__).resolve().is_relative_to(
            args.require_gpu_source.resolve()
        ):
            raise RuntimeError('Imported GPU4PySCF does not come from the required source checkout')
        if cupy.cuda.runtime.getDeviceCount() != 1:
            raise RuntimeError('Exactly one CUDA device must be visible')
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        physical = (
            subprocess.check_output(
                [
                    'nvidia-smi',
                    '-i',
                    '1',
                    '--query-gpu=uuid,pci.bus_id,driver_version',
                    '--format=csv,noheader,nounits',
                ],
                text=True,
            )
            .strip()
            .split(',')
        )
        actual_pci = cupy.cuda.Device(0).pci_bus_id
        normalize_pci = lambda value: (int(value.split(':')[0], 16), value.split(':', 1)[1].lower())
        if normalize_pci(physical[1].strip()) != normalize_pci(actual_pci):
            raise RuntimeError('Visible CUDA device is not physical GPU 1')
        gpu = {
            'name': properties['name'].decode(),
            'pci_bus_id': actual_pci,
            'uuid': physical[0].strip(),
            'driver_version': physical[2].strip(),
        }
        synchronize = cupy.cuda.get_current_stream().synchronize
    imports_seconds = time.perf_counter() - process_started
    started = time.perf_counter()
    atom, charge, multiplicity = CASES[args.case]
    mol = gto.M(atom=atom, basis=settings['basis'], charge=charge, spin=multiplicity - 1, verbose=4)
    mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
    mf.xc = settings['xc']
    if settings['density_fitting']:
        mf = mf.density_fit(auxbasis=settings['auxiliary_basis'])
    mf.grids.level = settings['grid_level']
    mf.nlcgrids.level = settings['nlc_grid_level']
    mf.conv_tol = settings['conv_tol']
    mf.conv_tol_grad = settings['conv_tol'] ** 0.5
    mf.max_cycle = settings['max_cycle']
    mf.chkfile = None
    mf.init_guess = args.guess
    cpu_mf = mf
    if args.backend == 'gpu':
        mf = mf.to_gpu()
        from gpu4pyscf.dft import numint
        from gpu4pyscf.scf import hf

        if args.overlap_cutoff is not None:
            hf.overlap_zero_eigenvalue_threshold = args.overlap_cutoff
    else:
        from pyscf.dft import numint
    records = []
    timings = defaultdict(list)
    scf_started = time.perf_counter()
    cpu_started = time.process_time()
    experiment = None
    grid_experiment = None

    def progress(env):
        synchronize()
        entry = {
            'cycle': int(env['cycle']) + 1,
            'energy_hartree': float(env['e_tot']),
            'orbital_gradient_norm': float(env['norm_gorb']),
            'elapsed_scf_seconds': time.perf_counter() - scf_started,
        }
        if args.vv10 == 'mixed':
            entry['vv10_precision'] = experiment.precisions[-1]
            if (abs(env['e_tot'] - env['last_hf_e']) < 1e-5 and entry['orbital_gradient_norm'] < 1e-3) or len(
                records
            ) >= 19:
                experiment.use_float32 = False
        if args.grid_contract == 'mixed':
            entry['grid_precision'] = 'float32' if grid_experiment.use_float32 else 'float64'
            if int(env['cycle']) >= 4:
                grid_experiment.use_float32 = False
        records.append(entry)

    mf.callback = progress
    profiler = cProfile.Profile()
    captured = {}
    verification = None

    def restart(dm):
        mf.max_cycle = settings['max_cycle'] - len(records)
        if mf.max_cycle < 1:
            raise RuntimeError('The total benchmark SCF cycle budget is exhausted')
        return float(mf.kernel(dm0=dm))

    with ExitStack() as stack:
        if args.grid_tasks == 'inline':
            from grid_experiment import inline_grid_tasks

            stack.enter_context(inline_grid_tasks())
        if args.grid_contract != 'baseline':
            from grid_experiment import install as install_blas

            grid_experiment = stack.enter_context(install_blas(args.grid_contract))
        if args.capture_vv10:
            original_vv10 = numint._vv10nlc

            def capture(rho, coords, weights, pars):
                captured.update(rho=rho, coords=coords, weights=weights, pars=pars)
                return original_vv10(rho, coords, weights, pars)

            numint._vv10nlc = capture
            stack.callback(setattr, numint, '_vv10nlc', original_vv10)
        if args.vv10 != 'baseline':
            from vv10_experiment import install

            experiment = stack.enter_context(install(args.vv10, args.vv10_block))
        if args.profile:
            for obj, name in (
                (mf, 'get_init_guess'),
                (mf, 'get_jk'),
                (mf, 'eig'),
                (mf._numint, 'nr_rks' if multiplicity == 1 else 'nr_uks'),
                (mf._numint, 'nr_nlc_vxc'),
                (numint, '_vv10nlc'),
                (mf._numint, 'eval_xc_eff'),
            ):
                stack.enter_context(_timed_method(obj, name, timings, synchronize))
            profiler.enable()
        dm0 = cpu_mf.get_init_guess() if args.backend == 'gpu' and args.guess != 'minao' else None
        energy = float(mf.kernel(dm0=dm0))
        if args.grid_contract == 'mixed' and grid_experiment.last_precision == 'float32':
            grid_experiment.use_float32 = False
            if args.vv10 == 'mixed':
                experiment.use_float32 = False
            energy = restart(mf.make_rdm1())
        if args.vv10 == 'mixed' and experiment.precisions[-1] != 'float64':
            experiment.use_float32 = False
            energy = restart(mf.make_rdm1())
        if args.vv10 == 'verify':
            experiment.use_float32 = False
            dm = mf.make_rdm1()
            hcore, overlap = mf.get_hcore(), mf.get_ovlp()
            veff = mf.get_veff(mol, dm)
            fock = mf.get_fock(hcore, overlap, veff, dm)
            full_energy = float(mf.energy_tot(dm, hcore, veff))
            gradient = float(cupy.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
            verification = {
                'float32_energy_hartree': energy,
                'float64_energy_hartree': full_energy,
                'energy_change_hartree': full_energy - energy,
                'float64_orbital_gradient_norm': gradient,
                'passed': abs(full_energy - energy) < mf.conv_tol and gradient < mf.conv_tol_grad,
            }
            verification['fallback_scf'] = not verification['passed']
            energy = full_energy
            if not verification['passed']:
                energy = restart(dm)
            verification['accepted_float64_gradient_norm'] = (
                gradient if verification['passed'] else records[-1]['orbital_gradient_norm']
            )
        synchronize()
        if args.profile:
            profiler.disable()
    scf_seconds = time.perf_counter() - scf_started
    cpu_seconds = time.process_time() - cpu_started
    total_seconds = time.perf_counter() - started
    if not mf.converged or not np.isfinite(energy):
        raise RuntimeError('Benchmark SCF did not converge')
    if args.profile:
        profiler.dump_stats(str(args.output.with_suffix('.prof')))
    dm = mf.make_rdm1()
    if args.density_output:
        np.save(args.density_output, dm.get() if args.backend == 'gpu' else dm)
    if captured:
        np.savez(
            args.capture_vv10,
            **{key: value.get() if hasattr(value, 'get') else value for key, value in captured.items()},
        )
    packages = ['pyscf', 'numpy', 'scipy']
    if args.backend == 'gpu':
        packages += ['gpu4pyscf-cuda13x', 'cupy-cuda13x', 'cutensor-cu13', 'cuda-toolkit']
    return {
        'benchmark_revision': _checkout_revision(Path(__file__).resolve().parents[2]),
        'benchmark_source_sha256': {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('gpu_performance.py', 'vv10_experiment.py', 'grid_experiment.py')
        },
        'gpu4pyscf_source': _source_provenance(gpu4pyscf, numint.libgdft) if args.backend == 'gpu' else None,
        'timestamp_utc': datetime.now(UTC).isoformat(),
        'backend': args.backend,
        'variant': args.vv10,
        'overlap_cutoff': args.overlap_cutoff,
        'vv10_block': args.vv10_block,
        'grid_contract': args.grid_contract,
        'grid_contraction_calls': {
            'float32': grid_experiment.float32_calls,
            'float64': grid_experiment.float64_calls,
        }
        if grid_experiment
        else None,
        'grid_tasks': args.grid_tasks,
        'pair_seconds': experiment.pair_seconds if experiment else None,
        'pair_precisions': experiment.precisions if args.vv10 in ('mixed', 'verify') else None,
        'verification': verification,
        'profiled': args.profile,
        'settings': settings,
        'settings_source_sha256': settings_hash,
        'case': args.case,
        'initial_guess': args.guess,
        'geometry_angstrom': atom.strip(),
        'charge': charge,
        'multiplicity': multiplicity,
        'effective_orbitals': int(mf.mo_coeff.shape[-1]),
        'last_cycle_orbital_gradient_norm': records[-1]['orbital_gradient_norm'],
        'platform': platform.platform(),
        'python': platform.python_version(),
        'versions': {name: version(name) for name in packages},
        'gpu': gpu,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'openmp_threads': lib.num_threads(),
        'thread_environment': {name: os.environ[name] for name in THREAD_VARIABLES},
        'threadpools': threadpool_info(),
        'pyscf_module': pyscf.__file__,
        'converged': bool(mf.converged),
        'energy_hartree': energy,
        'imports_seconds': imports_seconds,
        'scf_seconds': scf_seconds,
        'total_calculation_seconds': total_seconds,
        'process_seconds': time.perf_counter() - process_started,
        'process_cpu_seconds': cpu_seconds,
        'average_cpu_threads': cpu_seconds / scf_seconds,
        'cycles': len(records),
        'cycle_history': records,
        'nao': mol.nao,
        'naux': mf.with_df.auxmol.nao,
        'resolved_auxbasis': mf.with_df.auxbasis,
        'grid_points': len(mf.grids.weights),
        'nlc_grid_points': len(mf.nlcgrids.weights),
        'includes_nonlocal_correlation': bool(mf.do_nlc()),
        'profile_seconds': {key: {'total': sum(values), 'calls': values} for key, values in timings.items()},
        'timing_notes': 'Fresh process and mean-field; native benchmark SCF convergence '
        'behavior, guesses, grids and integrals. Imports timed separately. '
        'Profile runs synchronize method boundaries and are not speed comparisons.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--require-gpu-source',
        type=Path,
        help='Reject a GPU4PySCF import outside this checkout (use with PYTHONPATH)',
    )
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--case', choices=tuple(CASES), default='benzene')
    parser.add_argument('--guess', choices=('minao', 'huckel', 'mod_huckel', 'sap'), default='minao')
    parser.add_argument('--cpus', type=lambda value: [int(cpu) for cpu in value.split(',')], default=[4, 5, 6, 7])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--density-output', type=Path)
    parser.add_argument('--capture-vv10', type=Path)
    parser.add_argument('--overlap-cutoff', type=float)
    parser.add_argument(
        '--vv10',
        choices=('baseline', 'compiled', 'refined', 'float32', 'mixed', 'verify'),
        default='baseline',
    )
    parser.add_argument('--vv10-block', choices=(64, 128, 256), type=int, default=128)
    parser.add_argument('--grid-contract', choices=('baseline', 'blas', 'mixed'), default='baseline')
    parser.add_argument('--grid-tasks', choices=('baseline', 'inline'), default='baseline')
    args = parser.parse_args()
    for path, suffix in (
        (args.output, '.json'),
        (args.density_output, '.npy'),
        (args.capture_vv10, '.npz'),
    ):
        if path is not None and path.suffix != suffix:
            parser.error(f'Expected a {suffix} artifact path: {path}')
    if args.output.exists():
        parser.error('Use a new output path; benchmark evidence must not be overwritten')
    for path in (
        args.density_output,
        args.capture_vv10,
        args.output.with_suffix('.prof') if args.profile else None,
    ):
        if path is not None and path.exists():
            parser.error(f'Use a new artifact path: {path}')
    if args.overlap_cutoff is not None and not 0 < args.overlap_cutoff < 1:
        parser.error('Overlap cutoff must be positive and below one')
    if args.backend == 'cpu' and (
        args.require_gpu_source is not None
        or args.vv10 != 'baseline'
        or args.overlap_cutoff is not None
        or args.capture_vv10
        or args.grid_contract != 'baseline'
        or args.grid_tasks != 'baseline'
    ):
        parser.error('VV10 experiments and overlap overrides require the GPU backend')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = _calculate(args)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    'backend',
                    'energy_hartree',
                    'scf_seconds',
                    'cycles',
                    'profile_seconds',
                )
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
