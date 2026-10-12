"""Compare opt-in TF32 early-grid contractions with verified FP32 molecular SCF."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from _extensions import load_extensions
from gpu_performance import CASES, _resources, _source_provenance


def _micro(cp, np, trace_library, artifact):
    from gpu4pyscf.dft._mixed_grid import _gemm, _GridPrecision, _make_contract

    rng = np.random.default_rng(812)
    rows = []
    for m, n, k in ((17, 31, 83), (32, 32, 512), (276, 276, 4096), (276, 276, 16384)):
        a, b = [cp.asarray(rng.normal(size=shape)) for shape in ((m, k), (n, k))]
        initial = cp.asarray(rng.normal(size=(m, n)))
        reference = _gemm(a, b, alpha=0.2, beta=-1.3, out=initial.copy())
        modes = {}
        for tf32 in (False, True):
            controller = _GridPrecision(True, tf32=tf32)
            contract = _make_contract(None, controller)
            actual = contract('ig,jg->ij', a, b, alpha=0.2, beta=-1.3, out=initial.copy())
            cp.cuda.get_current_stream().synchronize()
            max_error = float(cp.max(cp.abs(actual - reference)))
            relative_frobenius = float(cp.linalg.norm(actual - reference) / cp.linalg.norm(reference))
            if not bool(cp.isfinite(actual).all()) or relative_frobenius > (1e-3 if tf32 else 1e-5):
                raise RuntimeError('Grid contraction failed matrix accuracy check')
            # Check accumulation and native float64 restoration at awkward dimensions.
            if (m, n, k) == (17, 31, 83):
                controller.use_float32 = False
                recovered = contract('ig,jg->ij', a, b, alpha=0.2, beta=-1.3, out=initial.copy())
                cp.testing.assert_array_equal(recovered, reference)
                controller.use_float32 = True
            modes['tf32' if tf32 else 'fp32'] = {
                'contract': contract, 'out': cp.empty((m, n)),
                'max_absolute_error': max_error, 'relative_frobenius_error': relative_frobenius,
                'wall_ms': [], 'device_ms': [],
            }
        for repeat in range(6):
            for mode in (('fp32', 'tf32') if repeat % 2 == 0 else ('tf32', 'fp32')):
                item = modes[mode]
                cp.cuda.get_current_stream().synchronize()
                start, end = cp.cuda.Event(), cp.cuda.Event()
                started = time.perf_counter()
                start.record()
                for _ in range(50):
                    item['contract']('ig,jg->ij', a, b, out=item['out'])
                end.record()
                end.synchronize()
                item['wall_ms'].append((time.perf_counter() - started) * 1000 / 50)
                item['device_ms'].append(cp.cuda.get_elapsed_time(start, end) / 50)
        row = {'shape': [m, n, k]}
        for mode, item in modes.items():
            row[mode] = {key: value for key, value in item.items() if key not in ('contract', 'out')}
            row[mode]['median_wall_ms'] = statistics.median(item['wall_ms'])
            row[mode]['median_device_ms'] = statistics.median(item['device_ms'])
        row['wall_speedup'] = row['fp32']['median_wall_ms'] / row['tf32']['median_wall_ms']
        rows.append(row)
    trace = None
    if trace_library:
        library = ctypes.CDLL(str(trace_library))
        library.trace_stop.argtypes = [ctypes.c_char_p]
        a32, b32 = [cp.asarray(rng.normal(size=(276, 4096)), dtype=cp.float32) for _ in range(2)]
        trace = {}
        for tf32 in (False, True):
            mode = 'tf32' if tf32 else 'fp32'
            # Warm the exact cuBLAS operation before collecting names; never time tracing.
            _gemm(a32, b32, tf32=tf32)
            cp.cuda.get_current_stream().synchronize()
            status = library.trace_start()
            if status:
                trace[mode] = {'error': f'CUPTI start status {status}'}
                continue
            _gemm(a32, b32, tf32=tf32)
            cp.cuda.get_current_stream().synchronize()
            path = artifact / f'{mode}-kernels.tsv'
            status = library.trace_stop(os.fsencode(path))
            trace[mode] = {'status': status, 'records': path.read_text().splitlines() if path.exists() else []}
    return {'matrices': rows, 'kernel_trace': trace}


def _scf(cp, np, repeats, with_forces, artifact, tf32_cycles):
    from gpu4pyscf import dft
    from gpu4pyscf.dft import _mixed_grid, mixed_precision
    from gpu4pyscf.scf import hf
    from pyscf import gto

    hf.overlap_zero_eigenvalue_threshold = 1e-10
    records = []

    def calculate(case, mode, run):
        atom, charge, multiplicity = CASES[case]
        mol = gto.M(atom=atom, basis='def2-tzvpd', charge=charge, spin=multiplicity - 1, verbose=0)
        mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit',
        )
        mf.grids.level, mf.nlcgrids.level = 5, 3
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-8, 1e-4, 200
        mf.chkfile = None
        controllers = []
        grid_type = _mixed_grid._GridPrecision
        def factory(use_float32):
            grid = grid_type(use_float32, tf32=mode == 'tf32')
            controllers.append(grid)
            return grid
        def progress(env):
            cycle = int(env['cycle']) + 1
            if mode == 'tf32' and cycle >= tf32_cycles:
                for grid in controllers:
                    grid.tf32 = False
            print(json.dumps({
                'case': case, 'mode': mode, 'run': run, 'cycle': cycle,
                'energy_hartree': float(env['e_tot']), 'orbital_gradient': float(env['norm_gorb']),
            }), flush=True)
        mf.callback = progress
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        if mode == 'float64':
            energy = float(mf.kernel())
            check = mixed_precision._full_check(mf, energy)
            if not check['passed']:
                raise RuntimeError('Native float64 reference failed physical Fock verification')
            energy = check['float64_energy_hartree']
            info = {'accepted': True, 'final_verification': check, 'fallback_scf': False}
        else:
            # This factory patch is scoped to this isolated benchmark process only.
            with patch.object(_mixed_grid, '_GridPrecision', factory):
                energy = mixed_precision.run_verified_scf(mf)
            info = mf.mixed_precision_info
        cp.cuda.get_current_stream().synchronize()
        scf_seconds = time.perf_counter() - started
        force_seconds, forces = None, None
        if with_forces:
            started = time.perf_counter()
            gradient = mf.Gradients()
            gradient.grid_response = gradient.auxbasis_response = True
            forces = -gradient.kernel()
            if hasattr(forces, 'get'):
                forces = forces.get()
            forces = np.asarray(forces)
            cp.cuda.get_current_stream().synchronize()
            force_seconds = time.perf_counter() - started
        row = {
            'case': case, 'mode': mode, 'run': run, 'energy_hartree': energy,
            'scf_seconds': scf_seconds, 'force_seconds': force_seconds,
            'forces_hartree_bohr': None if forces is None else forces.tolist(),
            'cycles': int(mf.cycles), 'precision_info': info,
            'effective_orbitals': int(mf.mo_coeff.shape[-1]),
            'grid_points': len(mf.grids.weights), 'nlc_grid_points': len(mf.nlcgrids.weights),
        }
        records.append(row)
        (artifact / f'{case}-{mode}-{run}.json').write_text(json.dumps(row, indent=2) + '\n')
        print('RESULT ' + json.dumps({k: v for k, v in row.items() if k != 'forces_hartree_bohr'}), flush=True)
        cp.get_default_memory_pool().free_all_blocks()

    # Small synthetic closed/open-shell checks also warm kernels before benzene timing.
    for case in ('water', 'water-cation'):
        for mode in ('float64', 'fp32', 'tf32'):
            calculate(case, mode, 0)
    calculate('benzene', 'float64', 0)
    for run in range(repeats):
        for mode in (('fp32', 'tf32') if run % 2 == 0 else ('tf32', 'fp32')):
            calculate('benzene', mode, run)
    summary = {}
    for case in ('water', 'water-cation', 'benzene'):
        rows = [r for r in records if r['case'] == case]
        reference = next(r for r in rows if r['mode'] == 'float64')
        summary[case] = {}
        for mode in ('fp32', 'tf32'):
            group = [r for r in rows if r['mode'] == mode]
            energy_error = max(abs(r['energy_hartree'] - reference['energy_hartree']) for r in group)
            force_error = max(float(np.max(np.abs(np.asarray(r['forces_hartree_bohr']) -
                                np.asarray(reference['forces_hartree_bohr'])))) for r in group) if with_forces else None
            if energy_error > 1e-8 or (force_error is not None and force_error > 1e-5):
                raise RuntimeError('Accepted result differs from native reference beyond comparison thresholds')
            summary[case][mode] = {
                'scf_seconds': [r['scf_seconds'] for r in group],
                'median_scf_seconds': statistics.median(r['scf_seconds'] for r in group),
                'max_energy_error_hartree': energy_error, 'max_force_error_hartree_bohr': force_error,
                'fallbacks': sum(r['precision_info']['fallback_scf'] for r in group),
                'cycles': [r['cycles'] for r in group],
                'median_energy_force_seconds': statistics.median(r['scf_seconds'] + r['force_seconds'] for r in group)
                if with_forces else None,
            }
        summary[case]['scf_speedup'] = summary[case]['fp32']['median_scf_seconds'] / summary[case]['tf32']['median_scf_seconds']
    return {'summary': summary, 'runs': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--mode', choices=('micro', 'scf', 'all'), default='all')
    parser.add_argument('--forces', action='store_true')
    parser.add_argument('--trace-library', type=Path)
    parser.add_argument('--tf32-cycles', type=int, default=5, help='TF32 grid iterations before ordinary FP32')
    args = parser.parse_args()
    if not 1 <= args.tf32_cycles <= 5:
        parser.error('tf32-cycles must be between one and five')
    if args.repeats < 1 or args.output.exists():
        parser.error('Use a positive repeat count and a new output path')
    artifact = args.output.with_suffix('')
    if artifact.exists():
        parser.error('The artifact directory already exists; preserve it and choose a new output path')
    artifact.mkdir(parents=True, exist_ok=False)
    _resources([4, 5, 6, 7])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('NVIDIA_TF32_OVERRIDE') == '0':
        raise RuntimeError('Select physical GPU 1 and do not disable TF32 externally')
    import cupy as cp
    import gpu4pyscf
    import numpy as np
    from gpu4pyscf.dft import numint
    from pyscf import lib
    from threadpoolctl import threadpool_info

    lib.num_threads(4)
    if lib.num_threads() != 4 or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Expected four CPU threads and one visible GPU')
    for pool in threadpool_info():
        if pool['num_threads'] > 4:
            raise RuntimeError('A numerical pool exceeds the four-CPU allocation')
    physical = subprocess.check_output([
        'nvidia-smi', '-i', '1', '--query-gpu=uuid,pci.bus_id,name,driver_version', '--format=csv,noheader',
    ], text=True).strip().split(',')
    def pci(value):
        domain, location = value.strip().split(':', 1)
        return int(domain, 16), location.lower()
    if pci(physical[1]) != pci(cp.cuda.Device(0).pci_bus_id):
        raise RuntimeError('Visible CUDA device differs from physical GPU 1')
    report = {
        'method': {'functional': 'wb97m-v', 'basis': 'def2-tzvpd', 'auxbasis': 'def2-universal-jkfit',
                   'grid_level': 5, 'nlc_grid_level': 3, 'conv_tol': 1e-8, 'conv_tol_grad': 1e-4,
                   'max_cycle': 200, 'overlap_cutoff': 1e-10, 'solvent': None},
        'gpu': physical, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threadpools': threadpool_info(),
        'versions': {name: version(name) for name in ('pyscf', 'gpu4pyscf-cuda13x', 'cupy-cuda13x', 'numpy')},
        'numerical_provenance': _source_provenance(gpu4pyscf, numint.libgdft),
        'extension_hashes': load_extensions(args.extension_root),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
        'initial_density': 'Fresh native initial guess; no density reused between runs',
        'tf32_grid_cycles': args.tf32_cycles,
        'policy': 'TF32 grid schedule ends within five iterations; native float64 checks, recovery and derivatives',
    }
    if args.mode in ('micro', 'all'):
        report['micro'] = _micro(cp, np, args.trace_library, artifact)
        (artifact / 'micro.json').write_text(json.dumps(report['micro'], indent=2) + '\n')
    if args.mode in ('scf', 'all'):
        report['scf'] = _scf(cp, np, args.repeats, args.forces, artifact, args.tf32_cycles)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('SUMMARY ' + json.dumps({key: value['summary'] for key, value in report.items() if key == 'scf'}), flush=True)


if __name__ == '__main__':
    main()
