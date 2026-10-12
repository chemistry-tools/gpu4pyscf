"""Compare accuracy-preserving cuBLAS FP64 emulation with the existing mixed SCF."""

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

from _extensions import load_extensions
from gpu_performance import CASES, _resources, _source_provenance
from tf32_benchmark import _scf


def _micro(cp, np, artifact, trace_library):
    from gpu4pyscf.dft._mixed_grid import _gemm

    rng = np.random.default_rng(413)
    results = []
    for m, n, k in ((17, 31, 83), (276, 276, 4096), (276, 276, 16384), (1024, 1024, 1024)):
        a, b = [cp.asarray(rng.normal(size=shape)) for shape in ((m, k), (n, k))]
        initial = cp.asarray(rng.normal(size=(m, n)))
        reference = _gemm(a, b, alpha=0.2, beta=-1.3, out=initial.copy())
        row = {'shape': [m, n, k], 'native': {'times_ms': []}, 'emulated': {'times_ms': []}}
        outputs = {}
        for mode in ('native', 'emulated'):
            actual = _gemm(a, b, alpha=0.2, beta=-1.3, out=initial.copy(), fp64_emulation=mode == 'emulated')
            cp.cuda.get_current_stream().synchronize()
            error = float(cp.linalg.norm(actual - reference) / cp.linalg.norm(reference))
            if not bool(cp.isfinite(actual).all()) or error > 1e-12:
                raise RuntimeError('FP64 matrix accuracy check failed')
            row[mode]['relative_frobenius_error'] = error
            row[mode]['max_absolute_error'] = float(cp.max(cp.abs(actual - reference)))
            outputs[mode] = cp.empty((m, n), dtype=cp.float64)
        for repeat in range(6):
            for mode in (('native', 'emulated') if repeat % 2 == 0 else ('emulated', 'native')):
                cp.cuda.get_current_stream().synchronize()
                started = time.perf_counter()
                for _ in range(20):
                    _gemm(a, b, out=outputs[mode], fp64_emulation=mode == 'emulated')
                cp.cuda.get_current_stream().synchronize()
                row[mode]['times_ms'].append((time.perf_counter() - started) * 1000 / 20)
        for mode in ('native', 'emulated'):
            row[mode]['median_ms'] = statistics.median(row[mode]['times_ms'])
        row['speedup'] = row['native']['median_ms'] / row['emulated']['median_ms']
        if trace_library:
            library = ctypes.CDLL(str(trace_library))
            library.trace_stop.argtypes = [ctypes.c_char_p]
            for mode in ('native', 'emulated'):
                status = library.trace_start()
                if status:
                    raise RuntimeError(f'CUPTI start failed: {status}')
                _gemm(a, b, out=outputs[mode], fp64_emulation=mode == 'emulated')
                cp.cuda.get_current_stream().synchronize()
                path = artifact / f'{m}-{n}-{k}-{mode}.tsv'
                status = library.trace_stop(os.fsencode(path))
                if status:
                    raise RuntimeError(f'CUPTI stop failed: {status}')
                row[mode]['kernel_trace'] = path.read_text().splitlines()
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--mode', choices=('micro', 'scf', 'all'), default='all')
    parser.add_argument('--forces', action='store_true')
    parser.add_argument('--trace-library', type=Path)
    args = parser.parse_args()
    artifact = args.output.with_suffix('')
    if args.repeats < 1 or args.output.exists() or artifact.exists():
        parser.error('Use a positive repeat count and a new report and artifact directory')
    _resources([4, 5, 6, 7])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Select physical GPU 1')
    if os.environ.get('CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT') or os.environ.get('CUBLAS_EMULATE_DOUBLE_PRECISION'):
        raise RuntimeError('Use scoped emulation with automatic precision, not global/fixed-mantissa overrides')
    import cupy as cp
    import gpu4pyscf
    import numpy as np
    from cupy_backends.cuda.libs import cublas
    from gpu4pyscf.dft import numint
    from pyscf import lib
    from threadpoolctl import threadpool_info

    lib.num_threads(4)
    if lib.num_threads() != 4 or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Expected four CPU threads and one visible GPU')
    for pool in threadpool_info():
        if pool['num_threads'] > 4:
            raise RuntimeError('A numerical pool exceeds four threads')
    physical = subprocess.check_output([
        'nvidia-smi', '-i', '1', '--query-gpu=uuid,pci.bus_id,name,driver_version', '--format=csv,noheader',
    ], text=True).strip().split(',')
    def pci(value):
        domain, location = value.strip().split(':', 1)
        return int(domain, 16), location.lower()
    if pci(physical[1]) != pci(cp.cuda.Device(0).pci_bus_id):
        raise RuntimeError('Visible CUDA device differs from physical GPU 1')
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if 'libcublas.so.' in line})
    if len(paths) != 1:
        raise RuntimeError('Cannot identify the loaded cuBLAS library')
    library = ctypes.CDLL(paths[0])
    handle = cp.cuda.device.get_cublas_handle()
    settings = {}
    for name in ('cublasGetEmulationStrategy', 'cublasGetFixedPointEmulationMantissaControl',
                 'cublasGetFixedPointEmulationMaxMantissaBitCount', 'cublasGetFixedPointEmulationMantissaBitOffset'):
        fn = getattr(library, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        value = ctypes.c_int()
        if fn(handle, ctypes.byref(value)):
            raise RuntimeError(f'Cannot query {name}')
        settings[name] = value.value
    if settings['cublasGetFixedPointEmulationMantissaControl'] != 0 or settings['cublasGetFixedPointEmulationMantissaBitOffset']:
        raise RuntimeError('Automatic native-accuracy mantissa control is required')
    artifact.mkdir(parents=True)
    report = {
        'method': {'functional': 'wb97m-v', 'basis': 'def2-tzvpd', 'auxbasis': 'def2-universal-jkfit',
                   'grid_level': 5, 'nlc_grid_level': 3, 'conv_tol': 1e-8, 'conv_tol_grad': 1e-4,
                   'max_cycle': 200, 'overlap_cutoff': 1e-10, 'solvent': None},
        'gpu': physical, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threadpools': threadpool_info(),
        'versions': {name: version(name) for name in ('pyscf', 'gpu4pyscf-cuda13x', 'cupy-cuda13x', 'numpy', 'nvidia-cublas')},
        'cublas': {'path': paths[0], 'sha256': hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest(),
                   'api_version': cublas.getVersion(handle), 'settings': settings,
                   'environment': {name: os.environ.get(name) for name in (
                       'CUBLAS_EMULATION_STRATEGY', 'CUBLAS_EMULATE_DOUBLE_PRECISION',
                       'CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT',
                       'CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK', 'CUBLAS_WORKSPACE_CONFIG',
                   )}},
        'numerical_provenance': _source_provenance(gpu4pyscf, numint.libgdft),
        'extension_hashes': load_extensions(args.extension_root),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scf_driver_sha256': hashlib.sha256(Path(__file__).with_name('tf32_benchmark.py').read_bytes()).hexdigest(),
        'cases_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
        'policy': 'Automatic FP64 accuracy; native physical-Fock verification, recovery and analytic forces',
    }
    if args.mode in ('micro', 'all'):
        report['micro'] = _micro(cp, np, artifact, args.trace_library)
        (artifact / 'micro.json').write_text(json.dumps(report['micro'], indent=2) + '\n')
    if args.mode in ('scf', 'all'):
        report['scf'] = _scf(cp, np, args.repeats, args.forces, artifact, 5,
                              candidates=('emulated-grid', 'emulated-blas'))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('SUMMARY ' + json.dumps(report.get('scf', {}).get('summary', {})), flush=True)


if __name__ == '__main__':
    main()
