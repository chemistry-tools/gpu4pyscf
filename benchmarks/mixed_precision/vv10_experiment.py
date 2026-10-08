"""Experimental VV10 pair kernels with float64 and mixed-precision comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path


class _VV10:
    def __new__(cls, *args, **kwargs):
        from gpu4pyscf.dft._mixed_vv10 import _VV10 as kernel

        return kernel(*args, **kwargs)


class _AdaptiveVV10:
    def __new__(cls, *args, **kwargs):
        from gpu4pyscf.dft._mixed_vv10 import _AdaptiveVV10 as kernel

        return kernel(*args, **kwargs)


@contextmanager
def install(mode, block=128, backend='nvrtc', *, profile=False):
    from gpu4pyscf.dft import numint

    original = numint._vv10nlc
    kernel = (
        _AdaptiveVV10(original, block, backend, profile=profile)
        if mode in ('mixed', 'verify')
        else _VV10(mode, block, backend, profile=profile)
    )
    numint._vv10nlc = kernel
    try:
        yield kernel
    finally:
        numint._vv10nlc = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('nvrtc', 'nvcc'), default='nvrtc')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    from gpu_performance import _resources

    _resources([4, 5, 6, 7])
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import numint
    from gpu4pyscf.dft._mixed_vv10 import _SOURCE

    data = np.load(args.input)
    rho, coords, weights = [cp.asarray(data[key]) for key in ('rho', 'coords', 'weights')]
    pars = tuple(data['pars'])
    start = time.perf_counter()
    reference = numint._vv10nlc(rho, coords, weights, pars)
    cp.cuda.get_current_stream().synchronize()
    baseline = time.perf_counter() - start
    results = []
    for mode in ('compiled', 'refined', 'float32'):
        for block in (64, 128, 256):
            kernel = _VV10(mode, block, args.backend, profile=True)
            result = kernel(rho, coords, weights, pars)
            for _ in range(3):
                result = kernel(rho, coords, weights, pars)
            entry = {
                'mode': mode,
                'block': block,
                'compiler': args.backend,
                'active_grids': kernel.active_grids[-1],
                'pair_seconds_median': statistics.median(kernel.pair_seconds[-3:]),
                'exc_max_abs_error': float(cp.max(cp.abs(result[0] - reference[0]))),
                'vxc_max_abs_error': float(cp.max(cp.abs(result[1] - reference[1]))),
                'nlc_energy_error_hartree': float(cp.dot(rho[0] * weights, result[0] - reference[0])),
            }
            results.append(entry)
            print(json.dumps(entry), flush=True)
    packages = ['cupy-cuda13x', 'nvidia-cuda-nvrtc']
    if args.backend == 'nvcc':
        packages += ['nvidia-cuda-nvcc', 'nvidia-cuda-crt', 'nvidia-nvvm']
    args.output.write_text(
        json.dumps(
            {
                'baseline_seconds': baseline,
                'results': results,
                'versions': {name: version(name) for name in packages},
                'source_sha256': hashlib.sha256(_SOURCE.encode()).hexdigest(),
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
