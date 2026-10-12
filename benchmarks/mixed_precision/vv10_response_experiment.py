"""Compare fused float64 VV10 response against native pair passes on frozen fields."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from _extensions import load_extensions
from gpu_performance import _resources, _source_provenance


def native_response(coords, rw, omega, kappa, atoms, natm):
    """Evaluate the original U/W/E and atom-by-grid coordinate response."""
    import cupy as cp

    from gpu4pyscf.dft import numint

    n = len(rw)
    u, w, e = (cp.empty(n, dtype=cp.float64) for _ in range(3))
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.int32),
                              cp.cumsum(cp.bincount(atoms, minlength=natm), dtype=cp.int32)))
    response = cp.empty((natm, 3, n), dtype=cp.float64)

    def ptr(a):
        return ctypes.cast(a.data.ptr, ctypes.c_void_p)

    stream = ctypes.cast(cp.cuda.get_current_stream().ptr, ctypes.c_void_p)
    err = numint.libgdft.VXC_vv10nlc_fock_eval_UWE(
        stream, ptr(u), ptr(w), ptr(e), ptr(coords), ptr(rw), ptr(omega), ptr(kappa), ctypes.c_int(n))
    if err:
        raise RuntimeError('Native VV10 field kernel failed')
    err = numint.libgdft.VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal(
        stream, ptr(response), ptr(coords), ptr(rw), ptr(omega), ptr(kappa), ptr(atoms), ptr(offsets),
        ctypes.c_int(natm), ctypes.c_int(0), ctypes.c_int(n))
    if err:
        raise RuntimeError('Native VV10 coordinate response failed')
    for atom, (begin, end) in enumerate(zip(offsets[:-1].get(), offsets[1:].get())):
        response[atom, :, begin:end] = -cp.sum(response[:, :, begin:end], axis=0)
    return u, w, e, cp.einsum('Adg,g->Ad', response, rw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    _resources([4, 5, 6, 7])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Select physical GPU 1')
    import cupy as cp
    import numpy as np

    import gpu4pyscf
    from gpu4pyscf.dft import numint

    if cp.cuda.runtime.getDeviceCount() != 1 or not cp.cuda.Device(0).pci_bus_id.lower().endswith('05:00.0'):
        raise RuntimeError('Visible device must be physical GPU 1')
    extensions = load_extensions(args.extension_root, forces=True)
    from gpu4pyscf.dft._vv10_grad import _kernels, _pair_response

    with np.load(args.input, allow_pickle=False) as fields:
        arrays = [cp.asarray(fields[name]) for name in ('coords', 'rw', 'omega', 'kappa', 'atoms')]
        natm = len(fields['offsets']) - 1
        if int(fields['begin']) != 0 or int(fields['batch']) != len(arrays[1]):
            raise ValueError('Use full atom-ordered captured fields')
    n = len(arrays[1])
    report = {'grid_points': n, 'atoms': natm, 'affinity': sorted(os.sched_getaffinity(0)),
              'sources': _source_provenance(gpu4pyscf, numint.libgdft), 'extensions': extensions,
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'runs': [],
              'native_response_tensor_bytes': natm*3*n*8, 'fused_response_tensor_bytes': 3*n*8}

    def measure(name, call):
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        result = call()
        cp.cuda.get_current_stream().synchronize()
        seconds = time.perf_counter()-started
        row = {'mode': name, 'seconds': seconds}
        if name != 'native':
            errors = [float(cp.max(cp.abs(x-y))) for x, y in zip(result, reference)]
            row['max_absolute_errors'] = errors
            for x, y in zip(result[:3], reference[:3]):
                cp.testing.assert_allclose(x, y, atol=1e-13, rtol=2e-12)
            cp.testing.assert_allclose(result[3], reference[3], atol=1e-9, rtol=2e-10)
        report['runs'].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        return result

    # Compile without running a large pair traversal, outside timing.
    for block in (64, 128, 256):
        for kernel in _kernels(cp.cuda.Device().id, block):
            kernel.compile()
    reference = measure('native', lambda: native_response(*arrays, natm))
    for block in (64, 128, 256):
        measure(f'fused-{block}', lambda: _pair_response(*arrays, natm, block=block))
    best = min((row for row in report['runs'] if row['mode'].startswith('fused')), key=lambda row: row['seconds'])
    block = int(best['mode'].split('-')[1])
    for order in (('fused', 'native'), ('native', 'fused')):
        for mode in order:
            measure('native' if mode == 'native' else f'fused-{block}',
                    (lambda: native_response(*arrays, natm)) if mode == 'native'
                    else (lambda: _pair_response(*arrays, natm, block=block)))
    medians = {mode: statistics.median(row['seconds'] for row in report['runs'] if row['mode'] == mode)
               for mode in ('native', f'fused-{block}')}
    report['summary'] = {'block': block, 'median_seconds': medians,
                         'speedup': medians['native']/medians[f'fused-{block}']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
