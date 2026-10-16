"""Symmetric float64 Gram products for density-fitted exchange."""

from __future__ import annotations

from functools import lru_cache

_MIN_ORBITALS = 256
_MIN_REDUCTION = 4096
_SOURCE = r'''
extern "C" __global__ void mirror_exchange(double* out, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * n) return;
    int row = i / n, col = i % n;
    if (col > row) out[i] = out[(long long)col*n + row];
}
'''


@lru_cache(maxsize=16)
def _mirror(device):
    import cupy as cp

    return cp.RawKernel(_SOURCE, 'mirror_exchange')


def _symmetric_k_update(rhok, out):
    """Accumulate one real Gram product, returning False for the general fallback."""
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    if (
        not isinstance(rhok, cp.ndarray)
        or rhok.dtype != cp.float64
        or rhok.ndim != 4
        or not rhok.flags.c_contiguous
        or rhok.shape[0] == 0
        or rhok.shape[1] < _MIN_ORBITALS
        or rhok.shape[2]*rhok.shape[3] < _MIN_REDUCTION
        or not isinstance(out, cp.ndarray)
        or out.dtype != cp.float64
        or out.shape != (rhok.shape[0], rhok.shape[1], rhok.shape[1])
        or not out.flags.c_contiguous
        or rhok.device.id != out.device.id
    ):
        return False
    r0, o0 = rhok.data.ptr, out.data.ptr
    if r0 < o0+out.nbytes and o0 < r0+rhok.nbytes:
        return False

    m, k = rhok.shape[1], rhok.shape[2]*rhok.shape[3]
    product = cp.empty((m, m), dtype=cp.float64)
    mirror = _mirror(cp.cuda.Device().id)
    handle = cp.cuda.device.get_cublas_handle()
    previous_math = cublas.getMathMode(handle)
    previous_pointer = cublas.getPointerMode(handle)
    previous_stream = cublas.getStream(handle)
    one, zero = np.asarray(1., dtype=np.float64), np.asarray(0., dtype=np.float64)
    try:
        cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        cublas.setMathMode(handle, 2)  # CUBLAS_PEDANTIC_MATH
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        for index in range(rhok.shape[0]):
            # Column-major upper is row-major lower. A separate product keeps
            # any nonsymmetric pre-existing output unchanged during accumulation.
            cublas.dsyrk(handle, cublas.CUBLAS_FILL_MODE_UPPER, cublas.CUBLAS_OP_T,
                         m, k, one.ctypes.data, rhok[index].data.ptr, k,
                         zero.ctypes.data, product.data.ptr, m)
            mirror(((m*m+255)//256,), (256,), (product, np.int32(m)))
            out[index] += product
    finally:
        cublas.setStream(handle, previous_stream)
        cublas.setMathMode(handle, previous_math)
        cublas.setPointerMode(handle, previous_pointer)
    return True
