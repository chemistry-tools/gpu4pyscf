"""Symmetric float64 kinetic-energy-density matrix products."""

from __future__ import annotations

from functools import lru_cache

_MIN_ORBITALS = 512
_SOURCE = r"""
extern "C" __global__ void scaled_tau(
        double* out, const double* ao, const double* scale,
        const long long* indices, int m, int k, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)m * 3 * n) return;
    int row = i / (3*n), column = i % (3*n), derivative = column / n;
    int g = indices[column % n];
    out[i] = ao[((long long)(derivative+1)*m + row)*k + g] * scale[g];
}
extern "C" __global__ void mirror_tau(double* out, int n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * n) return;
    int r = i / n, c = i % n;
    if (c > r) out[i] = out[(long long)c*n + r];
}
"""


@lru_cache(maxsize=16)
def _kernels(device):
    import cupy as cp

    return cp.RawKernel(_SOURCE, 'scaled_tau'), cp.RawKernel(_SOURCE, 'mirror_tau')


def _eligible(bra, ket, weights, out):
    import cupy as cp

    # Small blocks are faster with the original three products. Distinct fields
    # require the general bilinear product, even when their values happen to agree.
    if (
        bra is not ket
        or not isinstance(bra, cp.ndarray)
        or bra.dtype != cp.float64
        or bra.ndim != 3
        or bra.shape[0] < 4
        or bra.shape[1] <= _MIN_ORBITALS
        or not bra.shape[2]
        or not bra.flags.c_contiguous
        or not isinstance(weights, cp.ndarray)
        or weights.dtype != cp.float64
        or weights.shape != (bra.shape[2],)
        or not weights.flags.c_contiguous
        or (
            out is not None
            and (
                not isinstance(out, cp.ndarray)
                or out.dtype != cp.float64
                or out.shape != (bra.shape[1], bra.shape[1])
                or not out.flags.c_contiguous
            )
        )
    ):
        return False
    # Splitting signed weights must not discard nonfinite or zero-times-NaN
    # contributions that the original products would propagate.
    return bool(cp.isfinite(weights).all()) and bool(cp.isfinite(bra[1:4]).all())


def _symmetric_tau(ao, weights, out=None):
    """Form half the signed weighted Gram matrix, including all three derivatives."""
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    m, k = ao.shape[1:]
    if out is None:
        out = cp.empty((m, m), dtype=cp.float64)
    scale = cp.sqrt(cp.abs(weights))
    groups = ((cp.nonzero(weights > 0)[0], 0.5), (cp.nonzero(weights < 0)[0], -0.5))
    scaled_kernel, mirror_kernel = _kernels(cp.cuda.Device().id)
    handle = cp.cuda.device.get_cublas_handle()
    previous_math = cublas.getMathMode(handle)
    previous_pointer = cublas.getPointerMode(handle)
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    beta = 0.0
    try:
        cublas.setMathMode(handle, 2)  # PEDANTIC_MATH, as in the float64 grid GEMM.
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        for indices, alpha in groups:
            n = len(indices)
            if not n:
                continue
            scaled = cp.empty((m, 3 * n), dtype=cp.float64)
            scaled_kernel(
                ((scaled.size + 255) // 256,),
                (256,),
                (scaled, ao, scale, indices, np.int32(m), np.int32(k), np.int32(n)),
            )
            alpha_value, beta_value = np.asarray(alpha, np.float64), np.asarray(beta, np.float64)
            # Column-major upper is row-major lower. Stacking derivatives lets
            # one SYRK per sign replace the three full square matrix products.
            cublas.dsyrk(
                handle,
                cublas.CUBLAS_FILL_MODE_UPPER,
                cublas.CUBLAS_OP_T,
                m,
                3 * n,
                alpha_value.ctypes.data,
                scaled.data.ptr,
                3 * n,
                beta_value.ctypes.data,
                out.data.ptr,
                m,
            )
            beta = 1.0
        if not beta:
            out.fill(0)
        mirror_kernel(((m * m + 255) // 256,), (256,), (out, np.int32(m)))
    finally:
        cublas.setMathMode(handle, previous_math)
        cublas.setPointerMode(handle, previous_pointer)
    return out
