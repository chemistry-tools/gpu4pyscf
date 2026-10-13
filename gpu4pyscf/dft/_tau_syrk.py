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


@lru_cache(maxsize=16)
def _finite_kernel(device):
    import cupy as cp

    # Fuse the finite check and reduction instead of caching a large boolean
    # array for every changing AO block shape in the device memory pool.
    return cp.ReductionKernel(
        'float64 x', 'bool finite', 'isfinite(x)', 'a && b', 'finite = a', 'true', 'tau_all_finite'
    )


def _workspace_size(nao, ngrids):
    """Reserve one bounded task buffer for the three stacked derivatives."""
    return (3 * nao if nao > _MIN_ORBITALS else nao) * ngrids


def _eligible(bra, ket, weights, out, *, minimum=None):
    import cupy as cp

    if minimum is None:
        minimum = _MIN_ORBITALS
    # Small blocks are faster with the original three products. Distinct fields
    # require the general bilinear product, even when their values happen to agree.
    if (
        bra is not ket
        or not isinstance(bra, cp.ndarray)
        or bra.dtype != cp.float64
        or bra.ndim != 3
        or bra.shape[0] < 4
        or bra.shape[1] <= minimum
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
    finite = _finite_kernel(cp.cuda.Device().id)
    return bool(finite(weights)) and bool(finite(bra[1:4]))


def _symmetric_tau(ao, weights, buf=None, out=None):
    """Form half the signed weighted Gram matrix, including all three derivatives."""
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    m, k = ao.shape[1:]
    if out is None:
        out = cp.empty((m, m), dtype=cp.float64)
    size = m * 3 * k
    if isinstance(buf, cp.ndarray) and buf.dtype == cp.float64 and buf.flags.c_contiguous and buf.size >= size:
        workspace = buf.reshape(-1)[:size]
    else:
        # Standalone callers must not fill the pool with differently sized
        # multi-megabyte blocks that later large direct allocations cannot reclaim.
        memory = cp.cuda.Memory(size * 8)
        workspace = cp.ndarray((size,), dtype=cp.float64, memptr=cp.cuda.MemoryPointer(memory, 0))
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
            scaled = workspace[: m * 3 * n].reshape(m, 3 * n)
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
