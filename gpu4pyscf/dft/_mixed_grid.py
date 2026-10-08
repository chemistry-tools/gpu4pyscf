"""Controlled grid matrix products for mixed-precision DFT integration."""

from __future__ import annotations


def _gemm(a, b, *, alpha=1.0, beta=0.0, out=None):
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    if a.dtype != b.dtype or a.dtype not in (cp.float32, cp.float64):
        raise TypeError('BLAS matrices must share float32 or float64 dtype')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Expected two matrices with the same contraction dimension')
    if not a.flags.c_contiguous or not b.flags.c_contiguous:
        raise ValueError('BLAS inputs must be contiguous')
    if out is None:
        if beta:
            raise ValueError('Accumulation requires an output matrix')
        out = cp.empty((a.shape[0], b.shape[0]), dtype=a.dtype)
    if out.shape != (a.shape[0], b.shape[0]) or out.dtype != a.dtype:
        raise ValueError('Invalid BLAS output')
    if not out.flags.c_contiguous:
        raise ValueError('BLAS output must be contiguous')
    handle = cp.cuda.device.get_cublas_handle()
    previous = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    previous_math = cublas.getMathMode(handle)
    alpha_value = np.asarray(alpha, dtype=a.dtype)
    beta_value = np.asarray(beta, dtype=a.dtype)
    gemm = cublas.dgemm if a.dtype == cp.float64 else cublas.sgemm
    try:
        # cublas_api.h defines PEDANTIC_MATH=2; CuPy does not expose this enum.
        cublas.setMathMode(handle, 2)
        # Row-major C = A B^T is column-major C^T = B A^T.
        gemm(
            handle,
            cublas.CUBLAS_OP_T,
            cublas.CUBLAS_OP_N,
            b.shape[0],
            a.shape[0],
            a.shape[1],
            alpha_value.ctypes.data,
            b.data.ptr,
            b.shape[1],
            a.data.ptr,
            a.shape[1],
            beta_value.ctypes.data,
            out.data.ptr,
            out.shape[1],
        )
    finally:
        cublas.setPointerMode(handle, previous)
        cublas.setMathMode(handle, previous_math)
    return out


def _make_contract(original, controller):
    import cupy as cp

    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None, **kwargs):
        if (
            pattern == 'ig,jg->ij'
            and a.dtype == b.dtype == cp.float64
            and a.flags.c_contiguous
            and b.flags.c_contiguous
            and not kwargs
            and (out is None or out.flags.c_contiguous)
        ):
            if controller.use_float32:
                controller.last_precision = 'float32'
                controller.float32_calls += 1
                product = _gemm(a.astype(cp.float32), b.astype(cp.float32)).astype(cp.float64)
                if out is None:
                    return alpha * product
                if beta:
                    out *= beta
                    out += alpha * product
                else:
                    out[:] = alpha * product
                return out
            controller.float64_calls += 1
            controller.last_precision = 'float64'
            return _gemm(a, b, alpha=alpha, beta=beta, out=out)
        return original(pattern, a, b, alpha=alpha, beta=beta, out=out, **kwargs)

    return contract


class _GridPrecision:
    def __init__(self, use_float32):
        self.use_float32 = use_float32
        self.float32_calls = 0
        self.float64_calls = 0
        self.last_precision = None
