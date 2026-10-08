"""Controlled grid matrix products for mixed-precision DFT integration."""

from __future__ import annotations

from threading import local


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
            and a.ndim == b.ndim == 2
            and a.shape[1] == b.shape[1]
            and a.size and b.size
            and a.flags.c_contiguous
            and b.flags.c_contiguous
            and not kwargs
            and (
                out is None
                or (out.flags.c_contiguous and out.dtype == cp.float64 and out.shape == (a.shape[0], b.shape[0]))
            )
        ):
            if controller.use_float32:
                controller.last_precision = 'float32'
                controller.float32_calls += 1
                if out is None:
                    if beta:
                        raise ValueError('Accumulation requires an output matrix')
                    out = cp.empty((a.shape[0], b.shape[0]), dtype=cp.float64)
                a32, b32, product = controller._scratch(a, b)
                cp.copyto(a32, a, casting='unsafe')
                cp.copyto(b32, b, casting='unsafe')
                _gemm(a32, b32, out=product)
                return controller._accumulate(product, alpha, beta, out)
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
        self._workspace = local()
        self._accumulation_kernel = None

    def _release_scratch(self):
        self._workspace = local()

    def _scratch(self, a, b):
        import cupy as cp

        stream = cp.cuda.get_current_stream()
        key = (cp.cuda.Device().id, stream.ptr)
        if not hasattr(self._workspace, 'streams'):
            self._workspace.streams = {}
        # Retain the stream while buffers exist; worker threads never share scratch.
        _, buffers = self._workspace.streams.setdefault(key, (stream, [None, None, None]))
        shapes = (a.shape, b.shape, (a.shape[0], b.shape[0]))
        views = []
        for index, shape in enumerate(shapes):
            size = shape[0] * shape[1]
            if buffers[index] is None or buffers[index].size < size:
                buffers[index] = cp.empty(size, dtype=cp.float32)
            views.append(buffers[index][:size].reshape(shape))
        return views

    def _accumulate(self, product, alpha, beta, out):
        import cupy as cp

        if self._accumulation_kernel is None:
            self._accumulation_kernel = cp.ElementwiseKernel(
                'float32 product, float64 alpha, float64 beta',
                'float64 out',
                '''
                double scaled = alpha * (double)product;
                if (beta == 0.0) out = scaled;
                else out = beta * out + scaled;
                ''',
                'mixed_grid_accumulate',
                # Match the separate float64 multiplies/add in the unfused path.
                options=('--fmad=false',),
            )
        return self._accumulation_kernel(product, alpha, beta, out)
