"""Float64 products that preserve auxiliary-response symmetry and sparsity."""

from __future__ import annotations

import math
from functools import lru_cache

_MIN_METRIC_AUX = 1024
_MIN_GRAM_AUX = 256
_MIN_REDUCTION = 4096
_MAX_MATRIX_BYTES = 512 * 1024**2


def _gram_layout(shape, strides):
    if len(shape) != 3 or len(strides) != 3 or min(shape) <= 0 or min(strides) <= 0:
        return None
    m, k = shape[0], shape[1]*shape[2]
    if (m < _MIN_GRAM_AUX or k < _MIN_REDUCTION or max(m, k) > 2**31-1
            or 8*m*m > _MAX_MATRIX_BYTES):
        return None
    if strides[1] == shape[2]*strides[2]:
        pair_stride = strides[2]
    elif strides[2] == shape[1]*strides[1]:
        pair_stride = strides[1]
    else:
        return None
    if strides[0] == 8 and pair_stride % 8 == 0 and pair_stride >= m*8:
        operation, leading = 0, pair_stride//8
    elif pair_stride == 8 and strides[0] % 8 == 0 and strides[0] >= k*8:
        operation, leading = 1, strides[0]//8
    else:
        return None
    if leading > 2**31-1:
        return None
    span = (m-1)*strides[0]+(k-1)*pair_stride+8
    return m, k, operation, leading, span


def _supported_device(device):
    import cupy as cp

    return cp.cuda.Device(device).compute_capability == '86'


def _available_memory():
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()


@lru_cache(maxsize=16)
def _checks(device):
    import cupy as cp

    return (cp.ReductionKernel('double x', 'bool valid', 'isfinite(x)', 'a && b',
                               'valid = a', 'true', 'finite_aux_response'),
            cp.ReductionKernel('double x, double y', 'bool valid',
                               'isfinite(x) && x == y', 'a && b', 'valid = a',
                               'true', 'symmetric_aux_response'))


class _SparseAuxTransform:
    """Apply the native sorted auxiliary coefficients without dense zero products."""

    @classmethod
    def create(cls, auxmol, sorting, coefficient):
        import cupy as cp

        from gpu4pyscf.gto.mole import SortedMole

        if (not isinstance(auxmol, SortedMole) or not auxmol.mol.cart
                or not isinstance(coefficient, cp.ndarray) or coefficient.ndim != 2
                or coefficient.dtype != cp.float64):
            return None
        n = coefficient.shape[0]
        device = cp.cuda.device.get_device_id()
        if (n < _MIN_METRIC_AUX or coefficient.shape != (n, n)
                or coefficient.nbytes > _MAX_MATRIX_BYTES
                or coefficient.device.id != device or not _supported_device(device)
                or coefficient.nbytes*8 > _available_memory()):
            return None
        indices = cp.asarray(sorting)
        if (indices.shape != (n,) or indices.dtype.kind not in 'iu'
                or not bool(cp.all(cp.sort(indices) == cp.arange(n)))):
            return None
        finite, _ = _checks(device)
        if (not bool(finite(coefficient))
                or not bool(cp.all(cp.count_nonzero(coefficient, axis=0) == 1)
                            & cp.all(cp.count_nonzero(coefficient, axis=1) == 1))):
            return None
        self = cls()
        self.auxmol, self.sorting, self.coefficient = auxmol, indices, coefficient
        self.device, self.stream = device, cp.cuda.get_current_stream()
        return self

    def apply(self, matrix):
        import cupy as cp

        if (not isinstance(matrix, cp.ndarray) or matrix.ndim != 2
                or matrix.dtype != cp.float64 or matrix.shape[0] != self.coefficient.shape[1]
                or matrix.device.id != self.device or cp.cuda.device.get_device_id() != self.device
                or cp.cuda.get_current_stream().ptr != self.stream.ptr):
            return self.coefficient.dot(matrix)
        # coefficient[sorting] was built from auxmol.ctr_coeff in this call.
        # Its single nonzero per row/column makes the native sparse operation
        # a scaled permutation with the same float64 products as dense GEMM.
        value = self.auxmol.C_dot_mat(matrix)
        result = cp.empty_like(value)
        result[self.sorting] = value
        return result

    def project_rhs(self, vectors, rhs):
        import cupy as cp

        expected = self.coefficient.T
        if (isinstance(rhs, cp.ndarray) and rhs.dtype == expected.dtype
                and rhs.shape == expected.shape and rhs.strides == expected.strides
                and rhs.data.ptr == expected.data.ptr and rhs.device.id == expected.device.id):
            return cp.ascontiguousarray(self.apply(vectors).conj().T)
        return vectors.conj().T @ rhs.reshape(vectors.shape[0], -1)


def _symmetric_aux_gram(values, out, alpha, beta):
    """Return the symmetric update, or None before changing an unsupported output."""
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    from gpu4pyscf.df._jk_contraction import _mirror

    if (not isinstance(values, cp.ndarray) or values.dtype != cp.float64
            or values.ndim not in (3, 4)):
        return None
    batches = 1 if values.ndim == 3 else values.shape[0]
    if batches not in (1, 2) or (values.ndim == 4 and values.strides[0] < 0):
        return None
    layout = _gram_layout(values.shape[-3:], values.strides[-3:])
    if layout is None or not math.isfinite(alpha) or not math.isfinite(beta):
        return None
    m, k, operation, leading, span = layout
    if values.ndim == 4:
        span += (batches-1)*values.strides[0]
    device = cp.cuda.device.get_device_id()
    if values.device.id != device or not _supported_device(device):
        return None
    finite, symmetric = _checks(device)
    if out is None:
        if beta != 0 or m*m*8 > _available_memory() // 2:
            return None
    else:
        if (not isinstance(out, cp.ndarray) or out.dtype != cp.float64
                or out.shape != (m, m) or not out.flags.c_contiguous or out.device.id != device):
            return None
        r0, o0 = values.data.ptr, out.data.ptr
        if r0 < o0+out.nbytes and o0 < r0+span:
            return None
        if beta != 0 and not bool(symmetric(out, out.T)):
            return None
    if not bool(finite(values)):
        return None
    if out is None:
        out = cp.empty((m, m), dtype=cp.float64)
    handle = cp.cuda.device.get_cublas_handle()
    previous = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
    scale, scale_out = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    one = np.asarray(1., dtype=np.float64)
    try:
        cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        cublas.setMathMode(handle, 2)  # CUBLAS_PEDANTIC_MATH
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
        # Column-major upper is row-major lower. Native dm_oo often already has
        # the leading auxiliary dimension contiguous; no large packing copy.
        for index in range(batches):
            pointer = values.data.ptr + (index*values.strides[0] if values.ndim == 4 else 0)
            cublas.dsyrk(handle, cublas.CUBLAS_FILL_MODE_UPPER, operation, m, k,
                         scale.ctypes.data, pointer, leading,
                         scale_out.ctypes.data if index == 0 else one.ctypes.data, out.data.ptr, m)
        _mirror(device)(((m*m+255)//256,), (256,), (out, np.int32(m)))
    finally:
        cublas.setStream(handle, previous[0])
        cublas.setMathMode(handle, previous[1])
        cublas.setPointerMode(handle, previous[2])
    return out
