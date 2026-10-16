"""Bounded float64 whitening that skips exactly zero coefficient rows."""

from __future__ import annotations

from functools import lru_cache

_MIN_AUX = 2048
_BLOCK_COLUMNS = 512
_MAX_ROWS = 4096
_MAX_WORKSPACE_BYTES = 512*1024**2
_MAX_RETAINED_FRACTION = .75
_SOURCE = r'''
extern "C" __global__ void whitening_rows(
    const double* coefficient, bool* masks, int* invalid,
    int n, int width, int blocks) {
    __shared__ int partial[8];
    int row = blockIdx.x % n, block = blockIdx.x / n;
    int first = block*width, last = min(first+width, n), flags = 0;
    for (int col = first+threadIdx.x; col < last; col += blockDim.x) {
        double value = coefficient[(long long)row*n+col];
        flags |= (value != 0 ? 1 : 0) | (!isfinite(value) ? 2 : 0);
    }
    for (int offset = 16; offset; offset >>= 1)
        flags |= __shfl_xor_sync(0xffffffff, flags, offset);
    if (!(threadIdx.x & 31)) partial[threadIdx.x >> 5] = flags;
    __syncthreads();
    if (threadIdx.x < 32) {
        flags = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
        for (int offset = 16; offset; offset >>= 1)
            flags |= __shfl_xor_sync(0xffffffff, flags, offset);
        if (!threadIdx.x) {
            masks[(long long)block*n+row] = flags & 1;
            if (flags & 2) atomicOr(invalid, 1);
        }
    }
}
'''


def _workspace_plan(n, block_rows, width=_BLOCK_COLUMNS):
    if (n <= 0 or n > 2**31-1 or width <= 0
            or len(block_rows) != (n+width-1)//width
            or any(k < 0 or k > n for k in block_rows)):
        return None
    elements = sum(k*min(width, n-first*width) for first, k in enumerate(block_rows))
    retained_bytes = elements*8+sum(block_rows)*4
    kmax = max(block_rows)
    if not kmax or elements > n*n*_MAX_RETAINED_FRACTION:
        return None
    rows = min(_MAX_ROWS, (_MAX_WORKSPACE_BYTES-retained_bytes)//(8*kmax))
    rows = rows//256*256
    if rows < 256:
        return None
    return rows, retained_bytes+rows*kmax*8


def _supported_device(device):
    import cupy as cp

    return cp.cuda.Device(device).compute_capability == '86'


def _available_memory():
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()[0]+cp.get_default_memory_pool().free_bytes()


@lru_cache(maxsize=16)
def _kernels(device):
    import cupy as cp

    return (cp.RawKernel(_SOURCE, 'whitening_rows'),
            cp.ReductionKernel('double x', 'bool valid', 'isfinite(x)',
                               'a && b', 'valid = a', 'true', 'finite_whitening'))


class _BlockedWhitening:
    """Own compact native coefficients and one scratch buffer for a DF build."""

    @classmethod
    def create(cls, coefficient, tag):
        import cupy as cp
        import numpy as np

        if (tag != 'cd' or not isinstance(coefficient, cp.ndarray)
                or coefficient.dtype != cp.float64 or coefficient.ndim != 2
                or not coefficient.flags.c_contiguous):
            return None
        n = coefficient.shape[0]
        device = cp.cuda.device.get_device_id()
        if (n < _MIN_AUX or coefficient.shape != (n, n)
                or coefficient.nbytes > _MAX_WORKSPACE_BYTES
                or coefficient.device.id != device or not _supported_device(device)):
            return None
        blocks = (n+_BLOCK_COLUMNS-1)//_BLOCK_COLUMNS
        if coefficient.nbytes > _available_memory()//2:
            return None
        row_masks, _ = _kernels(device)
        try:
            masks = cp.empty((blocks, n), dtype=cp.bool_)
            invalid = cp.zeros(1, dtype=cp.int32)
            row_masks((blocks*n,), (256,), (coefficient, masks, invalid,
                      np.int32(n), np.int32(_BLOCK_COLUMNS), np.int32(blocks)))
            if int(invalid[0]):
                return None
            indices = [cp.flatnonzero(mask).astype(cp.int32) for mask in masks]
            plan = _workspace_plan(n, [len(index) for index in indices], _BLOCK_COLUMNS)
            if plan is None or plan[1] > _available_memory()//2:
                return None
            self = cls()
            self.device, self.stream = device, cp.cuda.get_current_stream()
            self.n, self.rows, self.workspace_bytes = n, plan[0], plan[1]
            self.blocks = []
            for block, index in enumerate(indices):
                first, last = block*_BLOCK_COLUMNS, min((block+1)*_BLOCK_COLUMNS, n)
                # flatnonzero preserves the original reduction-row order.
                # Permuting into triangular order changes cancellation in the
                # native inverse and is deliberately avoided here.
                compact = cp.ascontiguousarray(coefficient[index, first:last])
                self.blocks.append((first, last, index, compact))
            self.work = cp.empty(self.rows*max(map(len, indices)), dtype=cp.float64)
        except cp.cuda.memory.OutOfMemoryError:
            return None
        return self

    def apply(self, raw, out):
        """Write a supported product, or return False before touching output."""
        import cupy as cp
        import numpy as np
        from cupy_backends.cuda.libs import cublas

        if (not isinstance(raw, cp.ndarray) or raw.dtype != cp.float64 or raw.ndim != 2
                or not raw.flags.c_contiguous or raw.shape[1] != self.n
                or not 0 < raw.shape[0] <= 2**31-1 or raw.device.id != self.device
                or not isinstance(out, cp.ndarray) or out.dtype != cp.float64
                or out.shape != raw.shape or not out.flags.c_contiguous
                or out.device.id != self.device or cp.cuda.device.get_device_id() != self.device
                or cp.cuda.get_current_stream().ptr != self.stream.ptr):
            return False
        r0, o0 = raw.data.ptr, out.data.ptr
        if r0 < o0+out.nbytes and o0 < r0+raw.nbytes:
            return False
        w0 = self.work.data.ptr
        if (r0 < w0+self.work.nbytes and w0 < r0+raw.nbytes
                or o0 < w0+self.work.nbytes and w0 < o0+out.nbytes):
            return False
        # Removing zero products must not hide NaN/Inf * 0 from the fallback.
        if not bool(_kernels(self.device)[1](raw)):
            return False
        handle = cp.cuda.device.get_cublas_handle()
        previous = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
        one, zero = np.asarray(1., dtype=np.float64), np.asarray(0., dtype=np.float64)
        try:
            cublas.setStream(handle, self.stream.ptr)
            cublas.setMathMode(handle, 2)  # CUBLAS_PEDANTIC_MATH
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            for row in range(0, len(raw), self.rows):
                end = min(row+self.rows, len(raw))
                for first, last, indices, coefficient in self.blocks:
                    k, width = len(indices), last-first
                    if not k:
                        out[row:end, first:last] = 0
                        continue
                    compact = self.work[:(end-row)*k].reshape(end-row, k)
                    cp.take(raw[row:end], indices, axis=1, out=compact)
                    # Row-major output has a fixed pitch of n; write directly
                    # into its columns without allocating a second output tile.
                    cublas.dgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
                                  width, end-row, k, one.ctypes.data,
                                  coefficient.data.ptr, width, compact.data.ptr, k,
                                  zero.ctypes.data, out.data.ptr+(row*self.n+first)*8, self.n)
        finally:
            cublas.setStream(handle, previous[0])
            cublas.setMathMode(handle, previous[1])
            cublas.setPointerMode(handle, previous[2])
        return True


def _whiten(raw, coefficient, out, prepared):
    if prepared is not None and prepared.apply(raw, out):
        return out
    return raw.dot(coefficient, out=out)
