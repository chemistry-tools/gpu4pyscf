# The splitting and accumulation follow ozIMMU, specialized here to real Gram
# products with one shared slice buffer and CuPy runtime compilation.
# https://github.com/enp1s0/ozIMMU/tree/08eea9231729d54dbfd92955f2cbfc21ec236856
#
# MIT License
# Copyright (c) 2023 enp1s0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Bounded INT8 Gram products for the approximate phase of verified SCF."""

from __future__ import annotations

from functools import lru_cache
from threading import local

_MIN_ORBITALS = 256
_MIN_REDUCTION = 4096
_MAX_REDUCTION = 131072
_MAX_WORKSPACE_BYTES = 512*1024**2
_SLICES = 9
_SOURCE = r'''
extern "C" __global__ void row_exponents(
    const double* a, int* exponents, int* flags, int m, int k, int padded_m) {
    __shared__ int maxima[8], invalid[8];
    int row = blockIdx.x % padded_m, batch = blockIdx.x / padded_m;
    int maximum = 0, bad = 0;
    if (row < m) {
        for (int col = threadIdx.x; col < k; col += blockDim.x) {
            unsigned long long bits = __double_as_longlong(a[((long long)batch*m+row)*k+col]);
            int e = (bits >> 52) & 2047;
            maximum = max(maximum, e);
            bad |= e == 2047 || (e == 0 && (bits & 0xfffffffffffffULL));
        }
    }
    for (int offset = 16; offset; offset >>= 1) {
        maximum = max(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
        bad |= __shfl_xor_sync(0xffffffff, bad, offset);
    }
    if (!(threadIdx.x & 31)) {
        maxima[threadIdx.x >> 5] = maximum;
        invalid[threadIdx.x >> 5] = bad;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        maximum = threadIdx.x < 8 ? maxima[threadIdx.x] : 0;
        bad = threadIdx.x < 8 ? invalid[threadIdx.x] : 0;
        for (int offset = 16; offset; offset >>= 1) {
            maximum = max(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
            bad |= __shfl_xor_sync(0xffffffff, bad, offset);
        }
        if (!threadIdx.x) {
            int e = maximum ? maximum-1022 : 0;
            exponents[blockIdx.x] = e;
            flags[blockIdx.x] = bad || e < -500 || e > 500;
        }
    }
}

extern "C" __global__ void split_rows(
    const double* a, signed char* slices, const int* exponents,
    int m, int k, int padded_m, int padded_k) {
    int row = blockIdx.x;
    int maximum = exponents[row]+1022;
    long long plane = (long long)padded_m*padded_k;
    for (int col = threadIdx.x; col < padded_k; col += blockDim.x) {
        unsigned long long bits = row < m && col < k ?
            __double_as_longlong(a[(long long)row*k+col]) : 0;
        int e = (bits >> 52) & 2047;
        unsigned long long mantissa = (bits & 0xfffffffffffffULL) | (e ? 1ULL << 52 : 0);
        #pragma unroll
        for (int s = 0; s < 9; s++) {
            int shift = e-maximum-53+7*(s+1);
            unsigned long long value = shift >= 0 ? mantissa << shift :
                (shift > -64 ? mantissa >> -shift : 0);
            int digit = value & 127;
            slices[s*plane+(long long)row*padded_k+col] = (bits >> 63) ? -digit : digit;
        }
    }
}

extern "C" __global__ void accumulate_pair(
    const int* partial, double* accum, long long size, int shift, int n, int paired) {
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (index < size) {
        double value = (double)partial[index];
        // The transpose supplies the other slice ordering. Convert before
        // adding: their sum can exceed INT32 even though each dot product fits.
        if (paired) value += (double)partial[(index%n)*n+index/n];
        accum[index] += ldexp(value, shift);
    }
}

extern "C" __global__ void finish_gram(
    const double* accum, const int* exponents, double* out, int m, int padded_m) {
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= (long long)m*m) return;
    int row = index/m, col = index%m;
    // Mirror the product, preserving any nonsymmetric pre-existing output.
    double value = accum[(long long)max(row,col)*padded_m+min(row,col)];
    out[index] += ldexp(value, exponents[row]+exponents[col]);
}
'''


def _workspace_plan(m, k, batches=1):
    if not (0 < m < 2**31-16 and 0 < k <= _MAX_REDUCTION and 0 < batches*m < 2**31-16):
        return None
    padded_m, padded_k = (m+15)//16*16, (k+15)//16*16
    size = _SLICES*padded_m*padded_k + 12*padded_m**2 + 8*batches*padded_m
    if size > _MAX_WORKSPACE_BYTES:
        return None
    return padded_m, padded_k, size


@lru_cache(maxsize=16)
def _int8_supported(device):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    return cp.cuda.Device(device).compute_capability == '86' and hasattr(cublas, 'CUBLAS_COMPUTE_32I')


@lru_cache(maxsize=16)
def _kernels(device):
    import cupy as cp

    return tuple(cp.RawKernel(_SOURCE, name) for name in
                 ('row_exponents', 'split_rows', 'accumulate_pair', 'finish_gram'))


def _available_memory():
    from gpu4pyscf.lib.cupy_helper import get_avail_mem

    return get_avail_mem()


class _Int8Gram:
    def __init__(self):
        self._workspace = local()

    def release(self):
        self._workspace = local()

    def _scratch(self, m, k, batches):
        import cupy as cp

        stream = cp.cuda.get_current_stream()
        key = cp.cuda.Device().id, stream.ptr
        if not hasattr(self._workspace, 'streams'):
            self._workspace.streams = {}
        _, buffers = self._workspace.streams.setdefault(key, (stream, [None]*5))
        specifications = (
            (_SLICES*m*k, cp.int8, 1), (m*m, cp.int32, 4), (m*m, cp.float64, 8),
            (batches*m, cp.int32, 4), (batches*m, cp.int32, 4),
        )
        retained = sum(max(size, buffer.size if buffer is not None else 0)*width
                       for buffer, (size, _, width) in zip(buffers, specifications))
        if retained > _MAX_WORKSPACE_BYTES:
            buffers[:] = [None]*5
        views = []
        for index, (size, dtype, _) in enumerate(specifications):
            if buffers[index] is None or buffers[index].size < size:
                buffers[index] = cp.empty(size, dtype=dtype)
            views.append(buffers[index][:size])
        return views

    def update(self, rhok, out):
        """Add an approximate Gram product, or leave output untouched and return False."""
        import cupy as cp
        import numpy as np
        from cupy_backends.cuda.libs import cublas

        if (not isinstance(rhok, cp.ndarray) or rhok.dtype != cp.float64 or rhok.ndim != 4
                or not rhok.flags.c_contiguous or rhok.shape[0] == 0
                or rhok.shape[1] < _MIN_ORBITALS or rhok.shape[2]*rhok.shape[3] < _MIN_REDUCTION
                or not isinstance(out, cp.ndarray) or out.dtype != cp.float64
                or out.shape != (rhok.shape[0], rhok.shape[1], rhok.shape[1])
                or not out.flags.c_contiguous or rhok.device.id != out.device.id
                or rhok.device.id != cp.cuda.Device().id):
            return False
        if rhok.data.ptr < out.data.ptr+out.nbytes and out.data.ptr < rhok.data.ptr+rhok.nbytes:
            return False
        batches, m = rhok.shape[:2]
        k = rhok.shape[2]*rhok.shape[3]
        plan = _workspace_plan(m, k, batches)
        if plan is None or not _int8_supported(rhok.device.id) or plan[2] > _available_memory()//4:
            return False
        padded_m, padded_k, _ = plan
        slices, partial, accum, exponents, flags = self._scratch(padded_m, padded_k, batches)
        row_exponents, split_rows, accumulate_pair, finish_gram = _kernels(rhok.device.id)
        row_exponents((batches*padded_m,), (256,),
                      (rhok, exponents, flags, np.int32(m), np.int32(k), np.int32(padded_m)))
        # Validate every batch before mutating any output; a rejected batch must
        # not leave a partial update that the native fallback would count twice.
        if int(flags.max()):
            return False
        handle = cp.cuda.device.get_cublas_handle()
        previous = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
        one, zero = np.asarray(1, dtype=np.int32), np.asarray(0, dtype=np.int32)
        plane = padded_m*padded_k
        try:
            cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
            cublas.setMathMode(handle, 1)  # CUBLAS_TENSOR_OP_MATH
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            for batch in range(batches):
                scale = exponents[batch*padded_m:(batch+1)*padded_m]
                split_rows((padded_m,), (256,),
                           (rhok[batch], slices, scale, np.int32(m), np.int32(k),
                            np.int32(padded_m), np.int32(padded_k)))
                accum.fill(0)
                # Seven signed magnitude bits bound each exact INT32 dot product:
                # padded_k*127**2 < 2**31 for every accepted reduction length.
                for diagonal in range(_SLICES):
                    # Qi Qj^T and Qj Qi^T share one integer product and its
                    # transpose: nine slices need 25 products instead of 45.
                    for left in range(diagonal//2+1):
                        right = diagonal-left
                        cublas.gemmEx(handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                                      padded_m, padded_m, padded_k, one.ctypes.data,
                                      slices.data.ptr+left*plane, 3, padded_k,
                                      slices.data.ptr+right*plane, 3, padded_k,
                                      zero.ctypes.data, partial.data.ptr, 10, padded_m,
                                      cublas.CUBLAS_COMPUTE_32I, cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
                        # CUDA datatype ABI: CUDA_R_8I=3 and CUDA_R_32I=10.
                        accumulate_pair(((padded_m*padded_m+255)//256,), (256,),
                                        (partial, accum, np.int64(padded_m*padded_m),
                                         np.int32(-7*(diagonal+2)), np.int32(padded_m), np.int32(left != right)))
                finish_gram(((m*m+255)//256,), (256,), (accum, scale, out[batch], np.int32(m), np.int32(padded_m)))
        finally:
            cublas.setStream(handle, previous[0])
            cublas.setMathMode(handle, previous[1])
            cublas.setPointerMode(handle, previous[2])
        return True
