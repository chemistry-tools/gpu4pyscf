"""Bounded sparse AO products using the native density-fitting pair list."""

from __future__ import annotations

import math
from functools import lru_cache

_MIN_ORBITALS = 256
_MAX_MAP_BYTES = 64*1024**2
_MAX_WORKSPACE_BYTES = 512*1024**2
_MAX_RETAINED_FRACTION = .85
_PANEL = 256
_SOURCE = r'''
template<typename T> __device__ void pack_projection(T* out, const double* src,
        const int* lookup, int ni, int nj, int nl, long long sl, long long sp) {
    long long x = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (x >= (long long)ni*nl*nj) return;
    int j=x%nj, l=x/nj%nl, i=x/nj/nl, p=lookup[i*nj+j];
    out[x] = p < 0 ? T(0) : T(src[l*sl+(long long)p*sp]);
}
extern "C" __global__ void projection64(double* out, const double* src, const int* lookup,
        int ni, int nj, int nl, long long sl, long long sp) {
    pack_projection(out, src, lookup, ni, nj, nl, sl, sp);
}
extern "C" __global__ void projection32(float* out, const double* src, const int* lookup,
        int ni, int nj, int nl, long long sl, long long sp) {
    pack_projection(out, src, lookup, ni, nj, nl, sl, sp);
}
template<typename T> __device__ __forceinline__ void pack_transposed(T* out, const double* src,
        const int* lookup, int ni, int nj, int nl, long long sp) {
    __shared__ T tile[32][33];
    int l = blockIdx.y*32+threadIdx.x;
    for (int step=0; step<32; step+=8) {
        int pair=blockIdx.x*32+threadIdx.y+step;
        int p=pair < ni*nj ? lookup[pair] : -1;
        tile[threadIdx.y+step][threadIdx.x] = l < nl && p >= 0 ? T(src[l+(long long)p*sp]) : T(0);
    }
    __syncthreads();
    int pair=blockIdx.x*32+threadIdx.x, i=pair/nj, j=pair%nj;
    for (int step=0; step<32; step+=8) {
        l=blockIdx.y*32+threadIdx.y+step;
        if (pair < ni*nj && l < nl) out[((long long)i*nl+l)*nj+j] = tile[threadIdx.x][threadIdx.y+step];
    }
}
extern "C" __global__ void projection_transposed64(double* out, const double* src,
        const int* lookup, int ni, int nj, int nl, long long sp) {
    pack_transposed(out, src, lookup, ni, nj, nl, sp);
}
extern "C" __global__ void projection_transposed32(float* out, const double* src,
        const int* lookup, int ni, int nj, int nl, long long sp) {
    pack_transposed(out, src, lookup, ni, nj, nl, sp);
}
template<typename T> __device__ void scatter_projection(double* out, const T* src,
        const int* rows, int ni, int nk, int nl, long long si, long long sk, long long sl) {
    long long x = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (x >= (long long)ni*nk*nl) return;
    int l=x%nl, k=x/nl%nk, i=x/nl/nk;
    out[rows[i]*si+k*sk+l*sl] = double(src[((long long)i*nl+l)*nk+k]);
}
extern "C" __global__ void result64(double* out, const double* src, const int* rows,
        int ni, int nk, int nl, long long si, long long sk, long long sl) {
    scatter_projection(out, src, rows, ni, nk, nl, si, sk, sl);
}
extern "C" __global__ void result32(double* out, const float* src, const int* rows,
        int ni, int nk, int nl, long long si, long long sk, long long sl) {
    scatter_projection(out, src, rows, ni, nk, nl, si, sk, sl);
}
extern "C" __global__ void pack_back(double* out, const double* src, const int* cols,
        int nk, int nj, int nl, long long sk, long long sj, long long sl) {
    long long x = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (x >= (long long)nk*nj*nl) return;
    int l=x%nl, j=x/nl%nj, k=x/nl/nj;
    out[x] = src[k*sk+cols[j]*sj+l*sl];
}
extern "C" __global__ void result_back(double* out, const double* src, const int* lookup,
        const int* rows, const int* cols, const double* dm, const double* aux,
        int ni, int nj, int nl, int nao, long long op, long long ol,
        long long di, long long dj, long long al, double alpha, double beta) {
    long long x = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (x >= (long long)ni*nj*nl) return;
    int l=x%nl, j=x/nl%nj, i=x/nl/nj, p=lookup[i*nj+j];
    if (p >= 0) {
        double value = alpha*src[x];
        if (beta != 0) value += beta*dm[rows[i]*di+cols[j]*dj]*aux[l*al];
        out[p*op+l*ol] = value;
    }
}
'''


def _cpu_plan(addresses, n, width, symmetric):
    import numpy as np

    if (not isinstance(n, int) or n <= 0 or n*n*4 > _MAX_MAP_BYTES
            or not isinstance(width, int) or not 0 < width <= 128):
        return None
    addresses = np.asarray(addresses)
    if (addresses.ndim != 1 or addresses.dtype.kind not in 'iu' or not len(addresses)
            or len(addresses) > n*n or np.any(addresses < 0) or np.any(addresses >= n*n)
            or len(np.unique(addresses)) != len(addresses)):
        return None
    rows, cols = divmod(addresses.astype(np.int64), n)
    directed = np.full((n, n), -1, dtype=np.int32)
    directed[rows, cols] = np.arange(len(addresses), dtype=np.int32)
    duplicates = None
    if symmetric:
        both = (directed >= 0) & (directed.T >= 0)
        i, j = np.where(np.triu(both, 1))
        duplicates = directed[i, j], directed[j, i]
        # Native diagonal shell blocks contain both orientations. Choose one
        # stored representative for both; the caller checks their compatibility.
        lookup = np.maximum(directed, directed.T)
    else:
        lookup = directed
    mask = lookup >= 0
    degrees = mask.sum(axis=1)
    bits = [int.from_bytes(np.packbits(row).tobytes(), 'little') for row in mask]
    remaining = set(np.flatnonzero(degrees).tolist())
    groups, elements = [], 0
    while remaining:
        seed = min(remaining, key=lambda i: (degrees[i], i))
        chosen, union = [seed], bits[seed]
        remaining.remove(seed)
        while remaining and len(chosen) < width:
            row = min(remaining, key=lambda i: ((union | bits[i]).bit_count(), i))
            union |= bits[row]
            chosen.append(row)
            remaining.remove(row)
        row = np.asarray(sorted(chosen), dtype=np.int32)
        col = np.flatnonzero(mask[row].any(axis=0)).astype(np.int32)
        mapping = np.ascontiguousarray(lookup[np.ix_(row, col)])
        groups.append((row, col, mapping))
        elements += len(row)*len(col)
    return {'groups': groups, 'duplicates': duplicates, 'elements': elements,
            'rows': int(np.count_nonzero(degrees)), 'n': n, 'pairs': len(addresses)}


def _supported_device(device):
    import cupy as cp

    return cp.cuda.Device(device).compute_capability == '86'


def _available_memory():
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()[0]+cp.get_default_memory_pool().free_bytes()


@lru_cache(maxsize=16)
def _kernels(device):
    import cupy as cp

    names = ('projection64', 'projection32', 'projection_transposed64', 'projection_transposed32',
             'result64', 'result32', 'pack_back', 'result_back')
    result = {name: cp.RawKernel(_SOURCE, name) for name in names}
    result['finite'] = cp.ReductionKernel('double x', 'bool valid', 'isfinite(x)',
                                          'a && b', 'valid = a', 'true', 'finite_sparse_df')
    result['compatible'] = cp.ReductionKernel(
        'double x, double y', 'bool valid',
        'isfinite(x) && isfinite(y) && abs(x-y) <= 1.4210854715202004e-14*max(1.,max(abs(x),abs(y)))',
        'a && b', 'valid = a', 'true', 'compatible_sparse_df')
    return result


def _array(value, ndim, device):
    import cupy as cp

    return (isinstance(value, cp.ndarray) and value.dtype == cp.float64 and value.ndim == ndim
            and value.device.id == device and all(n > 0 for n in value.shape)
            and all(s > 0 and s % 8 == 0 for n, s in zip(value.shape, value.strides) if n > 1))


def _span(value):
    return sum((n-1)*s for n, s in zip(value.shape, value.strides))+value.itemsize


def _overlap(left, right):
    return (left.data.ptr < right.data.ptr+_span(right)
            and right.data.ptr < left.data.ptr+_span(left))


def _nonoverlapping(value):
    covered = value.itemsize
    for stride, n in sorted((s, n) for s, n in zip(value.strides, value.shape) if n > 1):
        if stride < covered:
            return False
        covered += (n-1)*stride
    return True


def _launch(kernel, size, args):
    kernel(((size+255)//256,), (256,), args)


class _SparseAOPlan:
    """Keep only index metadata; geometry and density fields are never cached."""

    @classmethod
    def create(cls, owner, addresses, n, *, symmetric=True, width=8):
        import cupy as cp
        import numpy as np

        device = cp.cuda.device.get_device_id()
        if n < _MIN_ORBITALS or n*n*4 > _MAX_MAP_BYTES or not _supported_device(device):
            return None
        if (not isinstance(addresses, cp.ndarray) or addresses.ndim != 1
                or addresses.dtype.kind not in 'iu' or addresses.device.id != device):
            return None
        snapshot = cp.asnumpy(addresses)
        key = n, device, cp.cuda.get_current_stream().ptr, snapshot.dtype.str, snapshot.tobytes()
        cache = getattr(owner, '_sparse_ao_cache', None)
        if cache is None or cache[0] != key:
            cache = key, {}
            owner._sparse_ao_cache = cache
        variant = bool(symmetric), width
        if variant in cache[1]:
            return cache[1][variant]
        if len(cache[1]) >= 3:
            cache[1].clear()
        data = _cpu_plan(snapshot, int(n), width, symmetric)
        if data is None or data['elements'] > n*n*_MAX_RETAINED_FRACTION:
            cache[1][variant] = None
            return None
        try:
            self = cls()
            self.n, self.pairs, self.rows = n, data['pairs'], data['rows']
            self.device, self.stream = device, cp.cuda.get_current_stream()
            self.symmetric = symmetric
            self.groups = [(cp.asarray(r), cp.asarray(c), cp.asarray(p)) for r, c, p in data['groups']]
            self.duplicates = (tuple(cp.asarray(v, dtype=np.int32) for v in data['duplicates'])
                               if symmetric and len(data['duplicates'][0]) else None)
            self.index_bytes = sum(v.nbytes for group in self.groups for v in group)
            if self.duplicates:
                self.index_bytes += sum(v.nbytes for v in self.duplicates)
        except cp.cuda.memory.OutOfMemoryError:
            return None
        cache[1][variant] = self
        return self

    def prepare(self, factors, *, single=False):
        return _SparseAOProduct.create(self, factors, single=single)


class _SparseAOProduct:
    """Own per-call coefficient tiles and bounded GEMM buffers on one stream."""

    @classmethod
    def create(cls, plan, factors, *, single=False):
        import cupy as cp

        if (not _array(factors, 3, plan.device) or factors.shape[1] != plan.n
                or factors.shape[0] not in (1, 2) or factors.shape[2] > 2**31-1
                or cp.cuda.device.get_device_id() != plan.device
                or cp.cuda.get_current_stream().ptr != plan.stream.ptr
                or single and not plan.symmetric):
            return None
        if not bool(_kernels(plan.device)['finite'](factors)):
            return None
        batches, _, k = factors.shape
        dtype = cp.float32 if single else cp.float64
        width = 4 if single else 8
        coefficient_elements = sum((len(c) if plan.symmetric else len(r))*k*batches
                                   for r, c, _ in plan.groups)
        if plan.symmetric:
            packed = max(len(r)*len(c) for r, c, _ in plan.groups)*_PANEL
            product = max(len(r) for r, _, _ in plan.groups)*k*_PANEL
        else:
            packed = k*max(len(c) for _, c, _ in plan.groups)*_PANEL
            product = max(len(r)*len(c) for r, c, _ in plan.groups)*_PANEL
        workspace = plan.index_bytes+(coefficient_elements+packed+product)*width
        if workspace > _MAX_WORKSPACE_BYTES or workspace > _available_memory()//2:
            return None
        try:
            self = cls()
            self.plan, self.base, self.k, self.batches = plan, factors, k, batches
            self.single, self.workspace_bytes = single, workspace
            self.coefficients = []
            for r, c, _ in plan.groups:
                indices = c if plan.symmetric else r
                self.coefficients.append([cp.ascontiguousarray(cp.take(factor, indices, axis=0), dtype=dtype)
                                          for factor in factors])
            self.packed, self.product = cp.empty(packed, dtype=dtype), cp.empty(product, dtype=dtype)
        except cp.cuda.memory.OutOfMemoryError:
            return None
        return self

    def _current(self):
        import cupy as cp

        return (cp.cuda.device.get_device_id() == self.plan.device
                and cp.cuda.get_current_stream().ptr == self.plan.stream.ptr)

    def project(self, packed, out, batch_start=0):
        """Write (batch,AO,occupied,aux), or reject before changing output."""
        import cupy as cp
        import numpy as np
        from cupy_backends.cuda.libs import cublas

        plan = self.plan
        if (not plan.symmetric or not self._current() or not _array(packed, 2, plan.device)
                or not _array(out, 4, plan.device) or packed.shape[1] != plan.pairs
                or not isinstance(batch_start, int) or batch_start < 0
                or batch_start+out.shape[0] > self.batches
                or out.shape[1:] != (plan.n, self.k, packed.shape[0])
                or not _nonoverlapping(out)
                or any(_overlap(out, value) for value in (packed, self.base, self.packed, self.product))):
            return False
        kernels = _kernels(plan.device)
        raw = packed if packed.strides[1] == 8 else packed.T
        if not bool(kernels['finite'](raw)):
            return False
        if plan.duplicates:
            left, right = plan.duplicates
            if not bool(kernels['compatible'](packed[:, left], packed[:, right])):
                return False
        handle = cp.cuda.device.get_cublas_handle()
        state = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
        scalar_dtype = np.float32 if self.single else np.float64
        one, zero = np.asarray(1., dtype=scalar_dtype), np.asarray(0., dtype=scalar_dtype)
        gemm = cublas.sgemm if self.single else cublas.dgemm
        pack_kernel = kernels['projection32' if self.single else 'projection64']
        result_kernel = kernels['result32' if self.single else 'result64']
        try:
            cublas.setStream(handle, plan.stream.ptr)
            cublas.setMathMode(handle, 2)
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            if plan.rows != plan.n:
                out.fill(0)
            for first in range(0, packed.shape[0], _PANEL):
                nl = min(_PANEL, packed.shape[0]-first)
                source = packed[first:first+nl]
                for (rows, cols, lookup), coefficients in zip(plan.groups, self.coefficients):
                    ni, nj = len(rows), len(cols)
                    if source.strides[0] == 8:
                        tiled = kernels['projection_transposed32' if self.single else 'projection_transposed64']
                        tiled(((ni*nj+31)//32, (nl+31)//32), (32, 8),
                              (self.packed, source, lookup, np.int32(ni), np.int32(nj),
                               np.int32(nl), np.int64(source.strides[1]//8)))
                    else:
                        _launch(pack_kernel, ni*nl*nj, (self.packed, source, lookup, np.int32(ni),
                                np.int32(nj), np.int32(nl), np.int64(source.strides[0]//8),
                                np.int64(source.strides[1]//8)))
                    for batch, coefficient in enumerate(coefficients[batch_start:batch_start+out.shape[0]]):
                        gemm(handle, 0, 0, self.k, ni*nl, nj, one.ctypes.data,
                             coefficient.data.ptr, self.k, self.packed.data.ptr, nj,
                             zero.ctypes.data, self.product.data.ptr, self.k)
                        target = out[batch, :, :, first:first+nl]
                        _launch(result_kernel, ni*self.k*nl, (target, self.product, rows,
                                np.int32(ni), np.int32(self.k), np.int32(nl),
                                *(np.int64(s//8) for s in target.strides)))
        finally:
            cublas.setStream(handle, state[0])
            cublas.setMathMode(handle, state[1])
            cublas.setPointerMode(handle, state[2])
        return True

    def back(self, values, out, alpha, beta=0., density=None, auxiliary=None):
        """Write only the native directed AO pairs, including Coulomb response."""
        import cupy as cp
        import numpy as np
        from cupy_backends.cuda.libs import cublas

        plan = self.plan
        if (plan.symmetric or self.single or not self._current()
                or not _array(values, 4, plan.device) or not _array(out, 2, plan.device)
                or values.shape[:3] != (self.batches, self.k, plan.n)
                or out.shape != (plan.pairs, values.shape[3])
                or not _nonoverlapping(out)
                or not math.isfinite(alpha) or not math.isfinite(beta)
                or any(_overlap(out, v) for v in (values, self.base, self.packed, self.product))):
            return False
        if beta != 0 and (not _array(density, 2, plan.device) or density.shape != (plan.n, plan.n)
                          or not _array(auxiliary, 1, plan.device) or auxiliary.shape != (values.shape[3],)
                          or _overlap(out, density) or _overlap(out, auxiliary)):
            return False
        kernels = _kernels(plan.device)
        if not bool(kernels['finite'](values)):
            return False
        handle = cp.cuda.device.get_cublas_handle()
        state = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
        one, zero = np.asarray(1., dtype=np.float64), np.asarray(0., dtype=np.float64)
        try:
            cublas.setStream(handle, plan.stream.ptr)
            cublas.setMathMode(handle, 2)
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
            for first in range(0, values.shape[3], _PANEL):
                nl = min(_PANEL, values.shape[3]-first)
                for (rows, cols, lookup), coefficients in zip(plan.groups, self.coefficients):
                    ni, nj = len(rows), len(cols)
                    for batch, coefficient in enumerate(coefficients):
                        source = values[batch, :, :, first:first+nl]
                        _launch(kernels['pack_back'], self.k*nj*nl, (self.packed, source, cols,
                                np.int32(self.k), np.int32(nj), np.int32(nl),
                                *(np.int64(s//8) for s in source.strides)))
                        cublas.dgemm(handle, 0, 0, nj*nl, ni, self.k, one.ctypes.data,
                                      self.packed.data.ptr, nj*nl, coefficient.data.ptr, self.k,
                                      (zero if batch == 0 else one).ctypes.data,
                                      self.product.data.ptr, nj*nl)
                    target = out[:, first:first+nl]
                    d = density if beta else self.base
                    a = auxiliary[first:first+nl] if beta else self.base
                    strides = density.strides if beta else (8, 8)
                    _launch(kernels['result_back'], ni*nj*nl,
                            (target, self.product, lookup, rows, cols, d, a, np.int32(ni),
                             np.int32(nj), np.int32(nl), np.int32(plan.n),
                             *(np.int64(s//8) for s in target.strides),
                             *(np.int64(s//8) for s in strides),
                             np.int64(auxiliary.strides[0]//8 if beta else 1),
                             np.float64(alpha), np.float64(beta)))
        finally:
            cublas.setStream(handle, state[0])
            cublas.setMathMode(handle, state[1])
            cublas.setPointerMode(handle, state[2])
        return True
