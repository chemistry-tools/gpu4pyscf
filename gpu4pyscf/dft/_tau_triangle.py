"""Direct signed float64 tau products for medium RTX 3080 grid blocks."""

from __future__ import annotations

from functools import lru_cache

from gpu4pyscf.dft import _tau_syrk

_MIN_ORBITALS = 384
_MAX_ORBITALS = 512
_MIN_GRIDS = 2048
_MAX_GRIDS = 4096
_TILE = 16
_SOURCE = r'''
extern "C" __global__ void tau_triangle(
        double* out, const double* ao, const double* weights, int m, int n) {
    const int T = 16, K = 32;
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj > bi) return;
    // Padding keeps the column reads from colliding in shared-memory banks.
    __shared__ double a[T][K+1];
    __shared__ double b[T][K+1];
    int ti = threadIdx.y, tj = threadIdx.x;
    int tid = ti*T+tj;
    double value = 0.;
    for (int d=1; d<4; ++d) {
        for (int p=0; p<n; p+=K) {
            for (int q=tid; q<T*K; q+=T*T) {
                int row=q/K, k=q%K, g=p+k;
                int ri=bi*T+row, rj=bj*T+row;
                a[row][k] = (ri<m && g<n) ? ao[((long long)d*m+ri)*n+g] : 0.;
                b[row][k] = (rj<m && g<n) ? (.5*weights[g])*ao[((long long)d*m+rj)*n+g] : 0.;
            }
            __syncthreads();
            #pragma unroll
            for (int k=0; k<K; ++k) value = fma(a[ti][k], b[tj][k], value);
            __syncthreads();
        }
    }
    int ri=bi*T+ti, rj=bj*T+tj;
    if (ri<m && rj<m && rj<=ri) {
        out[(long long)ri*m+rj]=value;
        if (ri!=rj) out[(long long)rj*m+ri]=value;
    }
}
'''


@lru_cache(maxsize=16)
def _supported_device(device):
    import cupy as cp

    # This dispatch range is measured on GA102. Other architectures keep BLAS.
    return cp.cuda.Device(device).compute_capability == '86'


@lru_cache(maxsize=16)
def _kernel(device):
    import cupy as cp

    return cp.RawKernel(_SOURCE, 'tau_triangle')


def _eligible(bra, ket, weights, out):
    import cupy as cp

    if (
        not isinstance(bra, cp.ndarray)
        or bra.ndim != 3
        or not _MIN_ORBITALS <= bra.shape[1] <= _MAX_ORBITALS
        or not _MIN_GRIDS <= bra.shape[2] <= _MAX_GRIDS
        or not _supported_device(cp.cuda.Device().id)
        or not _tau_syrk._eligible(bra, ket, weights, out, minimum=_MIN_ORBITALS - 1)
    ):
        return False
    if out is not None:
        # Direct writes must not race reads from aliased AO or weight storage.
        start, end = out.data.ptr, out.data.ptr + out.nbytes
        if any(start < field.data.ptr + field.nbytes and field.data.ptr < end for field in (bra, weights)):
            return False
    return True


def _triangular_tau(ao, weights, out=None):
    import cupy as cp
    import numpy as np

    m, n = ao.shape[1:]
    if out is None:
        out = cp.empty((m, m), dtype=cp.float64)
    blocks = (m + _TILE - 1) // _TILE
    _kernel(cp.cuda.Device().id)(
        (blocks, blocks), (_TILE, _TILE), (out, ao, weights, np.int32(m), np.int32(n))
    )
    return out
