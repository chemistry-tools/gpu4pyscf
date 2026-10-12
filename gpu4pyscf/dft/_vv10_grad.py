"""Fused float64 VV10 fields and coordinate-response derivatives."""

from __future__ import annotations

from functools import lru_cache

# Pair expressions follow lib/gdft/vv10.cu (Apache-2.0).
_SOURCE = r"""
extern "C" __global__ void vv10_pair_response(
        double* U, double* W, double* E, double* response,
        const double* coords, const double* rw, const double* omega,
        const double* kappa, const int* atoms, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    bool active = i < n;
    double oi = active ? omega[i] : 0;
    double ki = active ? kappa[i] : 1;
    double x = active ? coords[i*3] : 0;
    double y = active ? coords[i*3+1] : 0;
    double z = active ? coords[i*3+2] : 0;
    int atom = active ? atoms[i] : -1;
    double u = 0, w = 0, e = 0;
    double gx = 0, gy = 0, gz = 0;
    __shared__ double ox[BLOCK], kx[BLOCK], rx[BLOCK];
    __shared__ double cx[BLOCK], cy[BLOCK], cz[BLOCK];
    __shared__ int ax[BLOCK];
    for (int base = 0; base < n; base += BLOCK) {
        int j = base + threadIdx.x;
        if (j < n) {
            ox[threadIdx.x] = omega[j]; kx[threadIdx.x] = kappa[j];
            rx[threadIdx.x] = rw[j]; cx[threadIdx.x] = coords[j*3];
            cy[threadIdx.x] = coords[j*3+1]; cz[threadIdx.x] = coords[j*3+2];
            ax[threadIdx.x] = atoms[j];
        }
        __syncthreads();
        int end = min(BLOCK, n-base);
        for (int t = 0; t < end; ++t) {
            double dx = x-cx[t], dy = y-cy[t], dz = z-cz[t];
            double r2 = dx*dx + dy*dy + dz*dz;
            double gi = oi*r2+ki, gj = ox[t]*r2+kx[t], gs = gi+gj;
            double inv = 1.0/(gi*gj*gs);
            double ei = -rx[t]*inv;
            double ui = ei*(gs+gi)*gj*inv;
            u += ui; w += ui*r2; e += ei;
            if (atom != ax[t]) {
                double dphi = (oi*gj*gs + ox[t]*gi*gs + (oi+ox[t])*gi*gj)*inv;
                double gr = ei*dphi;
                gx += gr*dx; gy += gr*dy; gz += gr*dz;
            }
        }
        __syncthreads();
    }
    if (active) {
        U[i] = -1.5*u; W[i] = -1.5*w; E[i] = 1.5*e;
        // rw_i*q_ij = -rw_j*q_ji: the native off-diagonal and diagonal
        // grid responses contribute equally to the atom containing point i.
        double factor = -3.0*rw[i];
        response[i] = factor*gx;
        response[n+i] = factor*gy;
        response[2*n+i] = factor*gz;
    }
}
extern "C" __global__ void vv10_reduce_response(
        double* result, const double* response, const int* offsets, int n) {
    int atom = blockIdx.x, direction = blockIdx.y;
    double value = 0;
    for (int i = offsets[atom]+threadIdx.x; i < offsets[atom+1]; i += blockDim.x)
        value += response[direction*n+i];
    __shared__ double scratch[256];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x/2; width > 0; width /= 2) {
        if (threadIdx.x < width) scratch[threadIdx.x] += scratch[threadIdx.x+width];
        __syncthreads();
    }
    if (threadIdx.x == 0) result[atom*3+direction] = scratch[0];
}
"""


@lru_cache(maxsize=24)
def _kernels(device, block):
    import cupy as cp

    options = ('-std=c++17', f'-DBLOCK={block}')
    return (
        cp.RawKernel(_SOURCE, 'vv10_pair_response', options=options),
        cp.RawKernel(_SOURCE, 'vv10_reduce_response', options=options),
    )


def _pair_response(coords, rw, omega, kappa, atoms, natm, *, block=128):
    """Return native U/W/E fields and the fully contracted coordinate response.

    Atom labels must be nonnegative and sorted, as in atom-ordered gradient grids.
    Same-atom pairs contribute to U/W/E but cancel in the coordinate response.
    The final reduction is deterministic and uses no floating-point atomics.
    """
    import cupy as cp
    import numpy as np

    if type(natm) is not int or natm < 1 or block not in (64, 128, 256):
        raise ValueError('Use a positive atom count and a supported tile size')
    n = len(rw)
    arrays = (coords, rw, omega, kappa)
    if any(not isinstance(a, cp.ndarray) or a.dtype != cp.float64 or not a.flags.c_contiguous for a in arrays):
        raise TypeError('VV10 fields must be contiguous float64 CUDA arrays')
    if coords.shape != (n, 3) or any(a.shape != (n,) for a in (rw, omega, kappa)):
        raise ValueError('VV10 fields must have matching grid dimensions')
    if not isinstance(atoms, cp.ndarray) or atoms.dtype != cp.int32 or not atoms.flags.c_contiguous:
        raise TypeError('Grid atom labels must be contiguous int32 CUDA arrays')
    if atoms.shape != (n,):
        raise ValueError('Grid atom labels must match the fields')
    if n > np.iinfo(np.int32).max:
        raise ValueError('VV10 grid exceeds the native int32 index range')
    u, w, e = (cp.empty(n, dtype=cp.float64) for _ in range(3))
    result = cp.zeros((natm, 3), dtype=cp.float64)
    if not n:
        return u, w, e, result
    if bool(cp.any((atoms < 0) | (atoms >= natm))) or bool(cp.any(atoms[1:] < atoms[:-1])):
        raise ValueError('Grid atom labels must be valid and sorted')
    counts = cp.bincount(atoms, minlength=natm)
    offsets = cp.empty(natm+1, dtype=cp.int32)
    offsets[0] = 0
    offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
    response = cp.empty((3, n), dtype=cp.float64)
    pair, reduce = _kernels(cp.cuda.Device().id, block)
    pair(((n+block-1)//block,), (block,),
         (u, w, e, response, coords, rw, omega, kappa, atoms, np.int32(n)))
    reduce((natm, 3), (256,), (result, response, offsets, np.int32(n)))
    return u, w, e, result
