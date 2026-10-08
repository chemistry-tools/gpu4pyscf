"""VV10 pair kernels for verified mixed-precision DFT."""

from __future__ import annotations

import ctypes

# Adapted from gpu4pyscf/lib/gdft/vv10.cu at v1.8.1 (Apache-2.0).
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
# https://github.com/pyscf/gpu4pyscf/blob/v1.8.1/gpu4pyscf/lib/gdft/vv10.cu
_SOURCE = r"""
#ifndef BLOCK
#define BLOCK 128
#endif
#ifdef FLOAT_PAIR
typedef float Real;
#else
typedef double Real;
#endif
__device__ __forceinline__ double reciprocal(double d) {
#ifdef REFINE_RECIPROCAL
    if (d < 0x1p-1022 || d > 0x1p1022) return 1.0 / d;
    double r;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(d));
    r = fma(r, fma(-d, r, 1.0), r);
    r = fma(r, fma(-d, r, 1.0), r);
    return r;
#else
    return 1.0 / d;
#endif
}
extern "C" __global__ void pair(
        double* U, double* W, double* E, const double* coords,
        const double* rw, const double* omega, const double* kappa, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    bool active = i < n;
    Real oi = active ? omega[i] : 0;
    Real ki = active ? kappa[i] : 1;
    Real x = active ? coords[i*3] : 0;
    Real y = active ? coords[i*3+1] : 0;
    Real z = active ? coords[i*3+2] : 0;
    // Sum short chunks separately in float32; combine in float64.
    double us = 0, ws = 0, es = 0;
    __shared__ Real ox[BLOCK], kx[BLOCK], rx[BLOCK];
    __shared__ Real cx[BLOCK], cy[BLOCK], cz[BLOCK];
    for (int base = 0; base < n; base += BLOCK) {
        int j = base + threadIdx.x;
        if (j < n) {
            ox[threadIdx.x] = omega[j]; kx[threadIdx.x] = kappa[j];
            rx[threadIdx.x] = rw[j]; cx[threadIdx.x] = coords[j*3];
            cy[threadIdx.x] = coords[j*3+1]; cz[threadIdx.x] = coords[j*3+2];
        }
        __syncthreads();
        Real ub = 0, wb = 0, eb = 0;
        int end = min(BLOCK, n-base);
        for (int t = 0; t < end; ++t) {
            Real dx = x-cx[t], dy = y-cy[t], dz = z-cz[t];
            Real r2 = dx*dx + dy*dy + dz*dz;
            Real gi = oi*r2+ki, gj = ox[t]*r2+kx[t], gs = gi+gj;
#ifdef FLOAT_PAIR
            Real inv = 1.0f/(gi*gj*gs);
#else
            Real inv = reciprocal(gi*gj*gs);
#endif
            Real e = -rx[t]*inv;
            Real u = e*(gs+gi)*gj*inv;
            ub += u; wb += u*r2; eb += e;
        }
        us += ub; ws += wb; es += eb;
        __syncthreads();
    }
    if (active) { U[i] = -1.5*us; W[i] = -1.5*ws; E[i] = 1.5*es; }
}
"""


class _VV10:
    def __init__(self, mode='refined', block=128, backend='nvrtc', *, profile=False):
        import cupy as cp

        if mode not in ('compiled', 'refined', 'float32') or block not in (64, 128, 256):
            raise ValueError('Unknown VV10 kernel mode or block size')
        self.block = block
        self.mode = mode
        self.profile = profile
        options = ['-std=c++17', f'-DBLOCK={block}']
        if mode == 'refined':
            options.append('-DREFINE_RECIPROCAL')
        if mode == 'float32':
            options.append('-DFLOAT_PAIR')
        self.kernel = cp.RawKernel(_SOURCE, 'pair', options=tuple(options), backend=backend)
        self.pair_seconds = []
        self.active_grids = []

    def __call__(self, rho_drho, coords, weights, nlc_pars):
        import cupy as cp
        import numpy as np

        from gpu4pyscf.dft import numint
        from gpu4pyscf.lib.cupy_helper import batched_vec_norm2

        if rho_drho.dtype != cp.float64 or coords.dtype != cp.float64:
            raise TypeError('VV10 input fields must remain float64')
        nfull = coords.shape[0]
        assert rho_drho.shape == (4, nfull) and weights.shape == (nfull,)
        idx = cp.where((rho_drho[0] >= numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD) & (cp.abs(weights) > 1e-14))[0]
        rho = rho_drho[0, idx]
        r = cp.ascontiguousarray(coords[idx])
        gamma = batched_vec_norm2(rho_drho[1:4, idx].T)
        n = len(idx)
        if not n:
            return cp.zeros(nfull), cp.zeros((2, nfull))
        omega, dor, dog = [cp.empty(n) for _ in range(3)]
        stream = cp.cuda.get_current_stream()

        def ptr(a):
            return ctypes.cast(a.data.ptr, ctypes.c_void_p)

        err = numint.libgdft.VXC_vv10nlc_fock_eval_omega_derivative(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ptr(omega),
            ptr(dor),
            ptr(dog),
            ptr(rho),
            ptr(gamma),
            ctypes.c_double(nlc_pars[1]),
            ctypes.c_int(n),
        )
        if err:
            raise RuntimeError('CUDA error in VV10 omega derivative')
        kp = nlc_pars[0] * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6)
        beta = 0.03125 * (3 / nlc_pars[0] ** 2) ** 0.75
        kappa = kp * rho ** (1 / 6)
        rw = rho * weights[idx]
        u, w, e = [cp.empty(n) for _ in range(3)]
        if self.profile:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
        self.kernel(
            ((n + self.block - 1) // self.block,),
            (self.block,),
            (u, w, e, r, rw, omega, kappa, np.int32(n)),
        )
        if self.profile:
            end.record()
            end.synchronize()
            self.pair_seconds.append(cp.cuda.get_elapsed_time(begin, end) / 1000)
        self.active_grids.append(n)
        exc, vxc = cp.zeros(nfull), cp.zeros((2, nfull))
        exc[idx] = beta + 0.5 * e
        vxc[0, idx] = beta + e + rho * (kp * (1 / 6) * rho ** (-5 / 6) * u + dor * w)
        vxc[1, idx] = rho * dog * w
        return exc, vxc


class _AdaptiveVV10:
    def __init__(self, original, block, backend, *, profile=False):
        self.original = original
        self.coarse = _VV10('float32', block, backend, profile=profile)
        self.use_float32 = True
        self.precisions = []

    @property
    def pair_seconds(self):
        return self.coarse.pair_seconds

    def __call__(self, *args):
        self.precisions.append('float32' if self.use_float32 else 'float64')
        return self.coarse(*args) if self.use_float32 else self.original(*args)
