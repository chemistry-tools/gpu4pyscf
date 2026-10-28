"""Measure native direct-screening survivors independently of integral timings."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_CUTOFFS = (1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)
_MODES = {'exchange': 0, 'force-jk': 1, 'force-k': 2}
_SOURCE = r'''
#include <cuda_runtime.h>
#include "create_tasks.cu"

extern "C" __global__ void stage_counts(
    const unsigned *pa, const float *qa, unsigned na,
    const unsigned *pb, const float *qb, unsigned nb,
    const float *dm, unsigned nbas, float dmax, int mode,
    const float *cuts, unsigned long long *out)
{
    unsigned long long sums[13] = {};
    unsigned long long size = (unsigned long long)na * nb;
    for (unsigned long long p = blockIdx.x * blockDim.x + threadIdx.x;
         p < size; p += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned a = p / nb, b = p % nb;
        unsigned ij = pa[a], kl = pb[b];
        if (ij < kl) continue;
        ++sums[0];
        unsigned i = ij / nbas, j = ij % nbas;
        unsigned k = kl / nbas, l = kl % nbas;
        float d;
        if (mode == 0) {
            d = fmaxf(fmaxf(dm[i*nbas+k], dm[j*nbas+k]),
                      fmaxf(dm[i*nbas+l], dm[j*nbas+l]));
        } else {
            d = fmaxf(dm[i*nbas+l]+dm[j*nbas+k], dm[i*nbas+k]+dm[j*nbas+l]);
            if (mode == 1) d = fmaxf(d, dm[ij]+dm[kl]);
        }
        for (unsigned c = 0; c < 6; ++c) {
            float bound = cuts[c] - qa[a];
            if (qb[b] + (mode == 0 ? dmax : 0.f) < bound) continue;
            ++sums[1+c*2];
            if (d > bound-qb[b]) ++sums[2+c*2];
        }
    }
    for (unsigned c = 0; c < 13; ++c) {
        unsigned long long v = sums[c];
        for (int delta = 16; delta; delta >>= 1)
            v += __shfl_down_sync(0xffffffff, v, delta);
        if (threadIdx.x % 32 == 0) atomicAdd(out+c, v);
    }
}

// Run the actual production queue helper, independently of the stage replay.
// Persistent blocks bound the scratch queue even for large pair panels.
extern "C" __global__ void native_survivors(
    unsigned *pa, float *qa, unsigned na,
    unsigned *pb, float *qb, unsigned nb,
    float *dm, unsigned nbas, float dmax, int mode, float cutoff,
    unsigned *pool, unsigned long long *out)
{
    __shared__ int ntasks, pair_kl0;
    __shared__ int swap[256];
    unsigned batches = (nb + QUEUE_DEPTH-513) / (QUEUE_DEPTH-512);
    RysIntEnvVars envs = {}; envs.nbas = nbas;
    BoundsInfo bounds = {};
    bounds.npairs_ij = na; bounds.npairs_kl = nb;
    bounds.pair_ij_mapping = pa; bounds.pair_kl_mapping = pb;
    bounds.dm_cond = dm; bounds.cutoff = cutoff;
    JKEnergy jk = {}; jk.j_factor = mode == 1; jk.k_factor = 1.;
    for (unsigned long long job = blockIdx.x; job < (unsigned long long)na*batches;
         job += gridDim.x) {
        unsigned a = job / batches;
        if (threadIdx.x == 0) pair_kl0 = (job % batches) * (QUEUE_DEPTH-512);
        __syncthreads();
        unsigned ij = pa[a];
        if (mode == 0)
            _fill_vk_tasks(ntasks, pair_kl0, pool+blockIdx.x*QUEUE_DEPTH,
                a, ij/nbas, ij%nbas, qa, qb, dmax, swap, envs, bounds);
        else
            _fill_ejk_tasks(ntasks, pair_kl0, pool+blockIdx.x*QUEUE_DEPTH,
                a, ij/nbas, ij%nbas, qa, qb, swap, jk, envs, bounds);
        if (threadIdx.x == 0) atomicAdd(out, (unsigned long long)ntasks);
        __syncthreads();
    }
}
'''


def screen_cpu(pa, qa, pb, qb, dm, mode, cutoffs=_CUTOFFS):
    """Provide an independent small-array oracle with native float32 comparisons."""
    pa, pb = np.asarray(pa), np.asarray(pb)
    qa, qb, dm = (np.asarray(x, dtype=np.float32) for x in (qa, qb, dm))
    nbas = dm.shape[0]
    counts = np.zeros((len(cutoffs), 3), dtype=np.uint64)
    for a, ij in enumerate(pa):
        i, j = divmod(int(ij), nbas)
        for b, kl in enumerate(pb):
            if ij < kl:
                continue
            k, l = divmod(int(kl), nbas)
            if mode == 'exchange':
                local = max(dm[i, k], dm[j, k], dm[i, l], dm[j, l])
            else:
                local = max(dm[i, l]+dm[j, k], dm[i, k]+dm[j, l])
                if mode == 'force-jk':
                    local = max(local, dm[i, j]+dm[k, l])
            for c, cutoff in enumerate(cutoffs):
                counts[c, 0] += 1
                bound = np.float32(np.float32(math.log(cutoff))-qa[a])
                if np.float32(qb[b]+(dm.max() if mode == 'exchange' else 0)) < bound:
                    continue
                counts[c, 1] += 1
                if local > np.float32(bound-qb[b]):
                    counts[c, 2] += 1
    return counts


class ScreeningCounter:
    """Replay full/LR screening and check survivors against the native helper."""

    def __init__(self, native_source):
        import cupy as cp

        from gpu4pyscf.scf.jk import QUEUE_DEPTH

        self.source = Path(native_source).resolve()
        self.source_sha256 = {name: hashlib.sha256((self.source/name).read_bytes()).hexdigest()
                              for name in ('create_tasks.cu', 'vhf.cuh')}
        self.module = cp.RawModule(code=_SOURCE, backend='nvcc',
                                   options=('--std=c++17', '-I'+str(self.source)))
        self.replay = self.module.get_function('stage_counts')
        self.native = self.module.get_function('native_survivors')
        self.cuts = cp.asarray(np.log(_CUTOFFS).astype(np.float32))
        self.workers = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']
        self.pool = cp.empty(self.workers*QUEUE_DEPTH, dtype=cp.uint32)

    def count(self, pair_a, q_a, pair_b, q_b, dm, mode):
        import cupy as cp

        if mode not in _MODES:
            raise ValueError('Unsupported screening mode')
        pa, pb = (cp.asarray(x, dtype=cp.uint32) for x in (pair_a, pair_b))
        qa, qb, dm = (cp.asarray(x, dtype=cp.float32, order='C') for x in (q_a, q_b, dm))
        if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
            raise ValueError('Density bounds must be a square matrix')
        if qa.shape != pa.shape or qb.shape != pb.shape or pa.ndim != 1 or pb.ndim != 1:
            raise ValueError('Pair and bound shapes must match')
        if any(not bool(cp.all(cp.isfinite(x))) for x in (qa, qb, dm)):
            raise ValueError('Screening bounds must be finite')
        if any(x.size and int(x.max()) >= dm.size for x in (pa, pb)):
            raise ValueError('Pair index exceeds density bounds')
        if pa.size*pb.size > np.iinfo(np.int64).max:
            raise ValueError('Pair product exceeds the counter range')
        if not pa.size or not pb.size:
            return {'nominal': 0, 'native_survivors': 0, 'stages': [[0, 0, 0] for _ in _CUTOFFS]}
        args = (pa, qa, np.uint32(pa.size), pb, qb, np.uint32(pb.size), dm,
                np.uint32(dm.shape[0]), np.float32(float(dm.max())), np.int32(_MODES[mode]))
        counts = cp.zeros(13, dtype=cp.uint64)
        self.replay((min(1024, (pa.size*pb.size+255)//256),), (256,), (*args, self.cuts, counts))
        native_count = cp.zeros(1, dtype=cp.uint64)
        self.native((self.workers,), (256,), (*args, np.float32(float(self.cuts[0])), self.pool, native_count))
        values = cp.asnumpy(counts).astype(object)
        actual = int(native_count[0])
        if actual != int(values[2]):
            raise RuntimeError(f'Native screening/replay mismatch: {actual} != {values[2]}')
        stages = [[int(values[0]), int(values[1+2*c]), int(values[2+2*c])] for c in range(6)]
        return {'nominal': pa.size*pb.size, 'native_survivors': actual, 'stages': stages}


@contextmanager
def _native_timings(library, name, slice_argument):
    import cupy as cp

    original = getattr(library, name)
    rows = []

    def measured(*args):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        status = original(*args)
        end.record()
        end.synchronize()
        if status:
            raise RuntimeError(f'{name} failed with status {status}')
        rows.append({'shell_slice': list(args[slice_argument]),
                     'gpu_seconds': cp.cuda.get_elapsed_time(start, end)*1e-3})
        return status

    setattr(library, name, measured)
    try:
        yield rows
    finally:
        setattr(library, name, original)


def profile_screening(mf, density, native_source, output):
    """Pair untouched native integral timings with separately evaluated counters."""
    import cupy as cp

    from gpu4pyscf.lib.cupy_helper import condense
    from gpu4pyscf.scf import jk

    counter = ScreeningCounter(native_source)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
    if getattr(mf, 'with_df', None) is not None or mf.mol.spin or omega <= 0 or mf.direct_scf_tol != 1e-13:
        raise ValueError('This diagnostic requires the qualified restricted full/LR direct recipe')
    delta = .001*(cp.asarray(mf.get_init_guess(mf.mol, 'minao'))-density)
    records = []
    for label, key, mode, dm in (('exchange-full-density', omega, 'exchange', density),
                               ('exchange-density-difference', omega, 'exchange', delta),
                               ('force-full', None, 'force-jk', density),
                               ('force-long-range', omega, 'force-k', density)):
        opt = mf._get_jk_opt(mf.mol, key)
        mol = opt.sorted_mol
        if opt.diffuse_cutoff != 10.:
            raise ValueError('Qualified regrouping was not selected')
        if mode == 'exchange':
            operation = lambda dm=dm: mf.get_k(mf.mol, dm, 1, omega, alpha, hyb)
            native_name, slice_argument = 'RYS_build_k', 8
        else:
            operation = lambda dm=dm, mode=mode, key=key: mf.Gradients().jk_energy_per_atom(
                dm, float(mode == 'force-jk'), hyb if mode == 'force-jk' else alpha-hyb, omega=key)
            native_name, slice_argument = 'RYS_per_atom_jk_ip1_contract', 11
        reference = cp.asarray(operation()).copy()
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        with _native_timings(jk.libvhf_rys, native_name, slice_argument) as timings:
            result = cp.asarray(operation())
        cp.cuda.runtime.deviceSynchronize()
        seconds = time.perf_counter()-started
        error = float(cp.max(cp.abs(result-reference)))
        cp.testing.assert_allclose(result, reference, atol=2e-8 if mode != 'exchange' else 2e-9, rtol=0)
        sorted_dm = opt.apply_coeff_C_mat_CT(dm[None])
        density_log = cp.log(condense('absmax', sorted_dm, mol.ao_loc)+1e-300).astype(cp.float32)
        offsets = np.append(0, np.cumsum(mol.l_ctr_counts))
        groups = {int(offset): i for i, offset in enumerate(offsets[:-1])}
        rows = []
        for timing in timings:
            indices = tuple(groups[s] for s in timing['shell_slice'][::2])
            i, j, k, l = indices
            pa, qa, _ = opt.bas_pair_cache[i, j]
            pb, qb, _ = opt.bas_pair_cache[k, l]
            values = counter.count(pa, qa, pb, qb, density_log, mode)
            pattern = mol.uniq_l_ctr[list(indices)]
            primitive_products = math.prod(int(p[1]) for p in pattern)
            cartesian_products = math.prod((int(p[0])+1)*(int(p[0])+2)//2 for p in pattern)
            rows.append(dict(timing, **values, group_indices=indices, pattern=pattern.tolist(),
                             primitive_products=primitive_products, cartesian_products=cartesian_products))
        record = {'label': label, 'operator_omega': key, 'mode': mode, 'seconds': seconds,
                  'native_kernel_seconds': sum(t['gpu_seconds'] for t in timings), 'repeat_max_abs': error,
                  'internal_nao': mol.nao, 'internal_shells': mol.nbas, 'rows': rows}
        records.append(record)
        np.save(output/(label+'.npy'), cp.asnumpy(result))
        (output/'screening-progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print(label, seconds, len(rows), flush=True)
    return {'records': records, 'cutoffs': _CUTOFFS, 'native_screening_source_sha256': counter.source_sha256,
            'counter_source_sha256': hashlib.sha256(_SOURCE.encode()).hexdigest(),
            'scope': 'Fixed-density direct K and full/LR J/K force components. Screening counts run separately from native timings. Looser-cutoff counts are hypothetical, not energies or measured speedups.'}
