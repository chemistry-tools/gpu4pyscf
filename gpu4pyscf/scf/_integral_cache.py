"""Bounded float64 integral groups for repeated native direct exchange builds."""

from __future__ import annotations

import ctypes
import hashlib
import math


def _allocation_size(size):
    # CuPy's ordinary device pool rounds allocations to 512-byte units.
    return (int(size) + 511) // 512 * 512


def plan_groups(groups, pair_counts, budget, ao_loc_bytes):
    """Admit pair rectangles; a None budget requests every supported group."""
    remaining = None if budget is None else budget - _allocation_size(ao_loc_bytes)
    candidates = []
    for i, (li, pi) in enumerate(groups):
        for j in range(i+1):
            lj, pj = groups[j]
            for k in range(i+1):
                lk, pk = groups[k]
                for l in range(k+1):
                    ll, pl = groups[l]
                    ni, nk = pair_counts[i, j], pair_counts[k, l]
                    primitives = int(pi)*int(pj)*int(pk)*int(pl)
                    if not ni or not nk or budget is not None and primitives <= 1:
                        continue
                    nf = tuple((int(x)+1)*(int(x)+2)//2 for x in (li, lj, lk, ll))
                    nfij, nfkl = nf[0]*nf[1], nf[2]*nf[3]
                    # The native fill ABI uses int32 pair offsets and row strides.
                    if max(ni*nfij, nk*nfkl) >= 2**31:
                        if budget is None:
                            raise NotImplementedError('All-VRAM integral group exceeds the native int32 fill ABI')
                        continue
                    values = ni*nk*nfij*nfkl*8
                    size = (_allocation_size(values) + _allocation_size(ni*4)
                            + _allocation_size(nk*4))
                    roots = (int(li)+int(lj)+int(lk)+int(ll))//2+1
                    score = primitives*roots / math.prod(nf)
                    candidates.append((-score, size, (i, j, k, l)))
    selected = {}
    for _, size, task in sorted(candidates):
        if remaining is None or size <= remaining:
            selected[task] = size
            if remaining is not None:
                remaining -= size
    return selected


def _plan_size(plan, ao_loc_bytes):
    return sum(plan.values()) + (_allocation_size(ao_loc_bytes) if plan else 0)


def _require_capacity(required, free, total):
    # Subsequent SCF operations still need workspace outside the integral arrays.
    reserve = max(256*1024**2, total//20)
    available = max(0, free-reserve)
    if required > available:
        raise MemoryError(
            f'All-VRAM integral cache requires {required/1024**3:.3f} GiB; '
            f'{available/1024**3:.3f} GiB available after workspace reserve. '
            'Use a bounded direct_scf_cache_bytes budget or a larger GPU.')


def _available_capacity(cp):
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    if getattr(cp.cuda.get_allocator(), '__self__', None) is pool:
        # Released caches normally stay in CuPy's pool. Counting driver-free
        # memory alone would reject a replacement that can reuse those blocks.
        free += pool.free_bytes()
        limit = pool.get_limit()
        if limit:
            free = min(free, max(0, limit-pool.used_bytes()))
    return min(free, total), total


def _context_key(opt, operator, device, stream):
    mol = opt.sorted_mol
    digest = hashlib.blake2b(digest_size=16)
    for array in (mol._atm, mol._bas, mol._env):
        digest.update(array.tobytes())
    return (id(mol), digest.digest(), float(opt.direct_scf_tol),
            tuple(float(x) for x in operator), device, stream)


_CONTRACT = r'''
extern "C" __global__
void cached_k(const double* eri, const unsigned int* pair_a,
              const unsigned int* pair_b, const float* qa, const float* qb,
              const int* ao, const float* dm_cond, const double* dm, double* vk,
              int na, int nb, int nbas, int nao, int ni, int nj, int nk, int nl,
              float cutoff, float penalty) {
    const long long stride = (long long)nb*nk*nl;
    const long long size = (long long)na*ni*nj*stride;
    for (long long x = (long long)blockIdx.x*blockDim.x+threadIdx.x;
         x < size; x += (long long)gridDim.x*blockDim.x) {
        const long long row = x/stride;
        const int pa = row/(ni*nj), pb = (x%stride)/(nk*nl);
        const unsigned int a = pair_a[pa], b = pair_b[pb];
        if (a < b) continue;
        const float kl_cutoff = cutoff-qa[pa];
        if (qb[pb]+penalty < kl_cutoff) continue;
        const int ish = a/nbas, jsh = a%nbas, ksh = b/nbas, lsh = b%nbas;
        const float d_cutoff = kl_cutoff-qb[pb];
        if (!(dm_cond[ish*nbas+ksh] > d_cutoff ||
              dm_cond[jsh*nbas+ksh] > d_cutoff ||
              dm_cond[ish*nbas+lsh] > d_cutoff ||
              dm_cond[jsh*nbas+lsh] > d_cutoff)) continue;
        const int ij = row%(ni*nj), kl = (x%stride)%(nk*nl);
        const long long i = ao[ish]+ij%ni, j = ao[jsh]+ij/ni;
        const long long k = ao[ksh]+kl%nk, l = ao[lsh]+kl/nk;
        double value = eri[x];
        if (ksh == lsh) value *= .5;
        if (a == b) value *= .5;
        atomicAdd(vk+i*nao+l, value*dm[j*nao+k]);
        atomicAdd(vk+i*nao+k, value*dm[j*nao+l]);
        if (ish != jsh) {
            atomicAdd(vk+j*nao+l, value*dm[i*nao+k]);
            atomicAdd(vk+j*nao+k, value*dm[i*nao+l]);
        }
    }
}
'''


class _IntegralCache:
    def __init__(self, opt, operator, key, budget, pairs, *, all_groups=False):
        import cupy as cp
        import numpy as np

        mol = opt.sorted_mol
        self.key, self.operator = key, operator
        self.requested_budget, self.all_groups = budget, all_groups
        groups = mol.uniq_l_ctr
        counts = {pair: int(values[0].size) for pair, values in pairs.items()}
        self.plan = plan_groups(groups, counts, None if all_groups else budget, mol.ao_loc.nbytes)
        self.planned_bytes = _plan_size(self.plan, mol.ao_loc.nbytes)
        self.budget = self.planned_bytes if all_groups else budget
        if all_groups:
            _require_capacity(self.planned_bytes, *_available_capacity(cp))
        self.entries = {}
        self.used_bytes = 0
        self.builds = self.hits = self.misses = 0
        self.ao_loc = None
        self.kernel = None
        if self.plan:
            self.ao_loc = cp.asarray(mol.ao_loc, dtype=np.int32)
            self.used_bytes = self.ao_loc.data.mem.size
            self.kernel = cp.RawKernel(_CONTRACT, 'cached_k', options=('--std=c++17',))

    def contract(self, opt, task, pairs, shls_slice, pool, dm, dm_cond, vk,
                 log_cutoff, dm_penalty, library, shm_size):
        import cupy as cp
        import numpy as np

        reservation = self.plan.get(task)
        if reservation is None:
            return False
        i, j, k, l = task
        pa, qa, _ = pairs[i, j]
        pb, qb, _ = pairs[k, l]
        angular = opt.sorted_mol.uniq_l_ctr[list(task), 0]
        nf = tuple((int(x)+1)*(int(x)+2)//2 for x in angular)
        entry = self.entries.get(task)
        if entry is None:
            self.misses += 1
            entry = self._build(opt, task, shls_slice, pa, pb, qa, qb, nf,
                                pool, library, shm_size)
            if entry is None:
                if self.all_groups:
                    opt._integral_cache = None
                    raise MemoryError('All-VRAM integral cache allocation failed; no disk fallback was used')
                # Allocation failure never removes an integral contribution.
                self.plan.pop(task)
                return False
            size = sum(array.data.mem.size for array in entry)
            if size > reservation or self.used_bytes+size > self.budget:
                raise RuntimeError('Integral cache allocator exceeded its reserved budget')
            self.entries[task] = entry
            self.used_bytes += size
            self.builds += 1
        else:
            self.hits += 1
        values = entry[0]
        blocks = min(65535, (values.size+255)//256)
        self.kernel((blocks,), (256,), (
            values, pa, pb, qa, qb, self.ao_loc, dm_cond, dm, vk,
            np.int32(pa.size), np.int32(pb.size), np.int32(opt.sorted_mol.nbas),
            np.int32(dm.shape[-1]), *(np.int32(n) for n in nf),
            np.float32(log_cutoff), np.float32(dm_penalty)))
        return True

    def _build(self, opt, task, shls_slice, pa, pb, qa, qb, nf, pool, library, shm_size):
        import cupy as cp
        import numpy as np

        nfij, nfkl = nf[0]*nf[1], nf[2]*nf[3]
        # No density-dependent admission or screening: future densities can grow.
        # Use the caller's existing native queue; no second queue or full tensor.
        try:
            values = cp.empty((pa.size*nfij, pb.size*nfkl), dtype=np.float64)
            loca = cp.arange(pa.size, dtype=np.int32)
            locb = cp.arange(pb.size, dtype=np.int32)
            loca *= np.int32(nfij)
            locb *= np.int32(nfkl)
        except cp.cuda.memory.OutOfMemoryError:
            return None
        envs = opt.rys_envs
        omega, lr, sr = self.operator
        def pointer(array):
            return ctypes.c_void_p(array.data.ptr)
        error = library.RYS_fill_int4c2e_k(
            pointer(values), ctypes.c_int(values.shape[1]),
            ctypes.c_double(omega), ctypes.c_double(lr), ctypes.c_double(sr),
            ctypes.byref(envs), (ctypes.c_int*8)(*shls_slice), ctypes.c_int(shm_size),
            ctypes.c_int(pa.size), ctypes.c_int(pb.size), pointer(pa), pointer(pb),
            pointer(qa), pointer(qb), pointer(loca), pointer(locb),
            ctypes.c_float(-math.inf), pointer(pool), opt.sorted_mol._bas.ctypes)
        if error:
            raise RuntimeError(f'Integral cache fill failed for {task}')
        return values, loca, locb

    def statistics(self):
        return {'mode': 'all' if self.all_groups else 'bounded',
                'budget_bytes': self.budget, 'owned_gpu_bytes': self.used_bytes,
                'planned_gpu_bytes': self.planned_bytes, 'planned_groups': len(self.plan),
                'all_groups_resident': self.all_groups and len(self.entries) == len(self.plan),
                'groups': len(self.entries), 'builds': self.builds,
                'hits': self.hits, 'misses': self.misses}


def get_cache(opt, dms, hermi, operator, pairs, library, device_count):
    """Return an instance-owned cache only for the supported native K operation."""
    budget = getattr(opt, '_direct_scf_cache_bytes', 0)
    all_groups = getattr(opt, '_direct_scf_cache_all', False)
    if not budget and not all_groups:
        return None
    import cupy as cp

    device = cp.cuda.device.get_device_id()
    stream = cp.cuda.get_current_stream().ptr
    if (device_count != 1 or hermi != 1 or dms.shape[0] != 1
            or dms.dtype != cp.float64 or not dms.flags.c_contiguous
            or stream != 0 or opt.h_shls or operator[1] == 0
            or not all(math.isfinite(x) for x in operator)):
        return None
    if not hasattr(library, 'RYS_fill_int4c2e_k'):
        if all_groups:
            raise RuntimeError('All-VRAM integral cache requires a native RYS_fill_int4c2e_k build')
        return None
    key = _context_key(opt, operator, device, stream)
    cache = getattr(opt, '_integral_cache', None)
    if (cache is None or cache.key != key or cache.requested_budget != budget
            or cache.all_groups != all_groups):
        # Release the former operator before admitting a replacement.
        opt._integral_cache = None
        cache = None
        try:
            cache = _IntegralCache(opt, operator, key, budget, pairs, all_groups=all_groups)
        except cp.cuda.memory.OutOfMemoryError:
            if all_groups:
                raise MemoryError('All-VRAM integral cache metadata allocation failed') from None
            return None
        opt._integral_cache = cache
    return cache
