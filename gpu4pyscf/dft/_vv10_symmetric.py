"""Deterministic symmetric tiles for the float64 VV10 pair response."""

from __future__ import annotations

from functools import lru_cache

_MIN_POINTS = 65536
_TILE = 96
_TILES_PER_BATCH = 128

# Float64 pair fields follow the equations in lib/gdft/vv10.cu (Apache-2.0).
_SOURCE = r"""
// Fused VV10 field and contracted coordinate response over unordered tile pairs.
#define LANES 16
#define THREADS (LANES * LANES)
#define PER_LANE (TILE / LANES)
#define STRIDE (3 * TILE + 8)
#define INPUTS 7

__device__ __forceinline__ double warp_sum(double val, int width) {
    for (int step = width / 2; step; step /= 2)
        val += __shfl_down_sync(0xffffffff, val, step, width);
    return val;
}

__device__ __forceinline__ void cache_tile(
    double cache[INPUTS][TILE], const double* coords, const double* rw,
    const double* omega, const double* kappa, int first, int last, int tid
) {
    if (tid < TILE) {
        int point = first + tid;
        bool active = point < last;
        cache[0][tid] = active ? coords[3 * point] : 0.;
        cache[1][tid] = active ? coords[3 * point + 1] : 0.;
        cache[2][tid] = active ? coords[3 * point + 2] : 0.;
        cache[3][tid] = active ? rw[point] : 0.;
        cache[4][tid] = active ? omega[point] : 0.;
        cache[5][tid] = active ? kappa[point] : 1.;
        cache[6][tid] = -3. * cache[3][tid] * cache[4][tid];
    }
}

extern "C" __global__ void pair_tiles(
    double* from_cols, double* from_rows, const double* coords, const double* rw,
    const double* omega, const double* kappa, const int* starts, const int* ends,
    const int* owners, int row_base, int col_base, int diagonal
) {
    int row = blockIdx.y, col = blockIdx.x;
    if (diagonal && row > col)
        return;
    bool both = !diagonal || row != col;
    int tid = threadIdx.x, a = tid / LANES, b = tid % LANES;
    int row_tile = row_base + row, col_tile = col_base + col;
    bool same_atom = owners[row_tile] == owners[col_tile];
    double* left = from_cols + (row * gridDim.x + col) * STRIDE;
    double* right = from_rows + (col * gridDim.y + row) * STRIDE;
    __shared__ double first[INPUTS][TILE], second[INPUTS][TILE];
    __shared__ double partial[3][LANES][TILE];
    __shared__ double grad[3][THREADS / 32];
    cache_tile(first, coords, rw, omega, kappa, starts[row_tile], ends[row_tile], tid);
    cache_tile(second, coords, rw, omega, kappa, starts[col_tile], ends[col_tile], tid);
    __syncthreads();

    double right_u[PER_LANE] = {}, right_w[PER_LANE] = {}, right_e[PER_LANE] = {};
    double gx = 0., gy = 0., gz = 0.;
    #pragma unroll
    for (int r = 0; r < PER_LANE; ++r) {
        int i = r * LANES + a;
        double oi = first[4][i], ki = first[5][i], wi = first[3][i];
        double ui = 0., wi_sum = 0., ei = 0.;
        #pragma unroll
        for (int c = 0; c < PER_LANE; ++c) {
            int j = c * LANES + b;
            double dx = first[0][i] - second[0][j];
            double dy = first[1][i] - second[1][j];
            double dz = first[2][i] - second[2][j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double oj = second[4][j], wj = second[3][j];
            double gi = oi * r2 + ki, gj = oj * r2 + second[5][j], sum = gi + gj;
            double product = gi * gj;
            double inv = 1. / (product * sum);
            double energy_i = -wj * inv, energy_j = -wi * inv;
            double u_i = energy_i * fma(gj, sum, product) * inv;
            double u_j = energy_j * fma(gi, sum, product) * inv;
            ui += u_i;
            wi_sum += u_i * r2;
            ei += energy_i;
            right_u[c] += u_j;
            right_w[c] += u_j * r2;
            right_e[c] += energy_j;
            if (!same_atom) {
                double response = first[6][i] * u_i + second[6][j] * u_j;
                gx += response * dx;
                gy += response * dy;
                gz += response * dz;
            }
        }
        ui = warp_sum(ui, LANES);
        wi_sum = warp_sum(wi_sum, LANES);
        ei = warp_sum(ei, LANES);
        if (b == 0) {
            left[i] = ui;
            left[TILE + i] = wi_sum;
            left[2 * TILE + i] = ei;
        }
    }

    if (both) {
        #pragma unroll
        for (int c = 0; c < PER_LANE; ++c) {
            int j = c * LANES + b;
            partial[0][a][j] = right_u[c];
            partial[1][a][j] = right_w[c];
            partial[2][a][j] = right_e[c];
        }
        __syncthreads();
        for (int p = tid; p < 3 * TILE; p += THREADS) {
            int field = p / TILE, j = p % TILE;
            double value = 0.;
            for (int lane = 0; lane < LANES; ++lane)
                value += partial[field][lane][j];
            right[p] = value;
        }
    }
    gx = warp_sum(gx, 32);
    gy = warp_sum(gy, 32);
    gz = warp_sum(gz, 32);
    if (tid % 32 == 0) {
        grad[0][tid / 32] = gx;
        grad[1][tid / 32] = gy;
        grad[2][tid / 32] = gz;
    }
    __syncthreads();
    if (tid < 3) {
        double value = 0.;
        for (int warp = 0; warp < THREADS / 32; ++warp)
            value += grad[tid][warp];
        left[3 * TILE + tid] = value;
        if (both)
            right[3 * TILE + tid] = -value;
    }
}

extern "C" __global__ void gather_tiles(
    double* fields, double* response, const double* partial, const double* opposite,
    const int* starts, const int* ends, int base, int partners, int points,
    int tiles, int diagonal
) {
    int tile = base + blockIdx.x, t = threadIdx.x;
    int point = starts[tile] + t;
    const double* row = partial + blockIdx.x * partners * STRIDE + t;
    const double* transposed = opposite + blockIdx.x * partners * STRIDE + t;
    if (point < ends[tile]) {
        double u = 0., w = 0., e = 0.;
        for (int j = 0; j < partners; ++j) {
            const double* term = (diagonal && j < blockIdx.x ? transposed : row) + j * STRIDE;
            u += term[0];
            w += term[TILE];
            e += term[2 * TILE];
        }
        fields[point] += u;
        fields[points + point] += w;
        fields[2 * points + point] += e;
    }
    if (t < 3) {
        double value = 0.;
        for (int j = 0; j < partners; ++j) {
            const double* src = diagonal && j < blockIdx.x ? transposed : row;
            value += src[j * STRIDE + 3 * TILE];
        }
        response[t * tiles + tile] += value;
    }
}
"""


@lru_cache(maxsize=16)
def _kernels(device):
    import cupy as cp

    module = cp.RawModule(code=_SOURCE, options=('-std=c++17', f'-DTILE={_TILE}'))
    return module.get_function('pair_tiles'), module.get_function('gather_tiles')


def _pair_tiles(coords, rw, omega, kappa, counts, reduce):
    import cupy as cp
    import numpy as np

    starts, ends, owners = [], [], []
    tile_offsets = [0]
    start = 0
    for atom, count in enumerate(cp.asnumpy(counts)):
        end = start + int(count)
        for point in range(start, end, _TILE):
            starts.append(point)
            ends.append(min(point + _TILE, end))
            owners.append(atom)
        tile_offsets.append(len(owners))
        start = end

    points, tiles = len(rw), len(owners)
    batch = min(_TILES_PER_BATCH, tiles)
    stride = 3 * _TILE + 8
    work_a, work_b = (cp.empty((batch * batch, stride)) for _ in range(2))
    starts, ends, owners, offsets = (
        cp.asarray(values, dtype=cp.int32) for values in (starts, ends, owners, tile_offsets)
    )
    fields = cp.zeros((3, points))
    responses = cp.zeros((3, tiles))
    result = cp.empty((len(counts), 3))
    pair, gather = _kernels(cp.cuda.Device().id)
    npoints, ntiles = np.int32(points), np.int32(tiles)
    for first in range(0, tiles, batch):
        rows = min(batch, tiles - first)
        for second in range(first, tiles, batch):
            cols = min(batch, tiles - second)
            diagonal = np.int32(first == second)
            pair((cols, rows), (256,), (
                work_a, work_b, coords, rw, omega, kappa, starts, ends, owners,
                np.int32(first), np.int32(second), diagonal,
            ))
            # Both tile orientations have contiguous fields. A diagonal batch
            # gathers the lower triangle from the transposed partials.
            gather((rows,), (_TILE,), (
                fields, responses, work_a, work_b, starts, ends,
                np.int32(first), np.int32(cols), npoints, ntiles, diagonal,
            ))
            if not diagonal:
                gather((cols,), (_TILE,), (
                    fields, responses, work_b, work_a, starts, ends,
                    np.int32(second), np.int32(rows), npoints, ntiles, diagonal,
                ))
    reduce((len(counts), 3), (256,), (result, responses, offsets, ntiles))
    fields *= cp.asarray((-1.5, -1.5, 1.5))[:, None]
    return fields[0], fields[1], fields[2], result


def _try_symmetric_response(coords, rw, omega, kappa, counts, reduce):
    import cupy as cp
    import numpy as np

    points = len(rw)
    if points < _MIN_POINTS or points > np.iinfo(np.int32).max // 3:
        return None
    device = cp.cuda.Device()
    if cp.cuda.runtime.is_hip or device.compute_capability != '86':
        return None
    if any(array.device.id != device.id for array in (coords, rw, omega, kappa, counts)):
        return None

    from gpu4pyscf.lib.cupy_helper import get_avail_mem

    # Reserve space for the caller's fitted integrals and grid cache. Scratch
    # does not grow with the number of point pairs.
    scratch = 16 * _TILES_PER_BATCH**2 * (3 * _TILE + 8)
    if scratch + 64 * points > get_avail_mem() // 8:
        return None
    try:
        # VV10 normally has positive kappa and nonnegative omega. Wide bounds
        # on the inputs retain headroom for the factored products and padding.
        limit = 2.0**30
        if not (bool(cp.all(cp.abs(coords) <= limit)) and bool(cp.all(cp.abs(rw) <= limit))
                and bool(cp.all((omega >= 0) & (omega <= limit)))
                and bool(cp.all((kappa >= 1 / limit) & (kappa <= limit)))):
            return None
        return _pair_tiles(coords, rw, omega, kappa, counts, reduce)
    except cp.cuda.memory.OutOfMemoryError:
        return None
    except cp.cuda.runtime.CUDARuntimeError as error:
        if error.status != 2:  # cudaErrorMemoryAllocation
            raise
        return None
