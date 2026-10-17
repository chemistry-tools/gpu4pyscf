"""Bounded float64 density products from validated occupied-orbital factors."""

from __future__ import annotations

from functools import lru_cache

_MAX_WORKSPACE_BYTES = 64 * 1024**2


@lru_cache(maxsize=16)
def _supported_device(device):
    import cupy as cp

    return cp.cuda.Device(device).compute_capability == '86'


class _DensityBlock:
    def __init__(self, owner, left, right):
        self.owner, self.left, self.right = owner, left, right

    def dot(self, field, *, transpose=False, out=None):
        import cupy as cp

        left, right = (self.right, self.left) if transpose else (self.left, self.right)
        size = right.shape[1] * field.shape[1]
        if self.owner._projection is None or self.owner._projection.size < size:
            retained = self.owner._retained_bytes + 8 * size
            if retained > _MAX_WORKSPACE_BYTES:
                return None
            self.owner._projection = cp.empty(size, dtype=cp.float64)
        projected = self.owner._projection[:size].reshape(right.shape[1], field.shape[1])
        right.T.dot(field, out=projected)
        return left.dot(projected, out=out)


class _DensityProducts:
    def __init__(self, left, right):
        import cupy as cp

        self.left, self.right = cp.ascontiguousarray(left), cp.ascontiguousarray(right)
        self._masked_left = self._masked_right = None
        self._retained_bytes = 4 * self.left.nbytes
        self._projection = None
        self._device = cp.cuda.Device().id
        self._stream = cp.cuda.get_current_stream().ptr

    @classmethod
    def create(cls, density):
        import cupy as cp

        if (not isinstance(density, cp.ndarray) or density.dtype != cp.float64
                or density.ndim != 2 or density.shape[0] != density.shape[1]
                or density.device.id != cp.cuda.Device().id
                or not _supported_device(density.device.id)):
            return None
        coeff, occ = getattr(density, 'mo_coeff', None), getattr(density, 'mo_occ', None)
        if (not isinstance(coeff, cp.ndarray) or not isinstance(occ, cp.ndarray)
                or coeff.dtype != cp.float64 or occ.dtype != cp.float64
                or coeff.ndim != 2 or occ.ndim != 1 or coeff.shape != (len(density), len(occ))
                or coeff.device.id != density.device.id or occ.device.id != density.device.id):
            return None
        if not bool(cp.all(cp.isfinite(occ) & (occ >= 0))):
            return None
        mask = occ > 0
        rank = int(cp.count_nonzero(mask))
        if (not rank or 3 * rank >= len(density)
                or density.nbytes + 32 * rank * len(density) > _MAX_WORKSPACE_BYTES):
            return None
        # Reconstruct with the native make_rdm1 ordering before changing layouts.
        # Stale/inconsistent orbital tags must never replace the supplied density.
        right = coeff[:, mask]
        left = right * occ[mask]
        if not bool(cp.array_equal(density, left.dot(right.T))) or not bool(cp.array_equal(density, density.T)):
            return None
        return cls(left, right)

    def sort_orbitals(self, opt):
        if 32 * opt._sorted_mol.nao * self.left.shape[1] > _MAX_WORKSPACE_BYTES:
            return None
        self.left = opt.sort_orbitals(self.left, axis=[0])
        self.right = opt.sort_orbitals(self.right, axis=[0])
        import cupy as cp

        self.left, self.right = cp.ascontiguousarray(self.left), cp.ascontiguousarray(self.right)
        self._masked_left = self._masked_right = None
        self._retained_bytes = 4 * self.left.nbytes
        return self

    def block(self, indices):
        import cupy as cp

        m, rank = len(indices), self.left.shape[1]
        if (3 * rank >= m or self._retained_bytes > _MAX_WORKSPACE_BYTES
                or cp.cuda.Device().id != self._device or cp.cuda.get_current_stream().ptr != self._stream):
            return None
        if self._masked_left is None:
            self._masked_left, self._masked_right = cp.empty_like(self.left), cp.empty_like(self.right)
        left, right = self._masked_left[:m], self._masked_right[:m]
        cp.take(self.left, indices, axis=0, out=left)
        cp.take(self.right, indices, axis=0, out=right)
        return _DensityBlock(self, left, right)


def _density_dot(density, field, factors=None, *, transpose=False, out=None):
    if factors is not None:
        result = factors.dot(field, transpose=transpose, out=out)
        if result is not None:
            return result
    return (density.T if transpose else density).dot(field, out=out)
