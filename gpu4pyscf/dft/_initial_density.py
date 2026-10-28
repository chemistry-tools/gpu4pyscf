"""Recover compact orbital factors only when they reconstruct an untagged density."""

from __future__ import annotations

from functools import lru_cache
from math import isfinite

_MIN_ORBITALS = 128
_MAX_WORKSPACE_BYTES = 64 * 1024**2


@lru_cache(maxsize=16)
def _supported_device(device):
    import cupy as cp

    return cp.cuda.Device(device).compute_capability == '86'


def _density_orbitals(density):
    """Return float64 orbital tags, or None without modifying the supplied matrix.

    The existing DF decomposition supplies candidate factors. Its eigenvalue
    screening alone is insufficient for ordinary grids: both maximum-element
    and Frobenius reconstruction errors must be at double-precision roundoff.
    Meaningful discarded components, signed densities and expensive ranks keep
    the native dense calculation.
    """
    import cupy as cp
    from cupyx import errstate

    if (not isinstance(density, cp.ndarray) or density.dtype != cp.float64
            or density.ndim != 2 or density.shape[0] != density.shape[1]
            or len(density) < _MIN_ORBITALS
            or 6 * density.nbytes > _MAX_WORKSPACE_BYTES
            or density.device.id != cp.cuda.Device().id
            or not _supported_device(density.device.id)
            or getattr(density, 'mo_coeff', None) is not None):
        return None
    if not bool(cp.all(cp.isfinite(density))):
        return None
    scale = float(cp.max(cp.abs(density)))
    if scale == 0:
        return None
    # Refuse a nonsymmetric matrix even when the caller claims hermi=1.
    tolerance = 64 * cp.finfo(cp.float64).eps
    if float(cp.max(cp.abs(density - density.T))) > tolerance * scale:
        return None

    from gpu4pyscf.df.df_jk import decompose_rdm1_svd

    try:
        with errstate(linalg='raise'):
            left, right = decompose_rdm1_svd(density, hermi=1)
        rank = left.shape[1]
        if not rank or 3 * rank >= len(density):
            return None
        occupations = cp.sum(left * right, axis=0)
        if not bool(cp.all(cp.isfinite(occupations) & (occupations > 0))):
            return None
        # Check the square-root factors actually used by _eval_rho2, including
        # their rounding, rather than only the unsquared eigendecomposition.
        factors = cp.ascontiguousarray(left * cp.sqrt(occupations))
        residual = factors.dot(factors.T) - density
        error = float(cp.max(cp.abs(residual)))
        residual_norm, density_norm = float(cp.linalg.norm(residual)), float(cp.linalg.norm(density))
        if (not all(isfinite(value) for value in (error, residual_norm, density_norm))
                or error > tolerance * scale or residual_norm > tolerance * density_norm):
            return None
    except (cp.linalg.LinAlgError, cp.cuda.memory.OutOfMemoryError):
        return None
    return factors, cp.ones(rank, dtype=cp.float64)
