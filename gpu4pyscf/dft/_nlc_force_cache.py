"""Bounded reuse of verified float64 VV10 fields for one full-response force."""

from __future__ import annotations

from contextlib import contextmanager

from gpu4pyscf.dft._native_grid_cache import _MAX_BYTES, _NativeGridReuse


def _canonical_indices(grids):
    import cupy as cp
    import numpy as np

    indices = cp.where(cp.abs(grids.weights) > 1e-14)[0]
    coords, atoms = cp.asnumpy(grids.coords[indices]), cp.asnumpy(grids.atm_idx[indices])
    # Sort only metadata on the CPU. The fields and pair traversal stay on device.
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], atoms))
    return indices[cp.asarray(order)]


def _grid_configuration(grids):
    import numpy as np

    return (grids.level, repr(grids.atom_grid), grids.alignment, grids.cutoff,
            tuple(id(getattr(grids, key)) for key in
                  ('radi_method', 'prune', 'becke_scheme', 'radii_adjust')),
            np.asarray(grids.atomic_radii).tobytes())


class _NLCForceReuse:
    def __init__(self, ni):
        from gpu4pyscf.dft import numint

        self.snapshot = _NativeGridReuse(ni)
        self.supported = (self.snapshot.supported and type(ni) is numint.NumInt
                          and 'nlc_coeff' not in ni.__dict__ and ni.libxc is numint.libxc
                          and getattr(ni, 'cutoff', None) == getattr(numint.NumInt(), 'cutoff', None))
        self.capture = False
        self.configuration = None
        self.statistics = {'builds': 0, 'hits': 0, 'misses': 0, 'retained_bytes': 0}

    def clear(self):
        self.snapshot.clear()
        self.configuration = None
        self.statistics['retained_bytes'] = 0

    @contextmanager
    def verification(self):
        self.clear()
        self.capture = self.supported
        try:
            yield
        finally:
            self.capture = False

    def potential(self, mol, grids, xc, density, rho, pars, *, native=None):
        """Compute one native potential and retain its fused coordinate response."""
        import cupy as cp

        try:
            return self._potential(mol, grids, xc, density, rho, pars, native=native)
        except cp.cuda.memory.OutOfMemoryError:
            self.clear()
            return None

    def _potential(self, mol, grids, xc, density, rho, pars, *, native):
        import cupy as cp

        from gpu4pyscf.dft import gen_grid, numint
        from gpu4pyscf.dft._vv10_grad import _pair_response
        from gpu4pyscf.lib.cupy_helper import get_avail_mem

        args = (mol, grids, xc, density)
        arrays = self.snapshot._arrays(args)
        if (not self.capture or type(grids) is not gen_grid.Grids or arrays is None or density.ndim != 2
                or grids.atm_idx is None or rho.dtype != cp.float64):
            return None
        # Snapshot inputs, canonical fields and bounded sort/pair scratch must fit.
        estimated = sum(a.nbytes for a in arrays if a is not None) + 160 * len(grids.weights)
        if estimated > _MAX_BYTES or estimated > get_avail_mem() // 8:
            return None
        indices = _canonical_indices(grids)
        atoms = cp.ascontiguousarray(grids.atm_idx[indices], dtype=cp.int32)
        if bool(cp.any((atoms < 0) | (atoms >= mol.natm))):
            return None
        active = rho[0, indices] >= numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
        source_active = cp.where((rho[0] >= numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD)
                                 & (cp.abs(grids.weights) > 1e-14))[0]
        ranks = cp.empty(len(grids.weights), dtype=cp.int64)
        ranks[source_active] = cp.arange(len(source_active))
        order = ranks[indices[active]]
        inverse = cp.argsort(order)
        fields = None

        def pair(coords, rw, omega, kappa):
            nonlocal fields
            fields = _pair_response(cp.ascontiguousarray(coords[order]), rw[order],
                                    omega[order], kappa[order], atoms[active], mol.natm)
            return tuple(value[inverse] for value in fields[:3])

        if native is None:
            native = numint._vv10nlc
        result = native(rho, grids.coords, grids.weights, pars, _pair=pair)
        payload = (grids.coords[indices], grids.weights[indices], atoms, rho[:, indices], fields)
        self.snapshot._capture('nlc-force', args, {}, payload)
        if self.snapshot._entry is not None:
            self.configuration = _grid_configuration(grids)
            self.statistics['builds'] += 1
            self.statistics['retained_bytes'] = (
                sum(a.nbytes for a in self.snapshot._buffers if a is not None)
                + self.snapshot._result_bytes(payload))
        return result

    def take(self, ni, mol, source_grids, force_grids, xc, density):
        """Consume fields only for identical state and exactly permuted grids."""
        import cupy as cp

        try:
            return self._take(ni, mol, source_grids, force_grids, xc, density)
        except cp.cuda.memory.OutOfMemoryError:
            self.statistics['misses'] += 1
            self.clear()
            return None

    def _take(self, ni, mol, source_grids, force_grids, xc, density):
        import cupy as cp

        from gpu4pyscf.dft import numint

        args = (mol, source_grids, xc, density)
        if (ni is not self.snapshot.ni or self.configuration != _grid_configuration(source_grids)
                or not self.snapshot._matches('nlc-force', args, {})):
            self.statistics['misses'] += 1
            self.clear()
            return None
        indices = _canonical_indices(force_grids)
        coords, weights, atoms, rho, fields = self.snapshot._entry[-1]
        if any(not bool(cp.array_equal(a, b)) for a, b in (
                (force_grids.coords[indices], coords), (force_grids.weights[indices], weights),
                (force_grids.atm_idx[indices], atoms))):
            self.statistics['misses'] += 1
            self.clear()
            return None
        full_rho = cp.zeros((4, len(force_grids.weights)), dtype=cp.float64)
        full_rho[:, indices] = rho
        active = rho[0] >= numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
        order = cp.argsort(indices[active])
        result = tuple(value[order] for value in fields[:3]) + (fields[3].copy(),)
        self.statistics['hits'] += 1
        self.clear()
        return full_rho, result
