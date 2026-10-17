"""Scoped exact-input reuse of native ordinary integration for SCF verification."""

from __future__ import annotations

from contextlib import contextmanager
from types import FunctionType, MethodType

_MAX_BYTES = 128 * 1024**2


class _NativeGridReuse:
    def __init__(self, ni):
        self.ni = ni
        self.enabled = self.read = False
        self.supported = not any(name in ni.__dict__ for name in (
            'eval_xc_eff', 'block_loop', 'eval_ao', '_xc_type', 'eval_rho', 'eval_rho2'))
        self.hits = self.misses = 0
        self._entry = None
        self._buffers = []

    def clear(self):
        self._entry = None
        self._buffers.clear()

    def _cookie(self, mol):
        import cupy as cp

        dispatch = tuple(id(getattr(function, '__func__', function)) for function in (
            getattr(self.ni, name, None) for name in ('eval_xc_eff', 'block_loop', 'eval_ao', '_xc_type')))
        return (getattr(self.ni, 'omega', None), getattr(self.ni, 'cutoff', None),
                id(getattr(self.ni, 'gdftopt', None)), mol.charge, mol.spin, mol.cart,
                dispatch, id(getattr(self.ni, 'libxc', None)),
                cp.cuda.Device().id, cp.cuda.get_current_stream().ptr)

    @staticmethod
    def _arrays(args):
        import cupy as cp

        if len(args) != 4:
            return None
        _, grids, xc, density = args
        if not isinstance(xc, (str, int)) or not isinstance(density, cp.ndarray):
            return None
        values = [density, grids.coords, grids.weights]
        values += [getattr(density, key, None) for key in ('mo_coeff', 'mo_occ', 'occ_coeff')]
        if any(a is not None and (not isinstance(a, cp.ndarray) or a.dtype != cp.float64
                                 or a.device.id != cp.cuda.Device().id) for a in values):
            return None
        return values

    @staticmethod
    def _spec(value):
        return None if value is None else (value.shape, value.strides, value.dtype.str, value.device.id)

    @staticmethod
    def _copy_result(result, previous=None):
        import cupy as cp
        import numpy as np

        if isinstance(result, cp.ndarray):
            if not isinstance(previous, cp.ndarray) or previous.shape != result.shape or previous.dtype != result.dtype:
                previous = cp.empty_like(result)
            cp.copyto(previous, result)
            return previous
        if isinstance(result, tuple):
            if not isinstance(previous, tuple) or len(previous) != len(result):
                previous = (None,) * len(result)
            return tuple(_NativeGridReuse._copy_result(x, y) for x, y in zip(result, previous))
        if isinstance(result, list):
            return [_NativeGridReuse._copy_result(x) for x in result]
        if isinstance(result, np.ndarray):
            return result.copy()
        return result

    @staticmethod
    def _result_bytes(result):
        if isinstance(result, (tuple, list)):
            return sum(_NativeGridReuse._result_bytes(x) for x in result)
        return getattr(result, 'nbytes', 0)

    def _capture(self, name, args, kwargs, result):
        import cupy as cp
        import numpy as np

        values = self._arrays(args)
        if values is None or kwargs:
            self.clear()
            return
        total = sum(a.nbytes for a in values if a is not None) + self._result_bytes(result)
        from gpu4pyscf.lib.cupy_helper import get_avail_mem

        if total > _MAX_BYTES or total > get_avail_mem() // 8:
            self.clear()
            return
        mol, grids, xc, _ = args
        old_result = self._entry[-1] if self._entry is not None else None
        self._entry = None
        try:
            if len(self._buffers) != len(values):
                self._buffers = [None] * len(values)
            for index, value in enumerate(values):
                if value is None:
                    self._buffers[index] = None
                    continue
                saved = self._buffers[index]
                if saved is None or saved.shape != value.shape or saved.dtype != value.dtype:
                    saved = self._buffers[index] = cp.empty_like(value)
                cp.copyto(saved, value)
            copied = self._copy_result(result, old_result)
        except cp.cuda.memory.OutOfMemoryError:
            self.clear()
            return
        molecular = tuple(np.array(getattr(mol, key), copy=True) for key in ('_atm', '_bas', '_env'))
        self._entry = (name, id(mol), id(grids), xc, self._cookie(mol),
                       tuple(self._spec(a) for a in values), molecular, copied)

    def _matches(self, name, args, kwargs):
        import cupy as cp
        import numpy as np

        if self._entry is None or kwargs:
            return False
        values = self._arrays(args)
        if values is None:
            return False
        mol, grids, xc, _ = args
        expected_name, mol_id, grid_id, expected_xc, cookie, specs, molecular, _ = self._entry
        if (name != expected_name or id(mol) != mol_id or id(grids) != grid_id or xc != expected_xc
                or cookie != self._cookie(mol) or specs != tuple(self._spec(a) for a in values)):
            return False
        if any(not np.array_equal(getattr(mol, key), old)
               for key, old in zip(('_atm', '_bas', '_env'), molecular)):
            return False
        return all(a is None or bool(cp.array_equal(a, saved)) for a, saved in zip(values, self._buffers))

    def wrap(self, name, original):
        def call(_ni, *args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            if self.read:
                if self._matches(name, args, kwargs):
                    self.hits += 1
                    return self._copy_result(self._entry[-1])
                self.misses += 1
            result = original(*args, **kwargs)
            self._capture(name, args, kwargs, result)
            return result
        # Preserve the private NumInt dispatch graph used by the precision scope.
        function = FunctionType(call.__code__, original.__func__.__globals__, name,
                                call.__defaults__, call.__closure__)
        function.__wrapped__ = original.__func__
        return MethodType(function, self.ni)

    @contextmanager
    def verification(self):
        self.read = True
        try:
            yield
        finally:
            self.read = False


@contextmanager
def _native_grid_reuse(ni):
    from gpu4pyscf.dft import numint

    cache = _NativeGridReuse(ni)
    cache.supported = cache.supported and type(ni) is numint.NumInt
    originals = {name: getattr(ni, name) for name in ('nr_rks', 'nr_uks')}
    try:
        for name, original in originals.items():
            setattr(ni, name, cache.wrap(name, original))
        yield cache
    finally:
        for name, original in originals.items():
            setattr(ni, name, original)
        cache.clear()
