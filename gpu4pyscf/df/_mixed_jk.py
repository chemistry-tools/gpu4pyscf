"""Private early float32 exchange products for verified global-hybrid GGA SCF."""

from __future__ import annotations

from contextlib import contextmanager
from threading import local
from types import FunctionType, MethodType, SimpleNamespace


class _ExchangePrecision:
    def __init__(self, grid):
        from gpu4pyscf.dft._mixed_grid import _GridPrecision

        self.grid = grid
        self.float32_calls = 0
        self.float64_calls = 0
        self._workspace = local()
        self._accumulator = _GridPrecision(True)
        self._has_scratch = False

    def release(self):
        if self._has_scratch:
            self._workspace = local()
            self._accumulator._release_scratch()
            self._has_scratch = False

    def _scratch(self, a, b, out):
        import cupy as cp

        stream = cp.cuda.get_current_stream()
        self._has_scratch = True
        key = cp.cuda.Device().id, stream.ptr
        if not hasattr(self._workspace, 'streams'):
            self._workspace.streams = {}
        _, buffers = self._workspace.streams.setdefault(key, (stream, [None, None, None]))
        views = []
        for index, shape in enumerate((a.shape, b.shape, out.shape)):
            if index == 1 and a is b:
                views.append(views[0])
                continue
            size = 1
            for dimension in shape:
                size *= dimension
            if buffers[index] is None or buffers[index].size < size:
                buffers[index] = cp.empty(size, dtype=cp.float32)
            views.append(buffers[index][:size].reshape(shape))
        return views

    def contract(self, original, pattern, a, b, alpha=1., beta=0., out=None, **kwargs):
        import cupy as cp

        from gpu4pyscf.lib.cutensor import contraction

        if (
            pattern not in ('Lij,njk->nikL', 'nikL,njkL->nij')
            or a.dtype != cp.float64 or b.dtype != cp.float64
            or out is None or out.dtype != cp.float64 or kwargs
        ):
            return original(pattern, a, b, alpha=alpha, beta=beta, out=out, **kwargs)
        if not self.grid.use_float32:
            self.release()
            self.float64_calls += 1
            return original(pattern, a, b, alpha=alpha, beta=beta, out=out)
        from cupy_backends.cuda.libs import cutensor

        a32, b32, product = self._scratch(a, b, out)
        cp.copyto(a32, a, casting='unsafe')
        if a is not b:
            cp.copyto(b32, b, casting='unsafe')
        # Explicit IEEE float32 products; a TF32 compute descriptor is not used.
        contraction(pattern, a32, b32, 1., 0., out=product, compute_desc=cutensor.COMPUTE_DESC_32F)
        self.float32_calls += 1
        return self._accumulator._accumulate(product, alpha, beta, out)

    def gram(self, original, contract, rhok, out):
        if not self.grid.use_float32:
            self.release()
            self.float64_calls += 1
            return original(rhok, out)
        return self.contract(contract, 'nikL,njkL->nij', rhok, rhok, beta=1., out=out)


@contextmanager
def _exchange_context(mf, grid):
    from gpu4pyscf.df import df, df_jk
    from gpu4pyscf.dft.mixed_precision import _private_functions

    with_df = getattr(mf, 'with_df', None)
    if (
        type(with_df) is not df.DF
        or getattr(with_df.get_jk, '__func__', None) is not df.DF.get_jk
        or not hasattr(df_jk, '_k_gram')
        or mf.do_nlc() or mf._numint._xc_type(mf.xc) != 'GGA'
    ):
        yield None
        return
    omega, _, hybrid = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
    if omega != 0 or hybrid == 0:
        yield None
        return
    controller = _ExchangePrecision(grid)

    def contract(*args, **kwargs):
        return controller.contract(df_jk.contract, *args, **kwargs)

    def gram(rhok, out):
        return controller.gram(df_jk._k_gram, df_jk.contract, rhok, out)

    private = _private_functions(df_jk.__dict__, {'contract': contract, '_k_gram': gram})
    original = with_df.get_jk.__func__
    namespace = dict(original.__globals__, df_jk=SimpleNamespace(**private))
    function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    had_attribute = 'get_jk' in with_df.__dict__
    previous = with_df.__dict__.get('get_jk')
    with_df.get_jk = MethodType(function, with_df)
    try:
        yield controller
    finally:
        if had_attribute:
            with_df.get_jk = previous
        else:
            del with_df.get_jk
        controller.release()
