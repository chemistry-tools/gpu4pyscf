"""Compare direct float64 BLAS contractions with GPU4PySCF tensor dispatch."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager


def _gemm(*args, **kwargs):
    from gpu4pyscf.dft._mixed_grid import _gemm as gemm

    return gemm(*args, **kwargs)


@contextmanager
def install(mode='blas'):
    from gpu4pyscf.dft import numint
    from gpu4pyscf.dft._mixed_grid import _GridPrecision, _make_contract

    original = numint.contract
    controller = _GridPrecision(mode == 'mixed')
    numint.contract = _make_contract(original, controller)
    try:
        yield controller
    finally:
        numint.contract = original
        controller._release_scratch()


class _InlineExecutor:
    def __init__(self, max_workers):
        if max_workers != 1:
            raise ValueError('Inline execution requires exactly one visible GPU')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as error:  # noqa: BLE001 - match Future's exception transport
            future.set_exception(error)
        return future


@contextmanager
def inline_grid_tasks():
    from gpu4pyscf.dft import numint

    if numint.num_devices != 1:
        raise ValueError('Inline execution requires exactly one visible GPU')
    original = numint.ThreadPoolExecutor
    numint.ThreadPoolExecutor = _InlineExecutor
    try:
        yield
    finally:
        numint.ThreadPoolExecutor = original
