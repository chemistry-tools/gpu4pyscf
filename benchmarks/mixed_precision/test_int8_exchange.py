"""Check bounded INT8 exchange products against independent numerical references."""

from __future__ import annotations

import importlib.util
import os
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

gpu = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location('int8_gram_plan', ROOT/'gpu4pyscf/df/_int8_gram.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('m,k,batches', [(1, 1, 1), (17, 257, 2), (502, 30000, 1), (846, 43776, 1)])
def test_workspace_is_bounded_and_integer_accumulation_cannot_overflow(m, k, batches):
    module = _module()
    pm, pk, size = module._workspace_plan(m, k, batches)
    assert pm >= m and pk >= k and pm % 16 == pk % 16 == 0
    assert pk*127**2 < 2**31
    assert size == module._SLICES*pm*pk+12*pm*pm+8*batches*pm
    assert size <= module._MAX_WORKSPACE_BYTES


@pytest.mark.parametrize('m,k,batches', [(0, 7, 1), (1, 0, 1), (1, 131073, 1),
                                       (20000, 4096, 1), (2**31, 1, 1), (16, 16, 0)])
def test_invalid_or_expensive_shapes_use_the_native_fallback(m, k, batches):
    assert _module()._workspace_plan(m, k, batches) is None


@pytest.fixture
def int8():
    from gpu4pyscf.df import _int8_gram

    return _int8_gram


@pytest.fixture
def small_int8(int8, monkeypatch):
    monkeypatch.setattr(int8, '_MIN_ORBITALS', 0)
    monkeypatch.setattr(int8, '_MIN_REDUCTION', 0)
    return int8


@gpu
@pytest.mark.parametrize('shape', [(1, 7, 3, 13), (2, 33, 9, 29), (1, 257, 17, 257)])
def test_signed_batched_gram_preserves_general_output(shape, small_int8):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(1717)
    a = cp.asarray(rng.normal(size=shape))
    a[:,0] = 0
    initial = cp.asarray(rng.normal(size=(shape[0], shape[1], shape[1])))
    flat = a.reshape(shape[0], shape[1], -1)
    expected = initial+flat @ flat.transpose(0, 2, 1)
    out = initial.copy()
    kernel = small_int8._Int8Gram()
    assert kernel.update(a, out)
    cp.testing.assert_allclose(out, expected, atol=4e-12, rtol=4e-13)
    cp.testing.assert_array_equal(out[:,0], initial[:,0])
    kernel.release()
    assert not hasattr(kernel._workspace, 'streams')


@gpu
def test_small_products_match_a_120_digit_decimal_oracle(small_int8):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(1718)
    matrix = rng.normal(0, .25, (7, 31))
    matrix[0] = 0
    matrix[1, ::2] *= 2**-20
    matrix[2, ::2] *= -1
    expected = np.empty((7, 7))
    with localcontext() as context:
        context.prec = 120
        values = [[Decimal.from_float(float(x)) for x in row] for row in matrix]
        for i in range(7):
            for j in range(7):
                expected[i, j] = float(sum(x*y for x, y in zip(values[i], values[j])))
    out = cp.zeros((1, 7, 7))
    assert small_int8._Int8Gram().update(cp.asarray(matrix)[None,:,None,:], out)
    cp.testing.assert_allclose(out[0], expected, atol=2e-14, rtol=2e-14)
    cp.testing.assert_array_equal(out, out.transpose(0, 2, 1))


@gpu
def test_largest_supported_reduction_does_not_overflow_int32(small_int8):
    import cupy as cp

    a = cp.full((1, 3, 1, 131072), 1.99999999999999, dtype=cp.float64)
    out = cp.zeros((1, 3, 3))
    assert small_int8._Int8Gram().update(a, out)
    expected = cp.asarray(131072*1.99999999999999**2)
    cp.testing.assert_allclose(out, expected, atol=1e-8, rtol=3e-13)


@gpu
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 1e-310, 1e200, 1e-200])
def test_invalid_rows_in_a_later_batch_leave_every_output_untouched(value, small_int8):
    import cupy as cp

    a = cp.ones((2, 17, 3, 13))
    if abs(value) in (1e200, 1e-200):
        a[1, 4] = value
    else:
        a[1, 4, 1, 1] = value
    out = cp.full((2, 17, 17), .375)
    assert not small_int8._Int8Gram().update(a, out)
    cp.testing.assert_array_equal(out, .375)


@gpu
def test_very_small_normal_terms_do_not_wrap_mantissa_shifts(small_int8):
    import cupy as cp

    a = cp.ones((1, 17, 3, 13))
    a[0, 4, 1, 1] = 1e-200
    out = cp.zeros((1, 17, 17))
    assert small_int8._Int8Gram().update(a, out)
    flat = a.reshape(17, -1)
    cp.testing.assert_allclose(out[0], flat@flat.T, atol=2e-13, rtol=2e-13)


@gpu
def test_layout_alias_architecture_and_memory_guards_leave_output_untouched(small_int8, monkeypatch):
    import cupy as cp

    a = cp.ones((1, 17, 7, 13))
    out = cp.full((1, 17, 17), .25)
    kernel = small_int8._Int8Gram()
    assert not kernel.update(a.astype(cp.float32), out)
    assert not kernel.update(a[..., ::2], out)
    alias = a.reshape(-1)[:17*17].reshape(1, 17, 17)
    assert not kernel.update(a, alias)
    cp.testing.assert_array_equal(alias, 1)
    with monkeypatch.context() as patch:
        patch.setattr(small_int8, '_int8_supported', lambda device: False)
        assert not kernel.update(a, out)
    with monkeypatch.context() as patch:
        patch.setattr(small_int8, '_available_memory', lambda: 0)
        assert not kernel.update(a, out)
    assert not hasattr(kernel._workspace, 'streams')
    cp.testing.assert_array_equal(out, .25)


@gpu
def test_current_stream_state_restoration_and_separate_workspaces(small_int8):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    a = cp.arange(17*39, dtype=cp.float64).reshape(1, 17, 3, 13)/100
    expected = a.reshape(17, -1) @ a.reshape(17, -1).T
    handle = cp.cuda.device.get_cublas_handle()
    state = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
    kernel = small_int8._Int8Gram()
    out = cp.zeros((1, 17, 17))
    assert kernel.update(a, out)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        other = cp.zeros_like(out)
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        try:
            assert kernel.update(a, other)
            assert cublas.getPointerMode(handle) == cublas.CUBLAS_POINTER_MODE_DEVICE
            assert cublas.getMathMode(handle) == state[1]
            assert cublas.getStream(handle) == state[0]
        finally:
            cublas.setPointerMode(handle, state[2])
    stream.synchronize()
    assert len(kernel._workspace.streams) == 2
    cp.testing.assert_allclose(other[0], expected, atol=2e-12, rtol=3e-13)


@gpu
def test_cuda_failure_restores_blas_state(small_int8, monkeypatch):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    a = cp.ones((1, 17, 3, 13))
    out = cp.zeros((1, 17, 17))
    handle = cp.cuda.device.get_cublas_handle()
    state = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)

    def failure(*args):
        raise RuntimeError('injected integer GEMM failure')

    monkeypatch.setattr(cublas, 'gemmEx', failure)
    with pytest.raises(RuntimeError, match='injected integer GEMM failure'):
        small_int8._Int8Gram().update(a, out)
    assert (cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)) == state


@gpu
def test_changing_shapes_cannot_accumulate_unbounded_scratch(small_int8, monkeypatch):
    import cupy as cp

    monkeypatch.setattr(small_int8, '_MAX_WORKSPACE_BYTES', 30000)
    kernel = small_int8._Int8Gram()
    for m, k in ((16, 160), (32, 32), (16, 160)):
        out = cp.zeros((1, m, m))
        assert kernel.update(cp.ones((1, m, 1, k)), out)
        cp.testing.assert_array_equal(out, k)
        assert len(kernel._workspace.streams) == 1
        _, buffers = next(iter(kernel._workspace.streams.values()))
        assert sum(buffer.nbytes for buffer in buffers) <= 30000
