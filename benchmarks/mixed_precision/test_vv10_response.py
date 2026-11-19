"""Validate fused VV10 fields, coordinate derivatives and resource-safe edge cases."""

from __future__ import annotations

import os

import pytest
from vv10_response_experiment import native_response

pytestmark = pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('block', [64, 128, 256])
@pytest.mark.parametrize('n,natm', [(1, 1), (67, 1), (257, 6)])
def test_fused_fields_and_signed_weight_response(block, n, natm):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft._vv10_grad import _pair_response

    rng = np.random.default_rng(193)
    coords = cp.asarray(rng.normal(size=(n, 3)))
    rw = cp.asarray(rng.normal(scale=.05, size=n))
    rw[0] = 0
    omega = cp.asarray(rng.uniform(.2, 2., n))
    kappa = cp.asarray(rng.uniform(.1, 1., n))
    atoms = cp.asarray(np.sort(rng.choice([0, 2, 3] if natm == 6 else [0], n)), dtype=cp.int32)
    reference = native_response(coords, rw, omega, kappa, atoms, natm)
    actual = _pair_response(coords, rw, omega, kappa, atoms, natm, block=block)
    for x, y in zip(actual, reference):
        cp.testing.assert_allclose(x, y, atol=1e-12, rtol=2e-12)
    cp.testing.assert_allclose(actual[3].sum(axis=0), 0, atol=1e-12)
    if natm == 1:
        cp.testing.assert_array_equal(actual[3], 0)


@pytest.mark.parametrize('symmetric', [False, True])
def test_coordinate_response_matches_independent_energy_difference(symmetric, monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import _vv10_symmetric
    from gpu4pyscf.dft._vv10_grad import _pair_response

    if symmetric:
        if cp.cuda.Device().compute_capability != '86':
            pytest.skip('Symmetric VV10 is qualified on compute capability 8.6')
        monkeypatch.setattr(_vv10_symmetric, '_MIN_POINTS', 0)

    rng = np.random.default_rng(219)
    coords = rng.normal(size=(29, 3))
    rw = rng.normal(scale=.05, size=29)
    omega, kappa = rng.uniform(.2, 2., (2, 29))
    atoms = np.sort(rng.integers(0, 4, size=29)).astype(np.int32)
    gradient = _pair_response(*(cp.asarray(a) for a in (coords, rw, omega, kappa, atoms)), 4)[3].get()

    def energy(positions):
        r2 = np.sum((positions[:, None, :] - positions[None, :, :])**2, axis=2)
        gi = omega[:, None]*r2 + kappa[:, None]
        gj = omega[None, :]*r2 + kappa[None, :]
        return -.75*np.sum(rw[:, None]*rw[None, :]/(gi*gj*(gi+gj)))

    step = 1e-5
    for atom in range(4):
        for direction in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[atoms == atom, direction] += step
            minus[atoms == atom, direction] -= step
            assert abs(gradient[atom, direction]-(energy(plus)-energy(minus))/(2*step)) < 1e-9


def test_empty_fields_and_invalid_labels():
    import cupy as cp

    from gpu4pyscf.dft._vv10_grad import _pair_response

    fields = [cp.empty((0, 3)), *(cp.empty(0) for _ in range(3)), cp.empty(0, dtype=cp.int32)]
    result = _pair_response(*fields, 3)
    assert all(a.size == 0 for a in result[:3])
    cp.testing.assert_array_equal(result[3], 0)
    fields = [cp.zeros((3, 3)), *(cp.ones(3) for _ in range(3))]
    for labels in ([1, 0, 2], [0, 1, 3], [-1, 0, 1]):
        with pytest.raises(ValueError):
            _pair_response(*fields, cp.asarray(labels, dtype=cp.int32), 3)
    with pytest.raises(TypeError):
        _pair_response(*fields, cp.arange(3, dtype=cp.int64), 3)


def test_independent_cuda_streams_do_not_share_response_buffers():
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft._vv10_grad import _pair_response

    rng = np.random.default_rng(84)
    first = [cp.asarray(rng.normal(size=(65, 3))), cp.asarray(rng.normal(scale=.05, size=65)),
             cp.ones(65), cp.ones(65), cp.asarray([0]*32+[1]*33, dtype=cp.int32)]
    second = [first[0]*2, first[1]*3, *first[2:]]
    cp.cuda.get_current_stream().synchronize()
    references = [native_response(*fields, 2) for fields in (first, second)]
    cp.cuda.get_current_stream().synchronize()
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    results = []
    for stream, fields in zip(streams, (first, second)):
        with stream:
            results.append(_pair_response(*fields, 2))
    for stream in streams:
        stream.synchronize()
    for result, reference in zip(results, references):
        for actual, expected in zip(result, reference):
            cp.testing.assert_allclose(actual, expected, atol=1e-12, rtol=2e-12)


def test_symmetric_batches_and_workspace_fallback(monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import _vv10_symmetric
    from gpu4pyscf.dft._vv10_grad import _pair_response
    from gpu4pyscf.lib import cupy_helper

    if cp.cuda.Device().compute_capability != '86':
        pytest.skip('Symmetric VV10 is qualified on compute capability 8.6')
    rng = np.random.default_rng(497)
    counts = [0, 105, 328, 0, 99, 100]
    size = sum(counts)
    coords = cp.asarray(rng.normal(size=(size, 3)))
    rw = cp.asarray(rng.normal(scale=.025, size=size))
    omega, kappa = (cp.asarray(x) for x in rng.uniform(.2, 2., (2, size)))
    atoms = cp.repeat(cp.arange(len(counts), dtype=cp.int32), cp.asarray(counts))
    inputs = coords, rw, omega, kappa, atoms, len(counts)
    reference = _pair_response(*inputs)
    monkeypatch.setattr(_vv10_symmetric, '_MIN_POINTS', 1)
    monkeypatch.setattr(_vv10_symmetric, '_TILES_PER_BATCH', 2)
    monkeypatch.setattr(cupy_helper, 'get_avail_mem', lambda: 1 << 30)
    native_tiles = _vv10_symmetric._pair_tiles
    calls = []

    def counted(*args):
        calls.append(True)
        return native_tiles(*args)

    monkeypatch.setattr(_vv10_symmetric, '_pair_tiles', counted)
    result = _pair_response(*inputs)
    assert calls == [True]
    for actual, expected in zip(result, reference):
        cp.testing.assert_allclose(actual, expected, atol=1e-12, rtol=2e-12)
    cp.testing.assert_allclose(result[3].sum(axis=0), 0., atol=1e-12)
    monkeypatch.setattr(cupy_helper, 'get_avail_mem', lambda: 0)
    for actual, expected in zip(_pair_response(*inputs), reference):
        cp.testing.assert_array_equal(actual, expected)
    assert calls == [True]
