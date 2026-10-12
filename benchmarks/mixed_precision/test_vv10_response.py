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


def test_coordinate_response_matches_independent_energy_difference():
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft._vv10_grad import _pair_response

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
