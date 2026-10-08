"""Resource enforcement and opt-in CUDA correctness for the benchmark experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from gpu_performance import THREAD_VARIABLES, _resources, _settings

gpu = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA experiment')


def test_four_cpu_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(os, 'sched_setaffinity', lambda pid, cpus: calls.append(cpus), raising=False)
    for key in (*THREAD_VARIABLES, 'OMP_DYNAMIC', 'OMP_MAX_ACTIVE_LEVELS'):
        monkeypatch.setenv(key, 'original')
    _resources([4, 5, 6, 7])
    assert calls == [[4, 5, 6, 7]]
    assert all(os.environ[key] == '4' for key in THREAD_VARIABLES)
    assert os.environ['OMP_DYNAMIC'] == 'FALSE'
    for cpus in ([1], [1, 2, 3, 4, 5], [1, 1, 2, 3], [6, 7, 8, 9]):
        with pytest.raises(ValueError):
            _resources(cpus)


def test_inline_task_errors_are_retained():
    from grid_experiment import _InlineExecutor

    with _InlineExecutor(1) as executor:
        assert executor.submit(lambda x: x + 1, 3).result() == 4
        future = executor.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            future.result()
    with pytest.raises(ValueError):
        _InlineExecutor(2)


def test_protocol_preserves_method_and_provenance():
    settings, checksum = _settings()
    assert settings['xc'] == 'wb97m-v'
    assert settings['basis'] == 'def2-tzvpd'
    assert settings['conv_tol'] == 1e-8
    assert settings['grid_level'] == 5 and settings['nlc_grid_level'] == 3
    assert checksum == hashlib.sha256(Path(__file__).with_name('protocol.json').read_bytes()).hexdigest()


@pytest.mark.parametrize(
    'key,value',
    [
        ('unknown', True),
        ('protocol_version', True),
        ('ecp', None),
        ('density_fitting', False),
        ('grid_level', 10),
        ('nlc_grid_level', True),
        ('conv_tol', -1),
        ('max_cycle', 0),
        ('basis', ''),
        ('auxiliary_basis', []),
    ],
)
def test_protocol_rejects_unsupported_settings(tmp_path, key, value):
    settings, _ = _settings()
    settings[key] = value
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        _settings(path)


def test_protocol_rejects_missing_settings(tmp_path):
    path = tmp_path / 'protocol.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        _settings(path)


@gpu
@pytest.mark.parametrize('mode', ['compiled', 'refined', 'float32'])
@pytest.mark.parametrize('block', [64, 128, 256])
def test_vv10_partial_tiles_and_signed_weights(mode, block):
    import cupy as cp
    import numpy as np
    from vv10_experiment import _VV10

    from gpu4pyscf.dft import numint

    rng = np.random.default_rng(31)
    rho = cp.asarray(np.vstack((rng.uniform(0.01, 2, 257), rng.normal(0, 0.02, (3, 257)))))
    coords = cp.asarray(rng.normal(0, 3, (257, 3)))
    weights = cp.asarray(rng.normal(0, 0.05, 257))
    rho[0, :3] = 0
    weights[3:6] = 0
    pars = (6.0, 0.01)
    reference = numint._vv10nlc(rho, coords, weights, pars)
    result = _VV10(mode, block)(rho, coords, weights, pars)
    tolerance = 1e-6 if mode == 'float32' else 1e-12
    for actual, expected in zip(result, reference):
        cp.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * 1e-2)
    assert bool(cp.all(result[0][:3] == 0))


@gpu
def test_vv10_empty_density():
    import cupy as cp
    from vv10_experiment import _VV10

    result = _VV10()(cp.zeros((4, 13)), cp.zeros((13, 3)), cp.ones(13), (6.0, 0.01))
    assert all(bool(cp.all(array == 0)) for array in result)


@gpu
def test_vv10_profiling_is_opt_in(monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft._mixed_vv10 import _AdaptiveVV10

    rng = np.random.default_rng(32)
    rho = cp.asarray(np.vstack((rng.uniform(0.01, 2, 129), rng.normal(0, 0.02, (3, 129)))))
    coords = cp.asarray(rng.normal(0, 3, (129, 3)))
    weights = cp.ones(129) * 0.05
    timed = _AdaptiveVV10(None, 128, 'nvrtc', profile=True)
    reference = timed(rho, coords, weights, (6.0, 0.01))
    assert len(timed.pair_seconds) == 1 and timed.pair_seconds[0] > 0
    untimed = _AdaptiveVV10(None, 128, 'nvrtc')

    def forbidden_event(*args, **kwargs):
        raise AssertionError('Production VV10 must not create profiling events')

    with monkeypatch.context() as patch:
        patch.setattr(cp.cuda, 'Event', forbidden_event)
        result = untimed(rho, coords, weights, (6.0, 0.01))
    for actual, expected in zip(result, reference):
        cp.testing.assert_array_equal(actual, expected)
    assert untimed.pair_seconds == [] and untimed.coarse.active_grids == [129]


@gpu
def test_direct_blas_rectangular_and_accumulation():
    import cupy as cp
    import numpy as np
    from grid_experiment import _gemm, install

    from gpu4pyscf.dft import numint

    rng = np.random.default_rng(72)
    a, b, initial = [cp.asarray(rng.normal(size=shape)) for shape in ((17, 83), (31, 83), (17, 31))]
    expected = 0.2 * (a @ b.T) - 1.3 * initial
    actual = _gemm(a, b, alpha=0.2, beta=-1.3, out=initial.copy())
    cp.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    cp.testing.assert_allclose(_gemm(a, b), a @ b.T, atol=1e-13, rtol=1e-13)
    cp.testing.assert_allclose(_gemm(a.astype(cp.float32), b.astype(cp.float32)), a @ b.T, atol=1e-5, rtol=1e-5)
    with install('mixed') as precision:
        out = initial.copy()
        actual = numint.contract('ig,jg->ij', a, b, alpha=0.2, beta=-1.3, out=out)
        assert precision.last_precision == 'float32'
        cp.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        precision.use_float32 = False
        actual = numint.contract('ig,jg->ij', a, b)
        assert precision.last_precision == 'float64'
        cp.testing.assert_allclose(actual, a @ b.T, atol=1e-13, rtol=1e-13)
    original = numint.contract
    with pytest.raises(RuntimeError), install():
        cp.testing.assert_allclose(numint.contract('ig,jg->ij', a, b), a @ b.T)
        raise RuntimeError('Check restoration')
    assert numint.contract is original


@gpu
@pytest.mark.parametrize('alpha,beta', [(1.0, 0.0), (0.2, -1.3), (-2.0, 1.0), (0.0, 1.0)])
def test_mixed_grid_fused_float64_accumulation(alpha, beta):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft._mixed_grid import _gemm, _GridPrecision, _make_contract

    rng = np.random.default_rng(73)
    a, b = [cp.asarray(rng.normal(size=shape)) for shape in ((17, 83), (31, 83))]
    initial = cp.asarray(rng.normal(size=(17, 31))) if beta else cp.full((17, 31), cp.nan)
    product = _gemm(a.astype(cp.float32), b.astype(cp.float32)).astype(cp.float64)
    expected = alpha * product
    if beta:
        expected += beta * initial
    contract = _make_contract(None, _GridPrecision(True))
    out = initial.copy()
    actual = contract('ig,jg->ij', a, b, alpha=alpha, beta=beta, out=out)
    assert actual is out and actual.dtype == cp.float64
    cp.testing.assert_array_equal(actual, expected)
    if not beta:
        actual = contract('ig,jg->ij', a, b, alpha=alpha)
        cp.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='output matrix'):
        contract('ig,jg->ij', a, b, beta=1.0)


@gpu
def test_mixed_grid_changing_blocks_and_streams_keep_outputs():
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft._mixed_grid import _gemm, _GridPrecision, _make_contract

    rng = np.random.default_rng(74)
    inputs = [
        (cp.asarray(rng.normal(size=(rows, grids))), cp.asarray(rng.normal(size=(cols, grids))))
        for rows, cols, grids in ((17, 31, 83), (5, 11, 13), (29, 7, 113), (17, 31, 83))
    ]
    cp.cuda.get_current_stream().synchronize()
    streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]
    contract = _make_contract(None, _GridPrecision(True))
    results = []
    for _ in range(3):
        for stream in streams:
            with stream:
                for a, b in inputs:
                    result = contract('ig,jg->ij', a, b)
                    expected = _gemm(a.astype(cp.float32), b.astype(cp.float32)).astype(cp.float64)
                    results.append((result, expected))
    for stream in streams:
        stream.synchronize()
    for actual, expected in results:
        cp.testing.assert_array_equal(actual, expected)


@gpu
def test_mixed_grid_scope_releases_scratch_without_collecting_results():
    import weakref

    import cupy as cp
    from grid_experiment import install

    from gpu4pyscf.dft import numint

    a = cp.ones((17, 83))
    b = cp.ones((31, 83))
    with install('mixed') as controller:
        result = numint.contract('ig,jg->ij', a, b)
        buffers = [weakref.ref(buffer) for _, values in controller._workspace.streams.values() for buffer in values]
        assert all(reference() is not None for reference in buffers)
    assert all(reference() is None for reference in buffers)
    cp.testing.assert_array_equal(result, cp.full((17, 31), 83.0))
