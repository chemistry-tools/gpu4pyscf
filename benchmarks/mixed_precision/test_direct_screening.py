"""Check screening counts against hand examples and native queue helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from direct_screening import ScreeningCounter, screen_cpu


@pytest.mark.parametrize('mode,expected', [('exchange', 2), ('force-jk', 0), ('force-k', 0)])
def test_separate_symmetry_coarse_and_density_rejections(mode, expected):
    pairs = np.array([3, 2, 0], dtype=np.uint32)
    q = np.array([0., -10., -30.], dtype=np.float32)
    dm = np.array([[-1., -10.], [-10., -40.]], dtype=np.float32)
    result = screen_cpu(pairs, q, pairs, q, dm, mode, cutoffs=(1e-13,))
    assert result.tolist() == [[6, 3, expected]]


def test_density_cutoff_is_strict_and_empty_panels_are_zero():
    pair, q = np.array([0], dtype=np.uint32), np.array([0.], dtype=np.float32)
    result = screen_cpu(pair, q, pair, q, np.array([[-1.]]), 'exchange', (float(np.exp(-1.)),))
    assert result.tolist() == [[1, 1, 0]]
    assert not screen_cpu(pair[:0], q[:0], pair, q, np.array([[-1.]]), 'exchange').any()
@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
@pytest.mark.parametrize('mode', ['exchange', 'force-jk', 'force-k'])
def test_gpu_stage_counts_match_cpu_and_production_helpers(mode):
    import cupy as cp

    source = Path(os.environ['DFT_SCREEN_NATIVE_SOURCE'])
    counter = ScreeningCounter(source)
    pairs = np.array([8, 7, 6, 4, 3, 0], dtype=np.uint32)
    q = np.array([0., -3., -6., -11., -20., -45.], dtype=np.float32)
    dm = np.array([[-1., -8., -20.], [-8., -6., -13.], [-20., -13., -30.]], dtype=np.float32)
    result = counter.count(pairs, q, pairs, q, dm, mode)
    np.testing.assert_array_equal(result['stages'], screen_cpu(pairs, q, pairs, q, dm, mode))
    assert result['native_survivors'] == result['stages'][0][2]
    assert counter.count(pairs[:0], q[:0], pairs, q, dm, mode)['nominal'] == 0
    with pytest.raises(ValueError, match='square'):
        counter.count(pairs, q, pairs, q, cp.zeros((3, 2)), mode)


@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
def test_native_queue_crosses_chunk_and_partial_thread_boundaries():
    source = Path(os.environ['DFT_SCREEN_NATIVE_SOURCE'])
    counter = ScreeningCounter(source)
    nbas = 266
    pa = np.array([nbas*nbas-1, nbas*nbas-2], dtype=np.uint32)
    pb = np.arange(70001, dtype=np.uint32)
    qa = np.array([0., -1.], dtype=np.float32)
    qb = np.linspace(0., -40., len(pb), dtype=np.float32)
    dm = np.full((nbas, nbas), -3., dtype=np.float32)
    actual = counter.count(pa, qa, pb, qb, dm, 'force-jk')
    # The CUDA replay is independently checked against the real queue helpers.
    assert actual['native_survivors'] == actual['stages'][0][2] > 0
    assert actual['stages'][0][0] == len(pa)*len(pb)
