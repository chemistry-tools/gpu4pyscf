"""Check auxiliary-response products against native matrices and full forces."""

from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path

import pytest

gpu = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location('response_product_plan', ROOT/'gpu4pyscf/df/_response_products.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('shape,strides,operation,leading', [
    ((5166, 171, 171), (8, 7067088, 41328), 0, 5166),
    ((256, 64, 64), (32768, 512, 8), 1, 4096),
    ((256, 64, 64), (32800, 512, 8), 1, 4100),
    ((256, 64, 64), (8, 2048, 131072), 0, 256),
])
def test_layout_preserves_native_or_pitched_storage(shape, strides, operation, leading):
    layout = _module()._gram_layout(shape, strides)
    assert layout[:4] == (shape[0], shape[1]*shape[2], operation, leading)
    assert layout[4] == sum((n-1)*stride for n, stride in zip(shape, strides))+8


@pytest.mark.parametrize('shape,strides', [
    ((0, 64, 64), (32768, 512, 8)), ((255, 64, 64), (32768, 512, 8)),
    ((256, 1, 2), (16, 16, 8)), ((256, 64, 64), (32768, 512, -8)),
    ((256, 64, 64), (32768, 520, 8)), ((8193, 64, 64), (32768, 512, 8)),
    ((256, 64), (512, 8)), ((256, 64, 64), (4, 512, 8)),
])
def test_unsupported_layouts_keep_native_contraction(shape, strides):
    assert _module()._gram_layout(shape, strides) is None


@pytest.fixture
def products(monkeypatch):
    from gpu4pyscf.df import _response_products

    monkeypatch.setattr(_response_products, '_MIN_METRIC_AUX', 0)
    monkeypatch.setattr(_response_products, '_MIN_GRAM_AUX', 0)
    monkeypatch.setattr(_response_products, '_MIN_REDUCTION', 0)
    return _response_products


@gpu
@pytest.mark.parametrize('layout', ['C', 'F', 'F3', 'pitched'])
@pytest.mark.parametrize('alpha,beta', [(-.1, 1.), (.7, -.3), (1., 0.)])
def test_signed_gram_matches_general_product_without_packing(layout, alpha, beta, products):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(1964)
    m, n, p = 33, 7, 11
    raw = cp.asarray(rng.normal(size=(m, n*p)))
    if layout == 'F':
        raw = cp.asfortranarray(raw)
    elif layout == 'pitched':
        storage = cp.empty((m, n*p+5))
        storage[:,:n*p] = raw
        raw = storage[:,:n*p]
    values = raw.reshape(m, n, p)
    assert values.data.ptr == raw.data.ptr
    if layout == 'F3':
        values = cp.asfortranarray(values)
        raw = values.reshape(m, n*p)
    initial = cp.asarray(rng.normal(size=(m, m)))
    initial = initial+initial.T
    expected = alpha*(raw @ raw.T)+beta*initial
    out = initial.copy()
    assert products._symmetric_aux_gram(values, out, alpha, beta) is out
    cp.testing.assert_allclose(out, expected, atol=2e-12, rtol=4e-13)
    cp.testing.assert_array_equal(out, out.T)
    if beta == 0:
        result = products._symmetric_aux_gram(values, None, alpha, beta)
        cp.testing.assert_allclose(result, expected, atol=2e-12, rtol=4e-13)


@gpu
def test_gram_guards_preserve_output_and_native_nonfinite_semantics(products, monkeypatch):
    import cupy as cp

    values = cp.ones((17, 7, 11))
    out = cp.full((17, 17), .5)
    assert products._symmetric_aux_gram(values.astype(cp.float32), out, 1, 1) is None
    assert products._symmetric_aux_gram(values[..., ::-1], out, 1, 1) is None
    alias = values.reshape(-1)[:17*17].reshape(17, 17)
    assert products._symmetric_aux_gram(values, alias, 1, 1) is None
    cp.testing.assert_array_equal(alias, 1)
    out[0, 1] = .7
    saved = out.copy()
    assert products._symmetric_aux_gram(values, out, 1, 1) is None
    cp.testing.assert_array_equal(out, saved)
    out[:] = .5
    values[1, 2, 3] = cp.nan
    assert products._symmetric_aux_gram(values, out, 1, 1) is None
    cp.testing.assert_array_equal(out, .5)
    values[:] = 1
    with monkeypatch.context() as patch:
        patch.setattr(products, '_supported_device', lambda device: False)
        assert products._symmetric_aux_gram(values, out, 1, 1) is None
    with monkeypatch.context() as patch:
        patch.setattr(products, '_available_memory', lambda: 0)
        assert products._symmetric_aux_gram(values, None, 1, 0) is None
    out[:] = cp.nan
    assert products._symmetric_aux_gram(values, out, 1, 0) is out
    cp.testing.assert_array_equal(out, 77)


@gpu
def test_gram_restores_blas_state_on_stream_and_error(products, monkeypatch):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    handle = cp.cuda.device.get_cublas_handle()
    original = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        values = cp.ones((17, 7, 11))
        out = cp.zeros((17, 17))
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        try:
            assert products._symmetric_aux_gram(values, out, 1, 1) is out
            state = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
            assert state == (*original[:2], cublas.CUBLAS_POINTER_MODE_DEVICE)

            def fail(*args):
                raise RuntimeError('injected SYRK failure')

            monkeypatch.setattr(cublas, 'dsyrk', fail)
            with pytest.raises(RuntimeError, match='injected SYRK failure'):
                products._symmetric_aux_gram(values, out, 1, 1)
            assert (cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)) == state
        finally:
            cublas.setPointerMode(handle, original[2])
    stream.synchronize()
    cp.testing.assert_array_equal(out, 77)


@gpu
@pytest.mark.parametrize('interleaved', [False, True])
def test_spin_sum_and_later_invalid_spin_are_transactional(interleaved, products):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(1966)
    m, n, p = 17, 7, 11
    raw = cp.asarray(rng.normal(size=(2, m, n, p)))
    if interleaved:
        storage = cp.empty((n*p, 2, m))
        values = storage.transpose(1, 2, 0).reshape(2, m, n, p)
        values[:] = raw
    else:
        values = raw
    flat = raw.reshape(2, m, n*p)
    initial = cp.full((m, m), .5)
    expected = -.2*(flat @ flat.transpose(0, 2, 1)).sum(axis=0)+.7*initial
    actual = initial.copy()
    assert products._symmetric_aux_gram(values, actual, -.2, .7) is actual
    cp.testing.assert_allclose(actual, expected, atol=2e-12, rtol=4e-13)
    values[1, 2, 3, 4] = cp.inf
    actual = initial.copy()
    assert products._symmetric_aux_gram(values, actual, -.2, .7) is None
    cp.testing.assert_array_equal(actual, initial)


def _method(case, charge=0, spin=0, cart=True):
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='6-31g*',
                charge=charge, spin=spin, cart=cart, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.small_rho_cutoff = 1, 0
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-11, 1e-9, 1e-15
    mf.max_cycle, mf.chkfile = 100, None
    return mf


@gpu
@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
def test_sparse_metric_matches_same_native_eigenpairs_bitwise(case, products, monkeypatch):
    import cupy as cp

    from gpu4pyscf.df.grad import rhf

    mf = _method(case)
    mf.with_df.build(build_cderi=False)
    opt = mf.with_df.intopt
    auxmol, sorting = opt.auxmol, opt.int3c2e_evaluator(reorder_aux=True)[1]
    coefficient = auxmol.ctr_coeff
    sorted_coefficient = cp.empty_like(coefficient)
    sorted_coefficient[sorting] = coefficient
    plan = products._SparseAuxTransform.create(auxmol, sorting, sorted_coefficient)
    assert plan is not None
    j2c = rhf.int2c2e(auxmol)
    w, v = rhf.eigh(j2c)
    monkeypatch.setattr(rhf, 'eigh', lambda *args: (w, v))
    expected = sorted_coefficient @ rhf._gen_metric_solver(j2c, 'ED')(sorted_coefficient.T)
    solver = rhf._gen_metric_solver(j2c, 'ED', _rhs_projector=plan.project_rhs)
    actual = plan.apply(solver(sorted_coefficient.T))
    cp.testing.assert_array_equal(actual, expected)
    other = cp.arange(len(coefficient)*3, dtype=cp.float64).reshape(len(coefficient), 3)/1000
    cp.testing.assert_array_equal(plan.project_rhs(v, other), v.T @ other)
    with monkeypatch.context() as patch:
        patch.setattr(products, '_available_memory', lambda: 0)
        assert products._SparseAuxTransform.create(auxmol, sorting, sorted_coefficient) is None


@gpu
@pytest.mark.parametrize('case,charge,spin', [('CH3CONH2', 0, 0), ('C5H5N', 0, 0),
                                           ('C2H6SO', 0, 0), ('CH3COF', 0, 0),
                                           ('CH3COCl', 0, 0), ('H2O', 1, 1)])
def test_complete_forces_match_unmodified_reference(case, charge, spin, products, monkeypatch):
    import cupy as cp
    import numpy as np
    from native_force_benchmark import _reference_df

    reference = os.getenv('DFT_REFERENCE_UHF_DF_GRADIENT' if spin else 'DFT_REFERENCE_DF_GRADIENT')
    assert reference, 'Retain the matching original DF gradient source for this test'
    mf = _method(case, charge, spin)
    mf.kernel()
    assert mf.converged
    dm = mf.make_rdm1()
    calls = {'metric': 0, 'gram': 0}
    apply = products._SparseAuxTransform.apply
    gram = products._symmetric_aux_gram

    def counted_apply(self, matrix):
        calls['metric'] += 1
        return apply(self, matrix)

    def counted_gram(*args):
        result = gram(*args)
        calls['gram'] += int(result is not None)
        return result

    from gpu4pyscf.df.grad import rhf, uhf

    monkeypatch.setattr(products._SparseAuxTransform, 'apply', counted_apply)
    monkeypatch.setattr(uhf if spin else rhf, '_symmetric_aux_gram', counted_gram)
    results = []
    for scope in (_reference_df(Path(reference), unrestricted=bool(spin)), contextlib.nullcontext()):
        mf.with_df.reset()
        cp.get_default_memory_pool().free_all_blocks()
        mf.get_veff(mf.mol, dm)
        gradient = mf.nuc_grad_method()
        gradient.grid_response = gradient.auxbasis_response = True
        with scope:
            results.append(gradient.kernel())
    assert calls['metric'] >= 2
    assert calls['gram'] > 0
    np.testing.assert_allclose(results[1], results[0], atol=2e-10, rtol=2e-10)


@gpu
@pytest.mark.parametrize('spin', [0, 1])
@pytest.mark.parametrize('hermi,j_factor,omega,response', [(1, 1, 0, True), (1, 0, .3, True),
                                                         (0, 1, 0, True), (0, 0, -.2, False)])
def test_general_density_and_range_operators_preserve_reference(hermi, j_factor, omega, response, spin, products):
    import cupy as cp
    import numpy as np
    from native_force_benchmark import _reference_df

    reference = Path(os.environ['DFT_REFERENCE_UHF_DF_GRADIENT' if spin else 'DFT_REFERENCE_DF_GRADIENT'])
    mf = _method('H2O', charge=spin, spin=spin)
    mf.with_df.build()
    rng = np.random.default_rng(1965)
    left, right = (rng.normal(size=(mf.mol.nao, 5)) for _ in range(2))
    density = cp.asarray(left @ (left if hermi == 1 else right).T/mf.mol.nao)
    if spin:
        density = cp.stack([density, density*.6])
    gradient = mf.nuc_grad_method()
    gradient.auxbasis_response = response
    results = []
    for scope in (_reference_df(reference, unrestricted=bool(spin)), contextlib.nullcontext()):
        mf.with_df.reset()
        cp.get_default_memory_pool().free_all_blocks()
        with scope:
            results.append(gradient.jk_energy_per_atom(density, j_factor=j_factor, k_factor=.7,
                                                       omega=omega, hermi=hermi))
    np.testing.assert_allclose(results[1], results[0], atol=2e-10, rtol=2e-10)


@gpu
def test_complete_force_matches_energy_finite_difference(products):
    mf = _method('CH3COF')
    mf.kernel()
    assert mf.converged
    gradient = mf.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    analytic = gradient.kernel()[0, 0]
    coordinates, step, energies = mf.mol.atom_coords(), 2e-4, []
    for sign in (-1, 1):
        displaced = coordinates.copy()
        displaced[0, 0] += sign*step
        mf.reset(mf.mol.set_geom_(displaced, unit='Bohr', inplace=False))
        energies.append(mf.kernel())
        assert mf.converged
    assert abs(analytic-(energies[1]-energies[0])/(2*step)) < 1e-5
