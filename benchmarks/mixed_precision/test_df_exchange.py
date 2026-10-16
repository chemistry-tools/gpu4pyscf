"""Check symmetric density-fitting exchange updates and native fallbacks."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.fixture
def symmetric(monkeypatch):
    from gpu4pyscf.df import _jk_contraction

    monkeypatch.setattr(_jk_contraction, '_MIN_ORBITALS', 0)
    monkeypatch.setattr(_jk_contraction, '_MIN_REDUCTION', 0)
    return _jk_contraction


@pytest.fixture
def integer(monkeypatch):
    from gpu4pyscf.df import _int8_gram

    monkeypatch.setattr(_int8_gram, '_MIN_ORBITALS', 0)
    monkeypatch.setattr(_int8_gram, '_MIN_REDUCTION', 0)
    return _int8_gram


@pytest.mark.parametrize('shape', [(1, 17, 7, 11), (2, 33, 9, 13)])
def test_symmetric_exchange_preserves_general_accumulation(shape, symmetric):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.lib.cupy_helper import contract

    rng = np.random.default_rng(1616)
    rhok = cp.asarray(rng.normal(size=shape))
    initial = cp.asarray(rng.normal(size=(shape[0], shape[1], shape[1])))
    expected = contract('nikL,njkL->nij', rhok, rhok, beta=1, out=initial.copy())
    actual = initial.copy()
    assert symmetric._symmetric_k_update(rhok, actual)
    cp.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-13)


def test_symmetric_exchange_restores_blas_state_and_uses_current_stream(symmetric):
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    rng = np.random.default_rng(1617)
    rhok = cp.asarray(rng.normal(size=(1, 23, 9, 11)))
    out = cp.zeros((1, 23, 23))
    expected = rhok.reshape(23, -1) @ rhok.reshape(23, -1).T
    cp.cuda.get_current_stream().synchronize()
    handle = cp.cuda.device.get_cublas_handle()
    state = cublas.getMathMode(handle), cublas.getPointerMode(handle), cublas.getStream(handle)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
        try:
            assert symmetric._symmetric_k_update(rhok, out)
            assert cublas.getPointerMode(handle) == cublas.CUBLAS_POINTER_MODE_DEVICE
            assert cublas.getMathMode(handle) == state[0]
            assert cublas.getStream(handle) == state[2]
        finally:
            cublas.setPointerMode(handle, state[1])
    stream.synchronize()
    cp.testing.assert_allclose(out[0], expected, atol=3e-12, rtol=3e-13)


def test_symmetric_exchange_rejects_aliases_and_other_layouts(symmetric):
    import cupy as cp

    a = cp.ones((1, 17, 7, 11))
    out = cp.zeros((1, 17, 17))
    assert not symmetric._symmetric_k_update(a.astype(cp.float32), out)
    assert not symmetric._symmetric_k_update(a[..., ::2], out)
    alias = a.reshape(-1)[:17*17].reshape(1, 17, 17)
    assert not symmetric._symmetric_k_update(a, alias)
    cp.testing.assert_array_equal(alias, 1)


@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
def test_complete_mixed_atom_density_fitted_jk_matches_full_product(case, symmetric, monkeypatch):
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.df import df_jk

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='6-31g*', cart=True, verbose=0)
    mf = dft.RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
    _, coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
    occ = cp.zeros(coeff.shape[1])
    occ[:mol.nelectron//2] = 2
    density = mf.make_rdm1(coeff, occ)
    with monkeypatch.context() as patch:
        patch.setattr(df_jk, '_symmetric_k_update', lambda *args: False)
        expected_j, expected_k = mf.get_jk(dm=density)
    calls = []

    def update(*args):
        result = symmetric._symmetric_k_update(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(df_jk, '_symmetric_k_update', update)
    actual_j, actual_k = mf.get_jk(dm=density)
    assert calls and all(calls)
    cp.testing.assert_allclose(actual_j, expected_j, atol=1e-11, rtol=1e-11)
    cp.testing.assert_allclose(actual_k, expected_k, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('xc,eligible', [('b3lyp', True), ('pbe', False), ('m06-l', False),
                                       ('cam-b3lyp', False), ('wb97x-v', False)])
def test_early_exchange_is_private_scoped_and_gated(xc, eligible):
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft.mixed_precision import _precision_context

    mol = gto.M(atom='O 0 0 0; H .75 0 .5; H -.75 0 .5', basis='6-31g*', verbose=0)
    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    other = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    original = mf.with_df.get_jk.__func__
    with pytest.raises(RuntimeError, match='scoped test failure'), _precision_context(mf) as (grid, _):
        assert (grid._df_exchange is not None) == eligible
        assert (mf.with_df.get_jk.__func__ is not original) == eligible
        assert other.with_df.get_jk.__func__ is original
        raise RuntimeError('scoped test failure')
    assert mf.with_df.get_jk.__func__ is original
    assert 'get_jk' not in mf.with_df.__dict__


def test_early_exchange_stream_scratch_and_float64_switch():
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cutensor
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft.mixed_precision import _precision_context
    from gpu4pyscf.lib.cutensor import contraction

    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mf = dft.RKS(mol, xc='b3lyp').density_fit()
    rng = np.random.default_rng(1618)
    a = cp.asarray(rng.normal(size=(7, 23, 23)))
    b = cp.asarray(rng.normal(size=(2, 23, 9)))
    initial = cp.asarray(rng.normal(size=(2, 23, 9, 7)))
    pattern = 'Lij,njk->nikL'
    product = contraction(pattern, a.astype(cp.float32), b.astype(cp.float32), 1., 0.,
                          compute_desc=cutensor.COMPUTE_DESC_32F)
    expected = .3*product.astype(cp.float64)-.2*initial
    cp.cuda.get_current_stream().synchronize()
    with _precision_context(mf) as (grid, _):
        control = grid._df_exchange
        actual = control.contract(contraction, pattern, a, b, alpha=.3, beta=-.2, out=initial.copy())
        cp.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            out = cp.empty_like(initial)
            control.contract(contraction, pattern, a, b, out=out)
        stream.synchronize()
        assert len(control._workspace.streams) == 2
        assert control.float32_calls == 2
        grid.use_float32 = False
        expected64 = contraction(pattern, a, b, 1., 0.)
        actual64 = control.contract(contraction, pattern, a, b, out=cp.empty_like(initial))
        cp.testing.assert_allclose(actual64, expected64, atol=2e-12, rtol=2e-13)
        assert control.float64_calls == 1 and control.float32_calls == 2
        assert not hasattr(control._workspace, 'streams')


@pytest.mark.parametrize('case,charge,spin', [('CH3CONH2', 0, 0), ('C5H5N', 0, 0),
                                            ('C2H6SO', 0, 0), ('CH3COF', 0, 0),
                                            ('CH3COCl', 0, 0), ('H2O', 1, 1)])
def test_verified_mixed_exchange_matches_float64_density(case, charge, spin, symmetric, integer, monkeypatch):
    import cupy as cp
    from ase.build import molecule
    from pyscf.data.nist import BOHR, HARTREE2EV
    from test_semilocal_precision import _calculator

    from gpu4pyscf.df import _mixed_jk, df_jk
    from gpu4pyscf.dft import mixed_precision

    verifying, integer_calls, controllers = False, [], []
    native_check = mixed_precision._full_check
    native_update = integer._Int8Gram.update
    finish = _mixed_jk._ExchangePrecision.finish_approximation

    def finish_approximation(control):
        finish(control)
        controllers.append(control)

    def check(*args):
        nonlocal verifying
        verifying = True
        assert controllers and controllers[-1].native_only and not controllers[-1].projection_float32
        try:
            return native_check(*args)
        finally:
            verifying = False

    def update(*args):
        assert not verifying, 'Final verification must use the native float64 Gram product'
        result = native_update(*args)
        integer_calls.append(result)
        return result

    monkeypatch.setattr(mixed_precision, '_full_check', check)
    monkeypatch.setattr(integer._Int8Gram, 'update', update)
    monkeypatch.setattr(_mixed_jk._ExchangePrecision, 'finish_approximation', finish_approximation)

    native, mixed = molecule(case), molecule(case)
    native.calc = _calculator(native, 'b3lyp', 'float64', charge=charge,
                              multiplicity=spin+1, density_fit=True)
    mixed.calc = _calculator(mixed, 'b3lyp', 'mixed', charge=charge,
                             multiplicity=spin+1, density_fit=True)
    with monkeypatch.context() as patch:
        patch.setattr(df_jk, '_symmetric_k_update', lambda *args: False)
        expected = native.get_potential_energy()
    actual = mixed.get_potential_energy()
    assert abs(actual-expected) < 1e-8
    report = mixed.calc.precision_info
    assert report['accepted'] and report['final_verification']['passed']
    assert report['df_exchange_contraction_calls']['float32'] > 0
    assert report['df_exchange_contraction_calls']['float64'] > 0
    assert report['df_exchange_contraction_calls']['int8'] > 0
    assert integer_calls and all(integer_calls)
    cp.testing.assert_allclose(mixed.calc.method_scan.make_rdm1(),
                               native.calc.method_scan.make_rdm1(), atol=1e-8, rtol=0)
    assert 'get_jk' not in mixed.calc.method_scan.with_df.__dict__
    # Derivatives execute after the private float32 dispatch has been removed.
    # Cartesian DF gradients amplify tiny differences between converged states.
    # Replay the accepted orbitals in a separate, entirely native calculator.
    reference = native.calc.method_scan
    for name in ('mo_coeff', 'mo_occ', 'mo_energy'):
        setattr(reference, name, getattr(mixed.calc.method_scan, name).copy())
    reference.e_tot = mixed.calc.method_scan.e_tot
    gradient = reference.Gradients()
    gradient.grid_response = gradient.auxbasis_response = True
    reference_forces = -cp.asarray(gradient.kernel()) * HARTREE2EV / BOHR
    cp.testing.assert_allclose(cp.asarray(mixed.get_forces()), reference_forces, atol=1e-5, rtol=0)
    assert mixed.calc.calculation_info['scf_reused']


def test_projection_tail_keeps_scratch_across_int8_gram_and_finishes_natively(integer):
    import cupy as cp
    from cupy_backends.cuda.libs import cutensor

    from gpu4pyscf.df._mixed_jk import _ExchangePrecision
    from gpu4pyscf.lib.cutensor import contraction

    class Grid:
        use_float32 = False

    control = _ExchangePrecision(Grid())
    control.projection_float32 = True
    rng = cp.random.RandomState(1619)
    a, b = rng.normal(size=(7, 23, 23)), rng.normal(size=(2, 23, 9))
    output = cp.empty((2, 23, 9, 7))
    pattern = 'Lij,njk->nikL'
    expected = contraction(pattern, a.astype(cp.float32), b.astype(cp.float32), 1., 0.,
                           compute_desc=cutensor.COMPUTE_DESC_32F).astype(cp.float64)

    def gram(rhok, out):
        return contraction('nikL,njkL->nij', rhok, rhok, alpha=1., beta=1., out=out)

    try:
        actual = control.contract(contraction, pattern, a, b, out=output)
        cp.testing.assert_array_equal(actual, expected)
        scratch = control._workspace.streams
        pointers = [buffer.data.ptr for _, buffers in scratch.values() for buffer in buffers]
        accumulated = cp.zeros((2, 23, 23))
        expected_gram = gram(actual, accumulated.copy())
        control.gram(gram, contraction, actual, accumulated)
        assert control.int8_calls == 1 and control.float32_calls == 1
        cp.testing.assert_allclose(accumulated, expected_gram, atol=1e-10, rtol=2e-13)
        assert control._workspace.streams is scratch
        control.contract(contraction, pattern, a, b, out=output)
        assert [buffer.data.ptr for _, buffers in scratch.values() for buffer in buffers] == pointers
        control.finish_approximation()
        assert not control.projection_float32 and not hasattr(control._workspace, 'streams')
        assert control._int8 is None
        # Even an externally changed search flag cannot re-enable approximation.
        control.projection_float32 = control.grid.use_float32 = True
        actual = control.contract(contraction, pattern, a, b, out=output)
        cp.testing.assert_allclose(actual, contraction(pattern, a, b, 1., 0.), atol=2e-12, rtol=2e-13)
        accumulated.fill(0)
        expected_gram = gram(actual, accumulated.copy())
        control.gram(gram, contraction, actual, accumulated)
        cp.testing.assert_allclose(accumulated, expected_gram, atol=2e-12, rtol=2e-13)
        assert control.float64_calls == 2 and control.int8_calls == 1 and control.float32_calls == 2
        assert not hasattr(control._workspace, 'streams') and control._int8 is None
    finally:
        control.release()


def test_integer_failure_scope_restores_df_and_releases_workspace(integer):
    import cupy as cp
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft.mixed_precision import _precision_context

    mol = gto.M(atom='O 0 0 0; H .75 0 .5; H -.75 0 .5', basis='6-31g*', verbose=0)
    mf = dft.RKS(mol, xc='b3lyp').density_fit()
    original = mf.with_df.get_jk.__func__
    with pytest.raises(ArithmeticError, match='integer scope failure'), _precision_context(mf) as (grid, _):
        grid.use_float32 = False
        control = grid._df_exchange
        control.projection_float32 = True
        rhok = cp.empty((1, 17, 3, 13))
        control.contract(df_jk.contract, 'Lij,njk->nikL', cp.ones((13, 17, 17)),
                         cp.ones((1, 17, 3)), out=rhok)
        out = cp.zeros((1, 17, 17))
        control.gram(lambda *args: out, None, rhok, out)
        assert control.int8_calls == 1
        backend = control._int8
        assert backend._workspace.streams
        raise ArithmeticError('integer scope failure')
    assert control._int8 is None and not hasattr(backend._workspace, 'streams')
    assert not hasattr(control._workspace, 'streams')
    assert 'get_jk' not in mf.with_df.__dict__ and mf.with_df.get_jk.__func__ is original


def test_integer_scope_restarts_for_changed_geometry_and_reuses_scf_for_forces(integer):
    from ase.build import molecule
    from test_semilocal_precision import _calculator

    atoms = molecule('CH3CONH2')
    atoms.calc = _calculator(atoms, 'b3lyp', 'auto', density_fit=True)
    atoms.get_potential_energy()
    first = atoms.calc.precision_info
    assert first['accepted'] and first['df_exchange_contraction_calls']['int8'] > 0
    atoms.positions[0, 0] += .08
    energy = atoms.get_potential_energy()
    second = atoms.calc.precision_info
    assert second is not first and second['accepted']
    assert second['df_exchange_contraction_calls']['int8'] > 0
    assert 'get_jk' not in atoms.calc.method_scan.with_df.__dict__
    reference = atoms.copy()
    reference.calc = _calculator(reference, 'b3lyp', 'float64', density_fit=True)
    assert abs(energy-reference.get_potential_energy()) < 1e-8
    atoms.get_forces()
    assert atoms.calc.precision_info is second and atoms.calc.calculation_info['scf_reused']
