"""Check signed float64 tau products, mixed dispatch and complete SCF derivatives."""

from __future__ import annotations

import os

import pytest
from _organic_cases import ASE_ORGANICS

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('signs', ['positive', 'negative', 'mixed', 'zero'])
@pytest.mark.parametrize('ngrids', [17, 83])
def test_signed_tau_matches_general_bilinear_product(signs, ngrids, monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import _tau_syrk, numint

    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 0)
    rng = np.random.default_rng(1302)
    ao = cp.asarray(rng.normal(size=(4, 19, ngrids)))
    weights = cp.asarray(rng.uniform(0.1, 1.0, size=ngrids))
    if signs == 'negative':
        weights *= -1
    elif signs == 'mixed':
        weights[::2] *= -1
        weights[::3] = 0
    elif signs == 'zero':
        weights.fill(0)
    expected = numint._tau_dot_full(ao, ao, weights)
    out = cp.full((19, 19), cp.nan)
    actual = numint._tau_dot(ao, ao, weights, out=out)
    assert actual is out
    cp.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    cp.testing.assert_array_equal(actual, actual.T)


@pytest.mark.parametrize('kind', ['small', 'distinct', 'complex', 'strided', 'weights', 'output', 'nonfinite'])
def test_unsupported_tau_preserves_general_product(kind, monkeypatch):
    import cupy as cp

    from gpu4pyscf.dft import numint

    ao = cp.ones((4, 513, 17))
    ket, weights, out = ao, cp.ones(17), None
    if kind == 'small':
        ao = ket = cp.ones((4, 512, 17))
    elif kind == 'distinct':
        ket = ao.copy()
    elif kind == 'complex':
        ao = ket = ao.astype(cp.complex128)
    elif kind == 'strided':
        ao = ket = ao[:, :, ::-1]
    elif kind == 'weights':
        weights = weights.astype(cp.float32)
    elif kind == 'output':
        out = cp.empty((513, 513), dtype=cp.float32)
    elif kind == 'nonfinite':
        weights[0] = cp.nan
    calls, sentinel = [], object()

    def full(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(numint, '_tau_dot_full', full)
    assert numint._tau_dot(ao, ket, weights, out=out) is sentinel
    assert len(calls) == 1


def test_zero_weight_nonfinite_fields_use_original_propagation(monkeypatch):
    import cupy as cp

    from gpu4pyscf.dft import _tau_syrk, numint

    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 0)
    ao, weights = cp.ones((4, 3, 17)), cp.ones(17)
    ao[1, 0, 0], weights[0] = cp.nan, 0
    assert not _tau_syrk._eligible(ao, ao, weights, None)
    actual, expected = numint._tau_dot(ao, ao, weights), numint._tau_dot_full(ao, ao, weights)
    cp.testing.assert_array_equal(cp.isnan(actual), cp.isnan(expected))
    cp.testing.assert_allclose(actual, expected, equal_nan=True)


def test_tau_outputs_survive_multiple_streams(monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import _tau_syrk, numint

    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 0)
    rng = np.random.default_rng(1303)
    inputs = [(cp.asarray(rng.normal(size=(4, 19, k))), cp.asarray(rng.normal(size=k))) for k in (17, 83)]
    cp.cuda.get_current_stream().synchronize()
    streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]
    results = []
    for stream in streams:
        with stream:
            for ao, weights in inputs:
                results.append((numint._tau_dot(ao, ao, weights), numint._tau_dot_full(ao, ao, weights)))
    for stream in streams:
        stream.synchronize()
    for actual, expected in results:
        cp.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_syrk_restores_blas_settings_on_failure(monkeypatch):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    from gpu4pyscf.dft import _tau_syrk

    handle = cp.cuda.device.get_cublas_handle()
    previous_math, previous_pointer = cublas.getMathMode(handle), cublas.getPointerMode(handle)

    def fail(*args):
        raise ArithmeticError('BLAS failure')

    monkeypatch.setattr(cublas, 'dsyrk', fail)
    with pytest.raises(ArithmeticError, match='BLAS failure'):
        _tau_syrk._symmetric_tau(cp.ones((4, 19, 17)), cp.ones(17))
    assert cublas.getMathMode(handle) == previous_math
    assert cublas.getPointerMode(handle) == previous_pointer


def test_tau_reuses_bounded_workspace_without_allocating_stacked_fields(monkeypatch):
    import cupy as cp
    import numpy as np
    from cupy_backends.cuda.libs import cublas

    from gpu4pyscf.dft import _tau_syrk, numint

    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 0)
    rng = np.random.default_rng(1305)
    ao, weights = cp.asarray(rng.normal(size=(4, 19, 83))), cp.asarray(rng.normal(size=83))
    expected = numint._tau_dot_full(ao, ao, weights)
    workspace, out = cp.empty(19 * 3 * 83), cp.empty((19, 19))
    _tau_syrk._symmetric_tau(ao, weights, buf=workspace, out=out)
    original_syrk, pointers = cublas.dsyrk, []

    def syrk(*args):
        pointers.append(args[6])
        return original_syrk(*args)

    monkeypatch.setattr(cublas, 'dsyrk', syrk)
    actual = _tau_syrk._symmetric_tau(ao, weights, buf=workspace, out=out)
    assert pointers == [workspace.data.ptr, workspace.data.ptr]
    cp.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_finite_guard_reduces_without_materializing_large_masks(monkeypatch):
    import cupy as cp

    from gpu4pyscf.dft import _tau_syrk

    ao, weights = cp.ones((4, 513, 17)), cp.ones(17)

    def materialized_mask(*args, **kwargs):
        raise AssertionError('AO finite checks must not allocate a full boolean mask')

    monkeypatch.setattr(cp, 'isfinite', materialized_mask)
    assert _tau_syrk._eligible(ao, ao, weights, None)
    ao[1, 0, 0] = cp.inf
    assert not _tau_syrk._eligible(ao, ao, weights, None)


def test_private_tau_keeps_float32_products_and_restores_dispatch(monkeypatch):
    import cupy as cp
    import numpy as np
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import _tau_syrk, mixed_precision, numint

    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 0)
    mf = dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0), xc='wb97m-v')
    original = mf._numint.nr_rks.__func__
    rng = np.random.default_rng(1304)
    ao, weights = cp.asarray(rng.normal(size=(4, 19, 83))), cp.asarray(rng.normal(size=83))
    with mixed_precision._precision_context(mf) as (grid, _):
        namespace = mf._numint.nr_rks.__func__.__globals__
        expected = namespace['_tau_dot_full'](ao, ao, weights)
        actual = namespace['_tau_dot'](ao, ao, weights)
        cp.testing.assert_array_equal(actual, expected)
        assert grid.float32_calls == 6 and grid.float64_calls == 0
        grid.use_float32 = False
        actual = namespace['_tau_dot'](ao, ao, weights)
        cp.testing.assert_allclose(actual, numint._tau_dot_full(ao, ao, weights), atol=2e-12, rtol=2e-12)
        assert grid.float64_calls == 0
    assert mf._numint.nr_rks.__func__ is original


@pytest.mark.parametrize('case', ['water', 'water-cation', *ASE_ORGANICS.values()])
def test_symmetric_tau_complete_scf_forces_and_energy_differences(case, monkeypatch):
    import numpy as np
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import _tau_syrk, mixed_precision
    from gpu4pyscf.scf import hf

    monkeypatch.setattr(hf, 'overlap_zero_eigenvalue_threshold', 1e-10)
    multiplicity = 2 if case == 'water-cation' else 1
    atoms = molecule('H2O' if case.startswith('water') else case)
    mol = gto.M(
        atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
        basis='def2-tzvpd',
        charge=multiplicity - 1,
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    # Force agreement requires tighter orbital convergence than production's
    # energy acceptance. Keep the stricter setting confined to correctness tests.
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-11, 1e-7, None
    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 10**6)
    reference_energy = mixed_precision.run_verified_scf(mf)
    gradient = mf.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    reference_force = np.asarray(gradient.kernel())
    monkeypatch.setattr(_tau_syrk, '_MIN_ORBITALS', 0)
    mf.reset(mol)
    actual_energy = mixed_precision.run_verified_scf(mf)
    assert mf.mixed_precision_info['accepted']
    actual_force = np.asarray(gradient.kernel())
    assert abs(actual_energy - reference_energy) < 1e-9
    np.testing.assert_allclose(actual_force, reference_force, atol=1e-7, rtol=1e-7)
    coords, step = mol.atom_coords(), 2e-4
    atom_index = atoms.get_chemical_symbols().index('H')
    energies = []
    for sign in (-1, 1):
        displaced = coords.copy()
        displaced[atom_index, 0] += sign * step
        mf.reset(mol.set_geom_(displaced, unit='Bohr', inplace=False))
        energies.append(mixed_precision.run_verified_scf(mf))
    assert abs(actual_force[atom_index, 0] - (energies[1] - energies[0]) / (2 * step)) < 2e-5
