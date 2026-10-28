"""Check roundoff-guarded initial-density factors and native grid fallbacks."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


def _density():
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(2917)
    factors = cp.asarray(rng.normal(size=(160, 17)))
    return factors.dot(factors.T)


def test_recovered_factors_preserve_density_and_stream():
    import cupy as cp

    from gpu4pyscf.dft._initial_density import _density_orbitals

    density = _density()
    original = density.copy()
    for stream in (cp.cuda.get_current_stream(), cp.cuda.Stream(non_blocking=True)):
        cp.cuda.get_current_stream().synchronize()
        with stream:
            factors, occ = _density_orbitals(density)
            reconstructed = (factors * occ).dot(factors.T)
        stream.synchronize()
        assert factors.shape == (160, 17)
        assert factors.dtype == cp.float64
        cp.testing.assert_allclose(reconstructed, original, atol=2e-12, rtol=2e-13)
        cp.testing.assert_array_equal(density, original)


@pytest.mark.parametrize('kind', ['float32', 'complex', 'cpu', 'shape', 'small', 'capacity', 'device',
                                 'nan', 'infinity', 'zero', 'signed', 'full_rank', 'asymmetric',
                                 'discarded_component', 'tagged'])
def test_unsupported_density_keeps_native_matrix(kind, monkeypatch):
    import cupy as cp

    from gpu4pyscf.dft import _initial_density
    from gpu4pyscf.lib.cupy_helper import tag_array

    density = _density()
    if kind == 'float32':
        density = density.astype(cp.float32)
    elif kind == 'complex':
        density = density.astype(cp.complex128)
    elif kind == 'cpu':
        density = cp.asnumpy(density)
    elif kind == 'shape':
        density = density[None]
    elif kind == 'small':
        density = density[:20, :20]
    elif kind == 'capacity':
        monkeypatch.setattr(_initial_density, '_MAX_WORKSPACE_BYTES', 1)
    elif kind == 'device':
        monkeypatch.setattr(_initial_density, '_supported_device', lambda device: False)
    elif kind == 'nan':
        density[0, 0] = cp.nan
    elif kind == 'infinity':
        density[0, 0] = cp.inf
    elif kind == 'zero':
        density.fill(0)
    elif kind == 'signed':
        density = -density
    elif kind == 'full_rank':
        density += cp.eye(len(density))
    elif kind == 'asymmetric':
        density[0, 1] += 1e-4
    elif kind == 'discarded_component':
        # The existing DF cutoff discards this component, but grids must not.
        density += cp.eye(len(density)) * 1e-9
    elif kind == 'tagged':
        density = tag_array(density, mo_coeff=cp.zeros((160, 17)), mo_occ=cp.ones(17))
    before = density.copy()
    assert _initial_density._density_orbitals(density) is None
    cp.testing.assert_array_equal(cp.asarray(density), cp.asarray(before))


@pytest.mark.parametrize('failure', ['solver', 'allocation', 'nonfinite_factors'])
def test_factorization_failure_keeps_density(failure, monkeypatch):
    import cupy as cp

    from gpu4pyscf.df import df_jk
    from gpu4pyscf.dft._initial_density import _density_orbitals

    density = _density()
    before = density.copy()

    def fail(*args, **kwargs):
        if failure == 'solver':
            raise cp.linalg.LinAlgError('test solver failure')
        if failure == 'allocation':
            raise cp.cuda.memory.OutOfMemoryError(1024, 0)
        return cp.full((160, 17), cp.inf), cp.zeros((160, 17))

    monkeypatch.setattr(df_jk, 'decompose_rdm1_svd', fail)
    assert _density_orbitals(density) is None
    cp.testing.assert_array_equal(density, before)


@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
@pytest.mark.parametrize('mixture', [False, True])
def test_complete_organic_integration_matches_dense(case, mixture, monkeypatch):
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import _initial_density, numint

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='def2-tzvpd', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.grids.level, mf.small_rho_cutoff = 3, 0
    _, coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
    occ = cp.zeros(coeff.shape[1])
    occ[:mol.nelectron//2] = 2
    density = mf.make_rdm1(coeff, occ).view(cp.ndarray)
    if mixture:
        density = .999 * density + .001 * cp.asarray(mf.get_init_guess())
    before = density.copy()
    mf.grids.build()
    with monkeypatch.context() as patch:
        patch.setattr(numint, '_density_orbitals', lambda density: None)
        expected = mf._numint.nr_rks(mol, mf.grids, mf.xc, density)
    hits = []

    def factorize(density):
        result = _initial_density._density_orbitals(density)
        hits.append(result is not None)
        return result

    monkeypatch.setattr(numint, '_density_orbitals', factorize)
    actual = mf._numint.nr_rks(mol, mf.grids, mf.xc, density)
    if not mixture:
        assert hits == [True]
    else:
        # A MINAO mixture can have meaningful small eigenvalues: the strict
        # reconstruction guard must be allowed to keep its dense evaluation.
        assert len(hits) == 1
    assert abs(actual[0] - expected[0]) < 2e-10
    assert abs(actual[1] - expected[1]) < 2e-10
    cp.testing.assert_allclose(actual[2], expected[2], atol=2e-11, rtol=2e-11)
    cp.testing.assert_array_equal(density, before)


@pytest.mark.parametrize('xc,hermi,tagged', [('lda,vwn', 1, False), ('pbe', 1, False),
                                         ('m06-l', 0, False), ('m06-l', 1, True)])
def test_other_integration_paths_do_not_recover_factors(xc, hermi, tagged, monkeypatch):
    import cupy as cp
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint

    mol = gto.M(atom='O 0 0 0; H .75 0 .5; H -.75 0 .5', basis='6-31g*', verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = 1
    _, coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
    occ = cp.zeros(coeff.shape[1])
    occ[:mol.nelectron//2] = 2
    density = mf.make_rdm1(coeff, occ)
    if not tagged:
        density = density.view(cp.ndarray)
    mf.grids.build()

    def forbidden(*args):
        raise AssertionError('This integration must retain its existing density representation')

    monkeypatch.setattr(numint, '_density_orbitals', forbidden)
    mf._numint.nr_rks(mol, mf.grids, mf.xc, density, hermi=hermi)
