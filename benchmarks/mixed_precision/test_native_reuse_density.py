"""Opt-in exact verification reuse and occupied-factor force regression checks."""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('kind', ['same', 'density', 'coords', 'weights', 'tags', 'molecule', 'cutoff', 'layout'])
def test_native_grid_reuse_requires_identical_scientific_inputs(kind):
    import cupy as cp

    from gpu4pyscf.dft import numint
    from gpu4pyscf.dft._native_grid_cache import _native_grid_reuse
    from gpu4pyscf.lib.cupy_helper import tag_array

    ni = numint.NumInt()
    ni.cutoff = 1e-10
    calls = []

    def integration(self, mol, grids, xc, density):
        calls.append(1)
        return 2., -.25, density.copy()

    ni.nr_rks = MethodType(integration, ni)
    original = ni.nr_rks
    mol = SimpleNamespace(_atm=np.zeros((1, 6)), _bas=np.zeros((1, 8)), _env=np.zeros(20),
                          charge=0, spin=0, cart=False)
    grids = SimpleNamespace(coords=cp.zeros((7, 3)), weights=cp.ones(7))
    density = tag_array(cp.eye(4), mo_coeff=cp.eye(4), mo_occ=cp.ones(4))
    with _native_grid_reuse(ni) as cache:
        cache.enabled = True
        ni.nr_rks(mol, grids, 'pbe', density)[2].fill(7)
        if kind == 'density':
            density[0, 0] += .1
        elif kind == 'coords':
            grids.coords[0, 0] += .1
        elif kind == 'weights':
            grids.weights[0] += .1
        elif kind == 'tags':
            density.mo_coeff[0, 0] += .1
        elif kind == 'molecule':
            mol._env[0] += .1
        elif kind == 'cutoff':
            ni.cutoff *= 2
        elif kind == 'layout':
            density = tag_array(cp.asfortranarray(density), mo_coeff=density.mo_coeff, mo_occ=density.mo_occ)
        with cache.verification():
            first = ni.nr_rks(mol, grids, 'pbe', density)
            cp.testing.assert_array_equal(first[2], density)
            first[2].fill(11)
            second = ni.nr_rks(mol, grids, 'pbe', density)
            cp.testing.assert_array_equal(second[2], density)
        assert len(calls) == (1 if kind == 'same' else 2)
        assert cache.hits == (2 if kind == 'same' else 1)
    assert ni.nr_rks is original and cache._entry is None and not cache._buffers


def test_native_grid_reuse_keeps_scf_calls_and_capacity_fallback(monkeypatch):
    import cupy as cp

    from gpu4pyscf.dft import _native_grid_cache, numint

    ni = numint.NumInt()
    calls = []

    def integration(self, mol, grids, xc, density):
        calls.append(1)
        return 2., -.25, density.copy()

    ni.nr_rks = MethodType(integration, ni)
    mol = SimpleNamespace(_atm=np.zeros((1, 6)), _bas=np.zeros((1, 8)), _env=np.zeros(20),
                          charge=0, spin=0, cart=False)
    grids = SimpleNamespace(coords=cp.zeros((7, 3)), weights=cp.ones(7))
    density = cp.eye(4)
    with _native_grid_cache._native_grid_reuse(ni) as cache:
        cache.enabled = True
        for _ in range(2):
            ni.nr_rks(mol, grids, 'pbe', density)
        assert len(calls) == 2 and cache.hits == 0
        monkeypatch.setattr(_native_grid_cache, '_MAX_BYTES', 1)
        cache.clear()
        with cache.verification():
            for _ in range(2):
                ni.nr_rks(mol, grids, 'pbe', density)
        assert len(calls) == 4 and cache._entry is None


@pytest.mark.parametrize('xctype', ['LDA', 'GGA', 'MGGA'])
def test_factored_density_products_and_derivatives_match_native(xctype):
    import cupy as cp

    from gpu4pyscf.grad import rks
    from gpu4pyscf.grad._density_products import _DensityProducts
    from gpu4pyscf.scf import hf

    rng = np.random.default_rng(818)
    coeff = cp.asarray(rng.normal(scale=.1, size=(31, 31)))
    occ = cp.asarray([2., 2., 1.] + [0.] * 28)
    density = hf.make_rdm1(coeff, occ)
    plan = _DensityProducts.create(density)
    assert plan is not None
    idx = cp.asarray([*range(23)], dtype=cp.int32)
    block = plan.block(idx)
    assert block is not None
    dm = density[idx[:, None], idx]
    ao = cp.asarray(rng.normal(size=(10, 23, 257)))
    expected_rho, expected_products = rks._rho_and_dm_ao(ao, dm, xctype)
    rho, products = rks._rho_and_dm_ao(ao, dm, xctype, factors=block)
    cp.testing.assert_allclose(rho, expected_rho, atol=2e-11, rtol=2e-12)
    cp.testing.assert_allclose(products, expected_products, atol=2e-12, rtol=2e-12)
    wv = cp.asarray(rng.normal(size=({'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype], 257)))
    expected = rks._contract_grad_response(ao, wv, dm, expected_products)
    actual = rks._contract_grad_response(ao, wv, dm, products, factors=block)
    cp.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-12)
    # The NLC second pass supplies no retained density products.
    cp.testing.assert_allclose(rks._contract_grad_response(ao, wv, dm, factors=block),
                               rks._contract_grad_response(ao, wv, dm), atol=2e-10, rtol=2e-12)


@pytest.mark.parametrize('kind', ['untagged', 'density', 'coefficients', 'occupancy', 'complex', 'capacity', 'device'])
def test_invalid_or_stale_factors_keep_native_density(kind, monkeypatch):
    import cupy as cp

    from gpu4pyscf.grad import _density_products
    from gpu4pyscf.scf import hf

    coeff = cp.eye(31)
    occ = cp.asarray([2.] * 3 + [0.] * 28)
    density = hf.make_rdm1(coeff, occ)
    if kind == 'untagged':
        density = density.view(cp.ndarray)
        assert not hasattr(density, 'mo_coeff')
    elif kind == 'density':
        density[0, 0] += .1
    elif kind == 'coefficients':
        density.mo_coeff[0, 0] += .1
    elif kind == 'occupancy':
        density.mo_occ[0] = -1
    elif kind == 'complex':
        density = density.astype(cp.complex128)
    elif kind == 'capacity':
        monkeypatch.setattr(_density_products, '_MAX_WORKSPACE_BYTES', 1)
    else:
        monkeypatch.setattr(_density_products, '_supported_device', lambda device: False)
    assert _density_products._DensityProducts.create(density) is None


def test_factor_workspace_stream_and_dense_block_fallback():
    import cupy as cp

    from gpu4pyscf.grad._density_products import _DensityProducts
    from gpu4pyscf.scf import hf

    density = hf.make_rdm1(cp.eye(31), cp.asarray([2.] * 3 + [0.] * 28))
    plan = _DensityProducts.create(density)
    assert plan.block(cp.arange(8)) is None
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        assert plan.block(cp.arange(23)) is None


@pytest.mark.parametrize('xc', ['b3lyp', 'wb97m-v'])
@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
def test_complete_factored_forces_match_native_density_products(case, xc, monkeypatch):
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.grad import _density_products

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='6-31g*', verbose=0)
    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 2, 1
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-11, 1e-7, None
    mf.kernel()
    assert mf.converged
    gradient = mf.Gradients()
    gradient.grid_response = gradient.auxbasis_response = True
    with monkeypatch.context() as native:
        native.setattr(_density_products._DensityProducts, 'create', classmethod(lambda cls, density: None))
        expected = np.asarray(gradient.kernel())
    original = _density_products._DensityBlock.dot
    accelerated = []

    def counted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result is not None:
            accelerated.append(1)
        return result

    monkeypatch.setattr(_density_products._DensityBlock, 'dot', counted)
    actual = np.asarray(gradient.kernel())
    assert accelerated
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    if case == 'CH3CONH2' and xc == 'wb97m-v':
        coords = mol.atom_coords()
        atom, direction = np.unravel_index(np.argmax(np.abs(actual)), actual.shape)
        energies = []
        step = 2e-4
        for sign in (-1, 1):
            moved = coords.copy()
            moved[atom, direction] += sign * step
            mf.reset(mol.set_geom_(moved, unit='Bohr', inplace=False))
            energies.append(mf.kernel())
            assert mf.converged
        assert abs(actual[atom, direction] - (energies[1] - energies[0]) / (2 * step)) < 1e-5
