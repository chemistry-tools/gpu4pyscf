"""Opt-in VV10 field reuse, exact-state invalidation and native fallback checks."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


def _inputs():
    import cupy as cp
    from pyscf import gto

    from gpu4pyscf.dft import gen_grid, numint
    from gpu4pyscf.dft._nlc_force_cache import _NLCForceReuse
    from gpu4pyscf.lib.cupy_helper import tag_array

    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    ni = numint.NumInt()
    grids = gen_grid.Grids(mol)
    rng = np.random.default_rng(512)
    grids.coords = cp.asarray(rng.normal(size=(13, 3)))
    grids.weights = cp.asarray([0., 1., -.3, .1, .4, -.2, .6, .8, 1e-18, .2, .7, .3, .5])
    grids.atm_idx = cp.asarray([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=cp.int32)
    density = tag_array(cp.eye(2), mo_coeff=cp.eye(2), mo_occ=cp.ones(2))
    rho = cp.asarray(rng.normal(scale=.1, size=(4, 13)))
    rho[0] = cp.asarray(rng.uniform(.1, 2, size=13))
    rho[0, 7] = 1e-20
    force = grids.copy()
    permutation = cp.asarray(rng.permutation(13))
    force.coords = cp.vstack((grids.coords[permutation], cp.zeros((2, 3))))
    force.weights = cp.hstack((grids.weights[permutation], cp.zeros(2)))
    force.atm_idx = cp.hstack((grids.atm_idx[permutation], cp.asarray([-1, -1], dtype=cp.int32)))
    pars = ni.nlc_coeff('wb97m-v')[0][0]
    return ni, mol, grids, force, density, rho, pars, _NLCForceReuse(ni)


def test_fields_survive_exact_permutation_and_zero_padding_once():
    import cupy as cp

    from gpu4pyscf.dft import numint
    from gpu4pyscf.dft._vv10_grad import _pair_response

    ni, mol, grids, force, density, rho, pars, cache = _inputs()
    expected = numint._vv10nlc(rho, grids.coords, grids.weights, pars)
    with cache.verification():
        actual = cache.potential(mol, grids, 'wb97m-v', density, rho, pars)
    for a, b in zip(actual, expected):
        cp.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
    assert 0 < cache.statistics['retained_bytes'] <= 128 * 1024**2
    full_rho, fields = cache.take(ni, mol, grids, force, 'wb97m-v', density)
    # Verify restored density values independently by coordinate identity.
    for i in range(13):
        if abs(float(grids.weights[i])) <= 1e-14:
            continue
        j = int(cp.where(cp.all(force.coords == grids.coords[i], axis=1))[0][0])
        cp.testing.assert_array_equal(full_rho[:, j], rho[:, i])
    cp.testing.assert_array_equal(full_rho[:, -2:], 0)
    # Pair fields agree with an independent traversal of the same sorted domain.
    active = (full_rho[0] >= numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD) & (cp.abs(force.weights) > 1e-14)
    coords, weights, r = force.coords[active], force.weights[active], full_rho[0, active]
    atoms = force.atm_idx[active]
    order = cp.argsort(atoms)
    gamma = cp.sum(full_rho[1:4, active]**2, axis=0)
    omega = cp.sqrt(pars[1] * (gamma / r**2)**2 + (4*np.pi/3) * r)
    kappa = pars[0]*1.5*np.pi*(9*np.pi)**(-1/6) * r**(1/6)
    reference = _pair_response(cp.ascontiguousarray(coords[order]), (r*weights)[order],
                               omega[order], kappa[order], atoms[order], mol.natm)
    for a, b in zip(fields[:3], reference[:3]):
        cp.testing.assert_allclose(a[order], b, atol=2e-12, rtol=2e-12)
    cp.testing.assert_allclose(fields[3], reference[3], atol=2e-12, rtol=2e-12)
    assert cache.statistics['hits'] == 1 and cache.statistics['retained_bytes'] == 0
    assert cache.take(ni, mol, grids, force, 'wb97m-v', density) is None


@pytest.mark.parametrize('change', ['density', 'tags', 'molecule', 'coords', 'weights',
                                    'force_coords', 'force_weights', 'atoms', 'cutoff',
                                    'partition', 'stream', 'xc'])
def test_changed_inputs_discard_fields(change):
    import cupy as cp

    ni, mol, grids, force, density, rho, pars, cache = _inputs()
    with cache.verification():
        assert cache.potential(mol, grids, 'wb97m-v', density, rho, pars) is not None
    if change == 'density':
        density[0, 0] += .01
    elif change == 'tags':
        density.mo_occ[0] += .01
    elif change == 'molecule':
        mol._env[-1] += .01
    elif change == 'coords':
        grids.coords[0, 0] += .01
    elif change == 'weights':
        grids.weights[0] += .01
    elif change == 'force_coords':
        j = int(cp.where(cp.abs(force.weights) > 1e-14)[0][0])
        force.coords[j, 0] += .01
    elif change == 'force_weights':
        force.weights[0] += .01
    elif change == 'atoms':
        force.atm_idx[:] = 0
    elif change == 'cutoff':
        ni.cutoff = 1e-8
    elif change == 'partition':
        grids.becke_scheme = lambda *args: None
    xc = 'b97m-v' if change == 'xc' else 'wb97m-v'
    if change == 'stream':
        cp.cuda.get_current_stream().synchronize()
        with cp.cuda.Stream(non_blocking=True):
            result = cache.take(ni, mol, grids, force, xc, density)
    else:
        result = cache.take(ni, mol, grids, force, xc, density)
    assert result is None and cache.statistics['misses'] == 1
    assert cache.snapshot._entry is None and not cache.snapshot._buffers


def test_capacity_and_custom_density_fall_back_before_pair_work(monkeypatch):
    from gpu4pyscf.dft import _nlc_force_cache

    _ni, mol, grids, _, density, rho, pars, cache = _inputs()
    monkeypatch.setattr(_nlc_force_cache, '_MAX_BYTES', 1)
    with cache.verification():
        assert cache.potential(mol, grids, 'wb97m-v', density, rho, pars) is None
    assert cache.statistics['builds'] == 0 and cache.snapshot._entry is None


@pytest.mark.parametrize('stage', ['potential', 'take'])
def test_optional_allocation_failure_falls_back(monkeypatch, stage):
    import cupy as cp

    from gpu4pyscf.dft import _nlc_force_cache

    ni, mol, grids, force, density, rho, pars, cache = _inputs()
    if stage == 'take':
        with cache.verification():
            assert cache.potential(mol, grids, 'wb97m-v', density, rho, pars) is not None

    def fail(*args):
        raise cp.cuda.memory.OutOfMemoryError(1024, 2048)

    monkeypatch.setattr(_nlc_force_cache, '_canonical_indices', fail)
    if stage == 'take':
        assert cache.take(ni, mol, grids, force, 'wb97m-v', density) is None
    else:
        with cache.verification():
            assert cache.potential(mol, grids, 'wb97m-v', density, rho, pars) is None
    assert cache.snapshot._entry is None and cache.statistics['retained_bytes'] == 0


def test_full_nlc_force_matches_independent_energy_difference():
    import cupy as cp
    from _organic_cases import ORGANIC_CASES
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft._nlc_force_cache import _NLCForceReuse
    from gpu4pyscf.dft.gen_grid import stratmann
    from gpu4pyscf.grad import rks

    geometry, charge, multiplicity = ORGANIC_CASES['acetyl-fluoride']
    mol = gto.M(atom=geometry, basis='def2-tzvpd', charge=charge,
                spin=multiplicity-1, verbose=0)

    def method(molecule):
        mf = dft.RKS(molecule, xc='wb97m-v')
        mf.nlcgrids.level = 3
        mf.nlcgrids.becke_scheme, mf.nlcgrids.radii_adjust = stratmann, None
        mf.nlcgrids.build()
        return mf

    mf = method(mol)
    density = cp.asarray(mf.get_init_guess(mol, 'minao'))
    cache = mf._numint._nlc_force_reuse = _NLCForceReuse(mf._numint)
    with cache.verification():
        mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, density)
    derivative, _ = rks.get_nlc_exc_full_response(mf._numint, mol, mf.nlcgrids, mf.xc, density)
    assert cache.statistics['hits'] == 1
    atom, axis = np.unravel_index(np.argmax(np.abs(derivative)), derivative.shape)
    assert abs(derivative[atom, axis]) > 1e-6
    step = 2e-4
    energies = []
    for direction in (-1, 1):
        coords = mol.atom_coords().copy()
        coords[atom, axis] += direction*step
        displaced = mol.copy().set_geom_(coords, unit='Bohr')
        moved = method(displaced)
        energies.append(moved._numint.nr_nlc_vxc(displaced, moved.nlcgrids, moved.xc, density)[1])
    numerical = (energies[1]-energies[0])/(2*step)
    np.testing.assert_allclose(derivative[atom, axis], numerical, atol=5e-7, rtol=0)


def test_ase_full_response_enables_reuse_and_keeps_energy_cache():
    from ase import Atoms
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.tools.ase_interface import PySCF

    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, .7, .6], [0, -.7, .6]])
    mol = gto.M(atom=list(zip(atoms.symbols, atoms.positions)), basis='def2-tzvpd', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit()
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-8, 1e-4, None
    atoms.calc = PySCF(method=mf, precision='vv10', grid_response=True, auxbasis_response=True)
    assert np.isfinite(atoms.get_forces()).all()
    info = atoms.calc.precision_info
    assert info['accepted'] and info['nlc_force_reuse']['hits'] == 1
    assert info['nlc_force_reuse']['retained_bytes'] == 0
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.calc.precision_info is info
