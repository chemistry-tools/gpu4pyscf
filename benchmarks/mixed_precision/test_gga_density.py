"""Check adaptive float64 GGA density evaluation against occupied-orbital products."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
@pytest.mark.parametrize('fractional', [False, True])
def test_tagged_gga_density_matches_orbital_path(case, fractional, monkeypatch):
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint, rks
    from gpu4pyscf.lib.cupy_helper import tag_array

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='6-31g*', cart=True, verbose=0)
    mf = dft.RKS(mol, xc='b3lyp')
    mf.grids.level, mf.small_rho_cutoff = 3, 0
    _, coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
    occ = cp.zeros(coeff.shape[1])
    occ[:mol.nelectron//2] = 2
    if fractional:
        occ[mol.nelectron//2-1:mol.nelectron//2+1] = cp.asarray([1.7, .3])
    # Native tagged-orbital integration ignores the supplied density values.
    density = tag_array(cp.zeros((mol.nao, mol.nao)), mo_coeff=coeff, mo_occ=occ)
    rks.initialize_grids(mf, mol, density)
    args = mol, mf.grids, mf.xc, density
    monkeypatch.setattr(numint, '_rho_gga_use_dm', lambda nao, nocc: False)
    expected = mf._numint.nr_rks(*args, hermi=1)
    calls = []
    original = numint.eval_rho

    def density_rho(*args, **kwargs):
        calls.append(kwargs['xctype'])
        return original(*args, **kwargs)

    monkeypatch.setattr(numint, '_rho_gga_use_dm', lambda nao, nocc: True)
    monkeypatch.setattr(numint, 'eval_rho', density_rho)
    actual = mf._numint.nr_rks(*args, hermi=1)
    assert calls and set(calls) == {'GGA'}
    assert expected[0] > mol.nelectron*.99
    assert abs(actual[0]-expected[0]) < 2e-11
    assert abs(actual[1]-expected[1]) < 2e-11
    cp.testing.assert_allclose(actual[2], expected[2], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('xc', ['lda,vwn', 'm06-l'])
def test_non_gga_retains_orbital_density(xc, monkeypatch):
    import cupy as cp
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import numint, rks

    mol = gto.M(atom='O 0 0 0; H .75 0 .5; H -.75 0 .5', basis='6-31g*', verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level, mf.small_rho_cutoff = 1, 0
    _, coeff = mf.eig(mf.get_hcore(), mf.get_ovlp())
    occ = cp.zeros(coeff.shape[1])
    occ[:mol.nelectron//2] = 2
    density = mf.make_rdm1(coeff, occ)
    rks.initialize_grids(mf, mol, density)
    calls = []

    def unexpected_density(*args):
        calls.append(args)
        return True

    monkeypatch.setattr(numint, '_rho_gga_use_dm', unexpected_density)
    actual = mf._numint.nr_rks(mol, mf.grids, mf.xc, density)
    assert cp.isfinite(actual[2]).all()
    assert not calls
