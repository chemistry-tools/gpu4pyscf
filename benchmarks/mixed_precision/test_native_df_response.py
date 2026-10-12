"""Check density-fitting derivative scheduling against native forces and energy differences."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest
from native_force_benchmark import _reference_df

pytestmark = pytest.mark.skipif(
    os.getenv('DFT_GPU_TEST') != '1' or not os.getenv('DFT_REFERENCE_DF_GRADIENT'),
    reason='opt-in CUDA test with matching native DF baseline source',
)


@pytest.mark.parametrize(
    'hermi,j_factor,omega,aux_response', [(1, 1, 0, True), (1, 0, 0.3, True), (0, 1, 0, True), (0, 0, -0.2, False)]
)
def test_density_fitting_derivative_operators(hermi, j_factor, omega, aux_response):
    import cupy as cp
    import numpy as np
    from pyscf import gto

    from gpu4pyscf import dft

    mol = gto.M(atom='O 0 0 0; H .7586 0 .5043; H -.7586 0 .5043', basis='def2-tzvpd', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.with_df.build()
    rng = np.random.default_rng(172)
    left, right = (rng.normal(size=(mol.nao, 5)) for _ in range(2))
    dm = left @ (left if hermi == 1 else right).T / mol.nao
    dm = cp.asarray(dm)
    gradient = mf.nuc_grad_method()
    gradient.auxbasis_response = aux_response
    results = []
    for scope in (_reference_df(Path(os.environ['DFT_REFERENCE_DF_GRADIENT'])), contextlib.nullcontext()):
        mf.with_df.reset()
        cp.get_default_memory_pool().free_all_blocks()
        with scope:
            results.append(gradient.jk_energy_per_atom(dm, j_factor=j_factor, k_factor=0.7, omega=omega, hermi=hermi))
    np.testing.assert_allclose(results[1], results[0], atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('xc', ['wb97m-v', 'pbe0'])
def test_complete_hybrid_forces_and_energy_differences(cart, xc, monkeypatch):
    import cupy as cp
    import numpy as np
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.scf import hf

    monkeypatch.setattr(hf, 'overlap_zero_eigenvalue_threshold', 1e-10)
    mol = gto.M(atom='O 0 0 0; H .7586 0 .5043; H -.7586 0 .5043', basis='def2-tzvpd', cart=cart, verbose=0)
    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-11, 1e-5, 200
    mf.chkfile = None
    mf.kernel()
    assert mf.converged and mf.mo_coeff.shape[-1] == mol.nao
    dm = mf.make_rdm1()
    results = []
    for scope in (_reference_df(Path(os.environ['DFT_REFERENCE_DF_GRADIENT'])), contextlib.nullcontext()):
        mf.with_df.reset()
        cp.get_default_memory_pool().free_all_blocks()
        mf.get_veff(mol, dm)
        gradient = mf.nuc_grad_method()
        gradient.grid_response = gradient.auxbasis_response = True
        with scope:
            results.append(np.asarray(gradient.kernel()))
    np.testing.assert_allclose(results[1], results[0], atol=2e-10, rtol=2e-10)
    coordinates, step, energies = mol.atom_coords(), 2e-4, []
    for sign in (-1, 1):
        displaced = coordinates.copy()
        displaced[1, 0] += sign * step
        mf.reset(mol.set_geom_(displaced, unit='Bohr', inplace=False))
        energies.append(mf.kernel())
        assert mf.converged
    assert abs(results[1][1, 0] - (energies[1] - energies[0]) / (2 * step)) < 1e-5
