"""Check exact force contractions and native gradient agreement."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from force_experiment import density_response_patch

pytestmark = pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
ROOT = Path(__file__).resolve().parents[2]


def test_atom_reduction_preserves_atoms_without_basis_functions():
    from types import SimpleNamespace

    import cupy as cp
    import numpy as np
    from pyscf import gto

    basis = np.zeros((2, gto.BAS_SLOTS), dtype=int)
    basis[:, gto.ATOM_OF] = [0, 2]
    mol = SimpleNamespace(natm=4, _bas=basis, ao_loc=np.asarray([0, 2, 3]))
    partial = cp.asarray([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    with density_response_patch(ROOT) as contract:
        actual = contract.__globals__['_reduce_to_atom'](mol, partial)
    np.testing.assert_array_equal(actual, [[5., 7., 9.], [0., 0., 0.], [7., 8., 9.], [0., 0., 0.]])


@pytest.mark.parametrize('components', [1, 4, 5])
@pytest.mark.parametrize('ngrids', [17, 257])
def test_contract_against_native_derivative_matrix(components, ngrids):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks

    rng = np.random.default_rng(92)
    ao = cp.asarray(rng.normal(size=(10, 23, ngrids)))
    wv = cp.asarray(rng.normal(size=(components, ngrids)))
    factors = rng.normal(size=(23, 7))
    dm = cp.asarray(factors @ factors.T)
    if components == 1:
        matrix = rks._d1_dot_(ao[1:4], numint._scale_ao(ao[0], wv[0]).T)
    else:
        matrix = rks._gga_grad_sum_(ao, wv[:4])
        if components == 5:
            rks._tau_grad_dot_(ao, wv[4], accumulate=True, out=matrix)
    reference = cp.einsum('dij,ji->id', matrix, dm)
    original = rks.get_exc_full_response
    with density_response_patch(ROOT) as contract:
        actual = contract(ao, wv, dm)
    assert rks.get_exc_full_response is original
    cp.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-12)


@pytest.mark.parametrize('charge', [0, 1])
def test_full_gradient_and_finite_difference(charge, monkeypatch):
    import cupy as cp
    import numpy as np
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import mixed_precision
    from gpu4pyscf.scf import hf

    monkeypatch.setattr(hf, 'overlap_zero_eigenvalue_threshold', 1e-10)
    mol = gto.M(atom='O 0 0 0; H .7586 0 .5043; H -.7586 0 .5043',
                basis='def2-tzvpd', charge=charge, spin=charge, verbose=0)
    mf = (dft.RKS if charge == 0 else dft.UKS)(mol, xc='wb97m-v').density_fit(
        auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-11, 1e-5, 100
    mf.chkfile = None
    mixed_precision.run_verified_scf(mf)
    gradient = mf.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    reference = np.asarray(gradient.kernel())
    with density_response_patch(ROOT):
        actual = np.asarray(gradient.kernel())
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
    step = 2e-4  # Bohr; numerical derivatives use the same full grids and method.
    coords = mol.atom_coords()
    energies = []
    for sign in (-1, 1):
        displaced = coords.copy()
        displaced[1, 0] += sign * step
        mf.reset(mol.set_geom_(displaced, unit='Bohr', inplace=False))
        energies.append(mixed_precision.run_verified_scf(mf))
    assert abs(actual[1, 0] - (energies[1] - energies[0]) / (2 * step)) < 1e-5
    cp.cuda.get_current_stream().synchronize()
