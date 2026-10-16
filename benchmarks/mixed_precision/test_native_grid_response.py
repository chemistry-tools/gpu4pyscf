"""Check native streaming XC forces and reused float64 density products."""

from __future__ import annotations

import ast
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('xctype', ['LDA', 'GGA', 'MGGA'])
@pytest.mark.parametrize('ngrids', [17, 257])
def test_density_and_reused_contractions_match_separate_native_products(xctype, ngrids):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks

    rng = np.random.default_rng(116)
    ao = cp.asarray(rng.normal(size=(10, 23, ngrids)))
    factors = rng.normal(size=(23, 7))
    dm = cp.asarray(factors @ factors.T)
    components = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    wv = cp.asarray(rng.normal(size=(components, ngrids)))
    density_ao = ao[0] if xctype == 'LDA' else ao[:4]
    reference_density = numint.eval_rho(None, density_ao, dm, xctype=xctype, hermi=1)
    density, products = rks._rho_and_dm_ao(ao, dm, xctype)
    assert products.shape[0] == (4 if xctype == 'MGGA' else 1)
    cp.testing.assert_array_equal(density, reference_density)
    reference_force = rks._contract_grad_response(ao, wv, dm)
    force = rks._contract_grad_response(ao, wv, dm, products)
    cp.testing.assert_allclose(force, reference_force, atol=2e-10, rtol=2e-12)


@contextmanager
def _two_pass_reference():
    from gpu4pyscf.grad import rks

    path = Path(os.environ['DFT_REFERENCE_GRADIENT'])
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_exc_full_response')
    namespace = rks.__dict__.copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    original = rks.get_exc_full_response
    rks.get_exc_full_response = namespace['get_exc_full_response']
    try:
        yield
    finally:
        rks.get_exc_full_response = original


@pytest.mark.skipif(not os.getenv('DFT_REFERENCE_GRADIENT'), reason='matching native baseline source required')
@pytest.mark.parametrize('xc', ['lda,vwn', 'pbe', 'b3lyp', 'wb97m-v'])
def test_streaming_complete_forces_and_energy_differences(xc, monkeypatch):
    import numpy as np
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.scf import hf

    monkeypatch.setattr(hf, 'overlap_zero_eigenvalue_threshold', 1e-10)
    mol = gto.M(atom='O 0 0 0; H .7586 0 .5043; H -.7586 0 .5043', basis='def2-tzvpd', verbose=0)
    mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-5
    mf.chkfile = None
    mf.kernel()
    assert mf.converged
    gradient = mf.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    with _two_pass_reference():
        reference = np.asarray(gradient.kernel())
    actual = np.asarray(gradient.kernel())
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
    coords = mol.atom_coords()
    step = 2e-4
    energies = []
    for sign in (-1, 1):
        displaced = coords.copy()
        displaced[1, 0] += sign * step
        mf.reset(mol.set_geom_(displaced, unit='Bohr', inplace=False))
        energies.append(mf.kernel())
        assert mf.converged
    assert abs(actual[1, 0] - (energies[1] - energies[0]) / (2 * step)) < 1e-5
