"""Opt-in nonlocal-only precision checks on mixed-element organic densities."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
def test_vv10_only_with_supplied_organic_density(case):
    import cupy as cp
    from ase.build import molecule
    from gpu4pyscf import dft
    from gpu4pyscf.dft import mixed_precision, numint
    from pyscf import gto

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='6-31g*', verbose=0)

    def method():
        mf = dft.RKS(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
        mf.grids.level, mf.nlcgrids.level = 2, 1
        mf.conv_tol, mf.conv_tol_grad = 1e-10, 1e-7
        mf.max_cycle, mf.chkfile = 100, None
        return mf

    reference = method()
    reference.kernel()
    assert reference.converged
    mixed = method()
    dm = .999 * reference.make_rdm1() + .001 * cp.asarray(mixed.get_init_guess())
    original_integrator, original_exchange = mixed._numint.nr_rks.__func__, mixed.with_df.get_jk.__func__
    original_pair = numint._vv10nlc
    policy = 'auto' if case in ('CH3CONH2', 'C2H6SO', 'CH3COCl') else 'vv10'
    energy = mixed_precision.run_verified_scf(mixed, dm0=dm, precision=policy)
    report = mixed.mixed_precision_info
    assert report['precision'] == 'vv10' and report['requested_precision'] == policy
    assert report['accepted'] and report['final_verification']['passed']
    assert report['grid_contraction_calls']['float32'] == 0
    assert report['df_exchange_contraction_calls']['float32'] == 0
    assert report['df_exchange_contraction_calls']['int8'] == 0
    assert 'float32' in report['vv10_precisions'] and 'float64' in report['vv10_precisions']
    assert mixed._numint.nr_rks.__func__ is original_integrator
    assert mixed.with_df.get_jk.__func__ is original_exchange
    assert numint._vv10nlc is original_pair
    assert abs(energy - reference.e_tot) < 1e-8
    cp.testing.assert_allclose(mixed.make_rdm1(), reference.make_rdm1(), atol=1e-6, rtol=0)
    gradients = []
    for mf in (reference, mixed):
        gradient = mf.Gradients()
        gradient.grid_response = gradient.auxbasis_response = True
        gradients.append(np.asarray(gradient.kernel()))
    np.testing.assert_allclose(gradients[0], gradients[1], atol=1e-6, rtol=0)
