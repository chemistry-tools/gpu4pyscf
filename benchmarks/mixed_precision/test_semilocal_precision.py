"""Opt-in ordinary grid-precision and native-force checks."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


def test_dense_grid_products_use_private_float32_then_native_float64():
    import cupy as cp
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import mixed_precision, numint
    from gpu4pyscf.dft._mixed_grid import _gemm

    mf = dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0), xc='b3lyp')
    original = mf._numint.nr_rks.__func__
    rng = np.random.default_rng(1610)
    bra, ket = (cp.asarray(rng.normal(size=(23, 257))) for _ in range(2))
    out = cp.empty((23, 23))
    with mixed_precision._precision_context(mf) as (grid, vv10):
        product = mf._numint.nr_rks.__func__.__globals__['_dot_ao_ao_dense']
        expected = _gemm(bra.astype(cp.float32), ket.astype(cp.float32)).astype(cp.float64)
        assert product(bra, ket, out=out) is out
        cp.testing.assert_array_equal(out, expected)
        assert grid.float32_calls == 1 and grid.float64_calls == 0
        grid.use_float32 = False
        cp.testing.assert_array_equal(product(bra, ket), numint._dot_ao_ao_dense(bra, ket))
        assert grid.float64_calls == 1 and not vv10.precisions
    assert mf._numint.nr_rks.__func__ is original


def _calculator(atoms, xc, precision, *, charge=0, multiplicity=1, density_fit=False):
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.tools.ase_interface import PySCF

    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='6-31g*', cart=True, charge=charge, spin=multiplicity - 1, verbose=0)
    mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc=xc)
    if density_fit:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = 3
    # Independent SCF paths need tighter orbital convergence for force comparisons.
    # Production tolerances are configured by the caller and are not changed.
    mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-9
    # Tight screening keeps incremental J/K drift below the residual checked here.
    mf.direct_scf_tol = 1e-15
    mf.max_cycle, mf.chkfile = 100, None
    return PySCF(method=mf, precision=precision, grid_response=True,
                 auxbasis_response=True if density_fit else None)


@pytest.mark.parametrize('case,xc,charge,multiplicity', [
    ('CH3CONH2', 'b3lyp', 0, 1),
    ('C5H5N', 'b3lyp', 0, 1),
    ('C2H6SO', 'b3lyp', 0, 1),
    ('CH3COF', 'b3lyp', 0, 1),
    ('CH3COCl', 'b3lyp', 0, 1),
    ('H2O', 'b3lyp', 1, 2),
    ('H2O', 'lda,vwn', 0, 1),
    ('H2O', 'pbe', 0, 1),
    ('H2O', 'm06-l', 0, 1),
    ('H2O', 'wb97x-v', 0, 1),
    ('H2O', '0.2*HF+0.08*LDA_X+0.72*GGA_X_B88,0.81*GGA_C_LYP+0.19*LDA_C_VWN_RPA', 0, 1),
])
def test_semilocal_mixed_scf_and_full_forces(case, xc, charge, multiplicity):
    from ase.build import molecule

    reference, mixed = molecule(case), molecule(case)
    reference.calc = _calculator(reference, xc, 'float64', charge=charge, multiplicity=multiplicity)
    mixed.calc = _calculator(mixed, xc, 'auto', charge=charge, multiplicity=multiplicity)
    energy, reference_energy = mixed.get_potential_energy(), reference.get_potential_energy()
    assert abs(energy - reference_energy) < 1e-8
    report = mixed.calc.precision_info
    assert report['accepted'] and report['final_verification']['passed']
    assert report['grid_contraction_calls']['float32'] > 0
    assert report['grid_contraction_calls']['float64'] > 0
    assert bool(report['vv10_precisions']) == bool(mixed.calc.method_scan.do_nlc())
    np.testing.assert_allclose(mixed.get_forces(), reference.get_forces(), atol=1e-5, rtol=0)
    assert mixed.calc.calculation_info['scf_reused']
    if os.getenv('DFT_REFERENCE_GRADIENT') and multiplicity == 1:
        from test_native_grid_response import _two_pass_reference

        gradient = reference.calc.method_scan.Gradients()
        gradient.grid_response = gradient.auxbasis_response = True
        with _two_pass_reference():
            native = np.asarray(gradient.kernel())
        np.testing.assert_allclose(gradient.kernel(), native, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize('case', ['CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COF', 'CH3COCl'])
def test_density_fitted_b3lyp_scf_and_same_density_forces(case, monkeypatch):
    import cupy as cp
    from ase.build import molecule

    reference, mixed = molecule(case), molecule(case)
    reference.calc = _calculator(reference, 'b3lyp', 'float64', density_fit=True)
    mixed.calc = _calculator(mixed, 'b3lyp', 'mixed', density_fit=True)
    assert abs(mixed.get_potential_energy() - reference.get_potential_energy()) < 1e-8
    assert mixed.calc.precision_info['final_verification']['passed']
    assert mixed.calc.precision_info['grid_contraction_calls']['float32'] > 0
    cp.testing.assert_allclose(mixed.calc.method_scan.make_rdm1(),
                               reference.calc.method_scan.make_rdm1(), atol=1e-8, rtol=0)
    # Cartesian DF derivatives amplify tiny differences between independent SCF
    # solutions. Test the new force algebra at exactly the same accepted density.
    if os.getenv('DFT_REFERENCE_GRADIENT'):
        from test_native_grid_response import _two_pass_reference

        from gpu4pyscf.grad import rks

        original = rks._rho_and_dm_ao
        reused = []

        def retained_density_products(ao, dm, xctype, **kwargs):
            rho, products = original(ao, dm, xctype, **kwargs)
            assert xctype == 'GGA' and len(products) == 1
            reused.append(True)
            return rho, products

        monkeypatch.setattr(rks, '_rho_and_dm_ao', retained_density_products)
        gradient = mixed.calc.method_scan.Gradients()
        gradient.grid_response = gradient.auxbasis_response = True
        with _two_pass_reference():
            native = np.asarray(gradient.kernel())
        np.testing.assert_allclose(gradient.kernel(), native, atol=2e-10, rtol=2e-10)
        assert reused


def test_b3lyp_native_force_matches_warm_geometry_energy_differences():
    from ase.build import molecule

    atoms = molecule('CH3CONH2')
    atoms.calc = _calculator(atoms, 'b3lyp', 'mixed')
    forces = atoms.get_forces()
    atom, direction = np.unravel_index(np.argmax(np.abs(forces)), forces.shape)
    original = atoms.positions[atom, direction]
    step = 1e-4
    energies = []
    for sign in (-1, 1):
        atoms.positions[atom, direction] = original + sign * step
        energies.append(atoms.get_potential_energy())
        assert atoms.calc.precision_info['accepted']
    assert abs(forces[atom, direction] + (energies[1] - energies[0]) / (2 * step)) < 2e-4
