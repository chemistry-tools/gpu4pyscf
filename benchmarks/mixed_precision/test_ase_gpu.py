"""Opt-in ASE energy, force and geometry-change checks on one CUDA device."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


def _calculator(atoms, charge, precision):
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.tools.ase_interface import PySCF

    mol = gto.M(
        atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
        basis='def2-tzvpd',
        charge=charge,
        spin=charge,
        verbose=0,
    )
    mf = (dft.RKS if charge == 0 else dft.UKS)(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.chkfile = None
    return PySCF(method=mf, precision=precision, grid_response=True, auxbasis_response=True)


@pytest.mark.parametrize('charge', [0, 1])
def test_ase_energy_force_and_warm_geometry_agreement(charge):
    from ase import Atoms

    from gpu4pyscf.dft import numint

    original_vv10, original_contract = numint._vv10nlc, numint.contract
    baseline = Atoms('OH2', positions=[[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]])
    mixed = baseline.copy()
    baseline.calc = _calculator(baseline, charge, 'float64')
    mixed.calc = _calculator(mixed, charge, 'auto' if charge == 0 else 'mixed')
    for displacement in (0.0, 0.015):
        baseline.positions[1, 0] += displacement
        mixed.positions[:] = baseline.positions
        reference_forces = baseline.get_forces()
        forces = mixed.get_forces()
        assert abs(mixed.get_potential_energy() - baseline.get_potential_energy()) < 1e-7
        np.testing.assert_allclose(forces, reference_forces, atol=1e-5, rtol=0)
        assert isinstance(forces, np.ndarray)
        assert mixed.calc.precision_info['accepted']
        assert mixed.calc.calculation_info['precision'] == 'mixed'
        assert numint._vv10nlc is original_vv10 and numint.contract is original_contract
        assert all(name not in mixed.calc.method_scan._numint.__dict__ for name in ('nr_rks', 'nr_uks', 'nr_nlc_vxc'))
    assert mixed.calc.method_scan.e_tot == mixed.calc.precision_info['final_verification']['float64_energy_hartree']


def test_mixed_force_matches_energy_finite_difference():
    from ase import Atoms

    atoms = Atoms('OH2', positions=[[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]])
    atoms.calc = _calculator(atoms, 0, 'mixed')
    force = atoms.get_forces()[1, 0]
    original = atoms.positions[1, 0]
    step = 1e-4
    energies = []
    for sign in (-1, 1):
        atoms.positions[1, 0] = original + sign * step
        energies.append(atoms.get_potential_energy())
    numeric_force = -(energies[1] - energies[0]) / (2 * step)
    assert abs(force - numeric_force) < 2e-4
