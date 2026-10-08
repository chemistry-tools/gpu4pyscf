"""Optimize synthetic water with verified mixed SCF and native float64 forces."""

from __future__ import annotations

import os


def main():
    # Set resource limits before numerical imports; expose one GPU in the launch environment.
    for variable in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '4'

    from ase import Atoms
    from ase.optimize import BFGS
    from pyscf import gto, lib

    from gpu4pyscf import dft
    from gpu4pyscf.tools.ase_interface import PySCF

    lib.num_threads(4)
    atoms = Atoms('OH2', positions=[[0, 0, 0], [0.7586, 0, 0.5043], [-0.7586, 0, 0.5043]])
    mol = gto.M(
        atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
        basis='def2-tzvpd',
        charge=0,
        spin=0,
    )
    mf = dft.RKS(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.chkfile = None
    atoms.calc = PySCF(method=mf, precision='mixed', grid_response=True, auxbasis_response=True)
    BFGS(atoms).run(fmax=0.005, steps=100)
    print('Energy (eV):', atoms.get_potential_energy())
    print('SCF verification:', atoms.calc.precision_info)
    # For analytic vibrational analysis, explicitly request a full float64 reference:
    # hessian = atoms.calc.get_hessian(atoms, auxbasis_response=2)


if __name__ == '__main__':
    main()
