# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from ase.calculators.calculator import Calculator, all_properties
except ImportError:
    print("""ASE is not found. Please install ASE via
pip3 install ase
          """)
    raise RuntimeError("ASE is not found")

import time

import cupy as cp
import numpy as np
from ase.units import Debye
from pyscf import lib
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscf.gto.mole import charge
from pyscf.pbc.gto.cell import Cell
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

# These functions are copied from the development branch of PySCF and will be
# provided by the pyscf.pbc.tools.pyscf_ase module in PySCF 2.11.

def pyscf_to_ase_atoms(pyscf_obj):
    '''
    Convert PySCF Cell/Mole object to ASE Atoms object
    '''
    from ase import Atoms
    from pyscf.gto import mole
    from pyscf.pbc import gto

    if isinstance(pyscf_obj, mole.MoleBase):
        cell = pyscf_obj
        method = None
    elif hasattr(pyscf_obj, 'mol'):
        cell = pyscf_obj.mol
        method = pyscf_obj
    else:
        cell = pyscf_obj.cell
        method = pyscf_obj

    symbols = cell.elements
    positions = cell.atom_coords() * BOHR
    if isinstance(cell, gto.Cell):
        a = cell.lattice_vectors() * BOHR
        atoms = Atoms(symbols, positions, cell=a, pbc=True)
    else:
        atoms = Atoms(symbols, positions, pbc=False)

    if method is not None:
        atoms.calc = PySCF(method=method)
    return atoms

def cell_from_ase(ase_atoms):
    '''Convert ASE atoms to PySCF Cell instance. The lattice vectors and atomic
    positions are defined in the Cell instance. It does not have any basis sets
    or pseudopotentials assigned. The Cell instance is not initialized with 'build()'.
    '''
    cell = Cell()
    cell.atom = ase_atoms_to_pyscf(ase_atoms)
    cell.a = np.asarray(ase_atoms.cell)
    return cell

def bandpath(cell, npoints=None):
    from ase.cell import Cell as ase_Cell
    a = cell.lattice_vectors() * BOHR # To Angstrom
    bp = ase_Cell(a).bandpath(npoints=npoints)
    return bp

def plot_band_structure(bandpath, e_kn, ax=None, color='k'):
    '''
    Args:
        bandpath:
            an ase.BandPath instance
        e_kn:
            eigenvalus for each k-points (in eV)
        ax:
            matplotlib Axis instance
        color:
            color for band
    '''
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    xcoords, sp_points, sp_labels = bandpath.get_linear_kpoint_axis()

    nbands = e_kn.shape[1]
    for b in range(nbands):
        plt.plot(xcoords, e_kn[:,b], color=color, lw=1)

    for x in sp_points:
        ax.axvline(x, color="gray", linewidth=0.8)
    ax.set_xticks(sp_points)
    ax.set_xticklabels(sp_labels)

    ax.set_xlim(xcoords[0], xcoords[-1])
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("k-vector")
    ax.set_title("Band Structure")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)  # Fermi level
    return ax

class PySCF(Calculator):
    implemented_properties = ['energy', 'forces', 'stress',
                              'dipole', 'magmom']

    default_parameters = {'precision': 'float64', 'grid_response': None,
                          'auxbasis_response': None}

    def __init__(self, restart=None, label='PySCF', atoms=None, directory='.',
                 method=None, precision='float64', grid_response=None,
                 auxbasis_response=None, **kwargs):
        """Construct PySCF-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'PySCF'.

        method: A PySCF method class
        precision: 'float64' (default) or 'mixed'
            Mixed molecular DFT uses early float32 grid/VV10 products, then verifies
            full float64 fields and reconverges in float64 if necessary.
        grid_response: bool or None
            Explicit gradient grid response; None retains the method's default.
        auxbasis_response: bool or None
            Explicit gradient auxiliary basis response; None retains its default.
        """
        Calculator.__init__(self, restart, label=label, atoms=atoms,
                            directory=directory, precision=precision,
                            grid_response=grid_response, auxbasis_response=auxbasis_response,
                            **kwargs)

        if not isinstance(method, lib.StreamObject):
            raise RuntimeError(f'{method} must be an instance of a PySCF method')

        self.method = method
        self.precision_info = None
        self.calculation_info = None
        self.hessian_info = None
        self.pbc = hasattr(method, 'cell')
        self.mesh = None
        if self.pbc:
            from gpu4pyscf.pbc.tools.discretization import freeze_mesh
            mol = method.cell
            self.mesh = freeze_mesh(method)
        else:
            mol = method.mol
        self.mol = mol
        self.method_scan = None
        if hasattr(method, 'as_scanner'):
            # Scanner can utilize the initial guess from previous calculations
            self.method_scan = method.as_scanner()

    def set(self, **kwargs):
        if 'precision' in kwargs and kwargs['precision'] not in ('float64', 'mixed'):
            raise ValueError("precision must be 'float64' or 'mixed'")
        for name in ('grid_response', 'auxbasis_response'):
            if name in kwargs and kwargs[name] is not None and type(kwargs[name]) is not bool:
                raise ValueError(f'{name} must be bool or None')
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_properties):
        Calculator.calculate(self, atoms)
        started = time.perf_counter()
        self.calculation_info = {}

        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        Z = np.array([charge(x) for x in self.mol.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))

        if self.pbc:
            self.mol.set_geom_(_atoms, a=np.asarray(atoms.cell), unit='Angstrom')
        else:
            self.mol.set_geom_(_atoms, unit='Angstrom')
        if self.pbc:
            from gpu4pyscf.pbc.tools.discretization import freeze_mesh
            base_method = self.method
            if self.method_scan is not None:
                base_method = self.method_scan
            freeze_mesh(base_method, self.mol, self.mesh)

        with_grad = 'forces' in properties or 'stress' in properties
        with_energy = with_grad or 'energy' in properties or 'dipole' in properties

        if with_energy:
            base_method = self.method if self.method_scan is None else self.method_scan
            self.precision_info = None
            scf_started = time.perf_counter()

            def evaluate():
                if self.method_scan is None:
                    self.method.reset(self.mol).run()
                    return self.method.e_tot
                return self.method_scan(self.mol)

            if self.parameters.precision == 'mixed':
                from gpu4pyscf.dft.mixed_precision import run_verified_scf
                e_tot = run_verified_scf(base_method, evaluate)
                self.precision_info = base_method.mixed_precision_info
            else:
                e_tot = evaluate()
            if not getattr(base_method, 'converged', True):
                raise RuntimeError(f'{base_method} not converged')
            self.results['energy'] = e_tot * HARTREE2EV
            self.calculation_info['scf_seconds'] = time.perf_counter() - scf_started

        if self.method_scan is None:
            base_method = self.method
        else:
            base_method = self.method_scan

        if with_grad:
            grad_obj = base_method.Gradients()
            if self.parameters.grid_response is not None:
                grad_obj.grid_response = self.parameters.grid_response
            if self.parameters.auxbasis_response is not None:
                if not hasattr(grad_obj, 'auxbasis_response'):
                    raise ValueError('auxbasis_response requires a density-fitted gradient')
                grad_obj.auxbasis_response = self.parameters.auxbasis_response

        if 'forces' in properties:
            force_started = time.perf_counter()
            forces = -grad_obj.kernel()
            if hasattr(forces, 'get'):
                forces = forces.get()
            self.results['forces'] = np.asarray(forces) * (HARTREE2EV / BOHR)
            self.calculation_info['force_seconds'] = time.perf_counter() - force_started

        if 'stress' in properties:
            stress = grad_obj.get_stress()
            if hasattr(stress, 'get'):
                stress = stress.get()
            self.results['stress'] = np.asarray(stress) * (HARTREE2EV / BOHR**3)

        if 'dipole' in properties:
            if self.pbc:
                raise NotImplementedError('dipole for PBC calculations')
            # in Gaussian cgs unit
            self.results['dipole'] = base_method.dip_moment() * Debye

        if 'magmom' in properties:
            magmom = self.mol.spin
            self.results['magmom'] = magmom

        self.calculation_info['total_seconds'] = time.perf_counter() - started

    def get_hessian(self, atoms=None, auxbasis_response=2):
        """Return an analytic molecular Cartesian Hessian in eV/Angstrom squared.

        Reconverge the current geometry with native float64 SCF before CPHF. This
        explicit method is separate from ASE's standard energy/force properties.
        Auxiliary response defaults to the full density-fitting response; pass
        None for a method without density fitting or to retain its native setting.
        """
        if self.pbc:
            raise NotImplementedError('ASE analytic Hessians currently support molecules only')
        if auxbasis_response is not None and (
            type(auxbasis_response) is not int or auxbasis_response not in (0, 1, 2)
        ):
            raise ValueError('Hessian auxbasis_response must be 0, 1, 2 or None')
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError('Atoms are required for a Hessian calculation')
        self.get_potential_energy(atoms)
        base = self.method if self.method_scan is None else self.method_scan
        self.hessian_info = None
        started = time.perf_counter()
        # Reconvergence can change the density; old derivative caches are invalid.
        self.results.clear()
        base.kernel(dm0=base.make_rdm1())
        if not base.converged:
            raise RuntimeError('Float64 Hessian reference did not converge')
        self.results['energy'] = float(base.e_tot) * HARTREE2EV
        hobj = base.Hessian()
        if self.parameters.grid_response is not None:
            hobj.grid_response = self.parameters.grid_response
        if auxbasis_response is not None:
            if not hasattr(hobj, 'auxbasis_response'):
                raise ValueError('Hessian auxbasis_response requires density fitting')
            hobj.auxbasis_response = auxbasis_response
        hessian = hobj.kernel()
        if hasattr(hessian, 'get'):
            hessian = hessian.get()
        hessian = np.asarray(hessian, dtype=np.float64)
        n = len(atoms)
        if hessian.shape != (n, n, 3, 3) or not np.isfinite(hessian).all():
            raise RuntimeError('Analytic Hessian must be finite with shape (N, N, 3, 3)')
        cartesian = hessian.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        cartesian = 0.5 * (cartesian + cartesian.T) * (HARTREE2EV / BOHR**2)
        self.hessian_info = {
            'seconds': time.perf_counter() - started,
            'float64_reference_energy_hartree': float(base.e_tot),
            'reference_scf_cycles': int(base.cycles),
            'grid_response': getattr(hobj, 'grid_response', None),
            'auxbasis_response': getattr(hobj, 'auxbasis_response', None),
        }
        return cartesian

    def get_fermi_level(self):
        method = self.method if self.method_scan is None else self.method_scan
        return method.get_fermi() * HARTREE2EV

    def get_eigenvalues(self, kpt=0, spin=0):
        method = self.method if self.method_scan is None else self.method_scan
        if method.istype('UHF'):
            e = method.mo_energy[spin]
        else:
            assert spin == 0
            e = method.mo_energy
        if method.istype('KSCF'):
            e = e[kpt]
        else:
            assert kpt == 0
        return e * HARTREE2EV

    def get_occupation_numbers(self, kpt=0, spin=0):
        method = self.method if self.method_scan is None else self.method_scan
        if method.istype('UHF'):
            occ = method.mo_occ[spin]
        else:
            assert spin == 0
            occ = method.mo_occ
        if method.istype('KSCF'):
            occ = occ[kpt]
        else:
            assert kpt == 0
        return occ

    def get_number_of_spins(self):
        method = self.method if self.method_scan is None else self.method_scan
        if method.istype('UHF'):
            nspins = 2
        else:
            nspins = 1
        return nspins

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import BandStructure
        method = self.method if self.method_scan is None else self.method_scan
        standard_path = self.atoms.cell.bandpath()
        band_kpts = method.cell.get_abs_kpts(standard_path.kpts)
        e_k = cp.asnumpy(cp.array(method.get_bands(band_kpts)[0]))
        if not method.istype('UHF'):
            e_k = e_k[None]
        fermi = self.get_fermi_level()
        return BandStructure(standard_path, e_k, fermi)
