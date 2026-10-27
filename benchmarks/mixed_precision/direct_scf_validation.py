"""Validate complete direct-SCF energies and forces at the pinned DFT recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from gpu_performance import _resources, _settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--regroup', action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--geometry-report', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--finite-difference', action='store_true')
    parser.add_argument('--require-gpu-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.geometry_report) != bool(args.state):
        parser.error('External geometry and orbital state must be supplied together')
    _resources([int(x) for x in args.cpus.split(',')])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise RuntimeError('Unexpected physical GPU selection')
    args.output.mkdir(parents=True, exist_ok=False)
    settings, protocol_hash = _settings()

    import cupy as cp
    import numpy as np
    from ase import Atoms
    from pyscf import gto, lib
    from pyscf.data.nist import BOHR, HARTREE2EV
    from threadpoolctl import threadpool_info

    import gpu4pyscf
    from gpu4pyscf import dft
    from gpu4pyscf.dft.mixed_precision import _full_check
    from gpu4pyscf.scf import hf
    from gpu4pyscf.tools.ase_interface import PySCF

    package = Path(gpu4pyscf.__file__).absolute().parent
    if package != args.require_gpu_source.absolute() or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Use the declared matching native single-GPU installation')
    lib.num_threads(4)
    if any(p['num_threads'] > 4 for p in threadpool_info()):
        raise RuntimeError('Thread budget exceeded')
    if args.geometry_report:
        metadata = json.loads(args.geometry_report.read_text())
        if metadata['basis'] != settings['basis'] or metadata['functional'] != settings['xc']:
            raise ValueError('External state recipe differs from the pinned protocol')
        atom, charge, multiplicity = (metadata[k] for k in ('geometry_angstrom', 'charge', 'multiplicity'))
    else:
        from _organic_cases import ORGANIC_CASES
        atom, charge, multiplicity = ORGANIC_CASES[args.case]
    if multiplicity != 1:
        raise ValueError('This qualification panel supports restricted singlets')
    hf.overlap_zero_eigenvalue_threshold = 1e-10
    mol = gto.M(atom=atom, basis=settings['basis'], charge=charge, spin=multiplicity-1,
                cart=False, verbose=4, output=str(args.output/'pyscf.log'))

    def method(current_mol, *, tolerance=None):
        mf = dft.RKS(current_mol, xc=settings['xc'])
        mf.grids.level, mf.nlcgrids.level = settings['grid_level'], settings['nlc_grid_level']
        mf.conv_tol = settings['conv_tol'] if tolerance is None else tolerance
        mf.conv_tol_grad = 1e-4 if tolerance is None else 1e-6
        mf.max_cycle, mf.chkfile = settings['max_cycle'], None
        mf.direct_scf_tol, mf.direct_scf_regroup = 1e-13, args.regroup
        return mf

    mf = method(mol)
    guess = None
    if args.state:
        with np.load(args.state, allow_pickle=False) as saved:
            coeff, occ = cp.asarray(saved['mo_coeff']), cp.asarray(saved['mo_occ'])
        if coeff.shape != (mol.nao, mol.nao) or occ.shape != (mol.nao,):
            raise ValueError('External orbitals have the wrong basis dimension')
        overlap = cp.asarray(mol.intor_symmetric('int1e_ovlp'))
        if float(cp.max(cp.abs(coeff.T @ overlap @ coeff-cp.eye(mol.nao)))) > 1e-6:
            raise ValueError('External orbitals do not match this overlap matrix')
        guess = .999*((coeff*occ) @ coeff.T)+.001*cp.asarray(mf.get_init_guess(mol, 'minao'))
    cycles = []

    def progress(env):
        row = {'cycle': int(env['cycle'])+1, 'energy_hartree': float(env['e_tot']),
               'orbital_gradient': float(env['norm_gorb'])}
        cycles.append(row)
        print(json.dumps(row), flush=True)

    mf.callback = progress
    atoms = None
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    if guess is None:
        atoms = Atoms(mol.elements, positions=mol.atom_coords(unit='Angstrom'))
        atoms.calc = PySCF(method=mf, precision='float64', grid_response=True)
        energy = atoms.get_potential_energy()/HARTREE2EV
        mf = atoms.calc.method_scan
    else:
        energy = float(mf.kernel(dm0=guess))
    cp.cuda.runtime.deviceSynchronize()
    energy_seconds = time.perf_counter()-started
    if not mf.converged or mf.mo_coeff.shape[1] != mol.nao:
        raise RuntimeError('SCF must converge and retain the full basis')
    np.savez(args.output/'state.npz', mo_coeff=cp.asnumpy(mf.mo_coeff), mo_occ=cp.asnumpy(mf.mo_occ),
             mo_energy=cp.asnumpy(mf.mo_energy))
    (args.output/'scf.json').write_text(json.dumps({'energy_hartree': energy, 'cycles': cycles}, indent=2)+'\n')
    selected_cutoffs = {str(k): opt.diffuse_cutoff for k, opt in mf._opt_gpu.items() if opt is not None}
    if set(selected_cutoffs.values()) != {10. if args.regroup else .3}:
        raise RuntimeError(f'The requested representation was not exercised: {selected_cutoffs}')

    def gradient(current):
        grad = current.Gradients()
        grad.grid_response = True
        return np.asarray(cp.asnumpy(grad.kernel()))

    started = time.perf_counter()
    if atoms is not None:
        grad = -atoms.get_forces()*BOHR/HARTREE2EV
        if not atoms.calc.calculation_info['scf_reused'] or atoms.calc.calculation_info['scf_seconds'] != 0:
            raise RuntimeError('ASE forces must reuse the accepted unchanged-geometry SCF')
    else:
        grad = gradient(mf)
    cp.cuda.runtime.deviceSynchronize()
    force_seconds = time.perf_counter()-started
    np.save(args.output/'gradient.npy', grad)
    force_cutoffs = {str(k): opt.diffuse_cutoff for k, opt in mf._opt_gpu.items() if opt is not None}
    omega = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)[0]
    if (str(-omega) in force_cutoffs) == args.regroup:
        raise RuntimeError(f'Unexpected derivative operator dispatch: {force_cutoffs}')

    # Run controls after the timed force call so that verification cannot replace
    # the reusable SCF exchange optimizer before its derivative uses it.
    mf.direct_scf_regroup = False
    try:
        check = _full_check(mf, energy)
    finally:
        mf.direct_scf_regroup = args.regroup
    (args.output/'native-check.json').write_text(json.dumps(check, indent=2)+'\n')
    if not check['passed']:
        raise RuntimeError(f'Original-representation Fock verification failed: {check}')
    reference_error = None
    if args.regroup:
        mf.direct_scf_regroup = False
        try:
            reference_grad = gradient(mf)
        finally:
            mf.direct_scf_regroup = True
        np.save(args.output/'native-same-state-gradient.npy', reference_grad)
        reference_error = float(np.max(np.abs(reference_grad-grad)))
        np.testing.assert_allclose(grad, reference_grad, atol=2e-8, rtol=0)

    finite_difference = None
    if args.finite_difference:
        axis, atom_index, step = 0, 0, 2e-4
        displaced_energies = []
        for sign in (-1, 1):
            coordinates = mf.mol.atom_coords()
            coordinates[atom_index, axis] += sign*step
            displaced = method(mf.mol.set_geom_(coordinates, unit='Bohr', inplace=False), tolerance=1e-10)
            value = float(displaced.kernel(dm0=mf.make_rdm1()))
            if not displaced.converged:
                raise RuntimeError('Finite-difference displaced SCF did not converge')
            displaced_energies.append(value)
        numerical = (displaced_energies[1]-displaced_energies[0])/(2*step)
        finite_difference = {'atom': atom_index, 'axis': axis, 'step_bohr': step,
                             'energies_hartree': displaced_energies, 'gradient_hartree_bohr': numerical,
                             'analytic_hartree_bohr': float(grad[atom_index, axis]),
                             'conv_tol': 1e-10, 'conv_tol_grad': 1e-6, 'tolerance_hartree_bohr': 3e-6}
        (args.output/'finite-difference.json').write_text(json.dumps(finite_difference, indent=2)+'\n')
        if abs(numerical-grad[atom_index, axis]) > 3e-6:
            raise RuntimeError('Analytic force failed the independent energy finite difference')

    files = ['scf/hf.py', 'scf/jk.py', 'scf/_direct_scf.py', 'grad/rhf.py', 'grad/rks.py',
             'dft/rks.py', 'dft/numint.py', 'tools/ase_interface.py']
    report = {'case': args.case, 'regrouped': args.regroup, 'geometry_angstrom': atom,
              'charge': charge, 'multiplicity': multiplicity, 'basis': mol.basis, 'cart': mol.cart,
              'functional': mf.xc, 'grid_level': mf.grids.level, 'nlc_grid_level': mf.nlcgrids.level,
              'grid_response': True, 'conv_tol': mf.conv_tol, 'conv_tol_grad': mf.conv_tol_grad,
              'max_cycle': mf.max_cycle, 'direct_scf_tol': mf.direct_scf_tol,
              'protocol_sha256': protocol_hash, 'nao': mol.nao, 'energy_hartree': energy,
              'energy_seconds': energy_seconds, 'force_seconds': force_seconds, 'scf_cycles': cycles,
              'entry_point': 'ASE' if atoms is not None else 'kernel with controlled supplied density',
              'supplied_guess': None if guess is None else '99.9% retained DF density plus 0.1% MINAO',
              'original_representation_verification': check, 'selected_scf_cutoffs': selected_cutoffs,
              'selected_force_cutoffs': force_cutoffs,
              'same_state_force_max_abs': reference_error, 'finite_difference': finite_difference,
              'gradient_hartree_bohr': grad.tolist(), 'imported_package': str(package),
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threadpools': threadpool_info(),
              'physical_gpu': args.gpu, 'gpu_pci': cp.cuda.Device().pci_bus_id,
              'source_sha256': {name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in files},
              'native_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('lib/*.so')},
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
