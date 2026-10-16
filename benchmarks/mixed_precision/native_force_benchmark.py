"""Compare complete native forces with a retained density-fitting gradient baseline."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from _larger_cases import LARGER_CASES
from gpu_performance import _resources, _source_provenance


@contextlib.contextmanager
def _reference_df(path, *, unrestricted=False):
    if unrestricted:
        from gpu4pyscf.df.grad import uhf as rhf
    else:
        from gpu4pyscf.df.grad import rhf

    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_jk_energy_per_atom')
    namespace = rhf.__dict__.copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    original = rhf._jk_energy_per_atom
    rhf._jk_energy_per_atom = namespace['_jk_energy_per_atom']
    try:
        yield
    finally:
        rhf._jk_energy_per_atom = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(LARGER_CASES), required=True)
    parser.add_argument('--reference-df-grad', type=Path, required=True)
    parser.add_argument('--require-gpu-source', type=Path, required=True)
    parser.add_argument(
        '--state', type=Path, help='Previously verified SCF state directory with recipe.json and scf.npz'
    )
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists() or args.output.suffix != '.json':
        parser.error('Use at least two repeats and a new .json output path')
    artifact = args.output.with_suffix('')
    artifact.mkdir(parents=True, exist_ok=False)
    _resources([4, 5, 6, 7])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Select physical GPU 1')

    import cupy as cp
    import numpy as np
    from pyscf import gto, lib
    from threadpoolctl import threadpool_info

    import gpu4pyscf
    from gpu4pyscf import dft
    from gpu4pyscf.df.grad import rhf
    from gpu4pyscf.dft import mixed_precision, numint
    from gpu4pyscf.scf import hf

    source = Path(gpu4pyscf.__file__).resolve().parent
    if source != args.require_gpu_source.resolve():
        raise RuntimeError(f'Unexpected GPU4PySCF import: {source}')
    if cp.cuda.runtime.getDeviceCount() != 1 or not cp.cuda.Device(0).pci_bus_id.lower().endswith('05:00.0'):
        raise RuntimeError('Visible device must be physical GPU 1 on the designated host')
    lib.num_threads(4)
    if any(pool['num_threads'] > 4 for pool in threadpool_info()):
        raise RuntimeError('Numerical thread pools exceed the resource allowance')
    hf.overlap_zero_eigenvalue_threshold = 1e-10
    atom, charge, multiplicity = LARGER_CASES[args.case]
    recipe = {
        'case': args.case,
        'geometry_angstrom': atom,
        'charge': charge,
        'multiplicity': multiplicity,
        'functional': 'wb97m-v',
        'basis': 'def2-tzvpd',
        'auxbasis': 'def2-universal-jkfit',
        'grid_level': 5,
        'nlc_grid_level': 3,
        'conv_tol': 1e-8,
        'conv_tol_grad': 1e-4,
        'max_cycle': 200,
        'overlap_cutoff': 1e-10,
        'grid_response': True,
        'auxbasis_response': True,
    }
    mol = gto.M(atom=atom, basis=recipe['basis'], charge=charge, spin=multiplicity - 1, verbose=0)
    mf = dft.RKS(mol, xc=recipe['functional']).density_fit(auxbasis=recipe['auxbasis'])
    mf.grids.level, mf.nlcgrids.level = recipe['grid_level'], recipe['nlc_grid_level']
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = recipe['conv_tol'], recipe['conv_tol_grad'], recipe['max_cycle']
    mf.chkfile = None
    if args.state:
        saved = json.loads((args.state / 'recipe.json').read_text())
        if any(saved.get(key) != value for key, value in recipe.items()) or not saved['precision_info']['accepted']:
            raise RuntimeError('Saved SCF state must have the identical recipe and accepted verification')
        state_path = args.state / 'scf.npz'
        if hashlib.sha256(state_path.read_bytes()).hexdigest() != saved['state_sha256']:
            raise RuntimeError('Saved SCF state hash does not match')
        with np.load(state_path, allow_pickle=False) as data:
            mf.mo_coeff, mf.mo_occ, mf.mo_energy = (
                cp.asarray(data[key]) for key in ('mo_coeff', 'mo_occ', 'mo_energy')
            )
        mf.e_tot, mf.converged = saved['energy_hartree'], True
        check = mixed_precision._full_check(mf, mf.e_tot)
        if not check['passed']:
            raise RuntimeError(f'Saved SCF state failed fresh float64 verification: {check}')
        scf_seconds = saved['scf_seconds']
        precision_info = saved['precision_info']
        energy = check['float64_energy_hartree']
    else:
        mf.callback = lambda env: print(
            json.dumps({'cycle': int(env['cycle']) + 1, 'energy': float(env['e_tot'])}), flush=True
        )
        started = time.perf_counter()
        energy = mixed_precision.run_verified_scf(mf)
        cp.cuda.get_current_stream().synchronize()
        scf_seconds, precision_info = time.perf_counter() - started, mf.mixed_precision_info
        check = precision_info['final_verification']
    if not precision_info['accepted'] or mf.mo_coeff.shape[-1] != mol.nao:
        raise RuntimeError('SCF must pass verification and retain the full basis')
    state_path = artifact / 'scf.npz'
    np.savez(state_path, mo_coeff=mf.mo_coeff.get(), mo_occ=mf.mo_occ.get(), mo_energy=mf.mo_energy.get())
    saved = recipe | {
        'scf_seconds': scf_seconds,
        'energy_hartree': energy,
        'precision_info': precision_info,
        'state_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
    }
    (artifact / 'recipe.json').write_text(json.dumps(saved, indent=2) + '\n')
    report = saved | {
        'state_reused': str(args.state) if args.state else None,
        'fresh_verification': check,
        'nao': mol.nao,
        'effective_orbitals': int(mf.mo_coeff.shape[-1]),
        'aux_nao': mf.with_df.auxmol.nao,
        'grid_points': len(mf.grids.weights),
        'nlc_grid_points': len(mf.nlcgrids.weights),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'gpu_pci': cp.cuda.Device(0).pci_bus_id,
        'threadpools': threadpool_info(),
        'sources': _source_provenance(gpu4pyscf, numint.libgdft),
        'native_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('lib/*.so')},
        'df_gradient_sha256': hashlib.sha256(Path(rhf.__file__).read_bytes()).hexdigest(),
        'reference_df_gradient_sha256': hashlib.sha256(args.reference_df_grad.read_bytes()).hexdigest(),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runs': [],
    }
    dm, reference = mf.make_rdm1(), None
    omega = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)[0]
    for repeat in range(args.repeats):
        for mode in ('reference', 'current') if repeat % 2 == 0 else ('current', 'reference'):
            # Free unused pool blocks and rebuild full/range-separated buffers so that
            # memory-driven auxiliary batching starts from the same post-SCF state.
            mf.with_df.reset()
            cp.get_default_memory_pool().free_all_blocks()
            mf.get_jk(mol, dm)
            mf.get_jk(mol, dm, with_j=False, omega=omega)
            cp.cuda.get_current_stream().synchronize()
            free_memory = int(cp.cuda.runtime.memGetInfo()[0])
            gradient = mf.nuc_grad_method()
            gradient.grid_response = gradient.auxbasis_response = True
            scope = _reference_df(args.reference_df_grad) if mode == 'reference' else contextlib.nullcontext()
            with scope:
                started = time.perf_counter()
                values = np.asarray(gradient.kernel())
                cp.cuda.get_current_stream().synchronize()
                seconds = time.perf_counter() - started
            if reference is None:
                reference = values.copy()
            error = float(np.max(np.abs(values - reference)))
            row = {
                'mode': mode,
                'repeat': repeat,
                'seconds': seconds,
                'free_memory_before': free_memory,
                'gradient_hartree_bohr': values.tolist(),
                'max_error_hartree_bohr': error,
            }
            report['runs'].append(row)
            (artifact / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            if error > 1e-9:
                raise RuntimeError(f'Force agreement failed: {error} Hartree/Bohr')
            print(json.dumps({key: value for key, value in row.items() if key != 'gradient_hartree_bohr'}), flush=True)
    medians = {
        mode: statistics.median(row['seconds'] for row in report['runs'] if row['mode'] == mode)
        for mode in ('reference', 'current')
    }
    report['summary'] = {
        'median_seconds': medians,
        'speedup': medians['reference'] / medians['current'],
        'max_error_hartree_bohr': max(row['max_error_hartree_bohr'] for row in report['runs']),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
