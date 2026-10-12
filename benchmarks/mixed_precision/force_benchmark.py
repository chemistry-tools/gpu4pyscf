"""Compare exact force contractions at one accepted density with full grid response."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from _extensions import load_extensions
from _larger_cases import LARGER_CASES
from force_experiment import density_response_patch
from gpu_performance import _resources, _source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension-root', type=Path, required=True)
    parser.add_argument('--case', choices=tuple(LARGER_CASES), required=True)
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
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad import rks
    from gpu4pyscf.scf import hf

    if cp.cuda.runtime.getDeviceCount() != 1 or not cp.cuda.Device(0).pci_bus_id.lower().endswith('05:00.0'):
        raise RuntimeError('Visible device must be physical GPU 1 on the designated host')
    lib.num_threads(4)
    extensions = load_extensions(args.extension_root)
    from gpu4pyscf.dft import mixed_precision

    hf.overlap_zero_eigenvalue_threshold = 1e-10
    atom, charge, multiplicity = LARGER_CASES[args.case]
    mol = gto.M(atom=atom, basis='def2-tzvpd', charge=charge, spin=multiplicity - 1, verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-8, 1e-4, 200
    mf.chkfile = None

    def progress(env):
        print(json.dumps({'cycle': int(env['cycle']) + 1, 'energy': float(env['e_tot'])}), flush=True)

    mf.callback = progress
    report = {
        'case': args.case, 'geometry_angstrom': atom, 'charge': charge, 'multiplicity': multiplicity,
        'functional': mf.xc, 'basis': mol.basis, 'auxbasis': mf.with_df.auxbasis,
        'grid_level': 5, 'nlc_grid_level': 3, 'grid_response': True, 'auxbasis_response': True,
        'conv_tol': mf.conv_tol, 'conv_tol_grad': mf.conv_tol_grad, 'overlap_cutoff': 1e-10,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'openmp_threads': lib.num_threads(),
        'gpu_pci': cp.cuda.Device(0).pci_bus_id, 'threadpools': threadpool_info(),
        'sources': _source_provenance(gpu4pyscf, numint.libgdft), 'extensions': extensions,
        'installed_gradient_sha256': hashlib.sha256(Path(rks.__file__).read_bytes()).hexdigest(),
        'candidate_gradient_sha256': hashlib.sha256((args.extension_root / 'gpu4pyscf/grad/rks.py').read_bytes()).hexdigest(),
        'runs': [],
        'boundary': 'Reviewed density-contraction sites adapted into pinned release; full native checkout build unverified',
    }
    if report['openmp_threads'] != 4 or any(pool['num_threads'] > 4 for pool in report['threadpools']):
        raise RuntimeError('Numerical thread pools exceed the resource allowance')
    started = time.perf_counter()
    energy = mixed_precision.run_verified_scf(mf)
    cp.cuda.get_current_stream().synchronize()
    report.update(scf_seconds=time.perf_counter() - started, energy_hartree=energy,
                  precision_info=mf.mixed_precision_info, effective_orbitals=int(mf.mo_coeff.shape[-1]),
                  grid_points=len(mf.grids.weights), nlc_grid_points=len(mf.nlcgrids.weights))
    if not mf.mixed_precision_info['accepted'] or mf.mo_coeff.shape[-1] != mol.nao:
        raise RuntimeError('SCF must pass native verification and retain the full basis')
    dm = mf.make_rdm1()
    reference = None
    for repeat in range(args.repeats):
        modes = ('native', 'contracted') if repeat % 2 == 0 else ('contracted', 'native')
        for mode in modes:
            # Native gradients release DF buffers. Rebuild the same full/range-separated
            # buffers before each timed force call to match the post-SCF memory state.
            mf.get_veff(mol, dm)
            cp.cuda.get_current_stream().synchronize()
            gradient = mf.nuc_grad_method()
            gradient.grid_response = gradient.auxbasis_response = True
            scope = density_response_patch(args.extension_root) if mode == 'contracted' else contextlib.nullcontext()
            with scope:
                started = time.perf_counter()
                values = np.asarray(gradient.kernel())
                cp.cuda.get_current_stream().synchronize()
                seconds = time.perf_counter() - started
            if reference is None:
                if mode != 'native':
                    raise RuntimeError('Native force must establish the reference')
                reference = values.copy()
            error = float(np.max(np.abs(values - reference)))
            if error > 1e-9:
                raise RuntimeError(f'Force agreement failed: {error} Hartree/Bohr')
            row = {'mode': mode, 'repeat': repeat, 'seconds': seconds,
                   'gradient_hartree_bohr': values.tolist(), 'max_error_hartree_bohr': error}
            report['runs'].append(row)
            (artifact / f'{repeat}-{mode}.json').write_text(json.dumps(row, indent=2) + '\n')
            (artifact / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k != 'gradient_hartree_bohr'}), flush=True)
    medians = {mode: statistics.median(row['seconds'] for row in report['runs'] if row['mode'] == mode)
               for mode in ('native', 'contracted')}
    report['summary'] = {'median_seconds': medians, 'speedup': medians['native'] / medians['contracted'],
                         'max_error_hartree_bohr': max(row['max_error_hartree_bohr'] for row in report['runs'])}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
