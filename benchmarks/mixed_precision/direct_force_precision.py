"""Replay full forces at an identical retained density with the mixed direct CUDA pilot."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median

from gpu_performance import _resources, _settings


@contextmanager
def _mixed_force_dispatch(log_cutoff, counts):
    from gpu4pyscf.grad import rhf

    native = rhf.force_kernel
    counter = ctypes.c_void_p(counts.data.ptr if counts is not None else None)
    calls = []

    def select(opt, library):
        kernel = native(opt, library)
        if kernel.__name__ != 'RYS_per_atom_jk_ip1_contract':
            return kernel
        mixed = library.RYS_per_atom_jk_ip1_mixed
        mixed.restype = ctypes.c_int

        def launch(*args):
            calls.append(1)
            return mixed(*args, ctypes.c_float(log_cutoff), counter)

        return launch

    rhf.force_kernel = select
    try:
        yield calls
    finally:
        rhf.force_kernel = native


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-report', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--require-gpu-source', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--cutoffs', type=float, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--count-quartets', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.repeats < 1 or len(set(args.cutoffs)) != len(args.cutoffs)
            or any(not math.isfinite(x) or not 0 < x <= 1 for x in args.cutoffs)):
        parser.error('Use positive repeats and unique finite contribution cutoffs in (0, 1]')
    args.output.mkdir(exist_ok=False)
    _resources([int(cpu) for cpu in args.cpus.split(',')])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise RuntimeError('Unexpected physical GPU selection')

    import cupy as cp
    import gpu4pyscf
    import numpy as np
    from gpu4pyscf import dft
    from gpu4pyscf.dft.rks import initialize_grids
    from gpu4pyscf.grad import rhf
    from gpu4pyscf.scf import hf
    from gpu4pyscf.scf._direct_scf import REGROUPED_PRIMITIVE_CUTOFF
    from pyscf import gto, lib
    from threadpoolctl import threadpool_info

    package = Path(gpu4pyscf.__file__).absolute().parent
    if (package != args.require_gpu_source.absolute() or cp.cuda.runtime.getDeviceCount() != 1
            or cp.cuda.runtime.is_hip or cp.cuda.Device().compute_capability != '86'):
        raise RuntimeError('Select the matching single-device CUDA 8.6 build')
    if not hasattr(rhf.libvhf_rys, 'RYS_per_atom_jk_ip1_mixed'):
        raise RuntimeError('Native library lacks the experimental gradient entrypoint')
    gpu = subprocess.check_output(
        ['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid,pci.bus_id,name,power.limit',
         '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert cp.cuda.Device().pci_bus_id.lower()[-7:] == gpu[1].strip().lower()[-7:]
    lib.num_threads(4)
    if any(pool['num_threads'] > 4 for pool in threadpool_info()):
        raise RuntimeError('Thread budget exceeded')
    settings, protocol_hash = _settings()
    geometry = json.loads(args.geometry_report.read_text())
    if (geometry['functional'] != settings['xc'] or geometry['basis'] != settings['basis']
            or geometry['multiplicity'] != 1 or geometry.get('cart', False)):
        raise ValueError('Retained state must match the restricted spherical benchmark recipe')
    hf.overlap_zero_eigenvalue_threshold = 1e-10
    mol = gto.M(atom=geometry['geometry_angstrom'], basis=geometry['basis'], charge=geometry['charge'],
                spin=0, cart=False, verbose=4, output=str(args.output/'pyscf.log'))
    mf = dft.RKS(mol, xc=settings['xc'])
    mf.direct_scf_regroup = True
    mf.direct_scf_tol = 1e-13
    mf.grids.level, mf.nlcgrids.level = settings['grid_level'], settings['nlc_grid_level']
    mf.chkfile = None
    with np.load(args.state, allow_pickle=False) as data:
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = (cp.asarray(data[key]) for key in ('mo_coeff', 'mo_occ', 'mo_energy'))
    overlap = cp.asarray(mol.intor_symmetric('int1e_ovlp'))
    if (mf.mo_coeff.shape != (mol.nao, mol.nao) or mf.mo_occ.shape != (mol.nao,)
            or float(cp.max(cp.abs(mf.mo_coeff.T @ overlap @ mf.mo_coeff - cp.eye(mol.nao)))) > 1e-6):
        raise ValueError('Retained orbitals do not match the molecular overlap')
    dm = mf.make_rdm1()
    initialize_grids(mf, mol, dm)
    omega = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)[0]
    for component in (0., omega):
        opt = mf._get_jk_opt(mol, component)
        if opt.diffuse_cutoff != REGROUPED_PRIMITIVE_CUTOFF:
            raise RuntimeError('Sample does not activate the regrouped direct derivative')

    def evaluate(cutoff, *, count=False):
        counters = cp.zeros(2, dtype=cp.uint64) if count else None
        gradient = mf.nuc_grad_method()
        gradient.grid_response = True
        cp.cuda.runtime.deviceSynchronize()
        begin = time.perf_counter()
        if cutoff is None:
            result = gradient.kernel()
        else:
            with _mixed_force_dispatch(math.log(cutoff), counters) as calls:
                result = gradient.kernel()
            if not calls:
                raise RuntimeError('Mixed interface was not called')
        cp.cuda.runtime.deviceSynchronize()
        seconds = time.perf_counter() - begin
        result = cp.asnumpy(result)
        if result.shape != (mol.natm, 3) or not np.isfinite(result).all():
            raise RuntimeError(f'Invalid gradient at cutoff {cutoff}')
        return result, seconds, cp.asnumpy(counters).tolist() if count else None

    reference, first, _ = evaluate(None)
    base_seconds = [first]
    modes = {cutoff: {'seconds': [], 'max_error_hartree_per_bohr': [], 'counts': None} for cutoff in args.cutoffs}
    print(json.dumps({'baseline_seconds': first, 'case': geometry.get('case'), 'nao': mol.nao}), flush=True)
    for repeat in range(args.repeats):
        for cutoff in args.cutoffs if repeat % 2 == 0 else reversed(args.cutoffs):
            actual, seconds, counts = evaluate(cutoff, count=args.count_quartets and repeat == 0)
            mode = modes[cutoff]
            mode['seconds'].append(seconds)
            mode['max_error_hartree_per_bohr'].append(float(np.max(np.abs(actual - reference))))
            if counts is not None:
                mode['counts'] = dict(zip(('strong', 'weak'), counts))
            print(json.dumps({'cutoff': cutoff, **mode}), flush=True)
        if repeat + 1 < args.repeats:
            later, seconds, _ = evaluate(None)
            np.testing.assert_allclose(later, reference, atol=2e-8, rtol=0)
            base_seconds.append(seconds)
    for row in modes.values():
        row['median_seconds'] = median(row['seconds'])
        row['passes_same_state_tolerance'] = max(row['max_error_hartree_per_bohr']) <= 2e-8
    report = {'case': geometry.get('case'), 'nao': mol.nao, 'natm': mol.natm, 'functional': mf.xc,
              'basis': mol.basis, 'grid_levels': [mf.grids.level, mf.nlcgrids.level],
              'direct_scf_tol': mf.direct_scf_tol, 'protocol_sha256': protocol_hash,
              'geometry_sha256': _hash(args.geometry_report), 'state_sha256': _hash(args.state),
              'native_library_sha256': _hash(Path(rhf.libvhf_rys._name)),
              'gpu_uuid': gpu[0].strip(), 'gpu': args.gpu, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'baseline_seconds': base_seconds, 'baseline_median_seconds': median(base_seconds),
              'cutoffs': modes, 'force_tolerance_hartree_per_bohr': 2e-8,
              'timing_scope': 'Same-state full analytic force with prepared ordinary and VV10 grids and native optimizers.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
