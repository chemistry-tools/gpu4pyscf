"""Compare native direct and fitted fixed-density range-separated J/K builds."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import resource
import statistics
import time
from pathlib import Path

from gpu_performance import CASES, _resources, _settings


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('direct', 'df'), required=True)
    parser.add_argument('--case', default='benzene')
    parser.add_argument('--geometry-report', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--require-gpu-source', type=Path, required=True)
    parser.add_argument('--cpus', default='4,5,6,7')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--verify-cpu', action='store_true')
    parser.add_argument('--exchange', choices=('combined', 'split'), default='combined')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--regroup', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.geometry_report is None and args.state is not None:
        parser.error('Positive repeats and matching metadata for external states are required')
    if args.regroup is not None and args.backend != 'direct':
        parser.error('Regrouping controls are supported only for direct J/K')
    _resources([int(x) for x in args.cpus.split(',')])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Use physical GPU 1')
    args.output.mkdir(parents=True, exist_ok=False)
    settings, protocol_hash = _settings()

    import cupy as cp
    import numpy as np
    from pyscf import gto, lib
    from pyscf.scf import hf as cpu_hf
    from threadpoolctl import threadpool_info

    import gpu4pyscf
    from gpu4pyscf import dft
    from gpu4pyscf.lib.cupy_helper import tag_array
    from gpu4pyscf.scf import hf, jk

    package = Path(gpu4pyscf.__file__).absolute().parent
    if package != args.require_gpu_source.absolute():
        raise RuntimeError(f'Wrong native installation: {package}')
    if cp.cuda.runtime.getDeviceCount() != 1 or not cp.cuda.Device().pci_bus_id.lower().endswith('05:00.0'):
        raise RuntimeError('Unexpected benchmark GPU identity')
    lib.num_threads(4)
    if any(p['num_threads'] > 4 for p in threadpool_info()):
        raise RuntimeError('Thread budget exceeded')
    if args.geometry_report:
        metadata = json.loads(args.geometry_report.read_text())
        if metadata['basis'] != settings['basis'] or metadata['functional'] != settings['xc']:
            raise ValueError('External state recipe differs from the pinned benchmark')
        atom, charge, multiplicity = (metadata[k] for k in ('geometry_angstrom', 'charge', 'multiplicity'))
    else:
        from _organic_cases import ORGANIC_CASES
        atom, charge, multiplicity = (CASES | ORGANIC_CASES)[args.case]
    if multiplicity != 1:
        raise ValueError('This fixed-density pilot currently supports restricted singlets')
    mol = gto.M(atom=atom, basis=settings['basis'], charge=charge, spin=multiplicity-1,
                cart=False, verbose=6 if args.profile else 4, output=str(args.output/'pyscf.log'))
    hf.overlap_zero_eigenvalue_threshold = 1e-10
    mf = dft.RKS(mol, xc=settings['xc'])
    mf.chkfile = None
    mf.direct_scf_tol = 1e-13
    if args.regroup is not None:
        mf.direct_scf_regroup = args.regroup
    if args.backend == 'df':
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    if args.state:
        with np.load(args.state, allow_pickle=False) as state:
            coeff, occ = cp.asarray(state['mo_coeff']), cp.asarray(state['mo_occ'])
        if coeff.shape != (mol.nao, mol.nao) or occ.shape != (mol.nao,):
            raise ValueError('Saved state has a different basis size')
        overlap = cp.asarray(mol.intor_symmetric('int1e_ovlp'))
        if float(cp.max(cp.abs(coeff.T @ overlap @ coeff-cp.eye(mol.nao)))) > 1e-6:
            raise ValueError('Saved orbitals do not match this overlap matrix')
        density = tag_array((coeff*occ) @ coeff.T, mo_coeff=coeff, mo_occ=occ)
    else:
        density = mf.get_init_guess(mol, '1e')
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    samples = []

    def build(dm, label):
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        j = mf.get_j(mol, dm, hermi=1)
        cp.cuda.runtime.deviceSynchronize()
        after_j = time.perf_counter()
        if args.backend == 'df' or args.exchange == 'split':
            # Match the default DF get_veff route: full exchange plus positive-omega LR.
            k = mf.get_k(mol, dm, hermi=1)
            k *= hyb
            k += (alpha-hyb)*mf.get_k(mol, dm, hermi=1, omega=abs(omega))
        else:
            k = mf.get_k(mol, dm, hermi=1, omega=omega, lr_factor=alpha, sr_factor=hyb)
        cp.cuda.runtime.deviceSynchronize()
        row = {'label': label, 'j_seconds': after_j-started,
                   'k_seconds': time.perf_counter()-after_j}
        row['total_seconds'] = row['j_seconds']+row['k_seconds']
        samples.append(row)
        print(json.dumps(row), flush=True)
        return j, k

    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    j, k = build(density, 'cold-full')
    for _ in range(args.repeats):
        j, k = build(density, 'warm-full')
    delta_result = None
    if args.backend == 'direct':
        delta = .001*(cp.asarray(mf.get_init_guess(mol, 'minao'))-density)
        if float(cp.linalg.norm(delta)) == 0:
            raise ValueError('The density-difference test must be nonzero')
        for _ in range(args.repeats):
            dj, dk = build(delta, 'warm-delta')
        full_j, full_k = build(density+delta, 'increment-check-full')
        delta_result = {'j_max_abs': float(cp.max(cp.abs(full_j-j-dj))),
                            'k_max_abs': float(cp.max(cp.abs(full_k-k-dk)))}
        cp.testing.assert_allclose(full_j, j+dj, atol=2e-9, rtol=2e-10)
        cp.testing.assert_allclose(full_k, k+dk, atol=2e-9, rtol=2e-10)
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.output/'jk.prof'))
    reference = None
    if args.verify_cpu:
        if args.backend != 'direct':
            raise ValueError('CPU oracle checks native direct kernels, not the fitting approximation')
        cpu_dm = cp.asnumpy(density)
        cpu_j, cpu_k = cpu_hf.get_jk(mol, cpu_dm, hermi=1)
        _, cpu_lr = cpu_hf.get_jk(mol, cpu_dm, hermi=1, with_j=False, omega=omega)
        cpu_k = hyb*cpu_k+(alpha-hyb)*cpu_lr
        cpu_j, cpu_k = cp.asarray(cpu_j), cp.asarray(cpu_k)
        reference = {'j_max_abs': float(cp.max(cp.abs(j-cpu_j))),
                         'k_max_abs': float(cp.max(cp.abs(k-cpu_k)))}
        cp.testing.assert_allclose(j, cpu_j, atol=2e-9, rtol=2e-10)
        cp.testing.assert_allclose(k, cpu_k, atol=2e-9, rtol=2e-10)
    np.savez(args.output/'matrices.npz', density=cp.asnumpy(density), j=cp.asnumpy(j), k=cp.asnumpy(k))
    cderi_by_operator = {}
    dfobj = getattr(mf, 'with_df', None)
    if dfobj is not None:
        for label, obj in {'full': dfobj, **dfobj._rsh_df}.items():
            cderi = getattr(obj, '_cderi', None)
            if isinstance(cderi, dict):
                cderi_by_operator[label] = sum(x.nbytes for x in cderi.values())
            elif isinstance(cderi, (list, tuple)):
                cderi_by_operator[label] = sum(x.nbytes for x in cderi)
            elif hasattr(cderi, 'nbytes'):
                cderi_by_operator[label] = cderi.nbytes
    cderi_bytes = sum(cderi_by_operator.values())
    medians = {label: statistics.median(s['total_seconds'] for s in samples if s['label'] == label)
               for label in ('warm-full', 'warm-delta') if any(s['label'] == label for s in samples)}
    sources = ['scf/hf.py', 'scf/jk.py', 'scf/j_engine.py', 'scf/_direct_scf.py',
               'df/df.py', 'df/df_jk.py', 'dft/rks.py']
    report = {'case': args.case, 'backend': args.backend, 'basis': mol.basis, 'cart': mol.cart,
                  'exchange_assembly': args.exchange if args.backend == 'direct' else 'split',
                  'profiled': args.profile,
                  'direct_scf_regroup': mf.direct_scf_regroup,
                  'internal_basis': {str(key): {'nao': opt.sorted_mol.nao, 'nbas': opt.sorted_mol.nbas,
                                                'diffuse_cutoff': opt.diffuse_cutoff,
                                                'groups': opt.sorted_mol.uniq_l_ctr.tolist(),
                                                'group_counts': opt.sorted_mol.l_ctr_counts.tolist()}
                                     for key, opt in mf._opt_gpu.items() if opt is not None},
                  'geometry_angstrom': atom, 'charge': charge, 'multiplicity': multiplicity, 'nao': mol.nao,
                  'functional': mf.xc, 'omega': omega, 'long_range_fraction': alpha, 'short_range_fraction': hyb,
                  'direct_scf_tol': mf.direct_scf_tol, 'protocol_sha256': protocol_hash,
                  'scope': 'Fixed-density J/K only; no SCF, XC, VV10 or forces timed',
                  'density_source': 'saved orbitals' if args.state else 'one-electron Hamiltonian orbitals',
                  'density_sha256': hashlib.sha256(cp.asnumpy(density).tobytes()).hexdigest(),
                  'geometry_report_sha256': _hash(args.geometry_report) if args.geometry_report else None,
                  'state_sha256': _hash(args.state) if args.state else None,
                  'jk_energy_hartree': float(cp.einsum('ij,ji', density, .5*j-.25*k)),
                  'samples': samples, 'medians_seconds': medians, 'incremental_linearity': delta_result,
                  'independent_cpu_oracle': reference, 'retained_cderi_bytes': int(cderi_bytes),
                  'retained_cderi_by_operator': cderi_by_operator,
                  'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threadpools': threadpool_info(),
                  'gpu_pci': cp.cuda.Device().pci_bus_id, 'imported_package': str(package),
                  'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'gpu_pool_used_bytes': cp.get_default_memory_pool().used_bytes(),
                  'gpu_pool_reserved_bytes': cp.get_default_memory_pool().total_bytes(),
                  'source_sha256': {name: _hash(package/name) for name in sources},
                  'native_library_sha256': {p.name: _hash(p) for p in package.glob('lib/*.so')},
                  'benchmark_sha256': _hash(__file__)}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
