"""Check bounded admission and exact-density native exchange cache reuse."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _module(name='_integral_cache'):
    path = Path(__file__).parents[2]/f'gpu4pyscf/scf/{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('budget', [0, 511, 2048, 64*1024, 256*1024**2])
def test_admission_accounts_for_rectangles_and_metadata(budget):
    module = _module()
    groups = [(0, 6), (0, 1), (1, 3), (2, 1)]
    pairs = {(i, j): 13*(i+1)*(j+1) for i in range(4) for j in range(i+1)}
    plan = module.plan_groups(groups, pairs, budget, 137*4)
    if not plan:
        return
    total = module._allocation_size(137*4)
    for (i, j, k, l), size in plan.items():
        nf = [(groups[x][0]+1)*(groups[x][0]+2)//2 for x in (i, j, k, l)]
        expected = module._allocation_size(pairs[i, j]*pairs[k, l]*np.prod(nf)*8)
        expected += sum(module._allocation_size(pairs[p]*4) for p in ((i, j), (k, l)))
        assert size == expected
        assert np.prod([groups[x][1] for x in (i, j, k, l)]) > 1
        total += size
    assert total <= budget


def test_planner_skips_empty_primitive_and_int32_overflow_groups():
    module = _module()
    assert module.plan_groups([(0, 1)], {(0, 0): 10}, 1024**3, 8) == {}
    assert module.plan_groups([(0, 6)], {(0, 0): 0}, 1024**3, 8) == {}
    assert module.plan_groups([(3, 6)], {(0, 0): 2**30}, 2**63, 8) == {}


def test_planner_prefers_compact_contracted_work():
    module = _module()
    groups = [(0, 6), (1, 1)]
    pairs = {(0, 0): 10, (1, 0): 10, (1, 1): 10}
    full = module.plan_groups(groups, pairs, 1024**3, 12)
    task = (0, 0, 0, 0)
    tight = module.plan_groups(groups, pairs, full[task]+512, 12)
    assert tight == {task: full[task]}


def test_all_mode_includes_primitive_groups_and_accounts_for_every_rectangle():
    module = _module()
    groups = [(0, 6), (1, 1)]
    pairs = {(0, 0): 3, (1, 0): 4, (1, 1): 5}
    plan = module.plan_groups(groups, pairs, None, 80)
    assert len(plan) == 7 and (1, 1, 1, 1) in plan
    bounded = module.plan_groups(groups, pairs, 2**40, 80)
    assert bounded.keys() < plan.keys()
    assert module._plan_size(plan, 80) == sum(plan.values())+512
    with pytest.raises(NotImplementedError, match='int32'):
        module.plan_groups([(3, 1)], {(0, 0): 2**30}, None, 80)


def test_all_mode_capacity_preflight_leaves_room_for_scf():
    module = _module()
    gib = 1024**3
    module._require_capacity(7*gib, 8*gib, 10*gib)
    with pytest.raises(MemoryError, match='requires 8.000 GiB'):
        module._require_capacity(8*gib, 8*gib, 10*gib)
    with pytest.raises(MemoryError, match='bounded'):
        module._require_capacity(1, 256*1024**2, gib)


@pytest.mark.parametrize('value', [None, 0, 1, 'all'])
def test_all_toggle_requires_a_boolean(value):
    from test_direct_scf_regroup import _get, _method

    mf = _method()
    mf.direct_scf_cache_all = value
    with pytest.raises(ValueError, match='must be bool'):
        _get(mf)


def test_all_mode_requires_qualified_representation():
    from test_direct_scf_regroup import _get, _method

    mf = _method()
    mf.direct_scf_cache_all = True
    mf.direct_scf_regroup = False
    with pytest.raises(ValueError, match='qualified'):
        _get(mf)


def test_context_tracks_operator_geometry_basis_tolerance_device_and_stream():
    module = _module()
    mol = SimpleNamespace(_atm=np.zeros((2, 6), dtype=np.int32),
                          _bas=np.zeros((3, 8), dtype=np.int32), _env=np.arange(40.))
    opt = SimpleNamespace(sorted_mol=mol, direct_scf_tol=1e-13)
    operator = (-.3, 1., .15)
    key = module._context_key(opt, operator, 0, 0)
    assert module._context_key(opt, operator, 0, 0) == key
    for array in (mol._atm, mol._bas, mol._env):
        array.flat[-1] += 1
        assert module._context_key(opt, operator, 0, 0) != key
        array.flat[-1] -= 1
    assert module._context_key(opt, operator, 1, 0) != key
    assert module._context_key(opt, operator, 0, 1) != key
    assert module._context_key(opt, (-.3, .7, .15), 0, 0) != key
    opt.direct_scf_tol = 1e-12
    assert module._context_key(opt, operator, 0, 0) != key


@pytest.mark.parametrize('value', [-1, True, 1.5, None, '256'])
def test_invalid_budget_is_rejected(value):
    from test_direct_scf_regroup import _get, _method

    mf = _method()
    mf.direct_scf_cache_bytes = value
    with pytest.raises(ValueError, match='nonnegative integer'):
        _get(mf)


def test_budget_and_inplace_geometry_changes_rebuild_only_owner(monkeypatch):
    from test_direct_scf_regroup import _method, _Optimizer

    module, mf = _module('_direct_scf'), _method()
    monkeypatch.setattr(module, '_cache_source_key', lambda mol: mol.geometry)
    mf.mol.geometry = 0
    mf.direct_scf_cache_bytes = 256*1024**2

    def get(method):
        return module.get_optimizer(method, method.mol, .3, optimizer_type=_Optimizer,
                                    device_count=1, capability=(8, 6))

    first = get(mf)
    assert get(mf) is first and first._direct_scf_cache_bytes == 256*1024**2
    scanner = SimpleNamespace(**mf.__dict__)
    scanner.direct_scf_cache_bytes //= 2
    assert get(scanner) is not first and mf._opt_gpu[.3] is first
    scanner.direct_scf_cache_all = True
    full = get(scanner)
    assert full._direct_scf_cache_all is True and get(mf) is first
    mf.mol.geometry += 1
    assert get(mf) is not first
    mf.direct_scf_regroup = False
    assert get(mf)._direct_scf_cache_bytes == 0


@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
@pytest.mark.parametrize('name', ['CH3COF', 'CH3CONH2', 'C5H5N', 'C2H6SO', 'CH3COCl'])
def test_native_mixed_organic_reuse_with_growing_and_signed_densities(name):
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto
    from pyscf.scf import hf as cpu_hf

    from gpu4pyscf import dft
    from gpu4pyscf.scf import jk

    assert hasattr(jk.libvhf_rys, 'RYS_fill_int4c2e_k'), 'Rebuild native RYS for cache tests'
    atoms = molecule(name)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='def2-tzvpd', verbose=0)
    fast, direct = (dft.RKS(mol, xc='wb97m-v') for _ in range(2))
    fast.direct_scf_regroup = direct.direct_scf_regroup = True
    fast.direct_scf_cache_bytes = 32*1024**2
    dm = cp.asarray(direct.get_init_guess(mol, 'minao'))
    omega, alpha, hyb = fast._numint.rsh_and_hybrid_coeff(fast.xc, spin=0)

    def build(method, density, lr=alpha, sr=hyb, hermi=1):
        return method.get_k(mol, density, hermi, omega, lr, sr)

    # Build the geometry-only cache with a zero density, then activate every entry.
    cp.testing.assert_array_equal(build(fast, dm*0), cp.zeros_like(dm))
    cache = fast._opt_gpu[omega]._integral_cache
    assert cache.builds > 0 and cache.used_bytes <= fast.direct_scf_cache_bytes
    original_entries = dict(cache.entries)
    for density in (dm, -.001*dm, 2.*dm):
        actual = build(fast, density)
        cp.testing.assert_allclose(actual, build(direct, density), atol=2e-9, rtol=2e-10)
        assert fast._opt_gpu[omega]._integral_cache is cache
    assert cache.hits >= 3*cache.builds
    assert all(cache.entries[key][0] is entry[0] for key, entry in original_entries.items())
    cpu_dm = cp.asnumpy(dm)
    full = cpu_hf.get_jk(mol, cpu_dm, hermi=1, with_j=False)[1]
    lr = cpu_hf.get_jk(mol, cpu_dm, hermi=1, with_j=False, omega=omega)[1]
    cp.testing.assert_allclose(build(fast, dm), cp.asarray(hyb*full+(alpha-hyb)*lr),
                              atol=2e-9, rtol=2e-10)
    cp.testing.assert_allclose(build(fast, dm, .7, .2), build(direct, dm, .7, .2),
                              atol=2e-9, rtol=2e-10)
    assert fast._opt_gpu[omega]._integral_cache is not cache
    # Unsupported density symmetry must use the native path.
    nonsymmetric = dm+cp.tril(dm)*.01
    cp.testing.assert_allclose(build(fast, nonsymmetric, hermi=0),
                              build(direct, nonsymmetric, hermi=0), atol=2e-9, rtol=2e-10)
    fast.reset(mol)
    assert fast._opt_gpu == {None: None}


@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
def test_cache_allocation_miss_recomputes_and_old_native_library_falls_back(monkeypatch):
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.scf import _integral_cache, jk

    atoms = molecule('CH3COF')
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='def2-tzvpd', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.direct_scf_regroup = True
    dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)
    expected = mf.get_k(mol, dm, 1, omega, alpha, hyb)
    mf.direct_scf_cache_bytes = 16*1024**2
    monkeypatch.setattr(_integral_cache._IntegralCache, '_build', lambda *args: None)
    actual = mf.get_k(mol, dm, 1, omega, alpha, hyb)
    cp.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)
    opt = mf._opt_gpu[omega]
    assert opt._integral_cache.builds == 0 and opt._integral_cache.misses > 0
    assert _integral_cache.get_cache(opt, dm[None], 1, (-omega, alpha, hyb),
                                     opt.bas_pair_cache, SimpleNamespace(), 1) is None
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        assert _integral_cache.get_cache(opt, dm[None], 1, (-omega, alpha, hyb),
                                         opt.bas_pair_cache, jk.libvhf_rys, 1) is None


@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
def test_all_vram_exchange_scf_forces_and_capacity_failure(monkeypatch):
    import cupy as cp
    from pyscf import gto
    from pyscf.scf import hf as cpu_hf

    from gpu4pyscf import dft
    from gpu4pyscf.scf import hf

    hf.overlap_zero_eigenvalue_threshold = 1e-10
    mol = gto.M(atom='F 0 0 0; H 0 0 .92', basis='def2-tzvpd', verbose=0)
    mf, direct = (dft.RKS(mol, xc='wb97m-v') for _ in range(2))
    for method in (mf, direct):
        method.direct_scf_regroup = True
        method.grids.level, method.nlcgrids.level = 5, 3
        method.conv_tol, method.conv_tol_grad = 1e-8, 1e-4
    mf.direct_scf_cache_all = True
    dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)
    actual = mf.get_k(mol, dm, 1, omega, alpha, hyb)
    cache = mf._opt_gpu[omega]._integral_cache
    stats = cache.statistics()
    assert stats['mode'] == 'all' and stats['all_groups_resident']
    assert stats['groups'] == stats['planned_groups'] > 0
    assert stats['owned_gpu_bytes'] <= stats['planned_gpu_bytes']
    groups = mf._opt_gpu[omega].sorted_mol.uniq_l_ctr
    assert any(all(groups[x, 1] == 1 for x in task) for task in cache.entries)
    cpu_dm = cp.asnumpy(dm)
    full = cpu_hf.get_jk(mol, cpu_dm, hermi=1, with_j=False)[1]
    lr = cpu_hf.get_jk(mol, cpu_dm, hermi=1, with_j=False, omega=omega)[1]
    cp.testing.assert_allclose(actual, cp.asarray(hyb*full+(alpha-hyb)*lr), atol=2e-9, rtol=2e-10)
    cp.testing.assert_allclose(actual, direct.get_k(mol, dm, 1, omega, alpha, hyb),
                              atol=2e-9, rtol=2e-10)
    energy = mf.kernel(dm0=dm)
    assert mf.converged and cache.statistics()['hits'] > 0
    grad = mf.Gradients()
    grad.grid_response = True
    actual_gradient = grad.kernel()
    reference_energy = direct.kernel(dm0=dm)
    assert direct.converged and abs(energy-reference_energy) < 2e-8
    grad = direct.Gradients()
    grad.grid_response = True
    cp.testing.assert_allclose(actual_gradient, grad.kernel(), atol=2e-8, rtol=0)
    mf._opt_gpu[omega]._integral_cache = None
    monkeypatch.setattr(cp.cuda.runtime, 'memGetInfo', lambda: (0, 10*1024**3))
    with pytest.raises(MemoryError, match='All-VRAM integral cache requires'):
        mf.get_k(mol, dm, 1, omega, alpha, hyb)
    assert mf._opt_gpu[omega]._integral_cache is None
