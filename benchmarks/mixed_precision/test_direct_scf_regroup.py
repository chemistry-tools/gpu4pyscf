"""Validate qualified direct-SCF dispatch and instance-owned optimizer caches."""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = Path(__file__).parents[2]/'gpu4pyscf/scf/_direct_scf.py'
    spec = importlib.util.spec_from_file_location('direct_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Molecule:
    basis = 'def2-tzvpd'
    cart = False
    spin = 0
    omega = 0.
    elements = (1, 6, 7, 8, 9, 16, 17)

    def atom_charges(self):
        return self.elements

    @contextmanager
    def with_range_coulomb(self, omega):
        previous = self.omega
        self.omega = 0. if omega is None else omega
        try:
            yield
        finally:
            self.omega = previous


def _method():
    return SimpleNamespace(mol=_Molecule(), xc='wb97m-v', with_df=None,
                           direct_scf_regroup=True, direct_scf_tol=1e-13,
                           _opt_gpu={None: None}, istype=lambda name: name == 'RKS')


def _cutoff(mf, **kwargs):
    return _module().primitive_cutoff(mf, device_count=kwargs.get('device_count', 1),
                                    capability=kwargs.get('capability', (8, 6)),
                                    mol=kwargs.get('mol'))


def test_qualified_recipe_and_explicit_native_control():
    mf = _method()
    assert _cutoff(mf) == 10.
    mf.direct_scf_regroup = False
    assert _cutoff(mf) == .3


@pytest.mark.parametrize('field, value', [
    ('xc', 'b3lyp'), ('xc', lambda: None), ('with_df', object()),
    ('istype', lambda name: name == 'UKS'),
])
def test_other_methods_keep_native(field, value):
    mf = _method()
    setattr(mf, field, value)
    assert _cutoff(mf) == .3


@pytest.mark.parametrize('field, value', [
    ('basis', '6-31g*'), ('basis', {'C': 'def2-tzvpd'}),
    ('cart', True), ('spin', 1), ('elements', (1, 6, 15)),
])
def test_unqualified_molecules_keep_native(field, value):
    mf = _method()
    setattr(mf.mol, field, value)
    assert _cutoff(mf) == .3


def test_external_molecule_and_low_memory_method_keep_native():
    mf = _method()
    other = _Molecule()
    other.basis = 'sto-3g'
    assert _cutoff(mf, mol=other) == .3
    lowmem = type('LowMemory', (SimpleNamespace,), {'__module__': 'gpu4pyscf.dft.rks_lowmem'})
    assert _cutoff(lowmem(**mf.__dict__)) == .3


def test_periodic_and_legacy_methods_keep_native():
    mf = _method()
    mf.cell = mf.mol
    assert _cutoff(mf) == .3
    del mf.cell
    del mf.direct_scf_regroup
    assert _cutoff(mf) == .3


@pytest.mark.parametrize('alpha,hyb', [(1., .15), (.65, .25), (0., .25), (.65, 0.), (.25, .25)])
@pytest.mark.parametrize('enabled', [False, True])
def test_force_exchange_algebra_and_qualification(alpha, hyb, enabled):
    mf = _method()
    mf.direct_scf_regroup = enabled
    module = _module()
    full, omega, remainder = module.force_exchange(
        mf, .3, alpha, hyb, device_count=1, capability=(8, 6))
    assert (full, omega, remainder) == ((hyb, .3, alpha-hyb) if enabled else (alpha, -.3, hyb-alpha))
    # At any separation, full = SR + LR. Check both independent coefficients.
    if omega > 0:
        assert full == hyb and full+remainder == pytest.approx(alpha)
    else:
        assert full+remainder == pytest.approx(hyb) and full == alpha


@pytest.mark.parametrize('kwargs', [{'device_count': 2, 'capability': (8, 6)},
                                   {'device_count': 1, 'capability': (9, 0)}])
def test_force_exchange_other_devices_keep_original_split(kwargs):
    assert _module().force_exchange(_method(), .3, 1., .15, **kwargs) == (1., -.3, -.85)


def test_global_hybrid_force_and_fitted_control():
    module, mf = _module(), _method()
    assert module.force_exchange(mf, 0., .15, .15, device_count=1, capability=(8, 6)) == (.15, 0., 0.)
    mf.with_df = object()
    assert module.force_exchange(mf, .3, 1., .15, device_count=1, capability=(8, 6)) == (1., -.3, -.85)


@pytest.mark.parametrize('kwargs', [{'device_count': 2}, {'capability': (8, 0)}, {'capability': (9, 0)}])
def test_other_devices_keep_native(kwargs):
    assert _cutoff(_method(), **kwargs) == .3


@pytest.mark.parametrize('value', [None, 0, 1, 'auto', [], {}])
def test_invalid_public_controls_fail(value):
    mf = _method()
    mf.direct_scf_regroup = value
    with pytest.raises(ValueError, match='must be bool'):
        _cutoff(mf)


class _Optimizer:
    def __init__(self, mol, tolerance, *, diffuse_cutoff):
        self.mol, self.direct_scf_tol = mol, tolerance
        self.diffuse_cutoff = diffuse_cutoff
        self.build_omega = mol.omega

    def build(self):
        return self


def _get(mf, omega=.3, **kwargs):
    return _module().get_optimizer(mf, kwargs.get('mol', mf.mol), omega,
                                   optimizer_type=kwargs.get('optimizer_type', _Optimizer),
                                   device_count=1, capability=(8, 6))


@pytest.mark.parametrize('change', ['policy', 'tolerance', 'molecule'])
def test_cache_rebuilds_changed_inputs_and_keeps_other_entries(change):
    mf = _method()
    first = _get(mf)
    assert first.diffuse_cutoff == 10. and first.build_omega == .3
    assert first._direct_scf_regroup is True
    assert _get(mf) is first and mf.mol.omega == 0.
    other = _get(mf, -.3)
    if change == 'policy':
        mf.direct_scf_regroup = False
    elif change == 'tolerance':
        mf.direct_scf_tol = 1e-14
    else:
        mf.mol = _Molecule()
    second = _get(mf)
    assert second is not first
    assert second._direct_scf_regroup is (change != 'policy')
    assert mf._opt_gpu[-.3] is other


def test_scanner_copy_does_not_mutate_source_cache():
    source = _method()
    original = _get(source)
    scanner = SimpleNamespace(**source.__dict__)
    assert scanner._opt_gpu is source._opt_gpu
    scanner.direct_scf_regroup = False
    replacement = _get(scanner)
    assert scanner._opt_gpu is not source._opt_gpu
    assert replacement.diffuse_cutoff == .3
    assert replacement._direct_scf_regroup is False
    assert source._opt_gpu[.3] is original and original.diffuse_cutoff == 10.
    assert original._direct_scf_regroup is True


@pytest.mark.parametrize('qualified', [False, True])
@pytest.mark.parametrize('has_symbol', [False, True])
def test_force_kernel_qualification_and_old_library_fallback(qualified, has_symbol):
    module = _module()
    native, contracted = object(), object()
    library = SimpleNamespace(RYS_per_atom_jk_ip1=native)
    if has_symbol:
        library.RYS_per_atom_jk_ip1_contract = contracted
    opt = SimpleNamespace(_direct_scf_regroup=qualified)
    assert module.force_kernel(opt, library) is (contracted if qualified and has_symbol else native)
    # A matching primitive cutoff alone does not establish MF ownership/eligibility.
    assert module.force_kernel(SimpleNamespace(diffuse_cutoff=10.), library) is native


def test_failed_build_keeps_cache_and_restores_range_context():
    class Failed(_Optimizer):
        def build(self):
            raise RuntimeError('integral build failed')

    mf = _method()
    original_cache = mf._opt_gpu
    with pytest.raises(RuntimeError, match='build failed'):
        _get(mf, optimizer_type=Failed)
    assert mf._opt_gpu is original_cache and .3 not in mf._opt_gpu
    assert mf.mol.omega == 0.


@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
def test_native_instance_isolation_reset_and_coulomb_representation():
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft

    atoms = molecule('CH3COF')
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='def2-tzvpd', verbose=0)
    fast, native = (dft.RKS(mol, xc='wb97m-v') for _ in range(2))
    fast.direct_scf_regroup, native.direct_scf_regroup = True, False
    dm = cp.asarray(fast.get_init_guess(mol, 'minao'))
    omega, alpha, hyb = fast._numint.rsh_and_hybrid_coeff(fast.xc, spin=0)
    k_fast = fast.get_k(mol, dm, 1, omega, alpha, hyb)
    native_cache = native._opt_gpu
    assert fast._opt_gpu[omega].diffuse_cutoff == 10.
    assert native._opt_gpu is native_cache and native_cache == {None: None}
    k_native = native.get_k(mol, dm, 1, omega, alpha, hyb)
    cp.testing.assert_allclose(k_fast, k_native, atol=2e-9, rtol=2e-10)
    native_opt = native._opt_gpu[omega]
    fast.get_j(mol, dm)
    assert cp.all(cp.asarray(fast._opt_jengine[0.].sorted_mol._bas[:, 2]) == 1)
    fast.direct_scf_regroup = False
    fast.get_k(mol, dm, 1, omega, alpha, hyb)
    assert fast._opt_gpu[omega].diffuse_cutoff == .3
    assert native._opt_gpu[omega] is native_opt
    scanner = fast.as_scanner()
    scanner.direct_scf_regroup = True
    scanner.get_k(mol, dm, 1, omega, alpha, hyb)
    assert scanner._opt_gpu is not fast._opt_gpu
    assert fast._opt_gpu[omega].diffuse_cutoff == .3
    shifted = mol.set_geom_(mol.atom_coords()+.001, unit='Bohr', inplace=False)
    scanner.reset(shifted)
    assert scanner._opt_gpu == {None: None} and scanner.direct_scf_regroup is True
    scanner.get_k(shifted, dm, 1, omega, alpha, hyb)
    assert scanner._opt_gpu[omega].mol is shifted


@pytest.mark.skipif(os.environ.get('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
@pytest.mark.parametrize('has_symbol', [False, True])
def test_native_force_dispatch_and_older_library_control(monkeypatch, has_symbol):
    import cupy as cp
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.grad import rhf

    atoms = molecule('CH3COF')
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
                basis='def2-tzvpd', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.direct_scf_regroup = True
    dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
    opt = mf._get_jk_opt(mol, None)
    assert opt._direct_scf_regroup is True
    original = rhf.libvhf_rys.RYS_per_atom_jk_ip1
    contracted = getattr(rhf.libvhf_rys, 'RYS_per_atom_jk_ip1_contract', None)
    if has_symbol:
        assert contracted is not None, 'Rebuild native RYS to test contracted derivatives'
    counts = {'original': 0, 'contracted': 0}

    def wrap(kernel, name):
        def call(*args):
            counts[name] += 1
            return kernel(*args)
        return call

    library = SimpleNamespace(RYS_per_atom_jk_ip1=wrap(original, 'original'))
    if has_symbol:
        library.RYS_per_atom_jk_ip1_contract = wrap(contracted, 'contracted')
    monkeypatch.setattr(rhf, 'libvhf_rys', library)
    actual = rhf._jk_energy_per_atom(opt, dm, j_factor=1., k_factor=.15)
    assert counts['contracted' if has_symbol else 'original'] > 0
    assert counts['original' if has_symbol else 'contracted'] == 0
    # Removing owner qualification must use the original API, even at cutoff 10.
    monkeypatch.setattr(opt, '_direct_scf_regroup', False)
    reference = rhf._jk_energy_per_atom(opt, dm, j_factor=1., k_factor=.15)
    assert counts['original'] > 0
    cp.testing.assert_allclose(actual, reference, atol=2e-8, rtol=0)
