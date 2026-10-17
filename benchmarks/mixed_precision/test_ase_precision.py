"""CPU-only checks for precision verification, fallback and ASE lifecycle."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def precision(monkeypatch):
    module = _load('precision_under_test', 'gpu4pyscf/dft/mixed_precision.py')
    grid = SimpleNamespace(use_float32=True, float32_calls=3, float64_calls=4)
    vv10 = SimpleNamespace(use_float32=True, precisions=['float32'])

    @contextmanager
    def context(mf):
        grid.use_float32 = vv10.use_float32 = True
        yield grid, vv10

    monkeypatch.setattr(module, '_precision_context', context)
    return module, grid, vv10


class _MeanField:
    device = 'gpu'
    xc = 'wb97m-v'
    conv_tol = 1e-10
    conv_tol_grad = None
    check_convergence = None
    max_cycle = 100
    converged = False
    callback = None
    e_tot = -1.0
    _numint = SimpleNamespace(_xc_type=lambda xc: 'GGA',
                             nr_rks=SimpleNamespace(__globals__={'_dot_ao_ao_dense': None}))

    def __init__(self):
        self.calls = []
        self.dm = object()

    def do_nlc(self):
        return True

    def make_rdm1(self):
        return self.dm

    def kernel(self, dm0=None):
        self.calls.append((dm0, self.max_cycle))
        for cycle in range(min(3, self.max_cycle)):
            self.callback({'cycle': cycle, 'e_tot': self.e_tot, 'norm_gorb': 1e-7})
        self.converged = True
        return self.e_tot


def _check(passed, energy=-1.000000000001):
    return {
        'passed': passed,
        'float64_energy_hartree': energy,
        'energy_change_hartree': 1e-12,
        'float64_orbital_gradient_norm': 1e-7,
    }


class _Exchange:
    float32_calls = 8
    float64_calls = 4
    int8_calls = 0
    native_only = False
    projection_float32 = False

    def finish_approximation(self):
        self.native_only = True
        self.projection_float32 = False


def test_verified_energy_updates_method_and_restores_callback(precision, monkeypatch):
    module, grid, vv10 = precision
    mf = _MeanField()
    observed = []
    callback = observed.append
    mf.callback = callback
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(True))
    assert module.run_verified_scf(mf) == -1.000000000001
    assert mf.e_tot == -1.000000000001 and mf.converged
    assert len(observed) == 3 and mf.callback is callback
    assert mf.max_cycle == 100 and mf.cycles == 3
    assert not grid.use_float32 and not vv10.use_float32
    assert mf.mixed_precision_info['accepted']
    assert not mf.mixed_precision_info['fallback_scf']


def test_gpu_scanner_accepts_gpu_bases_despite_cpu_device_label(precision, monkeypatch):
    module, _, _ = precision
    gpu_type = type('GPUSCF', (_MeanField,), {'__module__': 'gpu4pyscf.scf.hf'})
    scanner_type = type('Scanner', (gpu_type,), {'__module__': 'pyscf.scf.hf', 'device': 'cpu'})
    mf = scanner_type()
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(True))
    module.run_verified_scf(mf)
    assert mf.mixed_precision_info['accepted']


def test_failed_verification_restarts_density_with_remaining_budget(precision, monkeypatch):
    module, _, vv10 = precision
    mf = _MeanField()
    checks = iter((_check(False), _check(True)))
    monkeypatch.setattr(module, '_full_check', lambda *args: next(checks))
    module.run_verified_scf(mf)
    assert mf.calls == [(None, 20), (mf.dm, 97)]
    assert not vv10.use_float32
    assert mf.mixed_precision_info['fallback_scf']
    assert [row['phase'] for row in mf.mixed_precision_info['cycles']] == ['mixed'] * 3 + ['float64'] * 3


def test_float64_failure_stops_and_restores_settings(precision, monkeypatch):
    module, _, _ = precision
    mf = _MeanField()
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(False))
    with pytest.raises(RuntimeError, match='Full float64 SCF failed'):
        module.run_verified_scf(mf)
    assert not mf.converged and mf.max_cycle == 100 and mf.callback is None
    assert not mf.mixed_precision_info['accepted']


def test_exhausted_total_budget_never_restarts(precision, monkeypatch):
    module, _, _ = precision
    mf = _MeanField()
    mf.max_cycle = 1
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(False))
    with pytest.raises(RuntimeError, match='budget is exhausted'):
        module.run_verified_scf(mf)
    assert len(mf.calls) == 1 and mf.max_cycle == 1 and not mf.converged


def test_kernel_exception_restores_settings(precision):
    module, _, _ = precision
    mf = _MeanField()

    def fail():
        raise ArithmeticError('SCF failure')

    with pytest.raises(ArithmeticError):
        module.run_verified_scf(mf, fail)
    assert mf.callback is None and mf.max_cycle == 100


def _dispatch_contract(x):
    return x + 1


def _dispatch_inner(x):
    return _dispatch_contract(x)


def _dispatch_outer(x):
    return _dispatch_inner(x)


def test_private_dispatch_leaves_original_functions_untouched():
    module = _load('precision_under_test', 'gpu4pyscf/dft/mixed_precision.py')
    first = module._private_functions(globals(), {'_dispatch_contract': lambda x: x + 10})
    second = module._private_functions(globals(), {'_dispatch_contract': lambda x: x + 20})
    assert _dispatch_outer(1) == 2
    assert first['_dispatch_outer'](1) == 11 and second['_dispatch_outer'](1) == 21


def test_tau_dispatch_preserves_early_float32_and_forwards_buffers():
    module = _load('grid_under_test', 'gpu4pyscf/dft/_mixed_grid.py')
    controller = SimpleNamespace(use_float32=True)
    calls = []

    def product(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name
        return call

    tau_dot = module._make_tau_dot(product('symmetric'), product('full'), controller)
    fields, weights, buffer, out = (object() for _ in range(4))
    assert tau_dot(fields, fields, weights, buf=buffer, out=out) == 'full'
    controller.use_float32 = False
    assert tau_dot(fields, fields, weights, buf=buffer, out=out) == 'symmetric'
    assert calls == [
        (name, (fields, fields, weights), {'buf': buffer, 'out': out}) for name in ('full', 'symmetric')
    ]


def test_lda_gga_product_dispatch_preserves_native_float64_and_output():
    module = _load('grid_under_test', 'gpu4pyscf/dft/_mixed_grid.py')
    controller = SimpleNamespace(use_float32=True)
    calls = []

    def native(bra, ket, out=None):
        calls.append(('native', bra, ket, out))
        return out

    def contract(pattern, bra, ket, out=None):
        calls.append((pattern, bra, ket, out))
        return out

    bra, ket, out = object(), object(), object()
    product = module._make_ao_dot(native, contract, controller)
    assert product(bra, ket, out=out) is out
    controller.use_float32 = False
    assert product(bra, ket, out=out) is out
    assert calls == [('ig,jg->ij', bra, ket, out), ('native', bra, ket, out)]


def test_non_vv10_schedule_switches_grid_products_and_checks_native_fields(precision, monkeypatch):
    module, grid, _ = precision
    mf = _MeanField()
    mf.xc, mf.do_nlc = 'b3lyp', lambda: False
    observed = []

    def kernel(dm0=None):
        for cycle in range(8):
            observed.append(grid.use_float32)
            mf.callback({'cycle': cycle, 'e_tot': mf.e_tot, 'norm_gorb': 1e-7})
        mf.converged = True
        return mf.e_tot

    def check(*args):
        assert not grid.use_float32
        return _check(True)

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', check)
    assert module.run_verified_scf(mf) == -1.000000000001
    assert observed == [True] * 5 + [False] * 3
    assert [row['phase'] for row in mf.mixed_precision_info['cycles']] == ['mixed'] * 5 + ['float64'] * 3
    assert mf.mixed_precision_info['accepted']


def test_non_vv10_failed_verification_recovers_from_current_density(precision, monkeypatch):
    module, _, _ = precision
    mf = _MeanField()
    mf.do_nlc = lambda: False
    checks = iter((_check(False), _check(True)))
    monkeypatch.setattr(module, '_full_check', lambda *args: next(checks))
    module.run_verified_scf(mf)
    assert mf.calls == [(None, 20), (mf.dm, 97)]
    assert mf.mixed_precision_info['fallback_scf'] and mf.mixed_precision_info['accepted']


@pytest.mark.parametrize('energy_cycle,gradient_cycle,switch', [
    (2, 2, 5), (2, 8, 8), (10, 2, 10), (None, 2, 12), (2, None, 12),
])
def test_adaptive_exchange_schedule_waits_for_both_residuals_and_is_bounded(
    precision, monkeypatch, energy_cycle, gradient_cycle, switch
):
    module, grid, _ = precision
    grid._df_exchange = _Exchange()
    mf = _MeanField()
    mf.xc, mf.do_nlc = 'b3lyp', lambda: False
    observed = []

    def kernel(dm0=None):
        for cycle in range(16):
            observed.append(grid.use_float32)
            energy_settled = energy_cycle is not None and cycle+1 >= energy_cycle
            gradient_settled = gradient_cycle is not None and cycle+1 >= gradient_cycle
            energy = mf.e_tot if energy_settled else mf.e_tot+.001
            mf.callback({'cycle': cycle, 'e_tot': energy, 'last_hf_e': mf.e_tot,
                         'norm_gorb': 1e-5 if gradient_settled else .1})
        mf.converged = True
        return mf.e_tot

    def check(*args):
        assert not grid.use_float32
        assert grid._df_exchange.native_only
        return _check(True)

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', check)
    module.run_verified_scf(mf)
    assert observed == [True]*switch+[False]*(16-switch)
    assert mf.mixed_precision_info['grid_precision_schedule'] == 'adaptive_hybrid_gga'
    assert mf.mixed_precision_info['accepted']


@pytest.mark.parametrize('adaptive_grid', [True, False])
def test_adaptive_failed_verification_keeps_recovery_float64_with_small_budget(precision, monkeypatch, adaptive_grid):
    module, grid, vv10 = precision
    grid._df_exchange = _Exchange()
    grid._df_exchange.adaptive_grid = adaptive_grid
    mf = _MeanField()
    mf.xc, mf.do_nlc, mf.max_cycle = ('b3lyp' if adaptive_grid else 'wb97m-v'), lambda: not adaptive_grid, 10
    observed, projections, checks = [], [], iter((_check(False), _check(True)))

    def kernel(dm0=None):
        mf.calls.append((dm0, mf.max_cycle))
        for cycle in range(mf.max_cycle):
            observed.append(grid.use_float32)
            projections.append(grid._df_exchange.projection_float32)
            mf.callback({'cycle': cycle, 'e_tot': mf.e_tot, 'last_hf_e': mf.e_tot+1,
                         'norm_gorb': .1})
        mf.converged = True
        return mf.e_tot

    def check(*args):
        assert not grid.use_float32
        assert not vv10.use_float32
        assert grid._df_exchange.native_only
        assert not grid._df_exchange.projection_float32
        return next(checks)

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', check)
    module.run_verified_scf(mf)
    assert observed == [True]*5+[False]*5
    assert projections == [True]*5+[False]*5
    assert mf.calls == [(None, 5), (mf.dm, 5)]
    assert mf.max_cycle == 10 and mf.callback is None
    assert mf.mixed_precision_info['fallback_scf'] and mf.mixed_precision_info['accepted']


@pytest.mark.parametrize('gradient_cycle,switch', [(2, 5), (9, 9), (None, 16)])
def test_projection_schedule_is_independent_bounded_and_one_way(precision, monkeypatch, gradient_cycle, switch):
    module, grid, _ = precision
    exchange = grid._df_exchange = _Exchange()
    mf = _MeanField()
    mf.xc, mf.do_nlc = 'b3lyp', lambda: False
    observed = []

    def kernel(dm0=None):
        for cycle in range(20):
            observed.append((grid.use_float32, exchange.projection_float32))
            settled = gradient_cycle is not None and cycle+1 >= gradient_cycle
            # The gradient rises again after the switch; neither schedule re-enables.
            norm = 1e-5 if settled and cycle < switch else 5e-4
            mf.callback({'cycle': cycle, 'e_tot': mf.e_tot, 'last_hf_e': mf.e_tot,
                         'norm_gorb': norm})
        mf.converged = True
        return mf.e_tot

    def check(*args):
        assert exchange.native_only and not exchange.projection_float32 and not grid.use_float32
        return _check(True)

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', check)
    module.run_verified_scf(mf)
    assert [value[0] for value in observed] == [True]*5+[False]*15
    assert [value[1] for value in observed] == [True]*switch+[False]*(20-switch)
    assert mf.mixed_precision_info['df_exchange_projection_schedule'] == {
        'minimum_cycles': 5, 'maximum_cycles': 16, 'orbital_gradient_threshold': 1e-4,
    }


@pytest.mark.parametrize('gradient_cycle,switch', [(2, 5), (9, 9), (None, 16)])
def test_nonlocal_exchange_preserves_first_five_grid_schedule(precision, monkeypatch, gradient_cycle, switch):
    module, grid, vv10 = precision
    exchange = grid._df_exchange = _Exchange()
    exchange.adaptive_grid = False
    mf = _MeanField()
    observed = []

    def kernel(dm0=None):
        for cycle in range(20):
            observed.append((grid.use_float32, exchange.projection_float32, vv10.use_float32))
            settled = gradient_cycle is not None and cycle+1 >= gradient_cycle
            mf.callback({'cycle': cycle, 'e_tot': mf.e_tot, 'last_hf_e': mf.e_tot,
                         'norm_gorb': 1e-5 if settled else .1})
        mf.converged = True
        return mf.e_tot

    def check(*args):
        assert exchange.native_only and not exchange.projection_float32
        assert not grid.use_float32 and not vv10.use_float32
        return _check(True)

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', check)
    module.run_verified_scf(mf)
    assert [v[0] for v in observed] == [True]*5+[False]*15
    assert [v[1] for v in observed] == [True]*switch+[False]*(20-switch)
    assert all(v[2] for v in observed)
    assert mf.mixed_precision_info['grid_precision_schedule'] == 'first_five'
    assert mf.mixed_precision_info['accepted']


def test_projection_phase_labels_follow_products_including_native_fallback(precision, monkeypatch):
    module, grid, _ = precision
    exchange = grid._df_exchange = _Exchange()
    exchange.float32_calls = 0
    mf = _MeanField()
    mf.xc, mf.do_nlc = 'b3lyp', lambda: False

    def kernel(dm0=None):
        grid.use_float32 = False
        for cycle in range(3):
            assert exchange.projection_float32
            if cycle != 1:
                exchange.float32_calls += 1
            mf.callback({'cycle': cycle, 'e_tot': mf.e_tot, 'last_hf_e': mf.e_tot,
                         'norm_gorb': .1})
        mf.converged = True
        return mf.e_tot

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(True))
    module.run_verified_scf(mf)
    assert [row['phase'] for row in mf.mixed_precision_info['cycles']] == ['mixed', 'float64', 'mixed']


@pytest.mark.parametrize('unreported_final_products', [False, True])
def test_int8_cycles_remain_approximate_and_recovery_never_reenables_them(
    precision, monkeypatch, unreported_final_products
):
    module, grid, _ = precision
    exchange = grid._df_exchange = _Exchange()
    mf = _MeanField()
    mf.xc, mf.do_nlc = 'b3lyp', lambda: False
    checks = iter((_check(False), _check(True)))

    def kernel(dm0=None):
        mf.calls.append((dm0, mf.max_cycle))
        grid.use_float32 = False
        for cycle in range(2):
            if not exchange.native_only:
                exchange.int8_calls += 1
            mf.callback({'cycle': cycle, 'e_tot': mf.e_tot, 'last_hf_e': mf.e_tot,
                         'norm_gorb': 1e-7})
        if unreported_final_products and not exchange.native_only:
            exchange.int8_calls += 1
            exchange.float32_calls += 1
        mf.converged = True
        return mf.e_tot

    def check(*args):
        assert exchange.native_only and not grid.use_float32
        return next(checks)

    mf.kernel = kernel
    monkeypatch.setattr(module, '_full_check', check)
    module.run_verified_scf(mf)
    report = mf.mixed_precision_info
    assert [row['phase'] for row in report['cycles']] == ['mixed']*2+['float64']*2
    assert report['df_exchange_contraction_calls']['int8'] == 2 + unreported_final_products
    assert mf.calls == [(None, 20), (mf.dm, 98)]
    assert report['fallback_scf'] and report['accepted']


def test_exchange_controller_finishes_and_releases_int8_before_native_calls(monkeypatch):
    grid_module = _load('grid_under_test', 'gpu4pyscf/dft/_mixed_grid.py')
    for name in ('gpu4pyscf', 'gpu4pyscf.df', 'gpu4pyscf.dft'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, 'gpu4pyscf.dft._mixed_grid', grid_module)
    backend = SimpleNamespace(update=lambda *args: True, released=False)

    def release():
        backend.released = True

    backend.release = release
    integer_module = ModuleType('gpu4pyscf.df._int8_gram')
    integer_module._Int8Gram = lambda: backend
    monkeypatch.setitem(sys.modules, integer_module.__name__, integer_module)
    exchange_module = _load('exchange_under_test', 'gpu4pyscf/df/_mixed_jk.py')
    grid = SimpleNamespace(use_float32=False)
    exchange = exchange_module._ExchangePrecision(grid)
    assert not exchange.projection_float32
    calls = []

    def native(a, out):
        calls.append(a)
        return out

    matrix, out = object(), object()
    assert exchange.gram(native, None, matrix, out) is out
    assert exchange.int8_calls == 1 and not calls
    exchange.projection_float32 = True
    exchange.finish_approximation()
    assert backend.released and exchange._int8 is None and not exchange.projection_float32
    grid.use_float32 = True
    assert exchange.gram(native, None, matrix, out) is out
    assert calls == [matrix] and exchange.float64_calls == 1


@pytest.fixture
def ase_calculator(precision, monkeypatch):
    from pyscf import gto, lib

    module, _, _ = precision
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(True))
    # Load only the Python interface, independently of CUDA or checkpoint availability.
    for name in ('gpu4pyscf', 'gpu4pyscf.dft', 'cupy'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, 'gpu4pyscf.dft.mixed_precision', module)
    interface = _load('ase_interface_under_test', 'gpu4pyscf/tools/ase_interface.py')

    class Method(_MeanField, lib.StreamObject):
        def __init__(self):
            super().__init__()
            self.mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
            self.scans = 0
            self.gradient_flags = []

        def as_scanner(self):
            return self

        def __call__(self, mol):
            self.mol = mol
            self.scans += 1
            if self.callback is None:
                self.callback = lambda env: None
                try:
                    return self.kernel()
                finally:
                    self.callback = None
            return self.kernel()

        def Gradients(self):
            base = self

            class Gradient:
                grid_response = False
                auxbasis_response = False

                def kernel(self):
                    base.gradient_flags.append((self.grid_response, self.auxbasis_response))
                    return np.ones((2, 3)) * 0.001

            return Gradient()

    return interface.PySCF, Method


def test_ase_geometry_changes_verify_each_scf_and_cache_properties(ase_calculator):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed', grid_response=True, auxbasis_response=True)
    forces = atoms.get_forces()
    first_info = atoms.calc.precision_info
    assert forces.shape == (2, 3) and isinstance(forces, np.ndarray)
    assert first_info['accepted'] and mf.gradient_flags == [(True, True)]
    atoms.get_potential_energy()
    assert mf.scans == 1
    atoms.positions[1, 2] += 0.01
    atoms.get_forces()
    assert mf.scans == 2 and atoms.calc.precision_info is not first_info
    assert all(row['phase'] == 'mixed' for row in atoms.calc.precision_info['cycles'])
    atoms.calc.set(precision='float64')
    atoms.get_forces()
    assert mf.scans == 3 and atoms.calc.precision_info is None


@pytest.mark.parametrize('policy', ['auto', 'mixed', 'float64', 'vv10'])
def test_ase_energy_then_forces_reuses_accepted_scf(ase_calculator, policy):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision=policy, grid_response=True, auxbasis_response=True)
    energy = atoms.get_potential_energy()
    info = atoms.calc.precision_info
    assert mf.scans == 1 and not atoms.calc.calculation_info['scf_reused']
    if policy == 'vv10':
        assert info['precision'] == 'vv10' and info['grid_precision_schedule'] == 'float64'
    forces = atoms.get_forces()
    assert mf.scans == 1 and mf.gradient_flags == [(True, True)]
    assert atoms.calc.precision_info is info
    assert atoms.calc.calculation_info['scf_reused']
    assert atoms.calc.calculation_info['scf_seconds'] == 0.0
    assert atoms.get_potential_energy() == energy
    np.testing.assert_array_equal(atoms.get_forces(), forces)
    assert mf.scans == 1 and len(mf.gradient_flags) == 1


@pytest.mark.parametrize('change', ['precision', 'grid_response', 'auxbasis_response', 'reset'])
def test_ase_settings_and_reset_invalidate_energy_only_state(ase_calculator, change):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed', grid_response=True, auxbasis_response=True)
    atoms.get_potential_energy()
    if change == 'reset':
        atoms.calc.reset()
    else:
        atoms.calc.set(**{change: 'float64' if change == 'precision' else False})
    atoms.get_forces()
    assert mf.scans == 2 and not atoms.calc.calculation_info['scf_reused']
    assert mf.gradient_flags == [(change != 'grid_response', change != 'auxbasis_response')]


@pytest.mark.parametrize('change', ['positions', 'numbers', 'cell', 'initial_charges', 'initial_magmoms'])
def test_ase_changed_atoms_cannot_reuse_energy_only_state(ase_calculator, change):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed')
    atoms.get_potential_energy()
    if change == 'positions':
        atoms.positions[1, 2] += 0.01
    elif change == 'numbers':
        atoms.set_atomic_numbers([2, 2])
    elif change == 'cell':
        atoms.set_cell([5, 5, 5])
    elif change == 'initial_charges':
        atoms.set_initial_charges([0.1, -0.1])
    else:
        atoms.set_initial_magnetic_moments([1, -1])
    # Compare the saved atoms even if a direct caller incorrectly reports no changes.
    atoms.calc.calculate(atoms, properties=['forces'], system_changes=[])
    assert mf.scans == 2 and not atoms.calc.calculation_info['scf_reused']


@pytest.mark.parametrize('change', ['convergence', 'energy', 'scanner'])
def test_ase_changed_electronic_state_is_not_reused(ase_calculator, change):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed')
    atoms.get_potential_energy()
    if change == 'convergence':
        mf.converged = False
    elif change == 'energy':
        mf.e_tot -= 0.1
    else:
        mf = atoms.calc.method_scan = method_type()
    previous_scans = mf.scans
    atoms.get_forces()
    assert mf.scans == previous_scans + 1 and not atoms.calc.calculation_info['scf_reused']


def test_ase_explicit_recalculation_does_not_reuse_scf(ase_calculator):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed')
    atoms.get_potential_energy()
    atoms.calc.calculate(properties=['forces'])
    assert mf.scans == 2 and not atoms.calc.calculation_info['scf_reused']


def test_ase_failed_force_evaluation_invalidates_reuse_and_old_forces(ase_calculator, monkeypatch):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed')
    atoms.get_forces()

    def fail():
        raise RuntimeError('Gradient failed')

    with monkeypatch.context() as patch:
        patch.setattr(mf, 'Gradients', lambda: SimpleNamespace(kernel=fail))
        with pytest.raises(RuntimeError, match='Gradient failed'):
            atoms.calc.calculate(properties=['forces'], system_changes=[])
    assert 'forces' not in atoms.calc.results and mf.scans == 1
    atoms.get_forces()
    assert mf.scans == 2 and not atoms.calc.calculation_info['scf_reused']


def test_ase_rejects_unknown_precision_and_response_types(ase_calculator):
    calculator, method_type = ase_calculator
    with pytest.raises(ValueError, match='precision'):
        calculator(method=method_type(), precision='float32')
    with pytest.raises(ValueError, match='grid_response'):
        calculator(method=method_type(), grid_response=1)


@pytest.mark.parametrize('mutation', ['unknown_root', 'unknown_recipe', 'unknown_case', 'atom_count', 'multiplicity'])
def test_workflow_input_rejects_identity_and_schema_changes(tmp_path, mutation):
    from ase_workflow import _RECIPE_KEYS, _input

    value = {
        'recipe': dict.fromkeys(_RECIPE_KEYS),
        'cases': [
            {
                'id': 'hydrogen',
                'numbers': [1],
                'positions': [[0, 0, 0]],
                'charge': 0,
                'multiplicity': 2,
            }
        ],
    }
    if mutation == 'unknown_root':
        value['other'] = True
    elif mutation == 'unknown_recipe':
        value['recipe']['other'] = True
    elif mutation == 'unknown_case':
        value['cases'][0]['other'] = True
    elif mutation == 'atom_count':
        value['cases'][0]['numbers'] = [1, 1]
    else:
        value['cases'][0]['multiplicity'] = 0
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        _input(path, 'hydrogen')


def test_ase_hessian_uses_native_reference_full_response_and_invalidates_forces(ase_calculator, monkeypatch):
    from ase import Atoms
    from pyscf.data.nist import BOHR, HARTREE2EV

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed', grid_response=True, auxbasis_response=True)
    atoms.get_forces()
    initial_density = mf.make_rdm1()
    observed = []
    raw_hessian = np.arange(36, dtype=float).reshape(2, 2, 3, 3)

    def kernel(dm0):
        assert dm0 is initial_density
        mf.converged = True
        mf.e_tot = -2.0
        return mf.e_tot

    class Hessian:
        grid_response = False
        auxbasis_response = 0

        def kernel(self):
            observed.append((self.grid_response, self.auxbasis_response))
            return raw_hessian

    monkeypatch.setattr(mf, 'kernel', kernel)
    monkeypatch.setattr(mf, 'Hessian', Hessian, raising=False)
    result = atoms.calc.get_hessian()
    reference = raw_hessian.transpose(0, 2, 1, 3).reshape(6, 6)
    reference = (reference + reference.T) * 0.5 * (HARTREE2EV / BOHR**2)
    np.testing.assert_array_equal(result, reference)
    assert observed == [(True, 2)]
    assert 'forces' not in atoms.calc.results
    assert atoms.calc.results['energy'] == -2.0 * HARTREE2EV
    assert atoms.calc.hessian_info['float64_reference_energy_hartree'] == -2.0


def test_ase_hessian_reference_failure_stops_without_cached_results(ase_calculator, monkeypatch):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed')
    atoms.get_forces()

    def kernel(dm0):
        mf.converged = False

    monkeypatch.setattr(mf, 'kernel', kernel)
    with pytest.raises(RuntimeError, match='Float64 Hessian reference'):
        atoms.calc.get_hessian(atoms)
    assert atoms.calc.results == {} and atoms.calc.hessian_info is None


def test_ase_hessian_reconvergence_invalidates_same_energy_scf_state(ase_calculator, monkeypatch):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf, precision='mixed')
    energy = atoms.get_potential_energy()
    verification = atoms.calc.precision_info

    def kernel(dm0):
        mf.converged = True
        return mf.e_tot

    with monkeypatch.context() as patch:
        patch.setattr(mf, 'kernel', kernel)
        patch.setattr(mf, 'Hessian', lambda: SimpleNamespace(
            auxbasis_response=0, kernel=lambda: np.zeros((2, 2, 3, 3))), raising=False)
        atoms.calc.get_hessian()
    assert atoms.get_potential_energy() == energy and mf.scans == 1
    atoms.get_forces()
    assert mf.scans == 2 and atoms.calc.precision_info is not verification
    assert not atoms.calc.calculation_info['scf_reused']


def test_initial_density_is_passed_to_native_scf(precision, monkeypatch):
    module, _, _ = precision
    mf = _MeanField()
    density = object()
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(True))
    module.run_verified_scf(mf, dm0=density)
    assert mf.calls == [(density, 20)]
    with pytest.raises(ValueError, match='not both'):
        module.run_verified_scf(mf, lambda: 0.0, dm0=density)


@pytest.mark.parametrize('recover', [False, True])
def test_vv10_only_keeps_grids_and_exchange_native(precision, monkeypatch, recover):
    module, grid, vv10 = precision
    exchange = _Exchange()
    grid._df_exchange = exchange
    mf = _MeanField()
    checks = iter((_check(False), _check(True)) if recover else (_check(True),))
    monkeypatch.setattr(module, '_full_check', lambda *args: next(checks))
    observed = []

    def callback(env):
        observed.append((grid.use_float32, exchange.native_only, exchange.projection_float32, vv10.use_float32))

    mf.callback = callback
    module.run_verified_scf(mf, dm0=mf.dm, precision='vv10')
    assert observed == [(False, True, False, True)] * 3 + ([(False, True, False, False)] * 3 if recover else [])
    report = mf.mixed_precision_info
    assert report['precision'] == 'vv10' and report['grid_precision_schedule'] == 'float64'
    assert report['df_exchange_projection_schedule'] is None
    assert report['fallback_scf'] == recover and report['accepted']
    assert mf.callback is callback and mf.max_cycle == 100


@pytest.mark.parametrize('policy', ['float32', 'float64', None])
def test_verified_runner_rejects_unknown_precision(precision, policy):
    module, _, _ = precision
    mf = _MeanField()
    with pytest.raises(ValueError, match='precision'):
        module.run_verified_scf(mf, precision=policy)
    assert not mf.calls


def test_vv10_only_rejects_method_without_nonlocal_term(precision):
    module, _, _ = precision
    mf = _MeanField()
    mf.do_nlc = lambda: False
    with pytest.raises(ValueError, match='nonlocal'):
        module.run_verified_scf(mf, precision='vv10')
    assert not mf.calls


def test_verification_failure_has_distinct_error_type(precision, monkeypatch):
    module, _, _ = precision
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(False))
    with pytest.raises(module.SCFVerificationError):
        module.run_verified_scf(_MeanField())


def test_ase_explicit_calculate_can_use_saved_atoms_and_reset_diagnostics(ase_calculator):
    from ase import Atoms

    calculator, method_type = ase_calculator
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=method_type(), precision='mixed')
    atoms.get_forces()
    atoms.calc.calculate(properties=['energy'])
    assert atoms.calc.precision_info['accepted'] and 'forces' not in atoms.calc.results
    changed = atoms.calc.set(precision='float64')
    assert changed == {'precision': 'float64'}
    assert atoms.calc.precision_info is atoms.calc.calculation_info is atoms.calc.hessian_info is None


def test_ase_failed_verification_retains_diagnostics_without_stale_results(ase_calculator, precision, monkeypatch):
    from ase import Atoms

    module, _, _ = precision
    calculator, method_type = ase_calculator
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=method_type(), precision='mixed')
    atoms.get_forces()
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(False))
    with pytest.raises(module.SCFVerificationError):
        atoms.calc.calculate(properties=['forces'])
    assert atoms.calc.results == {}
    assert not atoms.calc.precision_info['accepted']
    scans = atoms.calc.method_scan.scans
    monkeypatch.setattr(module, '_full_check', lambda *args: _check(True))
    atoms.get_forces()
    assert atoms.calc.method_scan.scans == scans + 1
    assert atoms.calc.precision_info['accepted'] and not atoms.calc.calculation_info['scf_reused']


def test_ase_defaults_to_verified_mixed_for_supported_gpu_dft(ase_calculator):
    from ase import Atoms

    calculator, method_type = ase_calculator
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=method_type())
    atoms.get_forces()
    assert atoms.calc.parameters.precision == 'auto'
    assert atoms.calc.calculation_info['precision'] == 'mixed'
    assert atoms.calc.precision_info['accepted']


@pytest.mark.parametrize('unsupported', ['cpu', 'unsupported_xc', 'legacy_grid', 'custom_convergence'])
def test_ase_auto_keeps_native_precision_for_unsupported_methods(ase_calculator, unsupported):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    if unsupported == 'cpu':
        mf.device = 'cpu'
    elif unsupported == 'unsupported_xc':
        mf.do_nlc = lambda: False
        mf._numint = SimpleNamespace(_xc_type=lambda xc: 'HF', nr_rks=mf._numint.nr_rks)
    elif unsupported == 'legacy_grid':
        mf.do_nlc = lambda: False
        mf._numint = SimpleNamespace(_xc_type=lambda xc: 'GGA', nr_rks=SimpleNamespace(__globals__={}))
    else:
        mf.check_convergence = lambda env: True
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf)
    atoms.get_forces()
    assert atoms.calc.calculation_info['precision'] == 'float64'
    assert atoms.calc.precision_info is None


@pytest.mark.parametrize('xctype', ['LDA', 'GGA', 'MGGA'])
def test_ase_auto_uses_verified_grid_precision_without_vv10(ase_calculator, xctype):
    from ase import Atoms

    calculator, method_type = ase_calculator
    mf = method_type()
    mf.do_nlc = lambda: False
    mf._numint = SimpleNamespace(_xc_type=lambda xc: xctype, nr_rks=mf._numint.nr_rks)
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calculator(method=mf)
    atoms.get_potential_energy()
    assert atoms.calc.calculation_info['precision'] == 'mixed'
    assert atoms.calc.precision_info['accepted']


def test_precision_setup_failure_invalidates_previous_convergence(precision, monkeypatch):
    module, _, _ = precision
    mf = _MeanField()
    mf.converged = True

    @contextmanager
    def fail(mf):
        raise ValueError('Unsupported integration dispatch')
        yield

    monkeypatch.setattr(module, '_precision_context', fail)
    with pytest.raises(ValueError, match='Unsupported integration'):
        module.run_verified_scf(mf)
    assert not mf.converged and not mf.mixed_precision_info['accepted']
    assert mf.callback is None and mf.max_cycle == 100
