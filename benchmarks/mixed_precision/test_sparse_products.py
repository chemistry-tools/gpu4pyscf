"""Check screened AO plans, packed products and native fallback semantics."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
gpu = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


def _module():
    spec = importlib.util.spec_from_file_location('sparse_df_plan', ROOT/'gpu4pyscf/df/_sparse_products.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _addresses(n, seed=1710):
    rng = np.random.default_rng(seed)
    mask = np.tril(rng.random((n, n)) < .25)
    np.fill_diagonal(mask, True)
    mask[0, 1] = mask[1, 0] = True
    return rng.permutation(np.flatnonzero(mask)).astype(np.int32)


@pytest.mark.parametrize('symmetric', [False, True])
@pytest.mark.parametrize('width', [1, 8, 16])
def test_plans_preserve_pair_order_and_sorted_reduction_columns(symmetric, width):
    n = 37
    addresses = _addresses(n)
    plan = _module()._cpu_plan(addresses, n, width, symmetric)
    lookup = np.full((n, n), -1, dtype=np.int32)
    lookup.flat[addresses] = np.arange(len(addresses))
    expected = np.where(lookup.T >= 0, lookup.T, lookup) if symmetric else lookup
    actual = np.full_like(lookup, -1)
    covered = []
    for rows, cols, mapping in plan['groups']:
        assert len(rows) <= width
        assert np.all(np.diff(rows) > 0) and np.all(np.diff(cols) > 0)
        actual[np.ix_(rows, cols)] = mapping
        covered.extend(rows)
    np.testing.assert_array_equal(actual, expected)
    assert sorted(covered) == list(range(n))
    assert plan['elements'] <= n*n
    if not symmetric:
        assert sorted(actual[actual >= 0]) == list(range(len(addresses)))


@pytest.mark.parametrize('addresses,n,width', [
    ([], 4, 2), ([0, 0], 4, 2), ([-1], 4, 2), ([16], 4, 2),
    ([1.5], 4, 2), ([[1]], 4, 2), ([0], 0, 2), ([0], 2**31, 2),
    ([0], 4, 0), ([0], 4, 129),
])
def test_invalid_plans_are_rejected(addresses, n, width):
    assert _module()._cpu_plan(addresses, n, width, True) is None


def test_empty_rows_remain_explicit_zero_outputs():
    plan = _module()._cpu_plan(np.array([0, 1, 5], dtype=np.int32), 4, 2, True)
    assert plan['rows'] == 2
    assert sorted(v for rows, _, _ in plan['groups'] for v in rows) == [0, 1]


@pytest.fixture
def sparse(monkeypatch):
    from gpu4pyscf.df import _sparse_products

    monkeypatch.setattr(_sparse_products, '_MIN_ORBITALS', 0)
    monkeypatch.setattr(_sparse_products, '_MAX_RETAINED_FRACTION', 1.01)
    return _sparse_products


@gpu
@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('batches', [1, 2])
def test_projection_matches_native_in_both_packed_layouts(single, transposed, batches, sparse):
    import cupy as cp
    from cupy_backends.cuda.libs import cutensor

    from gpu4pyscf.lib.cutensor import contraction

    n, k, nl = 31, 11, 277
    addresses = _addresses(n)
    rows, cols = divmod(addresses, n)
    rng = np.random.default_rng(1711)
    data = rng.normal(size=(n, n, nl))
    data = data+data.transpose(1, 0, 2)
    mask = np.zeros((n, n), dtype=bool)
    mask[rows, cols] = mask[cols, rows] = True
    data *= mask[:, :, None]
    full, factors = cp.asarray(data), cp.asarray(rng.normal(size=(batches, n, k)))
    raw = full.reshape(n*n, nl)[cp.asarray(addresses)]
    packed = raw.T if transposed else cp.ascontiguousarray(raw.T)
    plan = sparse._SparseAOPlan.create(SimpleNamespace(), cp.asarray(addresses), n, width=8)
    prepared = plan.prepare(factors, single=single)
    assert prepared is not None and prepared.workspace_bytes <= sparse._MAX_WORKSPACE_BYTES
    storage = cp.full((batches, k, n, nl+3), cp.nan)
    out = storage[:, :, :, :nl].transpose(0, 2, 1, 3)
    assert prepared.project(packed, out)
    if single:
        reference = contraction('Lij,njk->nikL', full.transpose(2, 0, 1).astype(cp.float32),
                                factors.astype(cp.float32), 1., 0., compute_desc=cutensor.COMPUTE_DESC_32F)
        cp.testing.assert_allclose(out, reference, atol=3e-5, rtol=3e-5)
    else:
        reference = contraction('Lij,njk->nikL', full.transpose(2, 0, 1), factors, 1., 0.)
        cp.testing.assert_allclose(out, reference, atol=3e-12, rtol=3e-13)
    cp.testing.assert_array_equal(storage[:, :, :, nl:], cp.nan)


@gpu
def test_last_bit_differences_in_diagonal_shell_orientations_follow_native_gradient(sparse):
    import cupy as cp

    from gpu4pyscf.lib.cupy_helper import contract

    n, nl = 17, 11
    addresses = _addresses(n, 1716)
    rows, cols = divmod(addresses, n)
    packed = cp.ones((len(addresses), nl))
    direct = int(np.flatnonzero(addresses == 1)[0])
    packed[direct] = np.nextafter(1., 2.)
    dense = cp.zeros((n, n, nl))
    dense[cp.asarray(cols), cp.asarray(rows)] = dense[cp.asarray(rows), cp.asarray(cols)] = packed
    factors = cp.eye(n)[None]
    expected = contract('pqr,pi->iqr', dense, factors[0])
    plan = sparse._SparseAOPlan.create(SimpleNamespace(), cp.asarray(addresses), n, width=8)
    prepared = plan.prepare(factors)
    out = cp.empty((1, n, n, nl))
    assert prepared.project(packed.T, out)
    cp.testing.assert_array_equal(out[0].transpose(1, 0, 2), expected)


@gpu
@pytest.mark.parametrize('batches', [1, 2])
@pytest.mark.parametrize('beta', [0., .7])
def test_back_projection_writes_only_directed_pairs_and_retains_pitches(batches, beta, sparse):
    import cupy as cp

    from gpu4pyscf.lib.cupy_helper import contract

    n, k, nl = 29, 9, 269
    addresses = _addresses(n, 1712)
    rng = np.random.default_rng(1712)
    factors = cp.asarray(rng.normal(size=(batches, n, k)))
    values = cp.asarray(rng.normal(size=(batches, k, n, nl)))
    density, auxiliary = cp.asarray(rng.normal(size=(n, n))), cp.asarray(rng.normal(size=nl))
    alpha = -.17
    expected = beta*density[:, :, None]*auxiliary
    expected = contract('niqr,npi->pqr', values, factors, alpha=alpha, beta=1., out=expected)
    plan = sparse._SparseAOPlan.create(SimpleNamespace(), cp.asarray(addresses), n, symmetric=False, width=8)
    prepared = plan.prepare(factors)
    storage = cp.full((len(addresses), nl+5), cp.nan)
    out = storage[:, :nl]
    assert prepared.back(values, out, alpha, beta, density if beta else None, auxiliary if beta else None)
    cp.testing.assert_allclose(out, expected.reshape(n*n, nl)[cp.asarray(addresses)], atol=3e-12, rtol=3e-13)
    cp.testing.assert_array_equal(storage[:, nl:], cp.nan)


@gpu
def test_stream_alias_nonfinite_and_duplicate_guards_leave_output_untouched(sparse):
    import cupy as cp

    n, k, nl = 17, 7, 11
    addresses = cp.asarray(_addresses(n, 1713))
    owner = SimpleNamespace()
    plan = sparse._SparseAOPlan.create(owner, addresses, n, width=8)
    factors = cp.ones((1, n, k))
    prepared = plan.prepare(factors)
    packed = cp.ones((nl, len(addresses)))
    out = cp.full((1, n, k, nl), 13.)
    assert prepared.project(packed, out)
    out.fill(13)
    for value in (cp.nan, cp.inf):
        packed[0, 0] = value
        assert not prepared.project(packed, out)
        cp.testing.assert_array_equal(out, 13)
    packed.fill(1)
    left, _ = plan.duplicates
    packed[0, int(left[0])] = 2
    assert not prepared.project(packed, out)
    cp.testing.assert_array_equal(out, 13)
    packed.fill(1)
    with cp.cuda.Stream(non_blocking=True):
        assert not prepared.project(packed, out)
    assert not prepared.project(packed, out[:, :, :, ::-1])
    overlapping = cp.lib.stride_tricks.as_strided(out, out.shape, (8, 8, 8, 8))
    assert not prepared.project(packed, overlapping)
    factors[0, 0, 0] = cp.nan
    assert plan.prepare(factors) is None


@gpu
def test_plans_track_index_contents_and_keep_only_three_owned_variants(sparse):
    import cupy as cp

    n = 23
    owner = SimpleNamespace()
    addresses = cp.asarray(_addresses(n, 1714))
    first = sparse._SparseAOPlan.create(owner, addresses, n, width=8)
    assert sparse._SparseAOPlan.create(owner, addresses, n, width=8) is first
    addresses[:] = addresses[::-1].copy()
    second = sparse._SparseAOPlan.create(owner, addresses, n, width=8)
    assert second is not first
    for width in (16, 32, 64):
        assert sparse._SparseAOPlan.create(owner, addresses, n, width=width) is not None
    assert len(owner._sparse_ao_cache[1]) <= 3


@gpu
def test_blas_state_is_restored_on_error_and_current_stream_is_used(sparse, monkeypatch):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    n, k, nl = 17, 7, 11
    addresses = cp.asarray(_addresses(n, 1715))
    factors, packed = cp.ones((1, n, k)), cp.ones((nl, len(addresses)))
    out = cp.zeros((1, n, k, nl))
    cp.cuda.runtime.deviceSynchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    handle = cp.cuda.device.get_cublas_handle()
    with stream:
        plan = sparse._SparseAOPlan.create(SimpleNamespace(), addresses, n, width=8)
        prepared = plan.prepare(factors)
        assert prepared.project(packed, out)
        state = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)

        def fail(*args):
            raise RuntimeError('injected sparse GEMM failure')

        with monkeypatch.context() as patch:
            patch.setattr(cublas, 'dgemm', fail)
            with pytest.raises(RuntimeError, match='injected sparse GEMM failure'):
                prepared.project(packed, out)
        assert (cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)) == state
    stream.synchronize()


@contextlib.contextmanager
def _reference_gradients():
    from gpu4pyscf.df.grad import rhf, uhf

    saved = []
    try:
        for module, variable in [(rhf, 'DFT_REFERENCE_DF_GRADIENT'),
                                 (uhf, 'DFT_REFERENCE_UHF_DF_GRADIENT')]:
            path = os.getenv(variable)
            if not path:
                pytest.fail(f'{variable} must name the retained pre-sparse gradient source')
            tree = ast.parse(Path(path).read_text())
            nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_jk_energy_per_atom']
            assert len(nodes) == 1
            namespace = dict(module.__dict__)
            # Only the explicitly retained native function is rebound for comparison.
            exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)  # noqa: S102
            saved.append((module, module._jk_energy_per_atom))
            module._jk_energy_per_atom = namespace['_jk_energy_per_atom']
        yield
    finally:
        for module, original in reversed(saved):
            module._jk_energy_per_atom = original


def _method(atoms, xc, charge=0, multiplicity=1):
    from pyscf import gto

    from gpu4pyscf import dft

    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='6-31g*',
                cart=True, charge=charge, spin=multiplicity-1, verbose=0)
    mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 0, 0
    mf.grids.prune = mf.nlcgrids.prune = None
    mf.small_rho_cutoff = 0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.chkfile = 1e-11, 1e-8, 150, None
    return mf


@gpu
@pytest.mark.parametrize('xc', ['b3lyp', 'wb97m-v'])
@pytest.mark.parametrize('case,charge,multiplicity', [
    ('CH3CONH2', 0, 1), ('C5H5N', 0, 1), ('C2H6SO', 0, 1),
    ('CH3COF', 0, 1), ('CH3COCl', 0, 1), ('H2O', 1, 2),
])
def test_complete_native_fields_and_forces_on_mixed_atom_panel(case, charge, multiplicity, xc,
                                                              sparse, monkeypatch):
    import cupy as cp
    from ase.build import molecule

    from gpu4pyscf.dft.mixed_precision import _full_check

    atoms = molecule(case)
    mf = _method(atoms, xc, charge, multiplicity)
    with monkeypatch.context() as patch:
        patch.setattr(sparse._SparseAOPlan, 'create', lambda *args, **kwargs: None)
        reference_energy = mf.kernel()
    assert mf.converged and mf.mo_coeff.shape[-1] == mf.mol.nao
    with _reference_gradients():
        gradient = mf.nuc_grad_method()
        gradient.grid_response = gradient.auxbasis_response = True
        reference_gradient = cp.asarray(gradient.kernel())
    calls = []
    original_project, original_back = sparse._SparseAOProduct.project, sparse._SparseAOProduct.back

    def project(self, *args):
        accepted = original_project(self, *args)
        calls.append(('projection', accepted))
        return accepted

    def back(self, *args):
        accepted = original_back(self, *args)
        calls.append(('back', accepted))
        return accepted

    monkeypatch.setattr(sparse._SparseAOProduct, 'project', project)
    monkeypatch.setattr(sparse._SparseAOProduct, 'back', back)
    check = _full_check(mf, reference_energy)
    assert check['passed'], check
    assert abs(check['energy_change_hartree']) < 1e-9
    assert any(name == 'projection' and accepted for name, accepted in calls)
    calls.clear()
    gradient = mf.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    actual_gradient = cp.asarray(gradient.kernel())
    assert any(name == 'back' and accepted for name, accepted in calls)
    cp.testing.assert_allclose(actual_gradient, reference_gradient, atol=3e-10, rtol=2e-9)
    if case in ('CH3CONH2', 'CH3COCl', 'H2O'):
        from pyscf.data.nist import BOHR

        step = 2e-4
        energies = []
        for sign in (-1, 1):
            displaced = atoms.copy()
            displaced.positions[0, 0] += sign*step*BOHR
            other = _method(displaced, xc, charge, multiplicity)
            energies.append(other.kernel())
            assert other.converged
        finite_difference = (energies[1]-energies[0])/(2*step)
        assert float(actual_gradient[0, 0]) == pytest.approx(finite_difference, abs=2e-5)


@gpu
@pytest.mark.parametrize('xc', ['b3lyp', 'wb97m-v'])
@pytest.mark.parametrize('charge,multiplicity', [(0, 1), (1, 2)])
def test_ase_verified_mixed_precision_and_force_reuse(charge, multiplicity, xc, sparse):
    from ase.build import molecule

    from gpu4pyscf.tools.ase_interface import PySCF

    atoms = molecule('H2O' if charge else 'CH3COCl')
    atoms.calc = PySCF(method=_method(atoms, xc, charge, multiplicity), precision='auto',
                      grid_response=True, auxbasis_response=True)
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    assert np.isfinite(energy) and np.isfinite(forces).all()
    info = atoms.calc.precision_info
    assert info['accepted'] and info['final_verification']['passed']
    assert atoms.calc.calculation_info['scf_reused']
    assert atoms.calc.calculation_info['scf_seconds'] == 0
    assert bool(info['vv10_precisions']) == (xc == 'wb97m-v')
    if xc == 'b3lyp':
        assert info['df_exchange_contraction_calls']['float32'] > 0
