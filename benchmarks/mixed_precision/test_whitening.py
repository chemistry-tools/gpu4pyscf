"""Check exact-zero float64 whitening, fallbacks and independent chemistry."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
gpu = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


def _module():
    spec = importlib.util.spec_from_file_location('whitening_plan', ROOT/'gpu4pyscf/df/_whitening.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('n,width', [(5166, 512), (4833, 512), (8192, 512)])
def test_workspace_is_bounded_and_counts_partial_columns(n, width):
    module = _module()
    counts = [min(first+width, n) for first in range(0, n, width)]
    rows, size = module._workspace_plan(n, counts, width)
    retained = sum(k*min(width, n-block*width)*8+k*4 for block, k in enumerate(counts))
    assert rows % 256 == 0 and 256 <= rows <= module._MAX_ROWS
    assert size == retained+rows*max(counts)*8 <= module._MAX_WORKSPACE_BYTES


@pytest.mark.parametrize('n,counts,width', [
    (0, [], 512), (-1, [], 512), (2**31, [], 512), (16, [16], 0),
    (16, [16], 8), (16, [-1, 8], 8), (16, [17, 8], 8),
    (1024, [1024, 1024], 512), (1024, [0, 0], 512),
])
def test_invalid_or_dense_plans_keep_original_gemm(n, counts, width):
    assert _module()._workspace_plan(n, counts, width) is None


@pytest.fixture
def whitening(monkeypatch):
    from gpu4pyscf.df import _whitening

    monkeypatch.setattr(_whitening, '_MIN_AUX', 0)
    monkeypatch.setattr(_whitening, '_BLOCK_COLUMNS', 64)
    monkeypatch.setattr(_whitening, '_MAX_ROWS', 256)
    return _whitening


@gpu
@pytest.mark.parametrize('permuted', [False, True])
@pytest.mark.parametrize('rows', [13, 269])
def test_partial_blocks_zero_rows_and_scaled_values_match_native(permuted, rows, whitening):
    import cupy as cp
    import numpy as np

    rng = np.random.default_rng(2001)
    n = 147
    coefficient = np.triu(rng.normal(size=(n, n)))
    coefficient[:, 64:128] = 0
    coefficient *= np.geomspace(1e-3, 1e3, n)
    if permuted:
        coefficient = coefficient[rng.permutation(n)]
    coefficient = cp.asarray(coefficient)
    raw = cp.asarray(rng.normal(size=(rows, n)))
    prepared = whitening._BlockedWhitening.create(coefficient, 'cd')
    assert prepared is not None and len(prepared.blocks[1][2]) == 0
    out = cp.full_like(raw, cp.nan)
    assert prepared.apply(raw, out)
    cp.testing.assert_allclose(out, raw@coefficient, atol=2e-10, rtol=3e-13)
    cp.testing.assert_array_equal(out[:, 64:128], 0)
    assert prepared.workspace_bytes <= whitening._MAX_WORKSPACE_BYTES


@gpu
def test_preparation_guards_do_not_drop_small_nonzero_terms(whitening, monkeypatch):
    import cupy as cp

    coefficient = cp.triu(cp.ones((147, 147)))
    assert whitening._BlockedWhitening.create(coefficient, 'ed') is None
    assert whitening._BlockedWhitening.create(coefficient.astype(cp.float32), 'cd') is None
    assert whitening._BlockedWhitening.create(coefficient.T, 'cd') is None
    assert whitening._BlockedWhitening.create(coefficient[:, :-1], 'cd') is None
    assert whitening._BlockedWhitening.create(cp.ones_like(coefficient), 'cd') is None
    coefficient[80, 1] = 1e-100
    prepared = whitening._BlockedWhitening.create(coefficient, 'cd')
    assert prepared is not None and bool(cp.any(prepared.blocks[0][2] == 80))
    for value in [cp.inf, cp.nan]:
        coefficient[0, 0] = value
        assert whitening._BlockedWhitening.create(coefficient, 'cd') is None
    coefficient[0, 0] = 1
    with monkeypatch.context() as patch:
        patch.setattr(whitening, '_supported_device', lambda device: False)
        assert whitening._BlockedWhitening.create(coefficient, 'cd') is None
    with monkeypatch.context() as patch:
        patch.setattr(whitening, '_available_memory', lambda: 0)
        assert whitening._BlockedWhitening.create(coefficient, 'cd') is None


@gpu
def test_rejected_inputs_do_not_mutate_output(whitening):
    import cupy as cp

    coefficient = cp.triu(cp.ones((147, 147)))
    prepared = whitening._BlockedWhitening.create(coefficient, 'cd')
    assert prepared is not None
    raw, out = cp.ones((37, 147)), cp.full((37, 147), .25)
    assert not prepared.apply(raw.astype(cp.float32), out)
    assert not prepared.apply(raw, out.astype(cp.float32))
    assert not prepared.apply(raw[:, ::-1], out)
    assert not prepared.apply(raw, out[:, ::-1])
    assert not prepared.apply(raw, raw)
    workspace_alias = prepared.work[:raw.size].reshape(raw.shape)
    assert not prepared.apply(workspace_alias, out)
    assert not prepared.apply(raw, workspace_alias)
    for value in [cp.nan, cp.inf, -cp.inf]:
        raw[0, 1] = value
        assert not prepared.apply(raw, out)
        cp.testing.assert_array_equal(out, .25)
    raw[:] = 1
    with cp.cuda.Stream(non_blocking=True):
        assert not prepared.apply(raw, out)
    cp.testing.assert_array_equal(out, .25)


@gpu
def test_stream_and_blas_state_are_preserved_on_success_and_error(whitening, monkeypatch):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        coefficient = cp.triu(cp.ones((147, 147)))
        prepared = whitening._BlockedWhitening.create(coefficient, 'cd')
        assert prepared is not None
        raw, out = cp.ones((269, 147)), cp.empty((269, 147))
        handle = cp.cuda.device.get_cublas_handle()
        previous = cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)
        try:
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
            assert prepared.apply(raw, out)
            expected = (*previous[:2], cublas.CUBLAS_POINTER_MODE_DEVICE)
            assert (cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)) == expected

            def fail(*args):
                raise RuntimeError('injected whitening GEMM failure')

            monkeypatch.setattr(cublas, 'dgemm', fail)
            with pytest.raises(RuntimeError, match='injected whitening'):
                prepared.apply(raw, out)
            assert (cublas.getStream(handle), cublas.getMathMode(handle), cublas.getPointerMode(handle)) == expected
        finally:
            cublas.setPointerMode(handle, previous[2])
    stream.synchronize()
    cp.testing.assert_allclose(out, cp.arange(1, 148)[None, :]*cp.ones_like(out))


def _method(case, spin=0, cart=True):
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='6-31g*',
                cart=cart, charge=int(bool(spin)), spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.small_rho_cutoff = 1, 0
    mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-11, 1e-9, 1e-15
    mf.max_cycle, mf.chkfile = 100, None
    return mf


@gpu
@pytest.mark.parametrize('case,spin,cart', [
    ('CH3CONH2', 0, True), ('C5H5N', 0, True), ('C2H6SO', 0, True),
    ('CH3COF', 0, True), ('CH3COCl', 0, True), ('H2O', 1, True),
    ('CH3CONH2', 0, False),
])
@pytest.mark.parametrize('host', [False, True])
def test_complete_host_and_device_tensors_match_original(case, spin, cart, host, whitening, monkeypatch):
    import cupy as cp
    import numpy as np

    mf = _method(case, spin, cart)
    mf.with_df.use_gpu_memory = not host
    mf.with_df.build()
    candidate = [np.asarray(cp.asnumpy(a)).copy() for a in mf.with_df._cderi]
    mf.with_df.reset(mf.mol)
    with monkeypatch.context() as patch:
        patch.setattr(whitening._BlockedWhitening, 'create', lambda *args: None)
        mf.with_df.build()
    for a, b in zip(candidate, mf.with_df._cderi):
        b = cp.asnumpy(b)
        # Ill-conditioned auxiliary inverses make individual factors sensitive
        # to float64 summation order. Compare their physical fitted integrals;
        # the separate complete SCF/force checks cover the full tensors.
        indices = np.linspace(0, a.shape[1]-1, 96, dtype=int)
        np.testing.assert_allclose(a[:, indices].T@a[:, indices], b[:, indices].T@b[:, indices],
                                   atol=5e-10, rtol=5e-11)


@gpu
@pytest.mark.parametrize('case,spin', [
    ('CH3CONH2', 0), ('C5H5N', 0), ('C2H6SO', 0), ('CH3COF', 0), ('CH3COCl', 0), ('H2O', 1),
])
def test_independent_scf_and_identical_density_forces(case, spin, whitening, monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft.mixed_precision import _full_check

    calls = []
    original = whitening._BlockedWhitening.apply

    def apply(self, raw, out):
        accepted = original(self, raw, out)
        calls.append(accepted)
        return accepted

    monkeypatch.setattr(whitening._BlockedWhitening, 'apply', apply)
    candidate = _method(case, spin)
    energy = candidate.kernel()
    assert candidate.converged and any(calls)
    gradient = candidate.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    force = gradient.kernel()
    with monkeypatch.context() as patch:
        patch.setattr(whitening._BlockedWhitening, 'create', lambda *args: None)
        candidate.with_df.reset(candidate.mol)
        check = _full_check(candidate, energy)
        # Tight SCF convergence fixes the density for the force comparison.
        # Independent native reconstruction uses a 1e-9 Ha comparison bound,
        # ten times stricter than the unchanged 1e-8 production verification.
        assert abs(check['energy_change_hartree']) < 1e-9, check
        assert check['float64_orbital_gradient_norm'] < 1e-8, check
        native = candidate.nuc_grad_method()
        native.grid_response = native.auxbasis_response = True
        np.testing.assert_allclose(force, native.kernel(), atol=2e-10, rtol=2e-10)
        reference = _method(case, spin)
        assert abs(reference.kernel()-energy) < 1e-9
        assert reference.converged
    del reference, candidate, gradient, native
    cp.get_default_memory_pool().free_all_blocks()


@gpu
def test_full_force_matches_independent_energy_differences(whitening):
    import cupy as cp
    import numpy as np

    method = _method('CH3COF')
    method.kernel()
    assert method.converged
    gradient = method.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    forces = -gradient.kernel()
    atom, direction = np.unravel_index(np.argmax(np.abs(forces)), forces.shape)
    positions = method.mol.atom_coords()
    for step in [6e-4, 2e-3]:
        energies = []
        for sign in [-1, 1]:
            displaced = positions.copy()
            displaced[atom, direction] += sign*step
            probe = _method('CH3COF')
            probe.mol.set_geom_(displaced, unit='Bohr')
            probe.reset(probe.mol)
            energies.append(probe.kernel())
            assert probe.converged
        assert abs(forces[atom, direction]+(energies[1]-energies[0])/(2*step)) < 4e-6
    cp.get_default_memory_pool().free_all_blocks()
