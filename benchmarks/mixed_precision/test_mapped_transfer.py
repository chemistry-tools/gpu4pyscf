"""Verify exact asynchronous density-fitting host writes and bounded lifetimes."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

gpu = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')
ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location('mapped_transfer_plan', ROOT/'gpu4pyscf/df/_mapped_transfer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('n,pairs', [(0, 5), (7, 0), (-1, 8), (3, 17), (8192, 65536), (2**31, 1)])
def test_invalid_small_and_expensive_workspaces_use_native_fallback(n, pairs):
    assert _module()._workspace_bytes(n, pairs) is None


@pytest.mark.parametrize('n,pairs', [(5166, 4061), (5166, 39374)])
def test_workspace_reserves_only_one_bounded_batch(n, pairs):
    module = _module()
    assert module._workspace_bytes(n, pairs) == 8*n*pairs
    assert module._workspace_bytes(n, pairs) <= module._MAX_WORKSPACE_BYTES


@pytest.fixture
def transfer(monkeypatch):
    from gpu4pyscf.df import _mapped_transfer

    monkeypatch.setattr(_mapped_transfer, '_MIN_ELEMENTS', 0)
    return _mapped_transfer


@gpu
def test_partial_overlapping_batches_reuse_source_and_preserve_all_bits(transfer):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.lib.cupy_helper import empty_mapped

    host = empty_mapped((17, 97))
    host[:] = .125
    expected = host.copy()
    writer = transfer._MappedCderiWriter.create(host, 51)
    assert writer is not None
    assert writer.workspace_bytes == 17*51*8
    source = cp.empty((51, 17))
    for index, (p0, p1) in enumerate(((3, 54), (43, 62), (70, 97))):
        bits = np.random.default_rng(1950+index).integers(0, 2**64-1, (p1-p0, 17), dtype=np.uint64)
        bits.flat[:4] = [0, 2**63, 0x7ff8000000000001, 0x7ff0000000000000]
        values = bits.view(np.float64)
        current = source.reshape(-1)[:values.size].reshape(values.shape)
        current.set(values)
        assert writer.enqueue(current, np.int64(p0), np.int64(p1))
        expected[:, p0:p1] = values.T
        source.fill(-9)
    writer.finish()
    np.testing.assert_array_equal(host.view(np.uint64), expected.view(np.uint64))
    assert not writer.pending
    writer.close()
    assert writer.buffer is None and writer.workspace_bytes == 0
    assert not writer.enqueue(source, 0, 51)


@gpu
def test_layout_memory_architecture_and_stream_guards_do_not_write(transfer, monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.lib.cupy_helper import empty_mapped

    host = empty_mapped((7, 51))
    host[:] = .5
    create = transfer._MappedCderiWriter.create
    assert create(np.empty_like(host), 17) is None
    assert create(host[:, ::2], 17) is None
    assert create(host, 52) is None
    with monkeypatch.context() as patch:
        patch.setattr(transfer, '_available_memory', lambda: 0)
        assert create(host, 17) is None
    with monkeypatch.context() as patch:
        patch.setattr(transfer, '_supported_device', lambda device: False)
        assert create(host, 17) is None
    writer = create(host, 17)
    assert writer is not None
    source = cp.ones((17, 7))
    assert not writer.enqueue(source.astype(cp.float32), 0, 17)
    assert not writer.enqueue(source[:, ::-1], 0, 17)
    assert not writer.enqueue(source, -1, 16)
    assert not writer.enqueue(source, 0, 18)
    assert not writer.enqueue(writer.buffer.reshape(17, 7), 0, 17)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        assert not writer.enqueue(source, 0, 17)
    writer.close()
    np.testing.assert_array_equal(host, .5)


@gpu
def test_nondefault_stream_and_copy_error_finish_before_release(transfer, monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.lib.cupy_helper import empty_mapped

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        host = empty_mapped((7, 51))
        host[:] = .5
        writer = transfer._MappedCderiWriter.create(host, 17)
        assert writer is not None
        source = cp.arange(17*7, dtype=cp.float64).reshape(17, 7)
        assert writer.enqueue(source, 3, 20)
        writer.finish()
        np.testing.assert_array_equal(host[:, 3:20], np.arange(17*7).reshape(17, 7).T)

        def fail(*args):
            raise RuntimeError('injected DMA failure')

        monkeypatch.setattr(cp.cuda.runtime, 'memcpy2DAsync', fail)
        try:
            with pytest.raises(RuntimeError, match='injected DMA failure'):
                writer.enqueue(source, 20, 37)
            assert writer.pending
        finally:
            writer.close()
        assert writer.buffer is None and not writer.pending
    stream.synchronize()


@gpu
@pytest.mark.parametrize('case,charge,spin', [('CH3CONH2', 0, 0), ('C5H5N', 0, 0),
                                           ('C2H6SO', 0, 0), ('CH3COF', 0, 0),
                                           ('CH3COCl', 0, 0), ('H2O', 1, 1)])
def test_host_fitted_integrals_are_bitwise_identical(case, charge, spin, transfer, monkeypatch):
    import numpy as np
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='6-31g*',
                cart=True, charge=charge, spin=spin, verbose=0)

    def build():
        mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
        mf.with_df.use_gpu_memory = False
        mf.with_df.build()
        return mf.with_df._cderi[0].copy()

    with monkeypatch.context() as patch:
        patch.setattr(transfer._MappedCderiWriter, 'create', lambda *args: None)
        reference = build()
    calls = []
    enqueue = transfer._MappedCderiWriter.enqueue

    def counted(self, *args):
        result = enqueue(self, *args)
        calls.append(result)
        return result

    monkeypatch.setattr(transfer._MappedCderiWriter, 'enqueue', counted)
    candidate = build()
    assert calls and all(calls)
    np.testing.assert_array_equal(candidate.view(np.uint64), reference.view(np.uint64))


@gpu
@pytest.mark.parametrize('case,charge,spin', [('CH3CONH2', 0, 0), ('H2O', 1, 1)])
def test_complete_host_backed_scf_and_forces(case, charge, spin, transfer, monkeypatch):
    import cupy as cp
    import numpy as np
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft

    atoms = molecule(case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='6-31g*',
                cart=True, charge=charge, spin=spin, verbose=0)

    def evaluate():
        mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
        mf.with_df.use_gpu_memory = False
        mf.grids.level, mf.small_rho_cutoff = 1, 0
        mf.conv_tol, mf.conv_tol_grad, mf.direct_scf_tol = 1e-11, 1e-9, 1e-15
        mf.max_cycle, mf.chkfile = 100, None
        energy = mf.kernel()
        assert mf.converged
        gradient = mf.nuc_grad_method()
        gradient.grid_response = gradient.auxbasis_response = True
        return energy, gradient.kernel()

    with monkeypatch.context() as patch:
        patch.setattr(transfer._MappedCderiWriter, 'create', lambda *args: None)
        reference_energy, reference_gradient = evaluate()
    cp.get_default_memory_pool().free_all_blocks()
    calls = []
    enqueue = transfer._MappedCderiWriter.enqueue

    def counted(self, *args):
        result = enqueue(self, *args)
        calls.append(result)
        return result

    monkeypatch.setattr(transfer._MappedCderiWriter, 'enqueue', counted)
    energy, gradient = evaluate()
    assert calls and all(calls)
    assert abs(energy-reference_energy) < 1e-10
    np.testing.assert_allclose(gradient, reference_gradient, atol=2e-6, rtol=2e-5)
