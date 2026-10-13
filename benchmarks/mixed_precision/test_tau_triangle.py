"""Check direct float64 tau products, guarded dispatch and mixed-atom derivatives."""

from __future__ import annotations

import os

import pytest
from _organic_cases import ASE_ORGANICS

pytestmark = pytest.mark.skipif(os.getenv('DFT_GPU_TEST') != '1', reason='opt-in CUDA test')


@pytest.mark.parametrize('shape', [(19, 17), (33, 83), (400, 2049)])
@pytest.mark.parametrize('signs', ['positive', 'negative', 'mixed', 'zero'])
def test_signed_triangle_matches_general_product(shape, signs):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import _tau_triangle, numint

    rng = np.random.default_rng(1313)
    m, k = shape
    ao = cp.asarray(rng.normal(size=(4, m, k)))
    weights = cp.asarray(rng.uniform(0.1, 1., size=k))
    if signs == 'negative':
        weights *= -1
    elif signs == 'mixed':
        weights[::2] *= -1
        weights[::3] = 0
    elif signs == 'zero':
        weights.fill(0)
    expected = numint._tau_dot_full(ao, ao, weights)
    out = cp.full((m, m), cp.nan)
    actual = _tau_triangle._triangular_tau(ao, weights, out=out)
    assert actual is out
    cp.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-12)
    cp.testing.assert_array_equal(actual, actual.T)


@pytest.mark.parametrize('m,k,expected', [(383, 2048, False), (384, 2048, True), (512, 4096, True),
                                      (513, 4096, False), (400, 2047, False), (400, 4097, False)])
def test_triangle_dispatch_size_boundaries(m, k, expected):
    import cupy as cp

    from gpu4pyscf.dft import _tau_triangle

    ao, weights = cp.ones((4, m, k)), cp.ones(k)
    assert _tau_triangle._eligible(ao, ao, weights, None) == expected


@pytest.mark.parametrize('kind', ['distinct', 'complex', 'strided', 'weights', 'output',
                                 'nonfinite_weights', 'zero_weight_nonfinite_ao', 'alias', 'device'])
def test_triangle_unsupported_inputs_keep_original_dispatch(kind, monkeypatch):
    import cupy as cp

    from gpu4pyscf.dft import _tau_triangle, numint

    ao, weights = cp.ones((4, 384, 2048)), cp.ones(2048)
    ket, out = ao, None
    if kind == 'distinct':
        ket = ao.copy()
    elif kind == 'complex':
        ao = ket = ao.astype(cp.complex128)
    elif kind == 'strided':
        ao = ket = ao[:, :, ::-1]
    elif kind == 'weights':
        weights = weights.astype(cp.float32)
    elif kind == 'output':
        out = cp.empty((384, 384), dtype=cp.float32)
    elif kind == 'nonfinite_weights':
        weights[0] = cp.inf
    elif kind == 'zero_weight_nonfinite_ao':
        ao[1, 0, 0], weights[0] = cp.nan, 0
    elif kind == 'alias':
        out = ao[1].reshape(-1)[:384**2].reshape(384, 384)
    elif kind == 'device':
        monkeypatch.setattr(_tau_triangle, '_supported_device', lambda device: False)
    calls, sentinel = [], object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(numint, '_tau_dot_full', original)
    assert not _tau_triangle._eligible(ao, ket, weights, out)
    assert numint._tau_dot(ao, ket, weights, out=out) is sentinel
    assert len(calls) == 1


def test_triangle_streams_and_supplied_output_need_no_workspace(monkeypatch):
    import cupy as cp
    import numpy as np

    from gpu4pyscf.dft import _tau_triangle, numint

    rng = np.random.default_rng(1314)
    ao, weights = cp.asarray(rng.normal(size=(4, 33, 83))), cp.asarray(rng.normal(size=83))
    expected = numint._tau_dot_full(ao, ao, weights)
    outputs = [cp.empty((33, 33)) for _ in range(2)]
    _tau_triangle._triangular_tau(ao, weights, out=outputs[0])
    cp.cuda.get_current_stream().synchronize()
    streams = [cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)]

    def allocate(*args, **kwargs):
        raise AssertionError('Supplied-output triangle calls must not allocate a stacked AO workspace')

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'empty', allocate)
        for stream, out in zip(streams, outputs):
            with stream:
                _tau_triangle._triangular_tau(ao, weights, out=out)
        for stream in streams:
            stream.synchronize()
    for out in outputs:
        cp.testing.assert_allclose(out, expected, atol=2e-12, rtol=2e-12)


def test_private_triangle_preserves_early_float32_dispatch(monkeypatch):
    import cupy as cp
    import numpy as np
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import mixed_precision, numint

    mf = dft.RKS(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0), xc='wb97m-v')
    rng = np.random.default_rng(1315)
    ao, weights = cp.asarray(rng.normal(size=(4, 400, 2048))), cp.asarray(rng.normal(size=2048))
    original, calls = numint._triangular_tau, []

    def triangle(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(numint, '_triangular_tau', triangle)
    with mixed_precision._precision_context(mf) as (grid, _):
        namespace = mf._numint.nr_rks.__func__.__globals__
        expected = namespace['_tau_dot_full'](ao, ao, weights)
        actual = namespace['_tau_dot'](ao, ao, weights)
        cp.testing.assert_array_equal(actual, expected)
        assert not calls and grid.float32_calls == 6 and grid.float64_calls == 0
        grid.use_float32 = False
        actual = namespace['_tau_dot'](ao, ao, weights)
        cp.testing.assert_allclose(actual, numint._tau_dot_full(ao, ao, weights), atol=2e-11, rtol=2e-12)
        assert calls == [1] and grid.float64_calls == 0


@pytest.mark.parametrize('case', [*ASE_ORGANICS.values(), 'water-cation'])
def test_triangle_complete_organic_scf_forces_and_finite_difference(case, monkeypatch):
    import numpy as np
    from ase.build import molecule
    from pyscf import gto

    from gpu4pyscf import dft
    from gpu4pyscf.dft import _tau_triangle, mixed_precision, numint
    from gpu4pyscf.scf import hf

    monkeypatch.setattr(hf, 'overlap_zero_eigenvalue_threshold', 1e-10)
    multiplicity = 2 if case == 'water-cation' else 1
    atoms = molecule('H2O' if case == 'water-cation' else case)
    mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)), basis='def2-tzvpd',
                charge=multiplicity - 1, spin=multiplicity - 1, verbose=0)
    mf = (dft.RKS if multiplicity == 1 else dft.UKS)(mol, xc='wb97m-v').density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level, mf.nlcgrids.level = 5, 3
    mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-11, 1e-7, None
    monkeypatch.setattr(_tau_triangle, '_MIN_ORBITALS', 10**6)
    reference_energy = mixed_precision.run_verified_scf(mf)
    gradient = mf.nuc_grad_method()
    gradient.grid_response = gradient.auxbasis_response = True
    reference_force = np.asarray(gradient.kernel())
    monkeypatch.setattr(_tau_triangle, '_MIN_ORBITALS', 1)
    monkeypatch.setattr(_tau_triangle, '_MIN_GRIDS', 1)
    original, calls = numint._triangular_tau, []

    def triangle(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(numint, '_triangular_tau', triangle)
    mf.reset(mol)
    actual_energy = mixed_precision.run_verified_scf(mf)
    assert calls and mf.mixed_precision_info['accepted']
    actual_force = np.asarray(gradient.kernel())
    assert abs(actual_energy - reference_energy) < 1e-9
    np.testing.assert_allclose(actual_force, reference_force, atol=1e-7, rtol=1e-7)
    if case in ('CH3CONH2', 'water-cation'):
        coords, step = mol.atom_coords(), 2e-4
        atom_index = atoms.get_chemical_symbols().index('H')
        energies = []
        for sign in (-1, 1):
            displaced = coords.copy()
            displaced[atom_index, 0] += sign * step
            mf.reset(mol.set_geom_(displaced, unit='Bohr', inplace=False))
            energies.append(mixed_precision.run_verified_scf(mf))
        assert abs(actual_force[atom_index, 0] - (energies[1] - energies[0]) / (2 * step)) < 2e-5
