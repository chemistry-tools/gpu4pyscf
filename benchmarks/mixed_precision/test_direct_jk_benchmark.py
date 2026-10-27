"""Check that direct-J/K representation experiments preserve other builders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load(monkeypatch):
    root = Path(__file__).parent
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('direct_pilot', root/'direct_jk_benchmark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('gpu4pyscf', 'gpu4pyscf.gto', 'gpu4pyscf.gto.mole'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    class SortedGTO:
        @classmethod
        def from_mol(cls, mol, **kwargs):
            return mol, kwargs

    sys.modules['gpu4pyscf.gto.mole'].SortedGTO = SortedGTO
    return module, SortedGTO


@pytest.mark.parametrize('kwargs, expected', [
    ({'decontract': True, 'diffuse_cutoff': .3}, 10.),
    ({'decontract': True, 'diffuse_cutoff': 1e200}, 1e200),
    ({'decontract': False, 'diffuse_cutoff': .3}, .3),
    ({'decontract': True, 'diffuse_cutoff': None}, None),
])
def test_exchange_split_preserves_coulomb_and_other_representations(monkeypatch, kwargs, expected):
    module, sorted_gto = _load(monkeypatch)
    original = sorted_gto.__dict__['from_mol']
    molecule = object()
    with module._primitive_split(10.):
        returned_mol, returned_kwargs = sorted_gto.from_mol(molecule, **kwargs)
        assert returned_mol is molecule
        assert returned_kwargs['diffuse_cutoff'] == expected
        assert returned_kwargs['decontract'] == kwargs['decontract']
    assert sorted_gto.__dict__['from_mol'] is original


def test_representation_restored_after_failure(monkeypatch):
    module, sorted_gto = _load(monkeypatch)
    original = sorted_gto.__dict__['from_mol']
    with pytest.raises(RuntimeError, match='kernel failure'), module._primitive_split(10.):
        raise RuntimeError('kernel failure')
    assert sorted_gto.__dict__['from_mol'] is original


def test_default_does_not_import_gpu_modules(monkeypatch):
    module, _ = _load(monkeypatch)
    monkeypatch.delitem(sys.modules, 'gpu4pyscf.gto.mole')
    with module._primitive_split(None):
        assert 'gpu4pyscf.gto.mole' not in sys.modules
