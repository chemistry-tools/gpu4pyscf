"""Prevent sparse force sampling from overstating benchmark timing or validation."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from tf32_benchmark import _df_storage, _scf_summary


def _records(first_only):
    def row(mode, seconds, forces, force_seconds):
        return {'case': 'synthetic', 'mode': mode, 'scf_seconds': seconds,
                'forces_hartree_bohr': forces, 'force_seconds': force_seconds,
                'energy_hartree': -10.0, 'precision_info': {'fallback_scf': False},
                'cycles': 7, 'effective_orbitals': 100, 'grid_points': 4096, 'nlc_grid_points': 2048}
    records = [row('float64', 30, [[0.0, 0.0, 0.0]], 2)]
    for mode, offset in (('fp32', 0), ('emulated-grid', 1)):
        for run, seconds in enumerate((20, 10, 15)):
            sampled = not first_only or run == 0
            records.append(row(mode, seconds + offset, [[0.0, 0.0, 3e-7]] if sampled else None,
                               2 if sampled else None))
    return records


@pytest.mark.parametrize('first_only', [True, False])
def test_force_sampling_counts_and_paired_timing(first_only):
    result = _scf_summary(np, _records(first_only), ('synthetic',), ('emulated-grid',), True)['synthetic']
    assert result['fp32']['scf_samples'] == 3
    assert result['fp32']['force_samples'] == (1 if first_only else 3)
    assert result['fp32']['median_scf_seconds'] == 15
    assert result['fp32']['median_energy_force_seconds'] == (22 if first_only else 17)
    assert result['emulated-grid']['max_force_error_hartree_bohr'] == pytest.approx(3e-7)


def test_unsampled_repeat_energy_is_still_verified():
    records = _records(True)
    records[2]['energy_hartree'] += 1e-6
    with pytest.raises(RuntimeError, match='beyond comparison thresholds'):
        _scf_summary(np, records, ('synthetic',), ('emulated-grid',), True)


def test_native_force_reference_is_required():
    records = _records(True)
    records[0]['forces_hartree_bohr'] = None
    with pytest.raises(RuntimeError, match='force measurements are required'):
        _scf_summary(np, records, ('synthetic',), ('emulated-grid',), True)


@pytest.mark.parametrize('field', ['effective_orbitals', 'grid_points', 'nlc_grid_points'])
def test_changed_basis_or_grid_is_rejected(field):
    records = deepcopy(_records(True))
    records[2][field] -= 1
    with pytest.raises(RuntimeError, match='same basis and grids'):
        _scf_summary(np, records, ('synthetic',), ('emulated-grid',), True)


@pytest.mark.parametrize('container', [list, tuple, lambda values: dict(enumerate(values))])
def test_df_storage_handles_native_device_buffer_containers(container):
    class GpuBuffer:
        nbytes = 200
    cpu = np.zeros((5, 5))
    ranged = SimpleNamespace(_cderi=container([cpu]))
    df = SimpleNamespace(_cderi=container([GpuBuffer()]), _rsh_df={'0.3': ranged})
    storage = _df_storage(SimpleNamespace(ndarray=GpuBuffer), np, SimpleNamespace(with_df=df))
    assert storage['full_range'] == {'gpu_bytes': 200, 'cpu_backed_bytes': 0}
    assert storage['omega=0.3'] == {'gpu_bytes': 0, 'cpu_backed_bytes': 200}
