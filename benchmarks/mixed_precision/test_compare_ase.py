"""Check scientific identity and geometry matching in ASE report comparisons."""

from __future__ import annotations

from copy import deepcopy

import pytest
from compare_ase import compare


def _reports():
    baseline = {
        'precision': 'float64',
        'unit_constants': {'hartree_ev': 27.0, 'bohr_angstrom': 0.5},
        'recipe': {'functional': 'example', 'basis': 'example'},
        'case': {'id': 'example', 'charge': 0, 'multiplicity': 1},
        'input_sha256': 'input',
        'versions': {'pyscf': 'example'},
        'gpu4pyscf_source': {'source_sha256': {'hf.py': 'example'}, 'native_library_sha256': 'native'},
        'python_extension_sha256': {'precision': 'example'},
        'benchmark_revision': 'example',
        'cpu_affinity': [0, 1, 2, 3],
        'openmp_threads': 4,
        'gpu_uuid': 'example',
        'requested_steps': 1,
        'fmax_target_ev_angstrom': 0.005,
        'calculation_seconds': 30.0,
        'optimization_converged': False,
        'frames': [
            {
                'positions_angstrom': [[0.0, 0.0, 0.0]],
                'energy_hartree': -1.0,
                'forces_ev_angstrom': [[0.0, 0.0, 0.0]],
                'calculator_timing': {'scf_seconds': 10.0, 'force_seconds': 5.0},
                'precision_info': None,
            }
        ],
    }
    baseline['frames'].append(deepcopy(baseline['frames'][0]))
    baseline['frames'][1]['positions_angstrom'][0][0] = 0.1
    candidate = deepcopy(baseline)
    candidate['precision'] = 'mixed'
    candidate['calculation_seconds'] = 20.0
    for frame in candidate['frames']:
        frame['precision_info'] = {'accepted': True, 'fallback_scf': False}
        frame['calculator_timing']['scf_seconds'] = 5.0
    return baseline, candidate


def test_matched_geometry_errors_exclude_divergent_optimizer_frames():
    baseline, candidate = _reports()
    candidate['frames'][0]['forces_ev_angstrom'][0][0] = 1e-6
    candidate['frames'][1]['positions_angstrom'][0][0] += 1e-9
    candidate['frames'][1]['forces_ev_angstrom'][0][0] = 100.0
    candidate['frames'][1]['precision_info']['fallback_scf'] = True
    result = compare(baseline, candidate)
    assert result['total_speedup'] == 1.5
    assert result['component_timing']['scf_seconds']['speedup'] == 2.0
    assert result['component_timing']['force_seconds']['speedup'] == 1.0
    assert len(result['matched_geometries']) == 1
    assert result['matched_geometries'][0]['max_atomic_force_difference_ev_angstrom'] == 1e-6
    assert result['unmatched_baseline_frames'] == result['unmatched_mixed_frames'] == 1
    assert result['mixed_fallback_frames'] == 1


@pytest.mark.parametrize('key', ['recipe', 'case', 'input_sha256', 'cpu_affinity', 'gpu_uuid', 'versions'])
def test_different_method_identity_or_resources_are_rejected(key):
    baseline, candidate = _reports()
    candidate[key] = 'different'
    with pytest.raises(ValueError, match=key):
        compare(baseline, candidate)


def test_native_source_changes_are_rejected():
    baseline, candidate = _reports()
    candidate['gpu4pyscf_source']['native_library_sha256'] = 'different'
    with pytest.raises(ValueError, match='native_library'):
        compare(baseline, candidate)


def test_unverified_mixed_frame_is_rejected():
    baseline, candidate = _reports()
    candidate['frames'][1]['precision_info']['accepted'] = False
    with pytest.raises(ValueError, match='verification'):
        compare(baseline, candidate)


def test_without_matching_geometry_accuracy_is_not_reported():
    baseline, candidate = _reports()
    for frame in candidate['frames']:
        frame['positions_angstrom'][0][1] += 1e-9
    with pytest.raises(ValueError, match='No identical geometries'):
        compare(baseline, candidate)


def test_force_unit_conversion_is_normalized_before_comparison():
    baseline, candidate = _reports()
    candidate['unit_constants']['hartree_ev'] *= 2
    for frame in baseline['frames']:
        frame['forces_ev_angstrom'][0][0] = 1.0
    for frame in candidate['frames']:
        frame['forces_ev_angstrom'][0][0] = 2.0
    result = compare(baseline, candidate)
    assert all(row['max_atomic_force_difference_ev_angstrom'] == 0 for row in result['matched_geometries'])
