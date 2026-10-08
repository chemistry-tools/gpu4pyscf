"""Compare matched ASE benchmark reports without mixing methods or geometries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _distance(first, second):
    if len(first) != 3 or len(second) != 3:
        raise ValueError('Forces must be Cartesian triples')
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def compare(baseline, candidate):
    """Summarize timing and identical-geometry errors for one frozen case."""
    if baseline['precision'] != 'float64' or candidate['precision'] != 'mixed':
        raise ValueError('Compare a float64 baseline with a mixed candidate')
    for key in (
        'recipe',
        'case',
        'input_sha256',
        'versions',
        'cpu_affinity',
        'openmp_threads',
        'gpu_uuid',
        'requested_steps',
        'fmax_target_ev_angstrom',
    ):
        if baseline[key] != candidate[key]:
            raise ValueError(f'Incompatible reports: {key}')
    for key in ('source_sha256', 'native_library_sha256'):
        if baseline['gpu4pyscf_source'][key] != candidate['gpu4pyscf_source'][key]:
            raise ValueError(f'Incompatible numerical sources: {key}')
    for report in (baseline, candidate):
        if not math.isfinite(report['calculation_seconds']) or report['calculation_seconds'] <= 0:
            raise ValueError('Calculation time must be finite and positive')
        if not report['frames']:
            raise ValueError('Reports must contain evaluated geometries')
    if any(not frame['precision_info']['accepted'] for frame in candidate['frames']):
        raise ValueError('Every mixed frame must pass float64 verification')
    reference_units = baseline['unit_constants']
    mixed_units = candidate['unit_constants']
    force_scale = (reference_units['hartree_ev'] / reference_units['bohr_angstrom']) / (
        mixed_units['hartree_ev'] / mixed_units['bohr_angstrom']
    )
    matched = []
    used = set()
    for candidate_index, frame in enumerate(candidate['frames']):
        for baseline_index, reference in enumerate(baseline['frames']):
            if baseline_index in used or frame['positions_angstrom'] != reference['positions_angstrom']:
                continue
            used.add(baseline_index)
            # Compare Hartree energies: interfaces can use different eV conversion constants.
            energy_difference = frame['energy_hartree'] - reference['energy_hartree']
            if not math.isfinite(energy_difference):
                raise ValueError('Matched energies must be finite')
            forces, reference_forces = frame['forces_ev_angstrom'], reference['forces_ev_angstrom']
            if len(forces) != len(reference_forces) or len(forces) != len(frame['positions_angstrom']):
                raise ValueError('Forces and positions must have matching atom counts')
            force_difference = max(_distance([x * force_scale for x in a], b) for a, b in zip(forces, reference_forces))
            if not math.isfinite(force_difference):
                raise ValueError('Matched forces must be finite')
            matched.append(
                {
                    'baseline_frame': baseline_index,
                    'mixed_frame': candidate_index,
                    'energy_difference_hartree': energy_difference,
                    'max_atomic_force_difference_ev_angstrom': force_difference,
                }
            )
            break
    if not matched:
        raise ValueError('No identical geometries: replay a frozen geometry before comparing accuracy')
    timing = {}
    for name in ('scf_seconds', 'force_seconds'):
        reference = sum(frame['calculator_timing'][name] for frame in baseline['frames'])
        mixed = sum(frame['calculator_timing'][name] for frame in candidate['frames'])
        timing[name] = {'float64': reference, 'mixed': mixed, 'speedup': reference / mixed}
    return {
        'case_id': baseline['case']['id'],
        'float64_seconds': baseline['calculation_seconds'],
        'mixed_seconds': candidate['calculation_seconds'],
        'total_speedup': baseline['calculation_seconds'] / candidate['calculation_seconds'],
        'component_timing': timing,
        'matched_geometries': matched,
        'unmatched_baseline_frames': len(baseline['frames']) - len(matched),
        'unmatched_mixed_frames': len(candidate['frames']) - len(matched),
        'mixed_fallback_frames': sum(bool(frame['precision_info']['fallback_scf']) for frame in candidate['frames']),
        'baseline_optimization_converged': baseline['optimization_converged'],
        'mixed_optimization_converged': candidate['optimization_converged'],
        'benchmark_revisions': [baseline['benchmark_revision'], candidate['benchmark_revision']],
        'python_extensions_identical': baseline['python_extension_sha256'] == candidate['python_extension_sha256'],
        'note': 'Single-run timings; shared-host contention and different optimizer paths can affect speedup.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--mixed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.mixed.read_text()))
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
