"""Measure AO tile occupancy without changing numerical screening or products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_TILES = ((16, 16), (16, 32), (32, 32), (64, 64))
DEFAULT_THRESHOLDS = (0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8)
COUNT_KEYS = (
    'entries',
    'negligible_entries',
    'tiles',
    'negligible_tiles',
    'tile_entries',
    'negligible_tile_entries',
    'gram_tile_products',
    'skippable_gram_tile_products',
    'gram_element_products',
    'skippable_gram_element_products',
)


def _tile_maxima(values, rows, columns):
    import numpy as np

    return np.maximum.reduceat(
        np.maximum.reduceat(values, np.arange(0, len(values), rows), axis=0),
        np.arange(0, values.shape[1], columns),
        axis=1,
    )


def _fractions(counts):
    return {
        'negligible_entry_fraction': counts['negligible_entries'] / counts['entries'],
        'negligible_tile_fraction': counts['negligible_tiles'] / counts['tiles'],
        'tile_covered_entry_fraction': counts['negligible_tile_entries'] / counts['tile_entries'],
        'skippable_gram_tile_product_fraction': counts['skippable_gram_tile_products'] / counts['gram_tile_products'],
        'skippable_gram_element_product_fraction': counts['skippable_gram_element_products']
        / counts['gram_element_products'],
    }


def measure_fields(ao, *, tiles=DEFAULT_TILES, thresholds=DEFAULT_THRESHOLDS):
    """Return joint-component occupancy and conservative Gram-work opportunities.

    Thresholds describe hypothetical entry pruning, not bounds on density, energy
    or force error. Gram counts retain a tile pair whenever both AO tiles have
    any active component; they do not assume overlap of their active entries.
    """
    import numpy as np

    ao = np.asarray(ao)
    if ao.ndim == 2:
        ao = ao[None]
    if (
        ao.ndim != 3
        or ao.shape[0] not in (1, 4, 10)
        or min(ao.shape[1:]) < 1
        or ao.dtype != np.float64
        or not np.isfinite(ao).all()
    ):
        raise ValueError('Use finite float64 AO fields with 1, 4 or 10 components and nonempty axes')
    if (
        not tiles
        or any(len(tile) != 2 or any(type(x) is not int or x < 1 for x in tile) for tile in tiles)
        or not thresholds
        or any(not np.isfinite(t) or t < 0 for t in thresholds)
    ):
        raise ValueError('Tile dimensions must be positive integers and thresholds finite/nonnegative')
    groups = {'value': (0, 1)}
    if len(ao) >= 4:
        groups.update(d1=(1, 4), value_d1=(0, 4))
    if len(ao) == 10:
        groups.update(d2=(4, 10), all=(0, 10))
    m, n = ao.shape[1:]
    result, maxima = [], {}
    for field, (start, end) in groups.items():
        values = np.max(np.abs(ao[start:end]), axis=0)
        maxima[field] = _tile_maxima(values, 16, 16)
        negligible = {float(eps): int(np.count_nonzero(values <= eps)) for eps in thresholds}
        for rows, columns in tiles:
            peak = _tile_maxima(values, rows, columns)
            row_sizes = np.minimum(rows, m - np.arange(0, m, rows))
            column_sizes = np.minimum(columns, n - np.arange(0, n, columns))
            r, k = peak.shape
            dense_gram = r * (r + 1) // 2 * k
            for eps in thresholds:
                inactive = peak <= eps
                active_rows = np.sum(~inactive, axis=0, dtype=np.int64)
                active_entries = row_sizes @ (~inactive).astype(np.int64)
                retained_gram = int(np.sum(active_rows * (active_rows + 1) // 2))
                retained_elements = int(np.sum(active_entries**2 * column_sizes))
                counts = {
                    'entries': m * n,
                    'negligible_entries': negligible[float(eps)],
                    'tiles': r * k,
                    'negligible_tiles': int(np.count_nonzero(inactive)),
                    'tile_entries': m * n,
                    'negligible_tile_entries': int(row_sizes @ inactive.astype(np.int64) @ column_sizes),
                    'gram_tile_products': dense_gram,
                    'skippable_gram_tile_products': dense_gram - retained_gram,
                    'gram_element_products': m * m * n,
                    'skippable_gram_element_products': m * m * n - retained_elements,
                }
                result.append(
                    {
                        'field': field,
                        'components': [start, end],
                        'tile': [rows, columns],
                        'threshold': float(eps),
                        'counts': counts,
                        'fractions': _fractions(counts),
                    }
                )
    return result, maxima


def summarize(blocks):
    """Aggregate counts with partial-tile sizes instead of averaging percentages."""
    groups = {}
    for block in blocks:
        for row in block['measurements']:
            key = row['field'], *row['tile'], row['threshold']
            if key not in groups:
                groups[key] = {
                    'field': row['field'],
                    'tile': row['tile'],
                    'threshold': row['threshold'],
                    'blocks': 0,
                    'counts': {name: 0 for name in COUNT_KEYS},
                }
            group = groups[key]
            group['blocks'] += 1
            for name in COUNT_KEYS:
                group['counts'][name] += row['counts'][name]
    return [row | {'fractions': _fractions(row['counts'])} for row in groups.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', default='4,5,6,7')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    from gpu_performance import _resources

    _resources([int(x) for x in args.cpus.split(',')])
    import numpy as np

    blocks = []
    for path in args.input:
        with np.load(path, allow_pickle=False) as data:
            ao = data['ao']
        rows, _ = measure_fields(ao)
        block = {
            'input': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'shape': list(ao.shape),
            'measurements': rows,
        }
        blocks.append(block)
        print(json.dumps({'input': str(path), 'shape': list(ao.shape)}), flush=True)
    report = {
        'schema_version': 1,
        'note': 'Occupancy only: nonzero thresholds are hypothetical pruning; no runtime or energy/force-error claim.',
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'blocks': blocks,
        'summary': summarize(blocks),
    }
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
