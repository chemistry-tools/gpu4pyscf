"""Check tile occupancy counts, partial tiles and derivative unions on CPU."""

from __future__ import annotations

import numpy as np
import pytest
from ao_tile_sparsity import measure_fields, summarize


def test_block_diagonal_gram_work_counts():
    ao = np.zeros((4, 4))
    ao[:2, :2] = 1
    ao[2:, 2:] = -1
    rows, _ = measure_fields(ao, tiles=((2, 2),), thresholds=(0.0,))
    row = rows[0]
    assert row['counts']['negligible_tiles'] == 2
    assert row['counts']['gram_tile_products'] == 6
    assert row['counts']['skippable_gram_tile_products'] == 4
    assert row['fractions']['skippable_gram_element_product_fraction'] == 0.75


def test_partial_tiles_exclude_padding_and_use_absolute_values():
    ao = np.zeros((3, 5))
    ao[2, 4] = -1
    rows, _ = measure_fields(ao, tiles=((2, 2),), thresholds=(0.0, 0.5, 1.0))
    assert rows[0]['counts']['tiles'] == 6
    assert rows[0]['counts']['negligible_entries'] == 14
    assert rows[0]['counts']['negligible_tile_entries'] == 14
    assert rows[1]['counts'] == rows[0]['counts']
    assert rows[2]['fractions']['tile_covered_entry_fraction'] == 1


def test_derivative_components_prevent_value_only_skips():
    ao = np.zeros((10, 3, 5))
    ao[2, 0, 0] = 2
    ao[9, 2, 4] = 3
    rows, maxima = measure_fields(ao, tiles=((2, 2),), thresholds=(0.0,))
    by_field = {row['field']: row for row in rows}
    assert by_field['value']['counts']['negligible_tiles'] == 6
    assert by_field['value_d1']['counts']['negligible_tiles'] == 5
    assert by_field['all']['counts']['negligible_tiles'] == 4
    assert maxima['all'].max() == 3


def test_aggregation_uses_counts_instead_of_mean_percentages():
    blocks = []
    for ao in [np.zeros((1, 1)), np.ones((3, 3))]:
        rows, _ = measure_fields(ao, tiles=((2, 2),), thresholds=(0.0,))
        blocks.append({'measurements': rows})
    row = summarize(blocks)[0]
    assert row['fractions']['negligible_entry_fraction'] == 0.1
    assert row['fractions']['negligible_tile_fraction'] == 0.2


def test_gram_estimate_matches_explicit_partial_tile_products():
    rng = np.random.default_rng(18)
    ao = rng.normal(size=(7, 11))
    for i in range(0, 7, 3):
        for j in range(0, 11, 4):
            if rng.random() < 0.6:
                ao[i : i + 3, j : j + 4] = 0
    rows, _ = measure_fields(ao, tiles=((3, 4),), thresholds=(0.0,))
    tile_products = element_products = 0
    for k in range(0, 11, 4):
        for i in range(0, 7, 3):
            for j in range(0, 7, 3):
                left = ao[i : i + 3, k : k + 4]
                right = ao[j : j + 3, k : k + 4]
                if not left.any() or not right.any():
                    element_products += left.shape[0] * right.shape[0] * left.shape[1]
                    if j <= i:
                        tile_products += 1
    assert rows[0]['counts']['skippable_gram_tile_products'] == tile_products
    assert rows[0]['counts']['skippable_gram_element_products'] == element_products


@pytest.mark.parametrize('kind', ['nan', 'complex', 'components', 'empty'])
def test_invalid_fields_are_rejected(kind):
    ao = np.zeros((4, 3, 5))
    if kind == 'nan':
        ao[0, 0, 0] = np.nan
    elif kind == 'complex':
        ao = ao.astype(np.complex128)
    elif kind == 'components':
        ao = ao[:3]
    else:
        ao = ao[:, :0]
    with pytest.raises(ValueError):
        measure_fields(ao)
