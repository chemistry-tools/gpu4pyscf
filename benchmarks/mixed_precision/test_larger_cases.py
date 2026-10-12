"""Check identity and valence of the synthetic size-comparison geometries."""

from __future__ import annotations

import math
from collections import Counter

import pytest
from _larger_cases import LARGER_CASES


@pytest.mark.parametrize('name,formula', [('naphthalene', {'C': 10, 'H': 8}),
                                        ('pyrene', {'C': 16, 'H': 10}),
                                        ('coronene', {'C': 24, 'H': 12})])
def test_fused_case_formula_and_valence(name, formula):
    geometry, charge, multiplicity = LARGER_CASES[name]
    atoms = [(parts[0], tuple(map(float, parts[1:]))) for entry in geometry.split(';')
             if (parts := entry.split())]
    assert Counter(symbol for symbol, _ in atoms) == formula
    assert (charge, multiplicity) == (0, 1)
    for index, (symbol, position) in enumerate(atoms):
        distances = [(other, math.dist(position, coordinates)) for j, (other, coordinates) in enumerate(atoms)
                     if index != j]
        assert min(distance for _, distance in distances) > 1.08
        bonded = [(other, distance) for other, distance in distances if distance < 1.5]
        assert len(bonded) == (3 if symbol == 'C' else 1)
        for other, distance in bonded:
            assert distance == pytest.approx(1.4 if symbol == other == 'C' else 1.09, abs=2e-8)
