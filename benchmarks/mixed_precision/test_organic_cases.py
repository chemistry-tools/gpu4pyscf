"""Check the mixed-atom organic panel and fixed generic geometry provenance."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from _organic_cases import ORGANIC_CASES


def test_organic_panel_covers_heteroatoms_and_neutral_singlets():
    elements = set()
    for geometry, charge, multiplicity in ORGANIC_CASES.values():
        atoms = [line.split()[0] for line in geometry.split(';')]
        assert 'C' in atoms and 'H' in atoms
        assert charge == 0 and multiplicity == 1
        elements.update(atoms)
    assert {'C', 'H', 'N', 'O', 'S', 'F', 'Cl'} <= elements


def test_larger_organic_keeps_formula_seed_and_fixed_coordinates():
    saved = json.loads(Path(__file__).with_name('organic_geometry.json').read_text())
    geometry, charge, multiplicity = ORGANIC_CASES['ethyl-4-acetamidobenzoate']
    assert Counter(line.split()[0] for line in geometry.split(';')) == {'C': 11, 'H': 13, 'N': 1, 'O': 3}
    assert geometry == saved['geometry_angstrom'] and (charge, multiplicity) == (0, 1)
    assert saved['seed'] == 75 and saved['smiles'] == 'CCOC(=O)c1ccc(NC(=O)C)cc1'
    assert saved['generator_version'] and 'fixed generic development geometry' in saved['generator']
