"""Fixed generic mixed-atom organic geometries for numerical development checks."""

from __future__ import annotations

import json
from pathlib import Path

from ase.build import molecule

ASE_ORGANICS = {
    'acetamide': 'CH3CONH2',
    'pyridine': 'C5H5N',
    'dimethyl-sulfoxide': 'C2H6SO',
    'acetyl-fluoride': 'CH3COF',
    'acetyl-chloride': 'CH3COCl',
}


def _ase_geometry(name):
    atoms = molecule(name)
    geometry = '; '.join(
        f'{symbol} {x:.10f} {y:.10f} {z:.10f}'
        for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)
    )
    return geometry, 0, 1


ORGANIC_CASES = {name: _ase_geometry(source) for name, source in ASE_ORGANICS.items()}
_larger = json.loads(Path(__file__).with_name('organic_geometry.json').read_text())
ORGANIC_CASES[_larger['name']] = _larger['geometry_angstrom'], _larger['charge'], _larger['multiplicity']
