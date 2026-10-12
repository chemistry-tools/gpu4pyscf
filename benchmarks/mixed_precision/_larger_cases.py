"""Deterministic idealized fused aromatics for development-only size comparisons."""

from __future__ import annotations

import math


def _fused_rings(centers, cc=1.4, ch=1.09):
    vertices = {}
    neighbors = {}
    for q, r in centers:
        x, y = math.sqrt(3) * cc * (q + r / 2), 1.5 * cc * r
        ring = []
        for index in range(6):
            angle = math.radians(30 + 60 * index)
            key = (round(x + cc * math.cos(angle), 8), round(y + cc * math.sin(angle), 8))
            vertices[key] = key
            neighbors.setdefault(key, set())
            ring.append(key)
        for index, first in enumerate(ring):
            second = ring[(index + 1) % 6]
            neighbors[first].add(second)
            neighbors[second].add(first)
    if any(len(adjacent) not in (2, 3) for adjacent in neighbors.values()):
        raise ValueError('Fused aromatic carbons must have two or three carbon neighbors')
    origin = [sum(point[index] for point in vertices) / len(vertices) for index in (0, 1)]
    atoms = []
    hydrogens = []
    for point in sorted(vertices):
        x, y = point[0] - origin[0], point[1] - origin[1]
        atoms.append(f'C {x:.8f} {y:.8f} 0')
        if len(neighbors[point]) == 2:
            vector = [sum(point[index] - other[index] for other in neighbors[point]) for index in (0, 1)]
            norm = math.hypot(*vector)
            hydrogens.append(f'H {x + ch * vector[0] / norm:.8f} {y + ch * vector[1] / norm:.8f} 0')
    return '; '.join(atoms + hydrogens), 0, 1


LARGER_CASES = {
    'naphthalene': _fused_rings(((0, 0), (1, 0))),
    'pyrene': _fused_rings(((0, 0), (1, 0), (0, 1), (1, 1))),
    'coronene': _fused_rings(((0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))),
    'ovalene': _fused_rings(
        ((0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, -1), (2, 0), (1, 1))
    ),
}
