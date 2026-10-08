"""Load only the new Python precision modules for installed-release compatibility tests."""

from __future__ import annotations

import hashlib
import importlib.util
import sys


def load_extensions(root, *, ase=False):
    """Test only these new Python modules against a matching installed release.

    Leave all existing numerical Python and native library paths intact. This is explicit
    compatibility testing, not evidence of rebuilding the fork's other master changes.
    """
    hashes = {}
    names = ['dft._mixed_grid', 'dft._mixed_vv10', 'dft.mixed_precision']
    if ase:
        names.append('tools.ase_interface')
    for name in names:
        path = root / 'gpu4pyscf' / (name.replace('.', '/') + '.py')
        qualified = 'gpu4pyscf.' + name
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes
