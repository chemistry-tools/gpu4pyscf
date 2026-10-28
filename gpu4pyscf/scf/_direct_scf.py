"""Select and cache the qualified molecular direct-integral representation."""

from __future__ import annotations

import hashlib

NATIVE_PRIMITIVE_CUTOFF = .3
REGROUPED_PRIMITIVE_CUTOFF = 10.
_REGROUPED_ELEMENTS = frozenset((1, 6, 7, 8, 9, 16, 17))


def primitive_cutoff(mf, *, device_count, capability, mol=None):
    """Keep unqualified recipes and explicit controls on the original representation."""
    enabled = getattr(mf, 'direct_scf_regroup', False)
    if type(enabled) is not bool:
        raise ValueError('direct_scf_regroup must be bool')
    if mol is None:
        mol = mf.mol
    xc = getattr(mf, 'xc', None)
    basis = mol.basis
    if (
        not enabled or device_count != 1 or capability != (8, 6) or hasattr(mf, 'cell')
        or not mf.istype('RKS') or getattr(mf, 'with_df', None) is not None
        or mol.cart or mol.spin != 0
        or not isinstance(xc, str) or xc.lower().replace('_', '-') != 'wb97m-v'
        or not isinstance(basis, str) or basis.lower().replace('-', '') != 'def2tzvpd'
        or not all(z in _REGROUPED_ELEMENTS for z in mol.atom_charges())
        or any(cls.__module__.endswith('_lowmem') for cls in type(mf).__mro__)
    ):
        return NATIVE_PRIMITIVE_CUTOFF
    return REGROUPED_PRIMITIVE_CUTOFF


def force_exchange(mf, omega, alpha, hyb, *, device_count, capability, mol=None):
    """Return full and remainder exchange weights for the native gradient passes."""
    if omega == 0:
        return hyb, 0., 0.
    if primitive_cutoff(mf, device_count=device_count, capability=capability, mol=mol) == REGROUPED_PRIMITIVE_CUTOFF:
        # hyb*full + (alpha-hyb)*LR avoids the doubled roots of the SR pass.
        # Keep the original split on unqualified methods and devices.
        return hyb, omega, alpha-hyb
    return alpha, -omega, hyb-alpha


def get_optimizer(mf, mol, omega, *, optimizer_type, device_count, capability):
    """Rebuild only the instance's affected cache entry when its settings change."""
    cutoff = primitive_cutoff(mf, device_count=device_count, capability=capability, mol=mol)
    budget = getattr(mf, 'direct_scf_cache_bytes', 0)
    if type(budget) is not int or budget < 0:
        raise ValueError('direct_scf_cache_bytes must be a nonnegative integer')
    cache_all = getattr(mf, 'direct_scf_cache_all', False)
    if type(cache_all) is not bool:
        raise ValueError('direct_scf_cache_all must be bool')
    if cutoff != REGROUPED_PRIMITIVE_CUTOFF:
        if cache_all:
            raise ValueError('All-VRAM integral cache requires qualified direct_scf_regroup=True')
        budget = 0
    source_key = _cache_source_key(mol) if budget or cache_all else None
    opt = mf._opt_gpu.get(omega)
    if (
        opt is None or opt.mol is not mol
        or opt.direct_scf_tol != mf.direct_scf_tol
        or getattr(opt, 'diffuse_cutoff', NATIVE_PRIMITIVE_CUTOFF) != cutoff
        or getattr(opt, '_direct_scf_cache_bytes', 0) != budget
        or getattr(opt, '_direct_scf_cache_all', False) != cache_all
        or (budget or cache_all) and getattr(opt, '_cache_source_key', None) != source_key
    ):
        with mol.with_range_coulomb(omega):
            opt = optimizer_type(mol, mf.direct_scf_tol, diffuse_cutoff=cutoff).build()
        opt._direct_scf_regroup = cutoff == REGROUPED_PRIMITIVE_CUTOFF
        opt._direct_scf_cache_bytes = budget
        opt._direct_scf_cache_all = cache_all
        opt._cache_source_key = source_key
        # Scanners/copies may initially share the old dictionary. Publish a new one
        # only after a successful build, without changing the source instance.
        mf._opt_gpu = {**mf._opt_gpu, omega: opt}
    return opt


def _cache_source_key(mol):
    from pyscf.gto import PTR_RANGE_OMEGA

    digest = hashlib.blake2b(digest_size=16)
    # The range operator belongs to the optimizer/call key, not the geometry.
    for array in (mol._atm, mol._bas, mol._env[:PTR_RANGE_OMEGA],
                  mol._env[PTR_RANGE_OMEGA+1:]):
        digest.update(array.tobytes())
    return digest.digest()


def force_kernel(opt, native_library):
    """Select contracted derivatives only for an owner-qualified optimizer."""
    if getattr(opt, '_direct_scf_regroup', False) is True:
        contracted = getattr(native_library, 'RYS_per_atom_jk_ip1_contract', None)
        if contracted is not None:
            return contracted
    return native_library.RYS_per_atom_jk_ip1
