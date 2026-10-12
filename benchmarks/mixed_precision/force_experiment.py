"""Test the density-contraction change against matching installed-release gradients."""

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager


def _replace(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('Installed gradient differs from the reviewed compatibility boundary')
    return source.replace(before, after)


@contextmanager
def density_response_patch(root, *, vv10_fused=False):
    """Patch only the reviewed contraction sites, retaining installed numerical code.

    This adapter tests new Python algebra against the pinned 1.8.1 wheel. It is
    not a native build or a replacement of that release's other Python sources.
    """
    from gpu4pyscf.grad import rks

    path = root / 'gpu4pyscf/grad/rks.py'
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_contract_grad_response')
    reduction = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_reduce_to_atom')
    namespace = rks.__dict__.copy()
    exec(compile(ast.Module(body=[helper, reduction], type_ignores=[]), str(path), 'exec'), namespace)
    original = {name: getattr(rks, name) for name in ('get_exc_full_response', 'get_nlc_exc_full_response')}
    for name, function in original.items():
        source = inspect.getsource(function)
        source = _replace(source, 'dvmat_orbital_response = cupy.zeros((3, nao, nao))',
                          'de_orbital_response = cupy.zeros((nao, 3))')
        if name == 'get_exc_full_response':
            begin = source.index("        if xctype == 'LDA':", source.index('    g0 = 0'))
            end = source.index('\n        g0 = g1', begin)
            candidate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            candidate_source = ast.get_source_segment(path.read_text(), candidate)
            new_begin = candidate_source.index('        split_wv = ', candidate_source.index('    g0 = 0'))
            new_end = candidate_source.index('\n        g0 = g1', new_begin)
            if 'dvmat_orbital_response' not in source[begin:end]:
                raise RuntimeError('Missing reviewed derivative-matrix contraction')
            source = source[:begin] + candidate_source[new_begin:new_end] + source[end:]
            source = _replace(source,
                              'rhf_grad.contract_h1e_dm(opt._sorted_mol, dvmat_orbital_response, dms, hermi=1)',
                              '_reduce_to_atom(opt._sorted_mol, de_orbital_response) * 2')
        else:
            source = _replace(source, '''        vtmp = _gga_grad_sum_(ao, wv)
        dvmat_orbital_response[numpy.ix_(range(3), idx, idx)] += vtmp
        dms_masked = take_last2d(dms_sorted, idx, out = dm_mask_buf)
        de_grid_response_rho[i_atom_of_grids] += cupy.einsum('xij,ji->x', vtmp, dms_masked) * 2

        del wv, vtmp''', '''        dms_masked = take_last2d(dms_sorted, idx, out = dm_mask_buf)
        partial = _contract_grad_response(ao, wv, dms_masked)
        de_orbital_response[idx] += partial
        de_grid_response_rho[i_atom_of_grids] += partial.sum(axis=0) * 2

        del wv, partial''')
            source = _replace(source,
                              'rhf_grad.contract_h1e_dm(_sorted_mol, dvmat_orbital_response, dms_sorted, hermi=1)',
                              '_reduce_to_atom(_sorted_mol, de_orbital_response) * 2')
            if vv10_fused:
                from gpu4pyscf.dft._vv10_grad import _pair_response

                namespace['_pair_response'] = _pair_response
                begin = source.index('    U_i = cupy.empty(ngrids)')
                end = source.index('\n    fw_rho_i =', begin)
                if source[begin:end].count('libgdft.VXC_vv10nlc_fock_eval_UWE(') != 1:
                    raise RuntimeError('Missing reviewed native VV10 field call')
                source = source[:begin] + '''    grid_to_atom_index_map = cupy.ascontiguousarray(grids.atm_idx[rho_nonzero_mask], dtype=cupy.int32)
    U_i, W_i, E_i, de_grid_response_phi = _pair_response(
        grids_coords, rho_weight_i, omega_i, kappa_i, grid_to_atom_index_map, natm)
''' + source[end:]
                begin = source.index('    grid_to_atom_index_map = grids.atm_idx')
                end = source.index('    del omega_i, kappa_i', begin)
                if source[begin:end].count('libgdft.VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal(') != 1:
                    raise RuntimeError('Missing reviewed native VV10 grid-response call')
                source = source[:begin] + source[end:]
        exec(compile(source, str(path) + ':installed-release-contraction-adapter', 'exec'), namespace)
    try:
        for name in original:
            setattr(rks, name, namespace[name])
        yield namespace['_contract_grad_response']
    finally:
        for name, function in original.items():
            setattr(rks, name, function)
