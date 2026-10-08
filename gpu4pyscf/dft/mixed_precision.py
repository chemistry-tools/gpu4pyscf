"""Verified mixed-precision SCF for molecular single-GPU DFT calculations."""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import FunctionType, MethodType


class SCFVerificationError(RuntimeError):
    """Mixed or recovery SCF failed the original float64 convergence criteria."""


def _private_functions(namespace, replacements):
    """Give an integration call graph private dispatch without changing module globals."""
    private = dict(namespace)
    for name, value in namespace.items():
        if isinstance(value, FunctionType) and value.__globals__ is namespace:
            cloned = FunctionType(value.__code__, private, value.__name__, value.__defaults__, value.__closure__)
            cloned.__kwdefaults__ = value.__kwdefaults__
            private[name] = cloned
    private.update(replacements)
    return private


@contextmanager
def _precision_context(mf):
    import cupy

    from gpu4pyscf.dft import numint
    from gpu4pyscf.dft._mixed_grid import _GridPrecision, _make_contract
    from gpu4pyscf.dft._mixed_vv10 import _AdaptiveVV10

    if numint.num_devices != 1 or cupy.cuda.runtime.getDeviceCount() != 1:
        raise ValueError('Mixed precision currently requires a single visible GPU')
    ni = mf._numint
    grid = _GridPrecision(True)
    vv10 = _AdaptiveVV10(numint._vv10nlc, 128, 'nvrtc')
    private = _private_functions(
        numint.__dict__,
        {
            'contract': _make_contract(numint.contract, grid),
            '_vv10nlc': vv10,
        },
    )
    originals = {}
    try:
        for name in ('nr_rks', 'nr_uks', 'nr_nlc_vxc'):
            bound = getattr(ni, name)
            if getattr(bound, '__func__', None) is not getattr(numint, name):
                raise ValueError(f'Mixed precision does not support a custom NumInt.{name}')
            originals[name] = (name in ni.__dict__, ni.__dict__.get(name))
            setattr(ni, name, MethodType(private[name], ni))
        yield grid, vv10
    finally:
        for name, (had_attribute, previous) in originals.items():
            if had_attribute:
                setattr(ni, name, previous)
            else:
                delattr(ni, name)
        # Private function namespaces form cycles; release buffers without waiting for GC.
        grid._release_scratch()


def _full_check(mf, approximate_energy):
    import cupy

    dm = mf.make_rdm1()
    hcore, overlap = mf.get_hcore(), mf.get_ovlp()
    veff = mf.get_veff(mf.mol, dm)
    # Test the physical Fock, without DIIS, damping or a level shift.
    fock = mf.get_fock(hcore, overlap, veff, dm, level_shift_factor=0)
    energy = float(mf.energy_tot(dm, hcore, veff))
    gradient = float(cupy.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
    gradient_tol = mf.conv_tol_grad
    if gradient_tol is None:
        gradient_tol = math.sqrt(mf.conv_tol)
    return {
        'float64_energy_hartree': energy,
        'energy_change_hartree': energy - approximate_energy,
        'float64_orbital_gradient_norm': gradient,
        'energy_tolerance_hartree': float(mf.conv_tol),
        'gradient_tolerance': float(gradient_tol),
        'passed': bool(mf.converged)
        and math.isfinite(energy)
        and math.isfinite(gradient)
        and abs(energy - approximate_energy) < mf.conv_tol
        and gradient < gradient_tol,
    }


def supports_mixed_precision(mf):
    """Whether the method supports automatic mixed precision before NumInt/device checks."""
    gpu = getattr(mf, 'device', None) == 'gpu' or any(
        cls.__module__.startswith('gpu4pyscf.') for cls in type(mf).__mro__
    )
    return (
        gpu
        and not hasattr(mf, 'cell')
        and hasattr(mf, 'xc')
        and callable(getattr(mf, 'do_nlc', None))
        and mf.do_nlc()
        and getattr(mf, 'check_convergence', None) is None
    )


def run_verified_scf(mf, calculate=None, *, dm0=None):
    """Run mixed SCF, check float64 fields and reconverge at the same geometry if needed.

    ``calculate`` may call a scanner to preserve its normal density reuse and geometry reset.
    All patches are per NumInt instance and restored before returning to derivatives.
    ``dm0`` supplies the native initial density when no scanner callable is used.
    ``mf.max_cycle`` remains the total budget across the mixed and float64 phases.
    Verification failures raise ``SCFVerificationError``; CUDA and callback errors propagate.
    """
    if calculate is not None and (not callable(calculate) or dm0 is not None):
        raise ValueError('Use a scanner callable or dm0, not both')
    # PySCF creates scanner classes in its CPU module even for GPU subclasses.
    gpu_method = getattr(mf, 'device', None) == 'gpu' or any(
        cls.__module__.startswith('gpu4pyscf.') for cls in type(mf).__mro__
    )
    if not gpu_method:
        raise ValueError('Mixed precision requires a GPU4PySCF mean-field object')
    if hasattr(mf, 'cell') or not hasattr(mf, 'xc'):
        raise ValueError('Mixed precision currently supports molecular RKS and UKS only')
    if not mf.do_nlc():
        raise ValueError('Mixed precision currently requires a VV10 functional')
    if getattr(mf, 'check_convergence', None) is not None:
        raise ValueError('Mixed precision requires the native SCF convergence criteria')
    budget = mf.max_cycle
    if type(budget) is not int or budget < 1:
        raise ValueError('max_cycle must be a positive integer')
    if not math.isfinite(mf.conv_tol) or mf.conv_tol <= 0:
        raise ValueError('conv_tol must be finite and positive')
    if mf.conv_tol_grad is not None and (not math.isfinite(mf.conv_tol_grad) or mf.conv_tol_grad <= 0):
        raise ValueError('conv_tol_grad must be finite and positive')
    callback = mf.callback
    records = []
    info = {'cycles': records, 'fallback_scf': False, 'accepted': False}
    mf.mixed_precision_info = info
    mf.converged = False

    with _precision_context(mf) as (grid, vv10):

        def progress(env):
            records.append(
                {
                    'phase': 'mixed' if vv10.use_float32 else 'float64',
                    'cycle': int(env['cycle']) + 1,
                    'energy_hartree': float(env['e_tot']),
                    'orbital_gradient_norm': float(env['norm_gorb']),
                }
            )
            if len(records) >= 5:
                grid.use_float32 = False
            if callable(callback):
                callback(env)

        mf.callback = progress
        try:
            # Reserve iterations for float64 recovery if approximate fields do not converge.
            mf.max_cycle = min(20, max(1, budget // 2))
            energy = float(calculate() if calculate is not None else mf.kernel(dm0=dm0))
            grid.use_float32 = vv10.use_float32 = False
            check = _full_check(mf, energy)
            info['initial_verification'] = check.copy()
            if not check['passed']:
                info['fallback_scf'] = True
                remaining = budget - len(records)
                if remaining < 1:
                    mf.converged = False
                    raise SCFVerificationError('Float64 verification failed and the SCF cycle budget is exhausted')
                mf.max_cycle = remaining
                energy = float(mf.kernel(dm0=mf.make_rdm1()))
                check = _full_check(mf, energy)
            info['final_verification'] = check
            info['grid_contraction_calls'] = {
                'float32': grid.float32_calls,
                'float64': grid.float64_calls,
            }
            info['vv10_precisions'] = vv10.precisions.copy()
            if not check['passed']:
                mf.converged = False
                raise SCFVerificationError('Full float64 SCF failed the original convergence criteria')
            mf.e_tot = check['float64_energy_hartree']
            mf.converged = info['accepted'] = True
            mf.cycles = len(records)
            return mf.e_tot
        except BaseException:
            mf.converged = False
            raise
        finally:
            mf.callback = callback
            mf.max_cycle = budget
