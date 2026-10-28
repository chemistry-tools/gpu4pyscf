"""Verified mixed-precision SCF for molecular single-GPU DFT calculations."""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
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

    from gpu4pyscf.df._mixed_jk import _exchange_context
    from gpu4pyscf.dft import numint
    from gpu4pyscf.dft._mixed_grid import _GridPrecision, _make_ao_dot, _make_contract, _make_tau_dot
    from gpu4pyscf.dft._mixed_vv10 import _AdaptiveVV10
    from gpu4pyscf.dft._native_grid_cache import _native_grid_reuse

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
    if '_dot_ao_ao_dense' in private:
        private['_dot_ao_ao_dense'] = _make_ao_dot(numint._dot_ao_ao_dense, private['contract'], grid)
    elif not mf.do_nlc():
        raise ValueError('LDA/GGA mixed precision requires the native SCF grid-product dispatch')
    # Older installed-release adapters have no symmetric tau path. For the native
    # engine, both alternatives retain this instance's private contract dispatch.
    if '_tau_dot_full' in private:
        private['_tau_dot'] = _make_tau_dot(private['_tau_dot'], private['_tau_dot_full'], grid)
    originals = {}
    try:
        for name in ('nr_rks', 'nr_uks', 'nr_nlc_vxc'):
            bound = getattr(ni, name)
            if getattr(bound, '__func__', None) is not getattr(numint, name):
                raise ValueError(f'Mixed precision does not support a custom NumInt.{name}')
            originals[name] = (name in ni.__dict__, ni.__dict__.get(name))
            setattr(ni, name, MethodType(private[name], ni))
        with _exchange_context(mf, grid) as exchange, _native_grid_reuse(ni) as reuse:
            grid._df_exchange = exchange
            grid._native_grid_reuse = reuse
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
    eligible = (
        gpu
        and not hasattr(mf, 'cell')
        and hasattr(mf, 'xc')
        and callable(getattr(mf, 'do_nlc', None))
        and getattr(mf, 'check_convergence', None) is None
    )
    if not eligible or mf.do_nlc():
        return eligible
    ni = mf._numint
    integration = getattr(ni.nr_rks, '__func__', ni.nr_rks)
    return (
        ni._xc_type(mf.xc) in ('LDA', 'GGA', 'MGGA')
        and '_dot_ao_ao_dense' in getattr(integration, '__globals__', {})
    )


def _select_precision(mf, precision, *, supplied_density=False, scanner=False):
    if precision == 'auto':
        orbital_guess = scanner and getattr(mf, 'mo_coeff', None) is not None
        return 'vv10' if mf.do_nlc() and (supplied_density or orbital_guess) else 'mixed'
    return precision


def run_verified_scf(mf, calculate=None, *, dm0=None, precision='auto', with_forces=False):
    """Run mixed SCF, check float64 fields and reconverge at the same geometry if needed.

    ``calculate`` may call a scanner to preserve its normal density reuse and geometry reset.
    All patches are per NumInt instance and restored before returning to derivatives.
    ``dm0`` supplies the native initial density when no scanner callable is used.
    ``precision='vv10'`` approximates only nonlocal pair products, keeping ordinary
    integration and exchange in float64. This can preserve rapid convergence from
    a good initial density. It requires a nonlocal functional.
    The default ``auto`` selects it for a supplied density or a scanner's retained
    orbitals with a nonlocal functional; other starts use the broader mixed schedule.
    VV10-only verification may reuse the last native ordinary integration when
    every input matches exactly; exchange and nonlocal fields are recomputed.
    ``with_forces=True`` requests bounded float64 VV10 field/coordinate-response
    reuse for the next full-response gradient at this exact state. Energy-only
    calls avoid that extra derivative work. Incompatible inputs use native forces.
    ``mf.max_cycle`` remains the total budget across the mixed and float64 phases.
    Verification failures raise ``SCFVerificationError``; CUDA and callback errors propagate.
    """
    if calculate is not None and (not callable(calculate) or dm0 is not None):
        raise ValueError('Use a scanner callable or dm0, not both')
    if type(with_forces) is not bool:
        raise ValueError('with_forces must be a boolean')
    if precision not in ('auto', 'mixed', 'vv10'):
        raise ValueError("precision must be 'auto', 'mixed' or 'vv10'")
    # PySCF creates scanner classes in its CPU module even for GPU subclasses.
    gpu_method = getattr(mf, 'device', None) == 'gpu' or any(
        cls.__module__.startswith('gpu4pyscf.') for cls in type(mf).__mro__
    )
    if not gpu_method:
        raise ValueError('Mixed precision requires a GPU4PySCF mean-field object')
    if hasattr(mf, 'cell') or not hasattr(mf, 'xc'):
        raise ValueError('Mixed precision currently supports molecular RKS and UKS only')
    has_nlc = bool(mf.do_nlc())
    requested_precision = precision
    precision = _select_precision(mf, precision, supplied_density=dm0 is not None, scanner=calculate is not None)
    vv10_only = precision == 'vv10'
    if vv10_only and not has_nlc:
        raise ValueError("precision='vv10' requires a nonlocal functional")
    if not has_nlc and mf._numint._xc_type(mf.xc) not in ('LDA', 'GGA', 'MGGA'):
        raise ValueError('Mixed precision requires an LDA, GGA or meta-GGA functional')
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
    info = {'requested_precision': requested_precision, 'precision': precision,
            'cycles': records, 'fallback_scf': False, 'accepted': False}
    nlc_reuse = getattr(mf._numint, '_nlc_force_reuse', None)
    if nlc_reuse is not None:
        nlc_reuse.clear()
        mf._numint._nlc_force_reuse = None
    nlc_reuse = None
    if with_forces and has_nlc:
        from gpu4pyscf.dft._nlc_force_cache import _NLCForceReuse

        nlc_reuse = _NLCForceReuse(mf._numint)
        mf._numint._nlc_force_reuse = nlc_reuse
        info['nlc_force_reuse'] = nlc_reuse.statistics
    mf.mixed_precision_info = info
    mf.converged = False

    with _precision_context(mf) as (grid, vv10):
        exchange = getattr(grid, '_df_exchange', None)
        reuse = getattr(grid, '_native_grid_reuse', None)
        if reuse is not None:
            reuse.enabled = vv10_only and reuse.supported
        if vv10_only:
            grid.use_float32 = False
            if exchange is not None:
                exchange.finish_approximation()
        adaptive = exchange is not None and not vv10_only
        adaptive_grid = adaptive and getattr(exchange, 'adaptive_grid', True)
        int8_seen = 0
        float32_exchange_seen = 0
        if adaptive:
            exchange.projection_float32 = True
        info['grid_precision_schedule'] = (
            'float64' if vv10_only else 'adaptive_hybrid_gga' if adaptive_grid else 'first_five'
        )
        info['df_exchange_projection_schedule'] = (
            {'minimum_cycles': 5, 'maximum_cycles': 16, 'orbital_gradient_threshold': 1e-4}
            if adaptive else None
        )

        def progress(env):
            nonlocal int8_seen, float32_exchange_seen
            int8_calls = getattr(exchange, 'int8_calls', 0)
            float32_exchange_calls = getattr(exchange, 'float32_calls', 0)
            approximate = (
                (vv10.use_float32 if has_nlc else grid.use_float32)
                or int8_calls > int8_seen or float32_exchange_calls > float32_exchange_seen
            )
            records.append(
                {
                    'phase': 'mixed' if approximate else 'float64',
                    'cycle': int(env['cycle']) + 1,
                    'energy_hartree': float(env['e_tot']),
                    'orbital_gradient_norm': float(env['norm_gorb']),
                }
            )
            int8_seen = int8_calls
            float32_exchange_seen = float32_exchange_calls
            # Preserve the original grid schedule independently of exchange.
            # Eligible Gram updates remain approximate until native verification.
            if adaptive_grid:
                if len(records) >= 12 or (len(records) >= 5
                        and abs(float(env['e_tot'] - env['last_hf_e'])) < 1e-5
                        and float(env['norm_gorb']) < 1e-3):
                    grid.use_float32 = False
            elif not vv10_only and len(records) >= 5:
                grid.use_float32 = False
            if adaptive and (len(records) >= 16 or (len(records) >= 5 and float(env['norm_gorb']) < 1e-4)):
                exchange.projection_float32 = False
            if callable(callback):
                callback(env)

        mf.callback = progress
        try:
            # Reserve iterations for float64 recovery if approximate fields do not converge.
            mf.max_cycle = min(20, max(1, budget // 2))
            energy = float(calculate() if calculate is not None else mf.kernel(dm0=dm0))
            grid.use_float32 = vv10.use_float32 = False
            if exchange is not None:
                exchange.finish_approximation()
                # Native SCF can evaluate a final approximate Fock after its last
                # callback. Do not attribute those products to a recovery cycle.
                int8_seen = exchange.int8_calls
                float32_exchange_seen = exchange.float32_calls
            with (reuse.verification() if reuse is not None else nullcontext(),
                  nlc_reuse.verification() if nlc_reuse is not None else nullcontext()):
                check = _full_check(mf, energy)
            info['initial_verification'] = check.copy()
            if not check['passed']:
                if nlc_reuse is not None:
                    nlc_reuse.clear()
                info['fallback_scf'] = True
                remaining = budget - len(records)
                if remaining < 1:
                    mf.converged = False
                    raise SCFVerificationError('Float64 verification failed and the SCF cycle budget is exhausted')
                mf.max_cycle = remaining
                energy = float(mf.kernel(dm0=mf.make_rdm1()))
                with (reuse.verification() if reuse is not None else nullcontext(),
                      nlc_reuse.verification() if nlc_reuse is not None else nullcontext()):
                    check = _full_check(mf, energy)
            info['native_grid_reuse'] = {'hits': reuse.hits, 'misses': reuse.misses} if reuse is not None else None
            info['final_verification'] = check
            info['grid_contraction_calls'] = {
                'float32': grid.float32_calls,
                'float64': grid.float64_calls,
            }
            info['df_exchange_contraction_calls'] = {
                'float32': exchange.float32_calls if exchange is not None else 0,
                'float64': exchange.float64_calls if exchange is not None else 0,
                'int8': exchange.int8_calls if exchange is not None else 0,
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
            if nlc_reuse is not None:
                nlc_reuse.clear()
            mf.converged = False
            raise
        finally:
            mf.callback = callback
            mf.max_cycle = budget
