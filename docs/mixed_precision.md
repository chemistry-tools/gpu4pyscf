# Verified mixed precision for molecular DFT

This fork accelerates molecular RKS/UKS SCF with VV10 on one visible GPU. Early grid
matrix products and VV10 pair products use float32, followed by verification with
native float64 energy and physical Fock. AO fields, densities, XC, exchange and
orbital diagonalization retain float64. Numerical settings are configured on the
mean-field object and preserved by the precision runner.

Use a matching native build of this checkout, following the [compilation guide](../README.md#compilation).
Do not combine current numerical Python sources with an older wheel's native libraries.
For a four-CPU allocation, apply CPU affinity, cap numerical thread counts at four,
and build with at most four jobs. Wheel builds can use `CMAKE_BUILD_ARGS="-j 4"`.
Only one GPU may be visible; set `CUDA_VISIBLE_DEVICES` before importing GPU4PySCF.

## Direct SCF

```python
from gpu4pyscf.dft.mixed_precision import SCFVerificationError, run_verified_scf

try:
    energy_hartree = run_verified_scf(mf, dm0=initial_density)
except SCFVerificationError:
    # Recovery failed the original SCF criteria; no energy is accepted.
    raise
forces = -mf.Gradients().kernel()  # native float64, Hartree/Bohr
report = mf.mixed_precision_info
```

`dm0` is optional and follows the native SCF initial-density convention. An optional
scanner callable can instead be supplied as the second argument; it preserves native
geometry reset and warm density reuse. A callable and an explicit density cannot be
supplied together. The native callback is retained through all SCF phases.

The runner uses float32 grid products for at most five iterations and float32 VV10
pair products during the approximate phase. It caps that phase at 20 iterations and
initially reserves half the configured `max_cycle` budget for recovery. After SCF,
it checks the full float64 energy change against `conv_tol` and the physical orbital
gradient against `conv_tol_grad`, or `sqrt(conv_tol)` when unset. Passing verification
stores and returns the checked float64 energy.

Failed verification restarts native float64 SCF from the current density at the same
geometry, using the remaining cycle budget. Failed recovery raises
`SCFVerificationError`; CUDA errors and callback exceptions propagate unchanged.
NumInt dispatch, the callback and the original cycle budget are restored on every
exit. The accepted flag remains false on failure. There are no module-level patches.
Custom NumInt overrides, custom convergence callbacks, periodic methods and multiple
visible GPUs are currently unsupported and rejected explicitly.

## ASE optimizers and analytic Hessians

```python
from ase.optimize import BFGS
from gpu4pyscf.tools.ase_interface import PySCF

atoms.calc = PySCF(method=mf, precision="mixed", grid_response=True,
                   auxbasis_response=True)
BFGS(atoms).run(fmax=0.005)
hessian = atoms.calc.get_hessian(atoms, auxbasis_response=2)
```

The calculator defaults to `precision="auto"`, selecting verified mixed precision for
supported GPU VV10 DFT and native float64 for other methods. `precision="float64"`
explicitly forces native SCF. Each changed geometry starts a new
precision schedule and the scanner reuses the last density. SCF fallback retains the
optimizer's geometry and history. Energies are in eV and NumPy forces in eV/Angstrom.
Optional gradient response controls retain native defaults when omitted.

`get_hessian()` reconverges the current density with native float64 SCF before the
analytic molecular Hessian. It returns a symmetric `(3N, 3N)` NumPy array in
eV/Angstrom² and invalidates force caches after reconvergence. Full auxiliary-basis
response is the default; use `None` for a method without density fitting. It never
falls back to finite differences. Hessians are exposed through this separate method
because they are not a standard ASE calculator property.

`calc.precision_info`, `calc.calculation_info` and `calc.hessian_info` expose SCF checks,
recovery, timing and Hessian-reference provenance. A failed SCF retains its diagnostic
report without stale energy or forces. `calc.set(precision="float64")` invalidates
cached results and reports before the next calculation.

See the [synthetic water example](../examples/dft/12-mixed_precision_ase.py) and the
[benchmark guide](../benchmarks/mixed_precision/ASE.md) for frozen-input comparisons.
Validation datasets must stay held out from tuning and implementation decisions.
Development checks can use designated test data and generic synthetic molecules;
private geometries and results belong outside Git.
