# Verified mixed precision for molecular DFT

This fork provides verified mixed precision for molecular RKS/UKS on one visible GPU,
including B3LYP and VV10 functionals. Early grid matrix products and, when present,
VV10 pair products use float32, followed by verification with
native float64 energy and physical Fock. Density-fitted global-hybrid GGAs also use
float32 exchange products during the early search, with float64 accumulation.
On compute capability 8.6, eligible later exchange Gram products use nine INT8
slices and float64 accumulation. This remains an approximate SCF step; the final
energy/Fock check, recovery and derivatives use native float64 throughout.
AO fields, densities, XC, integrals and orbital diagonalization retain float64. Numerical settings are configured on the
mean-field object and preserved by the precision runner.

Grid contractions reuse float32 scratch within each worker thread and CUDA stream,
and fuse conversion, scaling and accumulation into a float64 output. Scratch is released
when the precision scope exits. VV10 pair timing is disabled during normal calculations;
the benchmark's `--profile` option explicitly enables synchronized timing.

Full grid-response RKS forces evaluate density and orbital derivatives in one basis-function
pass. Real, exactly symmetric LDA and meta-GGA density matrices share their float64 density/
basis products within each block. Grid-weight derivatives keep their large batches. The
derivatives remain float64, and these block-local products are released after the call;
there is no reuse of density fields between geometries.

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

The runner uses float32 grid products for at most five iterations, with an adaptive
schedule for native density-fitted global-hybrid GGAs. Those methods retain float32
grid and exchange products until both the energy change falls below `1e-5` Hartree
and the orbital gradient below `1e-3`, after at least five iterations and at most
twelve. Once switched, grid and exchange-projection products remain float64;
eligible Gram products can use the INT8 path until final verification.
These are precision-switch criteria; the caller's SCF acceptance thresholds are unchanged.
Other methods retain the five-iteration grid schedule. VV10 pair products, when present,
use float32 during the approximate phase. It caps the first SCF phase at 20 iterations and
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

The exchange dispatch is private to the density-fitting instance. It uses explicit
float32 products, releases conversion buffers when switching to float64 and restores
the native method before derivatives. The report's `df_exchange_contraction_calls`
records `float32`, `int8` and native `float64` calls; zero counts identify calculations outside this path's
global-hybrid GGA and native density-fitting guards. Float64 real symmetric exchange
products use a triangular BLAS update for sufficiently large contiguous blocks.

The INT8 kernel is compiled by CuPy and requires no separate library or global
interception. It shares one signed seven-bit slice buffer for both sides of the
Gram product, pads integer matrices and bounds INT32 accumulation against overflow.
Scratch is bounded to 512 MiB per worker stream, checked against available memory
and released before native verification. Unsupported devices, layouts, scales and
larger workspaces retain the native float64 update. Approximate iterations remain
labelled `mixed` even after grid products switch to float64. Explicit
`precision="float64"` and ordinary native SCF calls do not use INT8.

The real-matrix splitting follows [ozIMMU](https://github.com/enp1s0/ozIMMU), with
its MIT notice retained in the specialized kernel source. Native verification is
independent of this approximate product and never re-enables it during recovery.

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
supported GPU LDA, GGA and meta-GGA DFT, including hybrids, and native float64 for other methods. `precision="float64"`
explicitly forces native SCF. Each changed geometry starts a new
precision schedule and the scanner reuses the last density. SCF fallback retains the
optimizer's geometry and history. Energies are in eV and NumPy forces in eV/Angstrom.
Optional gradient response controls retain native defaults when omitted.

The same ASE interface supports B3LYP without a nonlocal term:

```python
from gpu4pyscf import dft

mf = dft.RKS(mol, xc="b3lyp").density_fit()
atoms.calc = PySCF(method=mf, grid_response=True, auxbasis_response=True)
forces = atoms.get_forces()
```

Its precision report contains grid-product counts and an empty `vv10_precisions` list.
Non-VV10 support requires the current native grid-product dispatch; the older installed-release
compatibility adapter supports the original VV10 workflow only.

An energy request followed by forces at unchanged atoms reuses the accepted SCF
state. The [ASE guide](../benchmarks/mixed_precision/ASE.md) describes invalidation
and the per-call timing report.

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
