# Using verified mixed precision with ASE

The [library guide](../../docs/mixed_precision.md) owns the public SCF API and installation boundary.

The GPU4PySCF ASE calculator accepts `precision="mixed"` for molecular LDA/GGA/meta-GGA RKS/UKS on a
single visible GPU. Configure the method with the desired functional, basis, ECP, fitting basis,
grids, SCF tolerance and cycle budget first; the calculator preserves them.

```python
from gpu4pyscf.tools.ase_interface import PySCF

atoms.calc = PySCF(method=mf, precision="mixed", grid_response=True,
                   auxbasis_response=True)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # numpy array, eV/Angstrom
hessian = atoms.calc.get_hessian(atoms, auxbasis_response=2)  # (3N, 3N), eV/Angstrom squared
```

The default `precision="auto"` selects verified mixed SCF for supported molecular GPU DFT
and native float64 for other methods. `precision="float64"` explicitly forces native SCF. Each new geometry starts a fresh precision
schedule, while the normal scanner reuses the previous density. Grid products use float32 for
at most five SCF iterations; eligible density-fitted global-hybrid GGAs adapt the
float32 grid schedule to convergence progress, with a twelve-iteration cap. Other hybrids,
including range-separated MGGAs with VV10, retain the first-five grid schedule. Eligible
density-fitted hybrid exchange projections switch independently, after at least five iterations, when the orbital gradient
falls below `1e-4`, with a sixteen-iteration cap.
Eligible later exchange Gram products use the bounded INT8 path on compute capability 8.6.
Final verification, recovery and forces retain native float64.
The [precision guide](../../docs/mixed_precision.md#direct-scf) describes the switch criteria.
For nonlocal functionals with good starting densities, `precision="vv10"` keeps
ordinary grid products and exchange float64 throughout SCF and approximates only
VV10 pairs. It retains the same native verification and recovery. This explicit
option can preserve a short SCF path; it requires a nonlocal term. The workflow
comparison script also accepts `--precision vv10`.
VV10 pair products, when present,
use float32 partial sums combined in float64.
The calculator then computes the original float64 energy and physical Fock. Acceptance requires
both the original energy tolerance and orbital-gradient tolerance. The accepted float64 energy
is stored on the method and returned to ASE.

If verification fails, float64 SCF restarts from the current density at the same geometry,
using the remaining original cycle budget. At most 20 approximate iterations are used, with
at least half of the budget initially reserved for recovery. Failed float64 convergence raises
an error. The optimizer's geometry and history are retained; the whole optimizer is not restarted.

Integration and exchange dispatch are private to the method's NumInt and fitting instances. Module-level kernels and other
calculator instances remain unchanged. Native float64 routines are restored before gradients
or Hessians run. `grid_response` and `auxbasis_response` are optional explicit gradient controls;
`None` retains their native defaults. `calc.precision_info` records checks, fallback and cycles;
`calc.calculation_info` separates SCF and force timing. `calc.set(precision="float64")` clears
cached ASE results before the next calculation.

`calc.precision_info['df_exchange_contraction_calls']` reports float32 and float64
exchange-product counts. Integral construction, final verification and derivatives
retain float64; no optimizer restart or extra calculator setting is required.

An energy-only request followed by forces at unchanged atoms reuses the accepted
SCF state and its precision verification; derivatives still use native float64.
`calc.calculation_info['scf_reused']` reports this reuse, with `scf_seconds=0` for
that property call. Changed atoms, calculator settings, resets, failed calculations
and explicit `calc.calculate()` calls invalidate reuse. Call `calc.reset()` after
changing the underlying PySCF method directly; such changes are outside ASE's
property-cache tracking.

`calc.get_hessian()` explicitly reconverges native float64 SCF and runs the analytic molecular
Hessian with full auxiliary response by default. It returns a symmetric Cartesian NumPy matrix,
invalidates force caches after reconvergence and records the reference and timing in
`calc.hessian_info`. It is a separate method, since Hessians are not a standard ASE calculator
property. There is no finite-difference fallback. Methods without density fitting require
`auxbasis_response=None`. The grid response follows the calculator's explicit setting.

## Comparing an existing workflow

Use `ase_workflow.py` with a frozen JSON input outside the checkout. It contains a `recipe` object
and a `cases` list; optional `provenance` objects preserve source identities and hashes. The exact
accepted fields are defined by `_input` in the script. Each case supplies `id`, `numbers`,
`positions` in Angstrom, `charge` and `multiplicity`. The recipe explicitly supplies all numerical
and derivative settings; nonlocal grids retain the installed library's native default.

Run baseline and mixed calculations sequentially under the same environment and four-CPU limits
from [RUNNING.md](RUNNING.md):

```sh
CUDA_VISIBLE_DEVICES=1 python benchmarks/mixed_precision/ase_workflow.py \
  --input /outside/checkout/frozen-input.json --case example --precision float64 \
  --steps 2 --fmax 0.005 --output /outside/checkout/baseline.json
CUDA_VISIBLE_DEVICES=1 python benchmarks/mixed_precision/ase_workflow.py \
  --input /outside/checkout/frozen-input.json --case example --precision mixed \
  --steps 2 --fmax 0.005 --output /outside/checkout/mixed.json
```

Use `--steps 0` for identical-geometry energy/force comparisons. Larger step counts run BFGS
with the same force threshold and `maxstep=0.15` Angstrom. Independent optimizer paths can diverge;
compare matched-geometry forces as well as time, steps and final convergence. `--hessian` adds
the analytic float64 Hessian with the recipe's grid and auxiliary responses, after reconverging
the Hessian reference with native float64 SCF. It writes both Cartesian eV/Angstrom² and
atom-pair Hartree/Bohr² arrays. A short optimizer run alone is not a validated minimum.
Per-cycle progress is appended to
`scf-progress.jsonl` beside the frames and optimizer artifacts, without modifying earlier results.

Compare the resulting reports with:

```sh
python benchmarks/mixed_precision/compare_ase.py \
  --baseline /outside/checkout/baseline.json --mixed /outside/checkout/mixed.json \
  --output /outside/checkout/comparison.json
```

The comparator rejects differences in the frozen input, recipe, numerical sources, software
versions, GPU and CPU allocation. It reports timing, verified fallbacks and errors only at
exactly identical geometries; unmatched optimizer frames are counted separately. Energies are
compared in Hartree, and force conversion constants are normalized. Single-run speedups on a
shared host are preliminary measurements, not idle-host statistical estimates.

When compatibility-testing these new Python modules against an installed 1.8.1 wheel, run by
absolute path outside the checkout and add `--extension-root /path/to/this/fork`. The loader
replaces only the new precision modules and ASE interface; existing numerical Python sources
and native library paths remain from the installed release. Reports record both sets of hashes.
This does not establish compatibility of the fork's other master changes with old native binaries.
Use a matching native build for testing the entire source checkout.

Focused CPU-only tests and opt-in GPU force checks:

```sh
python -m pytest -q benchmarks/mixed_precision/test_ase_precision.py
DFT_GPU_TEST=1 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 python -m pytest -q \
  benchmarks/mixed_precision/test_ase_gpu.py \
  benchmarks/mixed_precision/test_vv10_precision.py
```

`test_semilocal_precision.py` adds B3LYP/Cartesian 6-31G* checks on nitrogen, oxygen,
sulfur, fluorine and chlorine organics, an open-shell case, LDA/GGA/meta-GGA coverage
and a force finite difference. It requires the current native grid-product dispatch.
Independent force comparisons use direct integrals with tighter correctness-only
screening and convergence. Density-fitted checks compare
SCF energies/densities and complete forces at an identical accepted density: the
existing Cartesian DF derivative path can amplify tiny differences between independently
converged densities. These checks do not establish independent DF force agreement.

The force checks include neutral and open-shell geometries, warm density reuse and a central
energy finite difference. They do not generalize to untested molecules or validate a TS/minimum.
