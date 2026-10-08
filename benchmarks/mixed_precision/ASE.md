# Using verified mixed precision with ASE

The GPU4PySCF ASE calculator accepts `precision="mixed"` for molecular RKS/UKS with VV10 on a
single visible GPU. Configure the method with the desired functional, basis, ECP, fitting basis,
grids, SCF tolerance and cycle budget first; the calculator preserves them.

```python
from gpu4pyscf.tools.ase_interface import PySCF

atoms.calc = PySCF(method=mf, precision="mixed", grid_response=True,
                   auxbasis_response=True)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # numpy array, eV/Angstrom
```

The default `precision="float64"` retains native SCF. Each new geometry starts a fresh precision
schedule, while the normal scanner reuses the previous density. Grid products use float32 for
at most five SCF iterations; VV10 pair products use float32 partial sums combined in float64.
The calculator then computes the original float64 energy and physical Fock. Acceptance requires
both the original energy tolerance and orbital-gradient tolerance. The accepted float64 energy
is stored on the method and returned to ASE.

If verification fails, float64 SCF restarts from the current density at the same geometry,
using the remaining original cycle budget. At most 20 approximate iterations are used, with
at least half of the budget initially reserved for recovery. Failed float64 convergence raises
an error. The optimizer's geometry and history are retained; the whole optimizer is not restarted.

Integration dispatch is private to the method's NumInt instance. Module-level kernels and other
calculator instances remain unchanged. Native float64 routines are restored before gradients
or Hessians run. `grid_response` and `auxbasis_response` are optional explicit gradient controls;
`None` retains their native defaults. `calc.precision_info` records checks, fallback and cycles;
`calc.calculation_info` separates SCF and force timing. `calc.set(precision="float64")` clears
cached ASE results before the next calculation.

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
the Hessian reference with native float64 SCF. It writes the Hessian in Hartree/Bohr² and never
falls back to finite differences. A short optimizer run alone is not a validated minimum.

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
  benchmarks/mixed_precision/test_ase_gpu.py
```

The force checks include neutral and open-shell geometries, warm density reuse and a central
energy finite difference. They do not generalize to untested molecules or validate a TS/minimum.
