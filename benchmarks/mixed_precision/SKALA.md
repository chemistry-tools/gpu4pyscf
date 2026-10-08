# Testing Skala through ASE

`skala_benchmark.py` compares the published Skala-1.1 GPU ASE calculator with native and
verified mixed ωB97M-V. It also compares direct ωB97M-V optimization with Skala preoptimization
followed by ωB97M-V optimization. These are different scientific methods: Skala uses its
published D3 correction, while ωB97M-V retains full VV10. Cross-method total-energy differences
are not accuracy estimates.

The harness uses only the synthetic cases in `gpu_performance.py`. Validation datasets remain
held out. Every calculation runs on physical GPU 1 with four logical CPUs, affinity `4,5,6,7`.
Run commands sequentially and keep environments, checkpoints, logs and JSON reports outside Git.

## Environment

Install [requirements-skala-cuda13.txt](requirements-skala-cuda13.txt) in a separate Linux
Python 3.12 environment. Skala 2026.9 requires PySCF >= 2.11; these comparisons use PySCF 2.14
for all methods, GPU4PySCF 1.8.1, PyTorch 2.13 CUDA 13, and ASE 3.27.0. The original DFT
environment remains independent.

Set the four-thread variables and CUDA library paths as described in [RUNNING.md](RUNNING.md),
using the new environment's site-packages. The tested wheel requires an unversioned cuSOLVER
link for `ctypes.util.find_library`. If that link is absent, create it in a private directory
outside the checkout, pointing `libcusolver.so` to the environment's `libcusolver.so.12`, and
prepend that directory to `LD_LIBRARY_PATH`. No system CUDA installation needs changing.

Run from outside the checkout so the installed numerical Python sources and native libraries
match. `--extension-root` explicitly loads only this fork's precision and ASE Python additions;
it does not establish that current master native code was built. Skala checkpoints are loaded
through the official hash-verified resolver. Reports retain their path/hash, installed versions,
source/native hashes, thread pools, geometry, charge, multiplicity and convergence history.

## Single-point energy and forces

Replace the interpreter, checkout and output paths with your own absolute paths:

```sh
/path/to/env/bin/python /path/to/fork/benchmarks/mixed_precision/skala_benchmark.py \
  --method native --extension-root /path/to/fork --output /tmp/benzene-native.json
/path/to/env/bin/python /path/to/fork/benchmarks/mixed_precision/skala_benchmark.py \
  --method mixed --extension-root /path/to/fork --output /tmp/benzene-mixed.json
/path/to/env/bin/python /path/to/fork/benchmarks/mixed_precision/skala_benchmark.py \
  --method skala --skala-grid-level 3 --extension-root /path/to/fork \
  --output /tmp/benzene-skala.json
```

Each invocation retains the first calculation separately and reports the median of three later
fresh-calculator calls using warm process caches. Timings include both SCF and forces, with
imports/model loading recorded separately. The Skala model remains loaded between samples.
ωB97M-V uses the unchanged protocol's ordinary grid level 5 and nonlocal grid level 3;
`--skala-grid-level` defaults to 5. Skala uses a different grid representation without density
pruning, so equal levels do not imply identical grids. The level-5 benzene Skala run exceeded
the tested RTX 3080's 10 GB memory; level 3 is an explicitly different configuration.

## Geometry changes and preoptimization

The published Skala GPU gradient reset retains the previous D3 geometry. A synthetic benzene
reset diagnostic reproduced this, and explicitly resetting the dispersion object matched a
fresh object. `--refresh-skala-dispersion` enables a benchmark-local ASE subclass that refreshes
that object before a new evaluation. It does not modify the installed Skala package. Use this
flag for geometry-update and optimization comparisons.

```sh
/path/to/env/bin/python /path/to/fork/benchmarks/mixed_precision/skala_benchmark.py \
  --method skala --case water --repeats 1 --check-forces --refresh-skala-dispersion \
  --extension-root /path/to/fork --output /tmp/skala-forces.json
/path/to/env/bin/python /path/to/fork/benchmarks/mixed_precision/skala_benchmark.py \
  --method mixed --optimize --extension-root /path/to/fork --output /tmp/direct-opt.json
/path/to/env/bin/python /path/to/fork/benchmarks/mixed_precision/skala_benchmark.py \
  --method cascade --optimize --skala-grid-level 3 --refresh-skala-dispersion \
  --extension-root /path/to/fork --output /tmp/cascade-opt.json
```

The force check compares a reused calculator with a fresh one after a geometry change and
checks one analytic force component against a central energy difference. Read its explicit
`passed` flag; a successfully written report alone does not establish force consistency.
Optimizations start from the same synthetic geometry scaled by 1.025, use BFGS with maximum
step 0.1 Å, and require forces below 0.005 eV/Å within 30 steps. Each stage records its own
frames, SCF cycles and elapsed time. Stage timing includes all optimizer evaluations; observer
frame timings can be cache reads and are not per-step wall times.

The cascade starts a fresh ωB97M-V calculator at the Skala geometry. It does not transfer a
Skala density between methods. No Hessians, frequencies or transition-state validation are
performed. See [PERFORMANCE.md](PERFORMANCE.md) for measured results and their limits.
