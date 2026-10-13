# Reproducing the experiments

Use Linux x86_64, Python 3.12 and a CUDA 13-compatible NVIDIA driver. Install the isolated
[pinned environment](requirements-cuda13.txt); it includes the tested GPU4PySCF 1.8.1 wheel.
No system-wide CUDA toolkit installation is needed. The host must provide OpenMP support.

```sh
python3.12 -m venv /tmp/gpu4pyscf-bench-env
/tmp/gpu4pyscf-bench-env/bin/python -m pip install -r benchmarks/mixed_precision/requirements-cuda13.txt
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export BLIS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export LD_LIBRARY_PATH=/tmp/gpu4pyscf-bench-env/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/gpu4pyscf-bench-env/lib/python3.12/site-packages/cutensor/lib
```

The harness requires physical GPU 1 and exactly four available logical CPUs, defaulting to
`4,5,6,7`. It verifies the GPU PCI identity, applies CPU affinity before numerical imports and
sets PySCF/OpenMP/BLAS thread limits. Choose four allowed CPU IDs near GPU 1 with `--cpus` if
this host uses a different CPU topology. Never run benchmarks, tests or builds simultaneously.
Keep raw results outside the checkout and use new output paths for every run.

## Installed release and source checkout

The original measurements used the installed 1.8.1 wheel. The shared kernels now live in the
library, so add `--extension-root /path/to/this/fork` to GPU benchmark commands when testing
these new Python modules against that wheel; see [ASE.md](ASE.md) for the import boundary. To reproduce that environment, run
these scripts by absolute path from a directory outside the source checkout, without a checkout
in `PYTHONPATH`. Both this fork's current master and the older release report version 1.8.1;
reports therefore also record the imported module path, source hashes and native library hash.

For testing library changes in this checkout, compile its matching native libraries first,
using the [native RTX 3080 build guide](NATIVE_BUILD.md) or the repository's
[compilation instructions](../../README.md#compilation) with at most
four build jobs and CPU affinity. Set `PYTHONPATH` to the checkout and pass
`--require-gpu-source /absolute/path/to/gpu4pyscf`. The harness rejects an import from elsewhere.
Do not combine current Python sources with native binaries from an older release.

## Fresh SCF comparisons

Replace `/absolute/path/to/gpu4pyscf` below with the checkout path. Run each command only after
the previous one exits. The timing includes final float64 verification and any fallback SCF,
while imports are reported separately. Add `--profile` for synchronized component timings and
a matching `.prof` artifact; use unprofiled calculations for speed comparisons.

```sh
/tmp/gpu4pyscf-bench-env/bin/python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/gpu_performance.py \
  --backend gpu --cpus 4,5,6,7 --overlap-cutoff 1e-10 --output /tmp/gpu-baseline.json
/tmp/gpu4pyscf-bench-env/bin/python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/gpu_performance.py \
  --backend gpu --cpus 4,5,6,7 --overlap-cutoff 1e-10 \
  --vv10 verify --grid-contract mixed --output /tmp/gpu-optimized.json
/tmp/gpu4pyscf-bench-env/bin/python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/gpu_performance.py \
  --backend cpu --cpus 4,5,6,7 --output /tmp/cpu-baseline.json
```

The overlap override retains all 276 benzene orbitals; stock 1.8.1 otherwise removes one at its
`1e-6` default. Backend grids and pruning remain native and can differ between CPU and GPU.
The explicit [protocol](protocol.json) fixes density fitting, ordinary grid level 5, nonlocal
grid level 3, full VV10, energy tolerance `1e-8` Hartree and gradient tolerance `1e-4`.
`--case water` and `--case water-cation` add neutral and open-shell checks. Charges, multiplicities
and geometry are fixed; no geometry preoptimization is included.

`--vv10 verify --grid-contract mixed` uses float32 grid matrix products for the first five
iterations, then float64. cuBLAS pedantic math prohibits TF32 products. VV10 pairs use float32
partial sums combined in float64 until SCF finishes, followed by an original float64 Fock
calculation. Acceptance requires both the energy change and the full float64 orbital gradient
at the original thresholds. AO fields, densities, XC, exchange and diagonalization stay float64.

`--vv10 mixed` instead switches VV10 to float64 when the energy change falls below `1e-5`
Hartree and the gradient below `1e-3`, or after 20 iterations. `--vv10 float32` is an approximate
diagnostic without final verification. `--grid-contract blas` and `--grid-tasks inline` test
float64 BLAS and synchronous single-GPU dispatch. Every patch is scoped and restored on exit.

## Kernel checks

CPU-only resource/protocol tests do not require GPU libraries:

```sh
python -m pytest -q benchmarks/mixed_precision/test_experiments.py
ruff check --config .ruff.toml benchmarks/mixed_precision
```

Run CUDA tests from outside the source checkout for the pinned wheel, or with the source import
configured after a native build. CPU affinity and environment limits also apply to tests:

```sh
DFT_GPU_TEST=1 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7 /tmp/gpu4pyscf-bench-env/bin/python -m pytest -q \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/test_experiments.py
```

Capture the final nonlocal fields by adding `--capture-vv10 /tmp/vv10-fields.npz` to an SCF run,
then run the pair-kernel microbenchmark:

```sh
/tmp/gpu4pyscf-bench-env/bin/python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/vv10_experiment.py \
  --input /tmp/vv10-fields.npz --output /tmp/vv10-kernels.json
```

It compares 64, 128 and 256-thread tiles with double arithmetic, float32 partial sums and a
[PTX approximate reciprocal](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp-approx-ftz-f64)
with two double corrections. No global fast-math flag is used. For `--backend nvcc`, the pinned
environment includes NVCC, CRT and NVVM 13.0.88; export its `nvidia/cu13` directory as `CUDA_PATH`
and add its `bin` directory to `PATH`. Compiler comparisons use identical inputs and settings.

## Exact force contractions

`force_benchmark.py` compares the original installed-release gradients with the new density-first
contraction on idealized synthetic pyrene or coronene. It retains the full functional, basis,
grids and both response terms. The accepted density is shared by both force implementations;
three alternating repetitions rebuild the same full/range-separated density-fitting buffers
outside the timed region, because native gradients release those buffers. Timing covers the
complete gradient call, without profiling synchronization inside its components.

Run from outside the checkout against the pinned wheel, with the resource limits above, and
keep each output path new. Finish one molecule before starting the next:

```sh
/tmp/gpu4pyscf-bench-env/bin/python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/force_benchmark.py \
  --extension-root /absolute/path/to/gpu4pyscf --case pyrene --output /tmp/pyrene-forces.json
/tmp/gpu4pyscf-bench-env/bin/python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/force_benchmark.py \
  --extension-root /absolute/path/to/gpu4pyscf --case coronene --output /tmp/coronene-forces.json
```

The scoped adapter changes only the reviewed derivative-matrix contraction sites and atom
reduction within the installed release's own functions. Its source guards reject incompatible
layouts. The remainder of the numerical Python and all native libraries stay from that release;
this is algebra compatibility evidence, not a full checkout build. Reports preserve source
hashes, fixed geometry, accepted SCF checks, resource limits, individual forces and timings.

After importing the installed release and loading the precision modules through `_extensions.py`,
run the opt-in `test_force_response.py` checks alongside the existing GPU tests. They compare
LDA/GGA/MGGA contractions against native derivative matrices and full neutral/open-shell
ωB97M-V forces against native gradients and finite-difference energies. With full grid response
enabled, the native source path applies the optimization directly in `gpu4pyscf/grad/rks.py`;
there is no additional optimization flag. Existing grid-response defaults are preserved.

## Fused VV10 response

The full-response gradient now evaluates the float64 U/W/E fields and explicit coordinate
derivative in one pair traversal. Pair antisymmetry combines the two atom contributions;
a deterministic reduction replaces the atom × direction × grid tensor. The kernel is compiled
through CuPy NVRTC and cached per device; its source is in `gpu4pyscf/dft/_vv10_grad.py`.
It uses no approximate reciprocal, reduced precision or cross-calculation field cache.

Add `--vv10-fused` to the force benchmark above to compare the already validated density-first
implementation with the additional fused response. The normal native source path enables it
automatically for full grid-response forces. For pinned-release compatibility tests, load
`_extensions.load_extensions(root, forces=True)` before using the scoped adapter with
`density_response_patch(root, vv10_fused=True)`. All remaining installed numerical code stays
from that release. The ordinary native installation is checked separately; see
[NATIVE_BUILD.md](NATIVE_BUILD.md) and the native results in [PERFORMANCE.md](PERFORMANCE.md).

`test_vv10_response.py` compares all fields and contracted derivatives with the original native
calls, and independently differentiates the pair energy by translating each atom's grid points.
It covers partial tiles, signed/zero weights, atoms without active points and separate CUDA
streams. `test_force_response.py` additionally compares complete neutral/open-shell forces and
finite-difference energies with and without the fused pass.

## Native density-fitting force comparisons

`native_force_benchmark.py` compares complete forces against a retained `df/grad/rhf.py`
from the same native-library revision. The reference function runs in the installed native
module's namespace; all other force code stays identical. Use a properly built installation
of this fork, preserve the baseline source outside the checkout, and require the actual
installed package path. The [native build guide](NATIVE_BUILD.md) explains the environment.

```sh
taskset -c 4-7 /tmp/gpu4pyscf-native-env/bin/python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/native_force_benchmark.py \
  --case ovalene --reference-df-grad /tmp/baseline-df-rhf.py \
  --require-gpu-source /tmp/gpu4pyscf-native-env/lib/python3.12/site-packages/gpu4pyscf \
  --output /tmp/ovalene-native-forces.json
```

Run under the host's shared benchmark lock, with the resource environment above, and finish
each calculation before starting another. Synthetic ovalene has 32 carbons and 14 hydrogens;
pyrene and coronene remain available. The recipe and full response terms are unchanged.
Three alternating repetitions rebuild the full/range-separated DF buffers outside timing
and release unused GPU-pool blocks first. Reports include each force, available memory,
source/native hashes and the complete-call medians. Component profiles are separate evidence.

The artifact directory contains `scf.npz` and `recipe.json`. Pass that directory through
`--state` to avoid repeating SCF. Reuse requires the identical recipe and geometry, a matching
state hash, retained basis orbitals and a fresh full float64 energy/orbital-gradient check.
The recorded SCF time then belongs to the original calculation, not the reload.

For native derivative checks, set `DFT_GPU_TEST=1` and `DFT_REFERENCE_DF_GRADIENT` to the
retained baseline source before running `test_native_df_response.py` sequentially. These
checks cover full/long/short-range operators, nonsymmetric densities, auxiliary response,
Cartesian/spherical hybrid forces and independent finite-difference energies.

## Symmetric float64 tau products

The native integrator uses signed symmetric rank updates for identical real float64 AO
fields with more than 512 active orbitals. Small, distinct, complex, noncontiguous or
nonfinite inputs retain the general bilinear product. Early mixed SCF retains its original
float32 products; final verification and recovery use the float64 path.
Native RKS/UKS tasks reserve one bounded stack workspace. Standalone callers release a
direct CUDA allocation instead of caching differently sized large buffers. Finite checks
use a fused reduction, avoiding large temporary masks that can exhaust memory before forces.

Run `test_tau_syrk.py` with `DFT_GPU_TEST=1` alongside the native checks. Its complete SCF,
force and finite-difference panel includes nitrogen, oxygen, sulfur, fluorine and chlorine
organics. The correctness tests exercise the symmetric path even on small molecules.
`_organic_cases.py` owns the generic panel; the larger ethyl-4-acetamidobenzoate geometry
and its deterministic generation provenance are in `organic_geometry.json`.

`native_tau_benchmark.py` measures fresh ASE energy-then-force requests at the fixed recipe.
It records accepted SCF verification, reuse, forces, source/native hashes and the actual
symmetric-kernel calls. The baseline option disables only the symmetric tau path in its
isolated benchmark process. Use matching native libraries and separate processes under
the shared lock; alternate the two modes between repeats:

```sh
taskset -c 4-7 /tmp/gpu4pyscf-native-env/bin/python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/native_tau_benchmark.py \
  --case ethyl-4-acetamidobenzoate --tau baseline \
  --require-gpu-source /tmp/gpu4pyscf-native-env/lib/python3.12/site-packages/gpu4pyscf \
  --output /tmp/organic-tau-baseline.json
taskset -c 4-7 /tmp/gpu4pyscf-native-env/bin/python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/native_tau_benchmark.py \
  --case ethyl-4-acetamidobenzoate --tau symmetric \
  --require-gpu-source /tmp/gpu4pyscf-native-env/lib/python3.12/site-packages/gpu4pyscf \
  --output /tmp/organic-tau-symmetric.json
```

Whole-workflow timing includes fresh SCF and one complete force evaluation. Grid-only
measurements do not establish a whole-workflow speedup. Validation datasets remain held out.
