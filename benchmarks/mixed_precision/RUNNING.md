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
using the repository's [compilation instructions](../../README.md#compilation) with at most
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

## TF32 early-grid experiment

`tf32_benchmark.py` compares the existing verified FP32 policy with explicit
`CUBLAS_COMPUTE_32F_FAST_TF32` grid products during the first five SCF iterations.
Use `--tf32-cycles 1` to limit TF32 to the first iteration, then retain ordinary
FP32 grid products for the rest of the existing five-iteration schedule.
VV10, the float64 energy/physical-Fock checks, recovery budget and native derivatives
retain the existing policy. This is an isolated benchmark option; ASE's `auto` default
does not enable TF32.

Run outside the checkout, with the same one-GPU/four-CPU limits and sequential
execution above. Use this branch's extension root when testing against the installed wheel:

```sh
python /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/tf32_benchmark.py \
  --extension-root /absolute/path/to/gpu4pyscf --forces --repeats 3 \
  --output /outside/checkout/tf32-comparison.json
```

The frozen synthetic cases are benzene, water and water cation. All use ωB97M-V,
def2-TZVPD, def2-universal-jkfit, grid levels 5/3, `conv_tol=1e-8`,
`conv_tol_grad=1e-4`, `max_cycle=200` and overlap cutoff `1e-10`. Runs start with
fresh native guesses. The report includes accepted float64 energy/force errors,
fallbacks, SCF cycles and three alternating benzene timing pairs. Matrix microbenchmarks
include conversion and accumulation costs, and check irregular dimensions separately.

Optional `--trace-library /outside/checkout/tensor_trace.so` records kernel names
in separate calls after matrix timing. Build `tensor_trace.cpp` against CUDA 13 CUPTI
headers and `libcupti.so.13`; the CUDA CRT headers must also be on the include path.
Tracing does not measure hardware utilization and is excluded from all timing runs.

## FP64 fixed-point emulation experiment

`fp64_emulation_benchmark.py` compares the existing verified mixed policy with two
opt-in requests for cuBLAS FP64 fixed-point emulation: `emulated-grid` covers the
later float64 grid products, and `emulated-blas` also enables it on the main SCF
thread's cuBLAS handle. Early FP32 grid products and mixed VV10 retain the existing
schedule. Before final physical-Fock verification, the harness disables both
emulation requests; verification, any recovery and analytic forces run natively.
The ASE `auto` default does not enable this experiment.

Use dynamic mantissa control with zero bit offset and the library's automatic
maximum. Do not set `CUBLAS_EMULATE_DOUBLE_PRECISION` or
`CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`: those global overrides would
invalidate the independent native check. `performant` lets cuBLAS select emulation
only when its heuristic expects a gain; `eager` requests it whenever possible.
Either can still dispatch to native FP64. See NVIDIA's
[emulation reference](https://docs.nvidia.com/cuda/cublas/#floating-point-emulation).

Run outside the checkout using the same four-CPU, GPU 1 and sequential-execution
constraints above. Each invocation requires a new report and artifact directory:

```sh
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT
CUBLAS_EMULATION_STRATEGY=performant taskset -c 4-7 python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/fp64_emulation_benchmark.py \
  --extension-root /absolute/path/to/gpu4pyscf --forces --repeats 3 \
  --output /outside/checkout/fp64-automatic.json
CUBLAS_EMULATION_STRATEGY=eager taskset -c 4-7 python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/fp64_emulation_benchmark.py \
  --extension-root /absolute/path/to/gpu4pyscf --forces --repeats 3 \
  --output /outside/checkout/fp64-eager.json
```

The frozen cases and scientific settings match the TF32 experiment above. Three
cyclically alternating benzene comparisons use fresh calculators and density
guesses. The report includes loaded cuBLAS path/hash, package/API versions,
mantissa settings, environment, extension/source hashes, native energy/force errors,
SCF cycles and recoveries. Requested grid-call counts do not prove emulated dispatch.

Matrix probes use six alternating batches of 20 products, include CPU submission,
and check irregular dimensions plus non-unit scaling/accumulation. They retain
FP64 inputs and outputs. Optional `--trace-library` records separate CUPTI calls
after timing, as above. Those traces identify kernels, not whole-SCF utilization.

For a private library comparison, install a pinned cuBLAS package into a new directory
outside the environment and preserve the original library paths. CuPy 14.2 uses
cuda-pathfinder, which prioritizes installed wheels over `LD_LIBRARY_PATH` and does
not use `PYTHONPATH` to find these libraries. Explicitly preload **both** private
cuBLAS libraries before Python starts, then require the expected loaded path:

```sh
dft_cublas_root=/outside/checkout/cublas138
taskset -c 4-7 python -m pip install --only-binary=:all: --no-deps \
  --target "$dft_cublas_root" nvidia-cublas==13.8.0.4
export PYTHONPATH="$dft_cublas_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$dft_cublas_root/nvidia/cu13/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$dft_cublas_root/nvidia/cu13/lib/libcublasLt.so.13:$dft_cublas_root/nvidia/cu13/lib/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
CUBLAS_EMULATION_STRATEGY=eager taskset -c 4-7 python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/fp64_emulation_benchmark.py \
  --extension-root /absolute/path/to/gpu4pyscf --forces --repeats 3 \
  --cublas-library "$dft_cublas_root/nvidia/cu13/lib/libcublas.so.13" \
  --output /outside/checkout/fp64-cublas138-eager.json
```

The harness rejects loaded API/package version mismatches before numerical runs,
and rejects a path mismatch when `--cublas-library` is supplied. Test this
configuration separately. CUDA toolkit and cuBLAS package versions are different:
the tested CUDA 13.4 Update 1 library is `nvidia-cublas==13.8.0.4`;
`13.6.1.10` predates Ozaki-II. The
[release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
describe Ozaki-II selection on Ampere and newer GPUs and the newer workspace pool.
Do not replace shared runtime libraries or interpret a package upgrade as a whole
fork/native build validation.

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
