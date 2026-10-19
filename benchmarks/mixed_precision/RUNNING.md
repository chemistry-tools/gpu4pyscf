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
this host uses a different CPU topology. Keep timed comparisons isolated from tests and
builds, and run them sequentially under the shared lock. Independent correctness checks may
use other compatible host GPUs with four separate logical CPUs per GPU; finish those checks
before measuring timings and do not compare elapsed times across different GPUs.
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

`ao_tile_sparsity.py` analyzes retained NumPy archives containing an `ao` array in
`(components, active_AOs, grid_points)` order. Accepted component counts are 1
(values), 4 (values and first derivatives), or 10 (also second derivatives).
This Linux CPU-only diagnostic needs NumPy, four allowed CPU IDs and no CUDA
installation. It preserves the supplied AO/grid order and records input hashes:

```sh
python benchmarks/mixed_precision/ao_tile_sparsity.py \
  --input /absolute/path/to/ao-block-1.npz /absolute/path/to/ao-block-2.npz \
  --cpus 4,5,6,7 --output /tmp/new-ao-occupancy.json
python -m pytest -q benchmarks/mixed_precision/test_ao_tile_sparsity.py
```

The report separates individual small values, wholly negligible tiles and
conservative Gram-work opportunities, with partial tiles weighted by their valid
sizes. Joint derivative masks retain a tile if any component exceeds the cutoff.
Nonzero cutoffs do not establish density, energy or force accuracy; captured XC
weights and densities can amplify small AO amplitudes. No pruning is enabled.
Sampling selected blocks does not establish whole-grid or whole-call savings.

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

The sparse density-fitting product checks use both retained pre-sparse DF gradient
sources to compare complete forces at the same density. Keep these files outside
the checkout and use the matching native installation described above:

```sh
DFT_GPU_TEST=1 CUDA_VISIBLE_DEVICES=1 \
DFT_REFERENCE_DF_GRADIENT=/absolute/path/to/reference/rhf.py \
DFT_REFERENCE_UHF_DF_GRADIENT=/absolute/path/to/reference/uhf.py \
taskset -c 4-7 /tmp/gpu4pyscf-bench-env/bin/python -m pytest -q \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/test_sparse_products.py
```

The tests explicitly exercise sparse paths on small molecules, including B3LYP,
ωB97M-V, mixed elements and an open shell. Production dispatch is restricted to
compute capability 8.6, at least 256 AOs, sparse enough pair masks and bounded
workspaces. Unsupported inputs retain native products. The existing SCF precision
schedule and final float64 acceptance checks are unchanged. Measure complete
energy-and-force calls separately; a passing kernel test does not establish speed.

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

`test_native_reuse_density.py` checks the bounded occupied-factor products used by
full RKS grid response, together with exact-input ordinary-integration reuse during
VV10-only SCF verification. Run it with `DFT_GPU_TEST=1` in the matching native
environment. Complete B3LYP and VV10 force comparisons include N/O/S/F/Cl organics
and an independent energy finite difference. Supplied density values must match
their orbital tags before force factorization is eligible; unsupported inputs retain
the native products. Cache reuse additionally requires identical density/tag and
grid values/order, molecular data, numerical dispatch, device and stream. Normal
SCF calls always evaluate ordinary integration, and verification still evaluates
native exchange and VV10. Reuse is confined to the verified SCF scope and reports
hit/miss counts in `mixed_precision_info['native_grid_reuse']`.

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

Float64 RKS GGA integration selects a density/AO product when the active AO count
is less than three times the occupied-orbital count. The density is formed from
the tagged occupied orbitals, preserving the native integration convention even
when a caller supplies inconsistent density-array values. Larger blocks retain
occupied-orbital projection. This dispatch is automatic and does not change grids
or SCF tolerances. `test_gga_density.py` compares both paths on mixed-atom molecules
and fractional occupations; run it with `DFT_GPU_TEST=1` in the matching native
environment before the complete semilocal SCF and force checks.

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

## Native auxiliary-response products

On a single RTX 3080, native DF gradients can use two float64 products in
`gpu4pyscf/df/_response_products.py`. Cartesian auxiliary coefficients with one
nonzero per row and column use the native sparse basis transform in metric
construction. The eigensystem and linear-dependence cutoff are unchanged.
Symmetric occupied-factor densities use one triangle of the auxiliary Gram
product and mirror it. Native auxiliary-contiguous and row-contiguous layouts
feed cuBLAS directly, including supported positive leading strides.

These paths are automatic in eligible gradients, including explicit float64
calculations. They do not use reduced precision. General/nonsymmetric factors,
incompatible layouts, nonfinite data and unsupported hardware retain native
products. Matrix sizes and memory headroom bound the sparse-transform dispatch.
The general right-hand-side solver remains available with its original behavior.

Run `test_response_products.py` with `DFT_GPU_TEST=1` and
`DFT_REFERENCE_DF_GRADIENT` pointing to the retained matching original gradient
source. Tests cover both Gram layouts, pitched input, signed accumulation,
alias and stream guards, error restoration, bitwise metric comparisons, complete
mixed-element/open-shell forces, range-separated operators, nonsymmetric
densities and energy finite differences. Complete-call performance comparisons
need unchanged recipes and both grid and auxiliary response enabled.

For unrestricted derivatives, also set `DFT_REFERENCE_UHF_DF_GRADIENT` to the
matching original `df/grad/uhf.py`. Both spin Gram products are checked before
updating output, preserving the native unrestricted exchange scaling.

## Exact-zero float64 density-fitting whitening

Single-GPU compute capability 8.6 builds can apply the native auxiliary inverse
in bounded column blocks, omitting only rows whose coefficients are exactly
zero throughout that block. Remaining reduction rows retain their original
order. The inverse, integral fields and matrix products remain float64;
SCF precision scheduling and native verification thresholds do not change.
The helper owns its coefficient copies and scratch buffer for one build.
Its workspace limit, minimum size and sparsity guard are defined in
`gpu4pyscf/df/_whitening.py`. Multiple GPUs, unsupported layouts/devices,
eigenvalue decompositions, dense matrices and invalid fields retain the
original product.

Run `test_whitening.py` with `DFT_GPU_TEST=1` in the matching native environment
under the resource limits above. Tests explicitly exercise the path on N/O/S/F/Cl
and open-shell molecules, compare host/GPU fitted integrals, independently rebuild
the original SCF integrals, compare full same-density forces and check energy
finite differences. Tight SCF convergence fixes the density for derivative
comparisons; the independent native-energy bound is `1e-9` Hartree, ten times
stricter than production verification. Individual inverse-derived factors can
differ through float64 cancellation, so factor identity is not an acceptance
criterion. Larger complete-call benchmarks must independently disable the new
helper and rebuild the original integrals before claiming native equivalence.

## Mapped density-fitting transfers

Single-GPU RTX 3080 builds that keep fitted integrals in mapped host storage
can transpose one batch on the GPU and copy it through CUDA DMA. The copy runs
on a separate stream and can overlap the next batch's integral evaluation and
matrix product. The scratch buffer is limited to 2 GiB and half the available
device memory; other devices, small batches and insufficient memory retain
the native mapped-write kernel. Both paths use the same float64 arithmetic.
Buffers remain owned until all host writes finish, including error exits.

Run `test_mapped_transfer.py` with `DFT_GPU_TEST=1` in the matching native
environment. It checks partial and overlapping writes, changing batch sizes,
NumPy integer offsets, bit preservation, stream ordering and failure cleanup.
Complete host-backed fitted tensors are compared bit for bit on the N/O/S/F/Cl
and open-shell panel, followed by native SCF/force comparisons. CPU-only runs
check the allocation bounds. Large-system timing must include a case whose
integrals actually spill to host memory; smaller systems can legitimately
retain the existing GPU-resident path.

## Symmetric float64 tau products

The native integrator uses signed symmetric rank updates for identical real float64 AO
fields with more than 512 active orbitals. Eligible medium blocks use the direct triangular
path below; distinct, complex, noncontiguous or nonfinite inputs retain the general bilinear
product. Early mixed SCF retains its original
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
  --case ethyl-4-acetamidobenzoate --tau baseline --triangle baseline \
  --require-gpu-source /tmp/gpu4pyscf-native-env/lib/python3.12/site-packages/gpu4pyscf \
  --output /tmp/organic-tau-baseline.json
taskset -c 4-7 /tmp/gpu4pyscf-native-env/bin/python \
  /absolute/path/to/gpu4pyscf/benchmarks/mixed_precision/native_tau_benchmark.py \
  --case ethyl-4-acetamidobenzoate --tau symmetric --triangle baseline \
  --require-gpu-source /tmp/gpu4pyscf-native-env/lib/python3.12/site-packages/gpu4pyscf \
  --output /tmp/organic-tau-symmetric.json
```

Whole-workflow timing includes fresh SCF and one complete force evaluation. Grid-only
measurements do not establish a whole-workflow speedup. Validation datasets remain held out.

## Direct medium-sized float64 tau products

`_tau_triangle.py` evaluates one matrix triangle directly with signed float64 weighted
products and mirrors its result. Shared-memory tiles replace the three full products
without splitting positive/negative weights or allocating stacked AO fields. Native
dispatch selects it only on compute capability 8.6 within the measured orbital/grid
range defined in that module. Compatibility, finite-field and output-alias guards
preserve the original fallbacks. Early float32 SCF products keep their existing dispatch.

Run `test_tau_triangle.py` with the opt-in native checks. It includes partially filled
tiles, signed/zero weights, range and architecture guards, aliased outputs, independent
CUDA streams and unchanged early float32 products. Its complete organic and open-shell
checks explicitly exercise the kernel below the production size guard, compare full
forces, and check energy finite differences at stricter correctness-only tolerances.

For fresh ASE comparisons against the preceding optimized engine, keep `--tau symmetric`
in both runs and alternate `--triangle baseline` with `--triangle native`. The default
is `native`; reports record both settings and all eligible triangle calls. The baseline
switch affects only this direct medium-block path in the benchmark process. A case can
legitimately have no eligible triangle blocks, so inspect the recorded coverage before
attributing a timing change to the kernel. The full native package, four-CPU limits and
shared sequential lock are required as above.
