# Fixed-method performance measurements

Measured on 2026-10-08 with one RTX 3080 10 GB and four logical CPUs near the GPU, on a dual
Xeon E5-2667 v4 server. Other GPUs were occupied. Every calculation ran sequentially in a fresh
process with CPU affinity and OpenMP/BLAS limits of four. The CPU reference averaged 3.82 active
CPU threads. These measurements describe this host and these molecules, not general throughput.

The method is the pinned benchmark protocol: ωB97M-V/def2-TZVPD with density fitting, ordinary grid level 5,
nonlocal grid level 3, full VV10, energy tolerance `1e-8` Hartree and orbital-gradient threshold
`1e-4`. The environment used GPU4PySCF 1.8.1, PySCF 2.10.0, CuPy 14.2.0, cuTENSOR 2.3.1,
CUDA 13.0.2 toolkit metadata, Python 3.12.13, NumPy 2.5.3 and SciPy 1.18.1.

## Benzene

| Calculation | SCF seconds | Cycles |
| --- | ---: | ---: |
| Native CPU PySCF, one reference run | 215.46 | 7 plus native extra verification |
| Native GPU4PySCF defaults, one run | 23.01 | 7 |
| Full-basis float64 GPU, median of three fresh runs | 22.60 | 7 |
| Mixed grid/VV10 with full float64 verification, median of three | 8.80 | 7 plus verification |

Full-basis float64 times were 22.596, 22.479 and 22.916 seconds. Optimized times were 8.707, 8.948
and 8.802 seconds. Alternating baseline/optimized repeat runs gave a **2.57× GPU speedup**, or
61% less SCF time. Imports and object construction added roughly 1.2 seconds. The four-thread
CPU reference is about 24.5 times the optimized GPU median, with native backend grid/pruning
differences and only one CPU sample.

GPU4PySCF's default `1e-6` overlap cutoff removed one of the 276 benzene basis orbitals. Lowering
the explicit benchmark cutoff to `1e-10` retained all orbitals and reduced the CPU/GPU energy
difference from `6.34e-6` to `4.55e-8` Hartree. Both baseline and optimized comparisons use this
same full-basis treatment. GPU grids contain 399,360 ordinary and 143,616 nonlocal points;
the CPU prunes more points. The precision experiment does not alter either grid.

The full-basis float64 GPU reference energy was `-232.226401813013` Hartree. All three optimized
results differed by less than `5.4e-11` Hartree. Final float64 orbital gradients were approximately
`1.58e-5`; their float32-to-float64 energy changes were below `7.8e-11` Hartree. Every final
verification passed at the original production thresholds.

## Costs and experiments

The original GPU profile spent 14.13 seconds in VV10 pair evaluation (62%), 6.57 seconds in
ordinary DFT integration (29%) and 0.74 seconds in J/K (3%). The implementation first accelerates
these two dominant terms. The [benchmark guide](README.md) explains precision switching,
verification, fallback, and reproduction commands. These remain explicit benchmark options;
the default library calculation path is unchanged.

On 129,570 active nonlocal points, the original float64 pair kernel took approximately 1.77
seconds. Float32 pair arithmetic with short sums combined in float64 took about 0.060 seconds.
The VV10 energy-contribution error at the captured density was `4.8e-11` Hartree. Raw derivative
errors are larger in low-density tails, so total-energy agreement is insufficient by itself;
the final full float64 Fock and orbital-gradient check remains required.

NVRTC and NVCC 13.0.88 gave approximately equal performance with identical numerical results.
The matching CRT and NVVM components were pinned to 13.0.88. A double reciprocal refined from
PTX's approximate reciprocal was slower, and a 256-thread tile was slower than 64/128-thread
tiles. Direct float64 BLAS dispatch and inline single-GPU tasks gave no material end-to-end
benefit. SAP and modified Huckel initial densities both retained seven iterations and did not
improve total time, including their CPU construction cost. No global fast-math flag was used.

## Additional chemistry checks

All checks used the same functional, basis, grids and final tolerances, independently initialized.

| Molecule | Float64 GPU seconds | Optimized seconds | Energy difference, Hartree |
| --- | ---: | ---: | ---: |
| Neutral water | 3.57 | 2.22 | `4.4e-13` |
| Water cation, doublet | 6.01 | 2.87 | `2.8e-14` |

Both passed the full float64 energy and gradient check. The kernel suite additionally checks
partial tiles, signed weights, zero-density points, rectangular matrix products, accumulation
and restoring patched functions. Broader molecular, geometry, spin and derivative coverage is
needed before integrating this policy into production. UMA geometry preoptimization was not used:
these comparisons require identical fixed geometries, and UMA does not supply an SCF density here.

The original measurements used the installed GPU4PySCF 1.8.1 release, upstream tag
`5b284c258a4260baef80e3d150b4e7a81a9dbd57`. This fork's starting `master` revision
`43dc42fca91c12d4c2fd28e47e4342705b3161fd` includes later changes despite the same version string.
The timing table above does not establish performance for those later library changes.
The original settings source SHA256 was
`0cf9ae9170ab4a739745428978be9097d962fbd8a1cba7d252796d318047fc1c`;
[protocol.json](protocol.json) preserves those scientific defaults without an external package.

Raw reports, logs, densities and profiles are retained outside the checkout. The reports record
exact source/settings hashes, software, affinity, thread pools, precision phases and verification.

## Fork port verification

The standalone harness at commit `094d71a00bf6bea38b1e3f8d3b06cc1f56f25aeb` passed all 25
focused tests on GPU 1: 14 CPU resource/protocol tests and 11 CUDA checks. A fresh sequential
benzene comparison using the same pinned 1.8.1 wheel took 22.7007 seconds for the float64
baseline and 8.7197 seconds for the optimized calculation, a 2.60× speedup in this single pair.
This confirms the port reproduces the experiment; it does not replace the three-run median.

The final energy difference was `5.3774e-11` Hartree, the float32-to-float64 verification change
was `7.6795e-11` Hartree, and the full float64 orbital gradient was `1.5823e-5`. Verification
passed without fallback. Reports confirmed CPU affinity `[4, 5, 6, 7]`, four OpenMP threads,
the experiment commit, actual wheel source hashes and the native library hash. The
`--require-gpu-source` guard also correctly rejected that wheel when the fork checkout was
required. Current master native-library changes remain outside this pinned-release check.

## Profiling and grid-contraction overhead

The changes after `6cd096f`, through `70a5162`, disable profiling-only VV10 event synchronization
in ordinary runs and reuse per-thread/per-stream float32 grid scratch. Conversion, scaling and
accumulation now use one float64 output kernel. Its multiply/add rounding matches the separate
operations; cuBLAS still uses pedantic math. Scratch is released when the precision scope exits.
The explicit benchmark `--profile` option retains synchronized VV10 timing.

Synthetic grid tests use deterministic normal fields (seed 75), `alpha=1`, `beta=1`, and six
alternating batches of 50 contractions per implementation. Compilation and the first call are
outside the timing. These are median per-call wall times, including CPU submission overhead:

| Matrix rows × columns; grid points | Previous, ms | Fused, ms | Speedup |
| --- | ---: | ---: | ---: |
| 32 × 32; 512 | 0.1923 | 0.1497 | 1.28× |
| 276 × 276; 4,096 | 0.2024 | 0.1581 | 1.28× |
| 276 × 276; 16,384 | 0.4406 | 0.4301 | 1.02× |

The outputs matched bit for bit, including repeated accumulation. The focused suite also checks
non-unit and negative scaling, zero-beta overwrite of NaN outputs, changing block shapes,
independent returned outputs, separate CUDA streams and releasing scratch without retaining it
through private dispatch cycles. All 71 focused checks passed, including native float64 force
comparisons and a finite-difference force check on synthetic water.

For the unchanged benzene protocol, three fresh alternating runs per implementation took
8.7995, 8.7238 and 8.7273 seconds before, and 8.6249, 8.7500 and 8.7150 seconds after. Medians
were **8.7273 and 8.7150 seconds**: no material end-to-end improvement is established. All six
full float64 energy/orbital-gradient checks passed without fallback; accepted energies differed
by less than `4e-13` Hartree. An earlier three-pair check showed about 1% improvement, also
within timing variation. Kernel gains must not be presented as equivalent whole-SCF gains.

## TF32 tensor-core experiment

The October 12 comparison uses an RTX 3080, physical GPU 1, CPU affinity `[4,5,6,7]`,
and four numerical threads. The installed environment contains GPU4PySCF CUDA 13 1.8.1,
PySCF 2.14.0, CuPy 14.2.0, NumPy 2.5.3 and cuBLAS 13.1.1.3. Only the fork's precision
modules are loaded over the installed numerical/native implementation. These results do
not establish compatibility of the complete fork with older native libraries.

The synthetic benzene recipe is ωB97M-V/def2-TZVPD with def2-universal-jkfit, grid
levels 5/3, full VV10, energy tolerance `1e-8` Hartree, orbital-gradient tolerance
`1e-4`, cycle budget 200 and overlap cutoff `1e-10`. Every run starts from a fresh
native density guess. Energy verification, physical-Fock checks, float64 recovery and
native float64 gradients retain the existing settings. The two schedules use separate
three-pair alternating comparisons against the existing FP32 mixed policy:

| TF32 grid schedule | Matched FP32 SCF median, s | TF32 SCF median, s | SCF cycles, FP32/TF32 | Energy + forces median, FP32/TF32, s |
| --- | ---: | ---: | --- | --- |
| First five iterations | 8.1297 | 9.9647 | 7 / 9 | 17.5206 / 19.3288 |
| First iteration only | 8.0287 | 7.9954 | 7 / 7 | 17.3554 / 17.4108 |

Five TF32 iterations add two SCF cycles and increase SCF time by **22.6%** and
energy-plus-force time by **10.3%**. The first-iteration schedule avoids extra cycles,
but its 0.4% SCF median reduction is within the run-to-run variation; total
energy-plus-force time is 0.3% higher. No material workflow speedup is established,
so the normal `auto` policy keeps TF32 disabled.

All accepted results pass the original float64 checks without recovery. The maximum
benzene energy differences against native float64 are `6.82e-10` Hartree for five
TF32 iterations and `1.81e-10` Hartree for one. Maximum force-component differences
are `5.08e-5` and `6.02e-5` eV/Angstrom respectively. Closed-shell water and doublet
water-cation energy/force checks also pass; their individual timings are correctness
checks, not repeated performance measurements.

Matrix microbenchmarks use fixed random fields, six alternating batches of 50
contractions, and include float64-to-float32 conversion and float64 accumulation:

| Rows × columns; grid points | FP32 median, ms | TF32 median, ms | Speedup |
| --- | ---: | ---: | ---: |
| 17 × 31; 83 | 0.1454 | 0.1477 | 0.98× |
| 32 × 32; 512 | 0.1475 | 0.1547 | 0.95× |
| 276 × 276; 4,096 | 0.1574 | 0.1602 | 0.98× |
| 276 × 276; 16,384 | 0.4113 | 0.3515 | 1.17× |

Separate CUDA 13 CUPTI activity traces record ordinary `ampere_sgemm` for the FP32
probe and `cutlass_80_tensorop_s1688gemm` for the TF32 probe, with zero dropped
records. This confirms tensor-core kernel selection for the representative matrix
operation; it is not a measurement of tensor-core utilization for the whole SCF.
Tracing is excluded from the timings. The focused correctness suite has 61 passing
checks, including irregular matrix dimensions, native float64 restoration, handle
restoration after injected errors, and real energy/force comparisons.

Use the [TF32 reproduction guide](RUNNING.md#tf32-early-grid-experiment). The
benchmark option remains experimental and is not exposed as an ASE precision default.

These checks used only synthetic fields, benzene and water; validation datasets were held out.
They ran sequentially on physical GPU 1 with affinity `[4, 5, 6, 7]` and four numerical threads.
SCF and kernel timing used the pinned CUDA 13 runtime described above; force/lifecycle checks
additionally used ASE 3.27.0. Only the new precision/ASE Python modules were explicitly loaded
over the matching installed 1.8.1 release. This does not verify a full native build of current
master. Raw reports, source hashes, scripts and startup-failure logs remain outside Git.

## FP64 fixed-point tensor-core emulation

The October 12 FP64 experiment uses the same synthetic benzene recipe and numerical
implementation as the TF32 comparison above: one RTX 3080, physical GPU 1,
CPU affinity `[4,5,6,7]`, four numerical threads, PySCF 2.14.0, GPU4PySCF CUDA 13
1.8.1 and CuPy 14.2.0. Only the fork's precision extensions are loaded over the
installed implementation. Native source/library hashes remain recorded; this does
not validate a complete native build of the fork.

The existing mixed baseline uses early FP32 grid products and mixed VV10. The
`emulated-grid` variant additionally requests cuBLAS FP64 fixed-point emulation
for later grid products. `emulated-blas` also enables the request on the main SCF
thread's cuBLAS handle. All variants disable the requests before final native
energy/physical-Fock verification, recovery and analytic forces. Dynamic mantissa
control, automatic maximum and zero bit offset retain native FP64 accuracy;
fixed-precision and global FP64-emulation overrides are absent. `eager` requests
emulation whenever possible; `performant` lets cuBLAS predict whether it helps.

Each configuration uses three cyclically alternating fresh-calculator benzene
comparisons. Timings include native verification; force timings remain native.
Baseline rows are matched within each library/policy comparison, rather than
reusing timings across upgraded libraries:

| cuBLAS package / selection | Mixed baseline SCF, s | Grid emulation SCF, s | Grid + main-handle emulation SCF, s | Energy + forces, baseline / grid / broader, s |
| --- | ---: | ---: | ---: | --- |
| 13.1.1.3 / eager | 8.1403 | 8.7985 | 11.3966 | 17.5458 / 18.1441 / 20.7397 |
| 13.8.0.4 / performant | 8.2798 | 8.2430 | 8.1820 | 17.7431 / 17.6807 / 17.6168 |
| 13.8.0.4 / eager | 8.1768 | 8.8676 | 12.5900 | 17.6299 / 18.3206 / 22.0344 |

Forced emulation in the installed library increases benzene SCF time by
about **8%** for grid products and **40%** for broader BLAS coverage. Requesting
emulation does not by itself establish that tensor-core emulation ran.
With 13.8.0.4, eager selection increases SCF time by **8.45%** and **53.97%**,
respectively, and energy-plus-force time by **3.92%** and **24.98%**. All benzene
runs retain seven SCF iterations and pass native verification without recovery.
Accepted energies differ from the native reference by less than `5.4e-11` Hartree;
maximum force-component errors stay below `3.7e-5` eV/Angstrom. Closed-shell water
and doublet water-cation energy/force checks also pass. The focused suite has 59
passing checks in both actual library configurations, plus manual negative
checks that reject wrong loaded paths and API/package mismatches before any
matrix/SCF benchmark.

Six alternating batches of 20 matrix products include CPU submission overhead,
with FP64 inputs/outputs and separate non-unit scaling/accumulation checks. On
13.1.1.3 with eager selection, a random 1024×1024×1024 product improves from
**5.1153 ms to 0.8220 ms (6.22×)**. The 276×276×16,384 probe improves from
6.1573 to 4.6586 ms (1.32×), while the irregular 17×31×83 probe slows from
0.0394 to 0.3394 ms. Maximum relative Frobenius error is `1.36e-15`. Separate
CUPTI traces show `cublasLt_fused_imma_dgemm_kernel_sm80` or its split-K variant,
with zero dropped records. Tracing is excluded from timing; these are matrix
kernel-selection measurements, not whole-SCF tensor-core utilization.

With the verified 13.8.0.4 library, eager selection uses Ozaki-II kernels
containing `oz2_int8_dgemm` and `nvjet_sm86_biu_mma` in the separate traces.
The 1024-cubed probe takes 5.1927 ms natively and 1.2906 ms with emulation
(4.02×); 276×276×4,096 instead slows from 1.8136 to 2.0282 ms, and
276×276×16,384 from 6.2635 to 7.2937 ms. The modular-conversion prologs take
most of the rectangular probe's traced time. This contrasts with the faster
square product and shows why a dense-kernel speedup cannot establish a molecular
workflow improvement. Maximum relative Frobenius error remains `1.36e-15`.
The automatic matrix probes select native FP64 and have approximately equal
timings. The automatic SCF median differences are 0.4–1.2%, comparable with
the measured variation; they do not establish a useful emulation speedup.

An additional untimed SCF diagnostic samples 14 distinct substantial benzene grid
products. Most have 220–270 rows and a 4,096-point contraction dimension; one has
2,048 points. Values include extremely small Gaussian tails. The dynamic mantissa
counter reports 91–1,020 bits, compared with roughly 60 in the initial random
probes. In 12 samples, the native FP64 kernel consumes more than 90% of traced
GPU kernel time while integer GEMMs take only about 5–6 microseconds. These
durations indicate native fallback after precision analysis; the other two samples
perform substantial Ozaki-II work. Kernel names alone can include conditionally
skipped emulation/native launches. This diagnostic explains why requesting
emulation on actual molecular fields adds setup work without the dense-probe gain;
its instrumented SCF times are excluded from every performance table.

No useful whole-workflow speedup is established for this recipe, so FP64 emulation
remains an opt-in benchmark and is disabled in the normal ASE `auto` policy.

The cuBLAS packages are installed in private external directories, leaving the
shared runtime unchanged. CUDA toolkit versions and cuBLAS package versions are
different. Early upgrade attempts changed package metadata but still loaded
**13.1.1.3**, because cuda-pathfinder searched the existing environment before
the supplied library paths. Their loaded paths, hashes and API version identify
that mismatch; these retained diagnostics are excluded from upgrade comparisons.
The guarded runs explicitly preload both private cuBLAS libraries and require
their loaded path and API to match the expected library/package. Separate
13.8.0.4 reports test CUDA 13.4 Update 1, whose
[release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
describe Ozaki-II selection and retained workspace memory. Do not infer an
algorithm or loaded library version from a filename or package metadata alone.

Use the [FP64 reproduction guide](RUNNING.md#fp64-fixed-point-emulation-experiment).
Raw reports, exact source snapshots, controller scripts, traces and intermediate
results remain outside Git. Validation datasets are held out.

## Larger synthetic size comparisons

The follow-up uses the same ωB97M-V/def2-TZVPD recipe, GPU 1, four-CPU affinity,
native implementation and independently generated idealized geometries described
in the [larger-system guide](RUNNING.md#larger-synthetic-systems). The cases do not
come from held-out datasets. Native float64 reference timings are single runs;
the mixed and emulated columns are medians of three cyclically alternating fresh
SCFs. Every SCF receives native energy/physical-Fock verification. Native forces
are measured and compared once per variant, on its first repeat.

Completed size comparisons, in seconds:

| Molecule | Basis functions | Actual cuBLAS | Native FP64 reference | Existing mixed | Grid emulation | Grid + main-handle emulation |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Naphthalene, C10H8 | 442 | 13.1.1.3 | 80.0077 | 27.5954 | 30.5523 | 37.5867 |
| Naphthalene, C10H8 | 442 | 13.8.0.4 | 81.1440 | 27.5676 | 30.6855 | 40.8705 |
| Pyrene, C16H10 | 682 | 13.1.1.3 | 188.9984 | 71.4960 | 75.9034 | 89.5331 |
| Pyrene, C16H10 | 682 | 13.8.0.4 | 188.9207 | 71.2234 | 75.4702 | 87.6566 |

For naphthalene, grid emulation increases SCF time by 10.7–11.3%, while broader
main-handle coverage increases it by 36.2–48.3%. Pyrene's increases are 6.0–6.2%
and 23.1–25.2%. The smaller overhead at 682 orbitals does not yet
establish a speedup. These completed runs retain all basis orbitals and ten SCF
iterations. Naphthalene energy errors are below `7e-13` Hartree and force-component
errors below `1.7e-5` eV/Angstrom, without recovery.
The completed pyrene comparison has energy errors below `7.9e-10` Hartree and
force-component errors below `1.93e-4` eV/Angstrom, within the original comparison
thresholds. This is fixed-geometry energy/force coverage, not Hessian or TS validation.

The observed full/range-separated density-fitting buffers for naphthalene and
pyrene remain on the GPU. Pyrene stores approximately 3.32 GiB across the two
caches; first-repeat native force calls take approximately 62.5 seconds, regardless
of SCF variant. Force timing has one sample per variant and must not be presented
as a three-run optimizer/workflow measurement. The initial 13.1 naphthalene
placement formatter did not handle array lists; its zero-byte storage fields are
unavailable, not evidence of zero use. Its numerical/timing results are unaffected,
and the original snapshot plus an explicit reporting note are retained. Later
reports use the corrected formatter.

Raw reports and logs remain outside Git. Library selection is checked against the
loaded path, API and package metadata; the shared environment is unchanged.

## Skala ASE comparison

An isolated Skala 2026.9 / Skala-1.1-rev1 experiment used Python 3.12, PySCF 2.14.0,
GPU4PySCF 1.8.1, PyTorch 2.13.0 CUDA 13 and ASE 3.27.0. All methods used the same installed
numerical sources/native libraries and this fork's explicit Python precision/ASE extensions.
The newer PySCF requirement means these are fresh comparisons, not ratios against the older
environment's timings above. [SKALA.md](SKALA.md) documents the harness and reproduction.

All cases were synthetic; validation datasets were untouched. Runs were sequential on the
10 GB RTX 3080, physical GPU 1, with affinity `[4,5,6,7]` and four numerical threads.
Benzene retained all 276 orbitals in def2-TZVPD. Density fitting used explicit
`def2-universal-jkfit`, energy tolerance `1e-8` Hartree and orbital-gradient tolerance `1e-4`.
Skala used its published D3 correction; ωB97M-V retained full VV10.

Skala's default dense model evaluation ran out of GPU memory at ordinary grid level 5,
before finishing an SCF. The completed Skala comparison uses its usual level 3, with 143,556
points. Both ωB97M-V variants retain level 5, with 399,360 points, and nonlocal level 3.
This is explicitly a comparison between different functionals and quadrature configurations.

These are medians of three fresh-calculator energy-and-forces calls after retaining the first
call separately. Imports/model loading are excluded; process caches and the Skala model remain
warm. SCF and gradient column medians need not sum exactly to the total median.

| Method | SCF, s | Forces, s | Energy + forces, s | SCF cycles |
| --- | ---: | ---: | ---: | ---: |
| ωB97M-V native float64, level 5 | 22.104 | 9.373 | 31.481 | 7 |
| ωB97M-V verified mixed, level 5 | 7.966 | 9.490 | 17.460 | 7 |
| Skala-1.1 + D3, level 3 | 7.747 | 14.295 | 22.062 | 10 |

The first calculation in those processes took 44.528, 18.812 and 26.414 seconds respectively;
imports/model loading took 1.278, 1.035 and 4.306 seconds. These are not cold-machine startup
measurements. Skala was faster than the native float64 baseline, but took 26% more time than
verified mixed ωB97M-V for energy and forces, even with its smaller grid. Its force evaluation
accounts for the difference; the measured SCF costs were similar.

A matched-start BFGS experiment scaled the same benzene geometry by 1.025 and required
forces below 0.005 eV/Å with maximum step 0.1 Å. This was one complete optimization per
workflow, including all force calls and setup within each stage, with imports excluded:

| Workflow | Stage time, s | Optimizer steps | SCF cycles |
| --- | ---: | ---: | ---: |
| Direct verified mixed ωB97M-V | 82.499 | 4 | 29 |
| Skala preoptimization | 145.238 | 6 | 47 |
| ωB97M-V refinement after Skala | 95.634 | 5 | 25 |
| Complete Skala → ωB97M-V workflow | 240.872 | 11 | 72 |

The cascade took 2.92× as long as direct DFT in this example. Final DFT forces were
0.000874 and 0.001104 eV/Å respectively, and final energies differed by `2.12e-7` eV.
Every DFT precision verification passed without fallback. These are force-converged geometry
optimizations, not frequency-verified minima or TS validation. No density was transferred
between functionals, and UMA preoptimization was not benchmarked in this experiment.

The published Skala GPU gradient reset also retained the preceding D3 geometry. A synthetic
3% benzene geometry change reproduced a `0.0108293` eV stale-dispersion error; explicit D3
reset matched a fresh object. The optimization harness uses a local ASE subclass with this
refresh enabled and records that choice. The installed Skala package was not changed.
Water geometry reuse and a central finite-difference force check passed with the refresh:
reused/fresh energy difference `6.92e-8` eV, maximum force difference `1.54e-4` eV/Å and
finite-difference force error `1.67e-4` eV/Å at a 0.001 Å displacement. Neutral and doublet
water energy/force smoke checks passed. This does not establish broader chemical accuracy.

Reports, harness snapshots, the pinned environment, source/model hashes, startup-failure
logs and the reset diagnostic are retained outside Git. Skala runs emitted interpreter
shutdown exceptions after writing valid reports and exiting with status zero; those messages
remain unresolved. The final formatted harness passed another water force/geometry check
with all SCFs converged, plus Ruff and 14 resource/protocol tests (18 CUDA checks skipped in
that CPU suite). This experiment does not make Skala a calculator default. A full native
build of the fork remains unverified as above.
