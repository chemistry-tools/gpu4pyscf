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

These checks used only synthetic fields, benzene and water; validation datasets were held out.
They ran sequentially on physical GPU 1 with affinity `[4, 5, 6, 7]` and four numerical threads.
SCF and kernel timing used the pinned CUDA 13 runtime described above; force/lifecycle checks
additionally used ASE 3.27.0. Only the new precision/ASE Python modules were explicitly loaded
over the matching installed 1.8.1 release. This does not verify a full native build of current
master. Raw reports, source hashes, scripts and startup-failure logs remain outside Git.

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
