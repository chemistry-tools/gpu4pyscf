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

## Force evaluation profile

On 2026-10-12, sequential native float64 force profiles used idealized synthetic pyrene
(26 atoms, 682 AOs) and coronene (36 atoms, 996 AOs), the same method/grids above, full grid
and auxiliary-basis response, physical GPU 1 and four CPUs. This later runtime used PySCF
2.14.0 and the pinned GPU4PySCF 1.8.1 CUDA 13 wheel. Component timings synchronize the GPU;
unprofiled repeated measurements are required for speed comparisons.

| Gradient component | Pyrene, seconds | Coronene, seconds |
| --- | ---: | ---: |
| Complete gradient | 62.18 | 140.75 |
| Semilocal XC full response | 24.94 | 58.33 |
| VV10 full response | 22.67 | 45.88 |
| VV10 U/W/E pairs, included in VV10 response | 8.58 | 16.72 |
| VV10 coordinate-response pairs, included in VV10 response | 9.70 | 19.06 |

Grid construction took 0.094/0.231 seconds, and the grid-weight derivative calls totaled
0.348/0.952 seconds. Semilocal/VV10 derivative contractions are better targets than these
small terms. The remaining complete-gradient time includes density-fitting exchange/Coulomb
derivatives and one-electron/nuclear terms; those were not individually timed in this profile.
The SCF and gradient routines rebuild grids in different orders, so their VV10 fields cannot
be reused by array shape alone.

## Density-first float64 force contractions

The full-response RKS XC and shared RKS/UKS VV10 routines now contract the density before
forming AO derivative matrices. For a meta-GGA, this replaces fifteen AO-square products
with five density/AO products and per-AO reductions. Every derivative remains float64;
the change preserves the density, all grid points and both response terms. It introduces
no SCF-to-gradient field cache.

Three alternating unprofiled force calls per implementation used the same accepted density
and rebuilt the native full/range-separated DF buffers before each timed gradient. The
synthetic molecules and resource limits match the profile above; validation datasets were
held out. These are complete-gradient medians:

| Molecule | Native, seconds | Density-first, seconds | Speedup | Less force time |
| --- | ---: | ---: | ---: | ---: |
| Pyrene | 62.5023 | 49.3319 | 1.267× | 21.1% |
| Coronene | 140.2340 | 109.6475 | 1.279× | 21.8% |

Pyrene native samples were 62.8358, 62.4885 and 62.5023 seconds; optimized samples were
49.3330, 49.3319 and 49.3267 seconds. Coronene native samples were 140.2365, 140.2289 and
140.2340 seconds; optimized samples were 109.6527, 109.6452 and 109.6475 seconds.
Maximum force-component differences from the first native call were `5.50e-12` and
`2.58e-13` Hartree/Bohr, respectively, including native repeat variation. The larger error
is about `2.83e-10` eV/Å. Both molecules retained the full basis and passed native SCF
verification without fallback.

The shared SCF runs took 71.3592/202.0493 seconds. Adding those to the force medians gives
estimated energy-plus-force reductions of 9.8%/8.9%; this sum is not a fresh whole-workflow
timing comparison. The VV10 U/W/E and coordinate-response pair kernels remain unchanged.

All 98 focused checks passed, including native derivative-matrix comparisons for LDA/GGA/MGGA,
atom reduction with missing basis functions, neutral/open-shell full forces and finite-difference
energies. Three additional ASE checks passed with the new force path active, including unchanged
geometry SCF reuse, geometry changes and finite-difference forces. Core Ruff checks and GitHub
checks passed. Only the reviewed contraction sites and atom reduction were adapted into the
matching installed-release gradient functions; their grid loops and native calls were retained.
A full native build was not verified in this experiment; the separate native installation
check below covers it. Raw reports, profiles,
fields, source snapshots and logs are retained outside Git. The benchmarked gradient source
SHA256 is `7450bda807002d78cfa14f8d7737c24003521aca021d910998a42442d9d65328`.

## Fused float64 VV10 pair response

The full-response gradient computes U/W/E and the explicit coordinate derivative in one
float64 pair traversal. It retains the native pair expressions and accumulation order for
U/W/E. Antisymmetry of the weighted coordinate pair derivative combines the two atom
contributions, followed by a deterministic atom reduction without floating-point atomics.
The CuPy/NVRTC kernel is cached per device; no input fields or forces are cached.

Frozen fields from the synthetic force profiles above were compared with both original native
pair calls and their final atom-by-grid reduction. These are three-call wall-time medians,
including allocation/reduction, after compilation, with all calculations sequential on GPU 1
and four CPUs:

| Molecule | Active points | Native pairs + reduction, s | Fused, s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Pyrene | 283,670 | 18.1755 | 12.5266 | 1.451× |
| Coronene | 393,810 | 35.7813 | 24.7116 | 1.448× |

All captured U/W/E fields matched bit for bit. Maximum coordinate-response differences were
`1.02e-16`/`1.83e-16` Hartree/Bohr. The response intermediate shrank from 168.81/324.49 MiB
to 6.49/9.01 MiB, respectively. Testing 64/128/256-thread tiles showed no clear benefit over
128 threads; that remains the default. These pair timings alone are not whole-gradient gains.

Three alternating complete-gradient calls then compared the validated density-first path
with that same path plus fused VV10. Each pair shared one accepted density, with identical
geometry, full basis, grid levels, response terms and reconstructed DF buffers:

| Molecule | Density-first, s | Plus fused VV10, s | Speedup | Additional force reduction |
| --- | ---: | ---: | ---: | ---: |
| Pyrene | 49.3816 | 43.6452 | 1.131× | 11.6% |
| Coronene | 110.0534 | 98.9623 | 1.112× | 10.1% |

Pyrene baseline samples were 49.4736, 49.3816 and 49.3607 seconds; fused samples were 43.6536,
43.6452 and 43.6397 seconds. Coronene baseline samples were 109.8401, 110.0594 and 110.0534
seconds; fused samples were 98.6789, 99.0123 and 98.9623 seconds. Maximum differences from
the first baseline gradient were `4.27e-12`/`2.94e-13` Hartree/Bohr, including baseline repeat
variation. The larger difference is about `2.20e-10` eV/Å. Native SCF verification passed
without fallback, and all 682/996 basis functions were retained.

The single shared SCF runs took 72.5729/202.4917 seconds. Adding those to the gradient medians
gives estimated additional energy-plus-force reductions of 4.7%/3.5%. Relative to the original
force medians reported above, both changes together reduce forces by approximately 30%; those
original and current medians come from separate benchmark batches under the same protocol.
These sums do not establish fresh whole-workflow timing medians.

All 112 focused checks passed, plus three ASE checks with fused VV10 active. Coverage includes
native U/W/E and coordinate-response comparisons, independently differentiated pair energies,
full neutral/open-shell forces and energy finite differences, partial tiles, signed/zero weights,
empty atom groups, independent CUDA streams, SCF reuse and geometry changes. Core Ruff and
GitHub checks passed. Only the reviewed gradient sites were adapted into the matching installed
release; remaining numerical Python/native code was retained. The new CuPy kernel itself was
compiled and tested directly from its production source. These installed-release measurements
did not verify a complete native rebuild; the separate native installation check below covers
it. Validation datasets were held out, and raw snapshots, fields,
reports and logs remain outside Git. The new kernel module SHA256 is
`bea0d52af4cbfb3053a8a36efc42afbd4467e49ce44f6e898546b94ff07ab62a`.

## Complete native RTX 3080 installation

The complete fork was rebuilt and installed as an ordinary wheel in a private Python 3.12
environment. All twelve fork libraries compiled, loaded and contained only `sm_86` device
code; their installed hashes matched the build outputs. Source modules imported from the
new installation's `site-packages`, with no installed-release compatibility adapter. The
numerical source is commit `85c406c`; the package frontend additionally contains the build-job
and argument-parsing fixes described in [NATIVE_BUILD.md](NATIVE_BUILD.md).

The build used CUDA 13.0.2/NVCC 13.0.88, GCC/G++/gfortran 14.2.0, CMake 4.4.4 and Ninja
1.13.2. Release mode used `-O3`, native host architecture flags and `86-real`. OpenMP,
CUTLASS, solvent, periodic, multigrid and ECP libraries remained enabled. Global fast math
was not enabled. LibXC remained the normal `gpu4pyscf-libxc-cuda13x==0.8.1` dependency,
providing CUDA LibXC 7.0.0; it was installed directly in the private environment. Other
Python/runtime dependencies were shared read-only from the retained environment, including
PySCF 2.14.0, CuPy 14.2.0, cuTENSOR 2.3.1 and cuBLAS 13.1.1.3. Builds, tests and calculations
ran sequentially with GPU 1 and CPU affinity `4,5,6,7`.

All 101 focused native checks passed. They cover the precision lifecycle and recovery,
ordinary ASE energy/force calls, unchanged-geometry SCF reuse, geometry changes,
neutral/open-shell finite differences, and fused VV10 fields/derivatives against the newly
built native calls. The separate compatibility-only force tests were not used for this
installation check. Core Ruff checks passed for the package frontend and benchmark metadata
change, using the repository's existing exclusions for legacy setup imports.

Direct complete-gradient comparisons then used the original upstream gradient functions
from `43dc42f`, executed with the same newly built libraries and grid loops as the optimized
installation. Neutral water, water cation and benzene matched within `3.11e-15`, `6.22e-15`
and `1.87e-14` Hartree/Bohr, respectively. Independent central energy differences on water
and water cation agreed with analytic gradients within `6.28e-9` and `7.37e-9` Hartree/Bohr.
Their single-call timings are accuracy checks rather than speed comparisons.

Pyrene used the same synthetic geometry, full 682-function basis, functional, auxiliary
basis, grid levels, response terms and scientific thresholds as the earlier measurements.
One verified density was shared, and DF buffers were reconstructed before each timed
gradient. Three alternating unprofiled calls per implementation gave:

| Molecule | Upstream native, s | Optimized native, s | Speedup | Less force time |
| --- | ---: | ---: | ---: | ---: |
| Pyrene | 61.2417 | 42.6445 | 1.436× | 30.4% |

Upstream samples were 61.4151, 61.2417 and 61.2384 seconds; optimized samples were 42.6645,
42.6445 and 42.3624 seconds. Maximum force difference, including repeat variation, was
`8.53e-14` Hartree/Bohr, about `4.39e-12` eV/Å. SCF took 71.4060 seconds and passed full
float64 verification without fallback; all small cases also passed without fallback.
This comparison isolates the force changes within the native installation. It does not
isolate a compiler speedup or establish new whole-ASE-workflow timing medians.

The complete native build and chosen-method force path are now verified. The earlier wheel
measurements remain separate evidence. No validation data was used. Build archives, compiler
configuration, the wheel, native/source hashes, raw forces, numerical checks and a runtime
launcher are retained outside Git. The tested wheel SHA256 is
`9a9e6c3e05f9cd5843c182622acc06c86ce98269c7fedca160572e95d6d5ae90`.

## Native profiling and streaming density products

On 2026-10-12–13, fresh profiles of the optimized native installation separated SCF,
the final float64 check and complete force evaluation. The same synthetic cases, recipe,
full basis, GPU 1 and four-CPU limits were retained. These synchronized timings are
inclusive; parent and child rows must not be added together. The module-level task wrapper
in these early profiles bypassed the private float32 grid dispatch; their SCF cost breakdown
does not represent production mixed SCF. The force and final float64-check timings below
remain useful. Speed comparisons use the separate unprofiled runs, which retained the
actual precision schedule. The later tau profile instruments the private namespace directly.

| Native force component | Pyrene, s | Coronene, s |
| --- | ---: | ---: |
| Complete force call | 42.41 | 95.97 |
| Semilocal XC response | 13.95 | 32.09 |
| VV10 response | 15.14 | 30.36 |
| Density-fitting J/K derivatives | 12.87 | 32.59 |
| Basis evaluation, included above | 0.46 | 0.81 |
| Density products, included above | 6.65 | 15.96 |
| Force density contractions, included above | 7.47 | 17.02 |

The final float64 SCF check took 15.41/33.49 seconds. Its native VV10 evaluation took
8.51/16.72 seconds, and ordinary XC integration took 5.44/12.54 seconds. During SCF,
float64 contractions accounted for substantial time. Basis construction and grid-weight
derivatives were small, so eliminating repeated density products was the useful force target.

Full-response RKS now evaluates density, XC and orbital derivatives in one higher-derivative
AO pass. Real, exactly symmetric LDA/meta-GGA densities share block-local float64 density/AO
products. A meta-GGA needs four such products instead of the previous nine across density
and force evaluation. Scalar grid weights are distributed over retained products; any
rounding change is checked against the previous complete gradient. GGA and other density
matrices retain separate products. Grid-weight responses still use large batches, and no
fields are cached across geometries or between SCF and forces.

Three alternating unprofiled complete-gradient calls per implementation used one accepted
density from the retained five-iteration precision runner. DF buffers were reconstructed
outside each timed gradient. Both implementations used the same native libraries:

| Molecule | Previous force path, s | Streaming products, s | Speedup | Less force time |
| --- | ---: | ---: | ---: | ---: |
| Pyrene | 42.3672 | 36.4876 | 1.161× | 13.9% |
| Coronene | 95.8000 | 82.0957 | 1.167× | 14.3% |

Pyrene previous samples were 42.4210, 42.3561 and 42.3672 seconds; streaming samples were
36.4876, 36.4795 and 36.4905 seconds. Coronene previous samples were 95.8000, 95.6860 and
95.8504 seconds; streaming samples were 81.9902, 82.0957 and 82.1135 seconds. Maximum force
differences, including repeat variation, were `3.27e-12` and `1.90e-13` Hartree/Bohr. Both
densities were exactly symmetric, all 682/996 orbitals were retained and the original SCF
verification passed without fallback. The shared SCF calls took 71.8722/201.8124 seconds.
Adding SCF to the force medians suggests about 5.1%/4.6% less combined time; these sums are
not fresh whole-ASE-workflow timing medians.

The force implementation is commit `3a9909f`, with gradient-source SHA256
`a1a13d69844d694b3ae4154ff24b8099fbca99f43a695892cc3a956fa8e95263`.
The initial candidate suite passed 114 checks, including four tests for an experimental
SCF policy that was subsequently discarded. A normal installation of the force-only wheel,
with the original precision policy, passed all 110 applicable native checks. Eleven separate
pinned-release force checks passed after retaining that release's original density/grid
loops in the compatibility adapter. Density tests matched native values bit for bit for
fixed AO inputs; contraction comparisons, LDA/GGA/meta-GGA complete forces and independent
energy finite differences also passed. Ruff and GitHub checks passed.

Only Python force code changed. Every native source file matched the earlier native build,
so packaging reused its twelve verified binaries; installed hashes and production Python
sources were checked again. The old installation and all source snapshots, profiles,
raw forces, wheels and logs were retained. Validation data was not used.

### Precision schedules that did not improve performance

Three isolated SCF trials retained full float64 acceptance/recovery and the original
tolerances, grids, basis and budget. The reference pyrene run took 71.5246 seconds and
ten iterations. Switching grid products at a convergence-based eighth iteration took
72.7064 seconds and twelve iterations. A bounded extension that switched at iteration six
took 74.4443 seconds and eleven iterations. Both passed verification without fallback,
but the extra iterations erased the cheaper products.

Keeping grid products coarse until final verification stalled at the twenty-iteration
limit. The native orbital gradient was `5.26e-4`, above the required `1e-4`, even though
the float32-to-float64 energy change was only `7.14e-11` Hartree. Verification rejected the
result, native float64 recovery converged in two iterations, and total time reached
137.1917 seconds. This demonstrates why the physical Fock check is required in addition
to energy agreement. These single-run trials were rejected, not promoted as improvements.
The production grid schedule remains float32 for at most five iterations.

## ASE property reuse

The `optimisation-v1` branch combines the verified mixed-precision path and grid
scratch/accumulation improvements above with reuse of accepted SCF state for
energy-then-force requests at unchanged atoms. That request pattern now makes one
SCF call, rather than two. No end-to-end timing ratio is claimed for this change.
The [ASE guide](ASE.md) describes invalidation and per-call timing.

On 2026-10-12, the consolidated branch passed 89 focused checks, including CUDA
contractions, neutral/open-shell water energy and force comparisons, warm geometry
changes, and a central finite-difference force check. The selected CI lint rules
also passed. Tests ran sequentially on GPU 1 with affinity `[4,5,6,7]` and four
numerical threads, using PySCF 2.14.0, ASE 3.27.0, CuPy 14.2.0 and the pinned
GPU4PySCF 1.8.1 CUDA 13 wheel. Only the new precision and ASE modules were loaded
from the branch; this remains installed-release compatibility testing rather than
verification of a full native checkout build. All inputs were synthetic, and
source hashes and logs are retained outside Git.

## Native density-fitting force scheduling

The 2026-10-13 native comparison retains the preceding optimized force implementation
(`147d304`) as its baseline. The density-fitting derivative kernel assigns up to 16 auxiliary
lanes to an orbital pair. Reducing shell-pair batch lengths from `4 * triplet_lanes` to
`triplet_lanes // 16` exposes more independent GPU blocks and shortens serial pair loops.
Integral screening, the original 54-element per-thread tile, float64 arithmetic, both
exchange operators and full response terms are unchanged.

Controlled pyrene J/K comparisons reconstructed the same full/range-separated DF buffers
before every call; available GPU memory was identical at 6,345,064,448 bytes. Three alternating
repetitions gave 13.3440 seconds for the original batching and 9.5430 seconds for smaller
batches. Narrowing the per-thread tile to nine added no benefit (9.5428 seconds), so the
original tile remains. Two further alternating repetitions with the selected pair-tile batches
gave **13.3418 → 5.2725 seconds**, 60.5% less J/K force time, with differences below
`3.5e-13` Hartree/Bohr. An initial uncontrolled-memory diagnostic is retained separately;
it showed approximately `1.02e-9` Hartree/Bohr variation that disappeared in the controlled comparisons.

Complete native force calls use one accepted density, freshly verified in float64. The
[native benchmark](native_force_benchmark.py) releases unused pool blocks and reconstructs
both DF buffers outside timing, then alternates the baseline and current complete gradient.
The ordinary/nonlocal grids, ωB97M-V/def2-TZVPD, def2-universal-jkfit, overlap cutoff,
convergence thresholds and full grid/auxiliary response are identical.

| Synthetic molecule | Baseline force seconds | Optimized force seconds | Reduction |
| --- | ---: | ---: | ---: |
| Pyrene, 26 atoms / 682 orbitals | 36.1476 | 28.1815 | 22.0% |
| Coronene, 36 atoms / 996 orbitals | 81.3481 | 63.1386 | 22.4% |
| Ovalene, 46 atoms / 1,310 orbitals | 141.2875 | 119.2368 | 15.6% |

These are medians of two alternating pairs, with individual baseline samples 36.0777/36.2175
and optimized samples 28.1373/28.2257 seconds for pyrene. Coronene's baseline samples were
81.4006/81.2955 seconds and optimized samples 63.1531/63.1242 seconds. Ovalene's baseline
samples were 141.4233/141.1518 seconds and optimized samples 119.2406/119.2330 seconds.
Maximum complete-force differences were `6.51e-13`, `8.74e-13` and `2.86e-12` Hartree/Bohr,
respectively.
SCF is unchanged; these force comparisons are not complete ASE
workflow medians. All measurements use physical GPU 1 (RTX 3080 10 GB), four CPU cores and
sequential runs. No validation-set inputs were used.

Ovalene adds a larger synthetic size check with 2,652 auxiliary functions. Its original fresh
verified SCF took 742.6351 seconds and 18 cycles without fallback; all 1,310 basis orbitals
were retained. Final energy was `-1227.995457370125` Hartree and the float64 orbital gradient
was `2.56e-5`. The synchronized baseline J/K profile took 58.0408 seconds: the derivative loop
accounted for 38.4682 seconds, including 7.5860 seconds forming its density tensors, while
the integral/density preparation loop took 14.9026 seconds. These inclusive component timings
are separate from the complete-call medians; they must not be added as independent costs.

The isolated ordinary wheel's DF gradient SHA256 is
`875b0d5d2aace272ee718260208af6139e6d4c2426344c8f1a84ea9b24e9334c`.
Every native-library source and all 12 compiled library hashes match the previous verified
native build. An independent installed-source check matched all **533 engine Python/data
files** to commit `ca62cd9`; the active local LibXC 7.0.0 library also matches the preserved
baseline's hash. The wheel passed **118 checks**, including full/long/short-range derivatives,
nonsymmetric densities, auxiliary-response control, Cartesian/spherical hybrid forces and
independent energy finite differences. Existing precision, ASE and VV10 checks are included.
The initial private environment's missing local LibXC installation and its failed import checks
are retained; the passing environment explicitly installs LibXC locally.

Raw profiles, trials, installed sources, wheel provenance, checks and complete force reports
are retained outside Git under `df-profile-next-06`, `df-schedule-next-07/08/10/11`,
`df-ovalene-next-09`, `native-package-df-next-12/13` and `native-df-forces-next-14`.
The exact committed source archive and `native-df-installed-next-15.json` retain the final
installed-source and dependency checks. GPU 1 was idle after all sequential calculations.

## Symmetric float64 tau products in native SCF

The 2026-10-13 implementation stacks the three AO derivatives into signed weighted Gram
products. Separate positive and negative weights allow float64 symmetric rank updates;
one matrix triangle is calculated and mirrored. Identical contiguous real fields with more
than 512 active orbitals use this path. Small, distinct, complex, noncontiguous or nonfinite
inputs retain the original bilinear products. Early mixed SCF retains its original three
float32 products and precision schedule. Full float64 verification/recovery and all native
force response terms remain required.

A fresh profile of the preceding native engine instruments its private NumInt namespace,
retaining 5,424 float32 and 5,424 float64 grid products. Pyrene's synchronized SCF/force calls
took 73.26/28.45 seconds; float64 tau products accounted for 20.89 seconds, including 3.49
seconds in verification. These are inclusive profile costs, not independent timing samples.
The earlier prototype's actual-grid potential differences were below `2.3e-15` Hartree.

### Memory and mixed-atom validation

The first implementation passed its small-molecule checks but exhausted memory in coronene's
subsequent VV10 gradient. Tau tasks now reserve one bounded workspace; standalone calls
release direct allocations and finite checks fuse their reduction without large masks.
The remaining gradient failure exposed a native allocator mismatch: grid-response batching
counted cached pool memory, while large allocations bypassed that pool. VV10 weight-response
batches now use actual free device memory, retaining the conservative half-memory budget
and rounding batch lengths down to their alignment. Grid points and response terms are
unchanged. A regression deliberately separates virtual pool capacity from device-free
memory, forcing multiple batches and comparing the complete VV10 response.

The permanent generic panel includes acetamide, pyridine, DMSO, acetyl fluoride and acetyl
chloride, covering C/H/N/O/S/F/Cl. Small-molecule correctness checks force the symmetric path,
so its size guard cannot hide missing coverage. The larger fixed C11H13NO3 organic geometry
also exercises the production guard. Initial force comparisons at orbital tolerance `1e-5`
found `1–2e-7` Hartree/Bohr differences in pyridine/DMSO; tighter correctness-only convergence
at `1e-7` separated SCF stopping variation from the matrix calculation. Production thresholds
remain `1e-8` Hartree and `1e-4` orbital gradient. Validation datasets were held out.

### Fresh native ASE comparisons

Each timing covers a fresh verified mixed SCF followed by native float64 forces at unchanged
geometry through ASE. The force request reuses the accepted SCF state. The baseline retains
all preceding optimizations and disables only symmetric tau products; both modes use the
corrected device-free-memory gradient budget. These are additional gains over the previous
optimized path, not comparisons with pristine upstream.

| Generic molecule | Baseline energy + forces, s | Symmetric tau, s | Reduction |
| --- | ---: | ---: | ---: |
| Pyrene, 682 orbitals | 100.5015 | 98.3582 | 2.13% |
| Coronene, 996 orbitals | 265.3642 | 246.4477 | 7.13% |
| C11H13NO3 organic, 681 orbitals | 106.8444 | 106.6395 | 0.19% |

Pyrene uses the medians of two alternating fresh pairs. Coronene and the larger organic use
one fresh pair each. Pyrene SCF medians were 71.9902 → 69.9641 seconds; coronene SCF took
201.7543 → 182.9636 seconds, a 9.31% reduction. Force times were approximately unchanged.
The organic's 0.19% difference is within timing variation; no material speedup is established.
Its spatially sparse blocks triggered only 112 symmetric calls, versus 498 per pyrene run
and 1,470 for coronene. The benefit depends on active orbital blocks, not total atom count.

All runs retained every basis orbital, used the same recipe, converged in 10/11/12 cycles
for pyrene/coronene/organic, and passed final verification without recovery. Maximum energy
differences across fresh runs were `2.39e-12`, `7.96e-13` and zero Hartree. Maximum complete-force
differences were `1.42e-7`, `2.98e-7` and `2.17e-8` Hartree/Bohr. These independent-SCF force
differences include stopping/trajectory variation; they are separate from shared-density
matrix comparisons and tighter-converged correctness checks.

The final installation passed **150 checks**, covering native fields, SCF verification and
recovery, ASE reuse, signed products, streams, failed BLAS restoration, bounded workspaces,
actual-memory gradient batching, mixed-atom forces and energy finite differences. The
installation audit matched all **534 engine Python/data files** to commit `ef55661`, all
12 native-library hashes to the earlier complete build, and the local LibXC 7.0.0 hash.
The final wheel SHA256 is
`45aeaa47b6cd3481a970b2bbcdf66fe16bf639d154defd8136ce9ff9da5420f4`.

All numerical work ran sequentially on physical GPU 1 (RTX 3080 10 GB), with affinity
`[4,5,6,7]` and four OpenMP/BLAS threads. No compiler or native-library sources changed.
Raw reports, accepted states, failed-allocation diagnostics and corrected installation
provenance remain outside Git under the 20261013 performance artifacts: `native-package-tau-18`,
`native-tau-memory-checks-19`, `native-tau-ase-20`, `native-tau-checks-21`,
`native-tau-installed-22.json` and `native-tau-comparison-23.json`. Earlier attempts remain
retained. The exact source archive and engine manifest establish the tested implementation.

## Current mixed-atom organic bottlenecks

Fresh paired plain/profile ASE calls on 2026-10-13 use the final native tau installation
above, the production recipe and fixed public development geometries. Each case runs in
a separate process; energy is followed by unchanged-geometry float64 forces. All numerical
work is sequential under the shared lock on GPU 1 with four CPUs. No engine code or
scientific settings changed for these measurements.

| Generic molecule | Orbitals | Plain SCF, s | Plain forces, s | Plain energy + forces, s |
| --- | ---: | ---: | ---: | ---: |
| Acetamide | 196 | 6.8404 | 3.3213 | 10.1623 |
| Pyridine | 267 | 10.6839 | 4.8247 | 15.5092 |
| DMSO | 214 | 7.7445 | 4.1262 | 11.8713 |
| Acetyl fluoride | 181 | 5.4866 | 2.6049 | 8.0919 |
| Acetyl chloride | 187 | 5.9873 | 2.9424 | 8.9303 |
| C11H13NO3 organic | 681 | 76.2537 | 28.1144 | 104.3687 |

These are single fresh timing samples, not speedup comparisons. Component profiles add
synchronization and Python profiling: total overhead is 5.74% for the larger organic and
9.81–13.22% for the smaller cases. The profiler instruments the private NumInt namespace
after precision dispatch is installed, retaining the native mixed calculation. Across the
panel, ordinary SCF grid integration accounts for 32.4–40.0% of profiled call time; SCF
VV10 integration plus its complete force response accounts for 31.6–38.7%.

The larger organic's 110.3632-second profiled call has these disjoint parent costs. SCF
rows include final verification; its cost must not be added again.

| Component | Seconds | Share of profiled call |
| --- | ---: | ---: |
| Ordinary SCF grid integration | 44.1358 | 40.0% |
| SCF VV10 integration | 23.2848 | 21.1% |
| Full VV10 force response | 16.5672 | 15.0% |
| SCF Coulomb/exchange, including DF setup | 13.2288 | 12.0% |
| Ordinary full-response XC forces | 7.2106 | 6.5% |
| DF Coulomb/exchange forces | 4.4688 | 4.0% |

Within ordinary integration, float64 tau products take **23.1789 seconds**. Only 1.7973
seconds use symmetric products. The 384–512 active-orbital block range accounts for 22.3787
seconds of the float64 ordinary matrix products, including both tau and other potential
terms; these blocks fall below the current tau guard. Typical costly products have
438–505 active orbitals and 4,096 grid points. This makes medium-sized float64 products
the first target for actual-field kernel/dispatch comparisons. Lowering the guard alone
is not established as beneficial and requires measurement, since its small-block fallback
was selected for performance.

Within VV10 forces, the pair traversal takes **14.3972 seconds** over 304,066 active points,
or 86.9% of the complete VV10 response. The final float64 verification takes **15.6689
seconds**, including 10.3969 seconds of VV10 integration; the native VV10 field evaluation
itself takes 9.7483 seconds. Exact pair-kernel improvements would therefore help both
verification and forces. Grid-weight response across ordinary and VV10 forces is only
**0.4202 seconds** (0.38% of the call), making it a lower-priority target.

All 12 calls pass verification, retain every basis orbital and reuse the accepted SCF
state for forces. No case needs recovery at production tolerances; DMSO's earlier recovery
under stricter correctness tolerances is not a production bottleneck in this panel. Plain
and profiled runs have identical cycle counts (9–12), energy differences below `6.83e-13`
Hartree and force differences below `1.96e-8` Hartree/Bohr. All 12 native-library hashes
match the verified installation, and every run uses the same profiler source hash.

Raw timing reports, component timings, product shapes, accepted states and Python profiles
remain outside Git under `20261013/organic-profile-25` with profiler
`organic-profile-harness-24.py`. The ordinary-component precision labels describe the
grid controller; `precision_info.vv10_precisions` records the independent VV10 schedule.
The batch completed successfully and released GPU 1 and the shared lock.

## Direct float64 tau products for medium organic blocks

The current symmetric BLAS path is not faster below its size guard. On 15 retained,
unmodified AO/weight blocks from the verified C11H13NO3 density, forcing that path was
slower than the original three float64 GEMMs at every sampled shape. For 438 orbitals
and 4,096 points, it took 15.806 versus 11.624 milliseconds; for 505 orbitals, 16.084
versus 15.439 milliseconds. The existing 512-orbital guard remains unchanged.

`_tau_triangle.py` instead contracts signed weights directly with float64 fused multiply
adds, computes one matrix triangle and mirrors it. Five tile/thread layouts were compared
on the same actual fields. The selected 16 × 16 tile needs no stacked AO allocation and
uses padded shared memory. It is 1.65–1.98× faster on the sampled 400–506-orbital blocks.
Maximum actual-field matrix error is `2.78e-16` Hartree. Twelve additional actual-field
prefix checks at 2,048, 3,073 and 4,096 points find 1.63–2.12× speedups, including the
partial final tile, with the same maximum error.

Native dispatch enables this path only on compute capability 8.6 for 384–512 active
orbitals and 2,048–4,096 grid points. Identical, contiguous, finite real float64 fields
are required, and aliased output storage retains the original product. Other sizes,
architectures and unsupported layouts keep their existing dispatch. Early mixed SCF
retains its original float32 products; large blocks retain symmetric BLAS. No precision
schedule, grid, pruning, convergence setting, response term or geometry changed.

Three alternating same-density complete ordinary-grid pairs took a median **4.2861 →
3.3136 seconds**, a 22.69% reduction. The candidate handled 153 of 232 blocks. Electron
count and XC energy agreed, and maximum complete potential-matrix differences were
`2.67e-15` Hartree. These fixed-density integration timings are separate from full SCF.

### Fresh ASE call comparisons

The baseline disables only the direct medium-block kernel; both modes retain the existing
large-block symmetric products and every previous optimization. Each independent process
runs fresh verified mixed SCF, then unchanged-geometry native float64 forces through ASE.
Two alternating pairs cover the larger organic; pyrene and coronene have one matched
pair each. Five smaller mixed-atom organics also have one pair each at the production guard.

| Generic molecule | Baseline energy + forces, s | Direct medium tau, s | Reduction |
| --- | ---: | ---: | ---: |
| C11H13NO3 organic, 681 orbitals | 105.8807 | 98.5395 | 6.93% |
| Pyrene, 682 orbitals | 98.5302 | 94.1228 | 4.47% |
| Coronene, 996 orbitals | 249.5507 | 243.7387 | 2.33% |

The organic's SCF median is **77.5002 → 70.0468 seconds**, a 9.62% reduction. Complete
forces remain approximately unchanged at 28.38/28.49 seconds. Both runs retain 112
large-block symmetric calls; the new path handles 1,224 medium blocks. All four organic
calls converge in 12 cycles, need no recovery and reuse the accepted SCF state for forces.
Maximum energy spread is `4.55e-13` Hartree; maximum force difference is `1.80e-8`
Hartree/Bohr. These are additional gains over the preceding optimized native installation,
not comparisons against pristine upstream or a change in SCF precision.

Pyrene SCF takes 70.0722 → 65.7007 seconds (6.24% reduction); coronene takes 185.3709 →
179.5620 seconds (3.13%). The new kernel handles 744/777 blocks respectively, with identical
10/11 cycle counts and no recovery. Complete force times remain approximately unchanged.
Maximum energy differences are `6.26e-12`/`9.10e-13` Hartree and maximum force differences
are `8.88e-8`/`3.15e-7` Hartree/Bohr. Independent SCF force differences include stopping and
trajectory variation; the same-density matrix comparisons and stricter derivative checks
above are separate correctness evidence.

| Smaller generic organic | Baseline energy + forces, s | Native dispatch, s | Triangle calls |
| --- | ---: | ---: | ---: |
| Acetamide | 10.2200 | 9.9836 | 0 |
| Pyridine | 15.4943 | 15.5371 | 0 |
| DMSO | 11.6736 | 11.6927 | 0 |
| Acetyl fluoride | 8.1068 | 8.1096 | 0 |
| Acetyl chloride | 8.7871 | 8.8295 | 0 |

The small molecules have no eligible production blocks; their timing variation is not
evidence of a kernel gain. Their correctness tests explicitly lower the guard and verify
accelerated forces separately. All 18 complete ASE calls pass verification, retain every
orbital, need no recovery and reuse SCF for forces. The five small-organic pairs preserve
their 9–10 cycles and differ by at most `3.42e-13` Hartree and `2.38e-8` Hartree/Bohr.

### Native installation and correctness

The normal isolated wheel installs engine commit `d2c5d0f`. Its independent audit matches
all **535 engine Python/data files**, the 12 unchanged compiled native-library hashes and
the active local LibXC 7.0.0 hash. The wheel SHA256 is
`9da8d3dca9d30c6251af3771c996d0f11fcc7ed825569a894b0df649b11f6bff`.
The new CUDA kernel compiles through CuPy NVRTC on the selected GPU; native library
sources did not change and match the retained complete sm86 build.

The installation passes **185 checks**, including all 35 new signed-product, partial-tile,
range/architecture/alias guard, stream, allocation, early-float32 and mixed-atom/open-shell
checks. Complete-force comparisons cover N/O/S/F/Cl chemistry, with independent energy
finite differences and stricter correctness-only SCF tolerances. Existing symmetric,
ASE verification/recovery/reuse, full grid/DF response and VV10 checks also pass. The first
suite's three reference-gradient tests failed because the harness pointed to a nonexistent
baseline path; correcting it to the retained matching `native-build-01/fork` source and
rerunning those three checks resolved the setup failure. Both attempts remain retained.

All numerical measurements here run sequentially on physical GPU 1 (RTX 3080 10 GB),
with affinity `[4,5,6,7]` and four OpenMP/BLAS threads. Validation datasets are held out.
Actual fields, prototype layouts, complete-grid comparisons, wheel/source provenance,
initial/repaired checks and fresh ASE reports remain outside Git under the 20261013
performance artifacts: `tau-actual-fields-27`, `tau-medium-triangle-28`,
`tau-medium-grid-29`, `native-package-triangle-30`, `native-triangle-installed-31.json`,
`native-triangle-validation-32`, `tau-triangle-tail-33` and `native-triangle-ase-34`.
The final batch completes successfully and releases GPU 1 and the shared benchmark lock.


## B3LYP native force products and separate projection scheduling

These measurements use B3LYP with VWN-RPA correlation, Cartesian `6-31g*`,
`def2-universal-jkfit` density fitting, grid level 3, `1e-8` Hartree energy and
`1e-5` orbital-gradient tolerances, with full grid and auxiliary-basis response.
All runs use physical GPU 1 (RTX 3080), four CPUs with affinity 4–7, matching native
libraries and both shared locks. Geometries stay fixed and validation datasets remain
held out; the tuning cases are synthetic/development molecules.

Native float64 auxiliary-response products in `6e3c956` use sparse scaled-permutation
basis transforms and symmetric BLAS Gram updates without packing occupied-pair tensors.
Relative to `1b45d1f`, complete forces improve by 6.3–8.7% across the three sizes;
whole-call reductions are 1.1–3.0%. At identical saved densities, the large-system DF
contributions agree with the retained original within `1.35e-13` Hartree/Bohr,
comparable to native repeat noise. Both spin channels retain native float64.

The following additional comparison uses projection-schedule commit `658ccf8` against
`6e3c956`; both already include those force optimizations and INT8 SCF Gram products.
Grid and exchange projections now have independent float32 searches. Projections
switch after at least five iterations when the orbital gradient is below `1e-4`, or
after sixteen iterations. Their scratch buffers survive intervening INT8 Gram blocks.
Native float64 verification, recovery and derivatives are unchanged.

| Molecule | Previous energy + forces, s | Separate projection schedule, s | Reduction |
| --- | ---: | ---: | ---: |
| Circumcoronene, 846 orbitals | 116.49 | 108.96 | 6.47% |
| Mixed organic, 753 orbitals | 70.22 | 66.47 | 5.34% |
| Mixed organic, 502 orbitals | 22.59 | 21.77 | 3.61% |

SCF time improves by 8.58%, 8.10% and 5.23%, respectively. All six calls pass the original
float64 energy/Fock check without recovery, retain all orbitals, and preserve their
14/15/14 iteration counts. Maximum final-energy differences are `5.46e-12` Hartree.
These are single fresh pairs, not timing medians or comparisons with pristine upstream.
Independent accepted states differ in forces by at most `1.22e-4` eV/Angstrom at these
production thresholds; this is separate from identical-state derivative validation.
Recipe, geometry, grid ordering, native-library hashes, source boundaries and resources
match across each pair.

The schedule passes 45 CUDA checks, including mixed N/O/S/F/Cl chemistry, open-shell
SCF, identical-state native force replay, warm ASE geometry changes, independent energy
finite differences, stream buffers and cleanup. A subsequent report-only fix accounts for
SCF products evaluated after the last callback, so native recovery cycles retain accurate
precision labels. The final state passes 122 CPU checks, repeats 64 state-machine checks
on the host and passes all 24 exchange CUDA checks again. That bookkeeping fix changes no numerical
products; the timing table retains the exact `658ccf8` source provenance.

Raw states, paired reports and source/native guards remain outside Git under
`b3lyp-force-response-native-20261016-v3`, `b3lyp-force-fixed-density-20261016-v1`,
`b3lyp-projection-native-20261016-v2` and `b3lyp-projection-final-20261016-v1`.


## Native exact-zero float64 DF whitening

Implementation `f04c720` applies the existing auxiliary inverse in bounded output-column
blocks, omitting exactly zero coefficient rows while retaining the remaining rows' original
order. This is a native float64 setup optimization; the inverse values, SCF precision schedules,
verification/recovery thresholds, derivatives and scientific recipe are unchanged. Automatic
dispatch currently targets single-device compute capability 8.6. Unsupported, dense, eigenvalue
and nonfinite cases retain the original product. The helper owns its buffers for one build and
bounds compact coefficients plus gather scratch to 512 MiB.

The baseline is `cf49707` with all preceding optimizations. Fresh processes alternate the original
and new whitening paths on physical GPU 1 (RTX 3080 10 GB), with four CPUs, affinity 4–7 and both
shared locks. Each timed call includes fresh density-fitting setup, verified mixed SCF and complete
native float64 forces, with grid and auxiliary response. The recipe is the B3LYP/VWN-RPA,
Cartesian `6-31g*`, `def2-universal-jkfit`, `1e-8` Hartree / `1e-5` gradient recipe above.

| Development molecule / grid | Original setup, s | New setup, s | Original energy + forces, s | New energy + forces, s | Whole-call reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circumcoronene, 846 orbitals, TC-style grid | 32.14 | 22.00 | 98.79 | 88.77 | 10.14% |
| N/O organic trimer, 753 orbitals, TC-style grid | 11.99 | 7.68 | 59.60 | 55.36 | 7.11% |
| Circumcoronene, 846 orbitals, native level 3 | 32.14 | 21.88 | 109.37 | 99.03 | 9.45% |

The first two rows are medians of three alternating pairs. The last row is one fresh pair.
The TC-style construction follows the retained TeraChem comparison and has 438,780 physical
points for circumcoronene; ordinary native level 3 has 938,916. This experiment changes neither
grid within a pair, and the production grid default remains unchanged. Setup reductions are
31.55%, 35.95% and 31.91%, respectively. Circumcoronene retains 14 SCF iterations and the organic
15; forces are approximately unchanged in elapsed time. These are additional gains over the
previous optimized engine, not comparisons with pristine upstream.

Deferred CUDA-event profiling preserves computation/copy overlap. The large-system whitening
operation, including the new gather and finite checks, takes **14.97 seconds**, versus **24.72
seconds** for the original dense product: a **39.45% reduction**. Block preparation takes 0.006
seconds and the retained coefficient fraction is 54.92%. Measured coefficient/scratch storage is
273.38 MiB. The 8.53 GB host tensor still takes approximately 3.36 seconds to allocate; actual
host copies take 0.78 seconds, largely overlapped with computation. Nested wait times are not
additional transfer costs. Profiled runs are separate from the timing medians above.

All 14 calls pass unchanged native float64 verification without recovery. Each first optimized
case/grid also rebuilds the original dense integrals outside timing and independently checks
the accepted state and its complete forces. Maximum final-energy differences are `3.02e-10`
Hartree; the largest same-density original-force difference is `8.18e-12` eV/Angstrom.
Independent accepted SCF states retain ordinary stopping/trajectory variation.

The implementation passes 106 local CPU checks and a 40-check host suite comprising 12 plan
checks and 28 opt-in CUDA checks. Coverage includes N/O/S/F/Cl and open-shell chemistry,
host/GPU storage, same-density original forces, energy finite differences, zero/partial blocks,
nonfinite/alias guards and BLAS stream/state restoration. Individual inverse-derived factors
can differ through float64 cancellation; physical fitted integrals, independent native energies
and forces determine acceptance. The final independent-energy correctness bound is `1e-9`
Hartree, stricter than production verification; tight solver convergence is retained for
derivative comparisons. Initial strict-factor/reconstruction failures, native-repeat controls
and unsuccessful alignment refinements remain retained alongside the final passing evidence.

All 15 staged product-source hashes match `f04c720`, and each complete call checks the base
535-file source/data manifest and 12 matching native-library hashes. Validation datasets stay
held out. Raw timings, numerical states, failed/final attempts and provenance remain outside
Git under `b3lyp-whitening-native-20261016-v2`; the original/final direct profiles are retained
under `b3lyp-setup-profile-20261016-v1` and `b3lyp-whitening-profile-20261016-v1`.


## Sparse density-fitting products

Implementation `eb3cfc2` projects the existing packed, screened AO pairs directly into
occupied-orbital products instead of unpacking a dense AO tensor. Row groups retain the
original order of their required reduction columns. The force back projection writes only
the directed pairs consumed by the native derivative kernels, including the Coulomb term.
Plans retain index metadata and prepare fresh coefficient tiles for each call. Unsupported or uneconomic
cases retain the original products; the dispatch and workspace guards live in
[`_sparse_products.py`](../../gpu4pyscf/df/_sparse_products.py).

Raw force forward projections remain native float64. Sparse variants introduced last-bit
differences that the inverse auxiliary metric amplified, including on long auxiliary
panels. On circumcoronene, the rejected variant differed from original forces by
`4.70e-7` eV/Angstrom even with matching batch budgets. Retaining the native forward
reduction and accelerating the back projection restores agreement near native repeat
noise. The earlier failed attempts and diagnostics are retained outside Git.

Implementation `7ae52c4` also enables the existing verified float32 projection and INT8
Gram schedule for density-fitted range-separated hybrid GGAs/MGGAs, including VV10
functionals. Only global-hybrid GGAs use adaptive grids; the other methods retain their
first-five grid schedule. Projection switching remains bounded to five through sixteen
iterations, with the existing `1e-4` orbital-gradient switch. Verification, recovery and
derivatives use float64, and acceptance thresholds are unchanged.

The reference is the previously optimized `f04c720` engine. Both functionals use
Cartesian `6-31g*`, `def2-universal-jkfit`,
ordinary grid level 3, nonlocal grid level 3 where applicable, `small_rho_cutoff=0`,
`1e-8` Hartree energy and `1e-5` orbital-gradient tolerances, and full grid/auxiliary
response. B3LYP uses the explicit VWN-RPA definition above. Fixed public development
geometries start from MINAO and retain every orbital. Validation datasets remain held out.
All calculations run sequentially on physical GPU 1 (RTX 3080), with four CPUs, affinity
4–7 and both shared locks.

The table reports complete fresh energy-plus-force calls, including fitting setup and
mandatory float64 verification. Imports and additional independent reference replays are
outside those timers. Each row uses one fresh reference process and one fresh optimized
process. Timing variation has not been averaged into medians.

| Development molecule | Functional | Orbitals | Previous engine, s | `7ae52c4`, s | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| Circumcoronene | B3LYP/VWN-RPA | 846 | 100.45 | 92.52 | 7.90% |
| N/O organic trimer | B3LYP/VWN-RPA | 753 | 62.25 | 54.79 | 11.97% |
| Circumcoronene | ωB97M-V | 846 | 500.90 | 409.43 | 18.26% |
| N/O organic trimer | ωB97M-V | 753 | 455.24 | 377.15 | 17.15% |

SCF iteration counts remain 14/15 for the two B3LYP cases and 15/14 for ωB97M-V.
The ωB97M-V energy/SCF stage improves by 26.13% and 26.05%; force times change only
slightly. Sparse products alone, before the range-separated precision extension, reduced
the complete ωB97M-V calls by 4.69% and 5.94%. The final combined improvement exceeds
the 10% whole-call target in both ωB97M-V cases and the large mixed-organic B3LYP case;
B3LYP circumcoronene remains below that target.

Additional B3LYP checks use complete public organic units from the same frozen geometry:

| Development molecule | Orbitals | Previous engine, s | `7ae52c4`, s | Reduction |
| --- | ---: | ---: | ---: | ---: |
| N/O organic monomer | 251 | 6.240 | 6.289 | -0.79% |
| N/O organic dimer | 502 | 20.787 | 20.577 | 1.01% |

The monomer uses the native path below the 256-AO sparse guard and has no sparse product
calls. The dimer exercises sparse float32/float64 SCF projections and the force back
projection, with only a small whole-call benefit. Both retain 14 SCF iterations. These
measurements show a size-dependent benefit in this panel.

All 16 final complete calls pass unchanged float64 verification without recovery. Across
the six final comparisons, independent accepted-state energies differ by at most
`1.99e-10` Hartree. Additional checks disable sparse SCF products and replay the retained
original gradient at the same accepted orbitals, with matching force batch budgets.
Their largest energy change is `9.69e-10` Hartree, below the stricter `1e-9` reference
bound, and their largest force difference is `8.68e-12` eV/Angstrom. Separate accepted
SCF states differ in forces by up to `1.78e-4` eV/Angstrom at the production stopping
thresholds; that is distinct from the same-state derivative check.

The final implementation passes 161 host checks, comprising 85 CPU checks and 76 opt-in
CUDA checks. Coverage includes B3LYP, CAM-B3LYP, ωB97M-V and `HYB_GGA_XC_LC_WPBE`,
N/O/S/F/Cl chemistry, open shells, original-density comparisons, native force replay,
energy finite differences, pitched/partial/batched products, stream/error restoration,
nonfinite/alias guards, index mutation and ASE verification/reuse. A preceding 32-check
native exchange/derivative regression suite also passes. Every complete call audits the
535-file native source/data manifest, 16 Python additions/overrides for the candidate,
12 matching native-library hashes, geometry, grids, import boundary and CPU/GPU resources.
All 16 candidate runtime source hashes match `7ae52c4`.

Raw inputs, states, timings, source proofs and failed/final attempts remain outside Git
under `sparse-df-native-20261017-v1` through `v9`, `sparse-df-call-bench-20261017-v1`
through `v4`, `sparse-df-regression-20261017-v1`, `sparse-df-roundoff-20261017-v1`
through `v3`, `sparse-df-large-force-20261017-v1`/`v2` and
`sparse-df-size-check-20261017-v1`. The installed-source audits, all paired assessments
and complete-call native reference checks pass.


## Short SCF paths with supplied guesses

For the approximately five-round workload, the verified nonlocal-only policy keeps
ordinary integration and exchange float64 throughout SCF, while VV10 pair products
use float32 partial sums accumulated in float64. Native float64 acceptance, recovery
and complete forces are unchanged. `auto` selects this policy for a nonlocal functional
with an explicit initial density or a scanner's retained orbitals. Fresh starts keep
the broader mixed policy; explicit `mixed` and `vv10` override selection. Guess quality
is assumed from the workflow, rather than estimated by the selector.

The development case is the fixed public ethyl-4-acetamidobenzoate geometry: 28 atoms,
681 spherical AOs, ωB97M-V/def2-tzvpd, def2-universal-jkfit, ordinary/nonlocal grid
levels 5/3, energy tolerance `1e-8` Hartree, orbital-gradient tolerance `1e-4`, overlap
cutoff `1e-10` and both full response terms. Every call receives the same retained
converged density blended with 0.1% MINAO. This controlled good-guess proxy measures
a short SCF path; it does not establish learned-guess accuracy. Validation data remain
held out. All runs use Bernd physical GPU 1, four CPUs, affinity 4–7 and both locks,
with sequential fresh processes and matching native libraries.

Times include fresh fitting setup, converged SCF, mandatory native verification for
approximate modes and complete forces. Imports, supplied-guess construction and an
additional independent diagnostic check are outside those timers. The float64 control
uses the same optimized native fork, based on `1d3eacd`, rather than an upstream wheel.

| Policy | Complete calls | SCF rounds | Energy stage, s | Forces, s | Total, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native float64, paired median | 2 | 4 | 83.23 | 29.72 | 112.95 |
| VV10-only, paired median | 2 | 4 | 49.78 | 29.09 | 78.87 |
| Auto selecting VV10-only | 1 | 4 | 49.91 | 29.02 | 78.93 |
| Previous broader mixed schedule, single control | 1 | 19 | 102.05 | 28.91 | 130.96 |

Two alternating native/VV10-only pairs show a **30.17% complete-call reduction**.
The automatic call is 39.73% faster than the single broader-mixed control, whose
approximate grids/exchange disturbed the already good guess. Force code is identical;
small force-time differences reflect run variation. These shared-host measurements
cover this recipe and controlled guess, rather than arbitrary sizes or guess quality.
All final candidate calls pass native verification without recovery. Their largest
energy difference from a paired native control is `2.28e-13` Hartree; the largest
complete-gradient difference is `2.28e-8` Hartree/Bohr.

The final automatic policy passes 87 CPU lifecycle/selection checks and 11 opt-in GPU
checks covering N/O/S/F/Cl organics, open shells, supplied densities, warm ASE geometry
changes, native full forces, energy finite differences and dispatch restoration. Core
Pyflakes/Pycodestyle checks pass. The broader Ruff default reports inherited ASE style
findings; the failed lint attempts are retained separately. Each complete benchmark
audits the 535-file engine manifest, Python overrides and 12 native-library hashes.
The timed SCF module matches `293213d`; a subsequent ASE forwarding fix preserves
explicit policies on warm scanner guesses, with additional regression checks.
The selected VV10 call forwards the same arguments as in the timings. No native
code changed.

A separate native profile at this same four-round input attributes 49.29 seconds to
SCF VV10 pairs, 21.28 to ordinary integration and 8.59 to fitting/J/K including 4.24
of fitting setup. Diagonalization takes 0.12 seconds. Full forces include 14.63 seconds
of VV10 pairs, 7.26 of ordinary grid response and 4.50 of fitted J/K response. These
synchronized inclusive component times are diagnostic, rather than speedup evidence.

Retained scripts, inputs, states, checks, source proofs and assessments are under
`/scratch/failip/gpu4pyscf-benchmarks/` in `provided-guess-20261017-v1`,
`good-guess-vv10-20261017-v1`, `good-guess-profile-20261017-v1` and
`good-guess-vv10-native-20261017-v1` through `v5`. The timing `v4/assessment.json`
records paired errors and timing scope. The exact-converged-guess tests were omitted
after selecting the approximately five-round target; actual accepted runs took four.
The final ASE explicit-policy regression results are retained in `v5`.


## Native verification reuse and occupied-factor forces

The follow-up in `cfb3c7c` starts from the verified VV10-only policy at `4707ff1`.
Its reductions are additional to the earlier 30.17% switch from native float64 SCF.
It reuses the last native ordinary integration during verification only when all
supported inputs match exactly, and uses validated occupied-orbital factors for
bounded float64 density products in full RKS ordinary/nonlocal grid response.
There is no rank truncation, new reduced precision or change to SCF tolerances.
Native exchange and VV10 are still evaluated during verification. The guards and
[reproduction guide](RUNNING.md#exact-force-contractions) describe eligible inputs
and native fallbacks; no extra calculator option is required.

Measurements use the same public development organic, fixed method and controlled
0.1% MINAO blend described above, plus the retained public coronene geometry
(36 atoms, 996 spherical AOs). Both use full grid and auxiliary response at the same
ωB97M-V/def2-tzvpd recipe, grid levels 5/3 and convergence thresholds. Timing includes
fitting setup, SCF, required native verification and complete forces. Imports, guess
construction and the additional independent diagnostic check remain outside the timers.
Runs use Bernd GPU 1, four CPUs with affinity 4–7, matching native libraries and both
locks. All GPU tests and calculations run serially; final timings follow validation.

| Case | Pairs | SCF rounds | Previous total, s | Current total, s | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed organic, 681 AOs | 5 | 4 | 78.81 | 70.75 | 10.23% |
| Coronene, 996 AOs | 2 | 3 | 189.85 | 168.08 | 11.47% |

Values are medians of individual complete calls. The organic panel retains both
initial alternating pairs, including the slower 74.90-second candidate, and three
predefined fresh confirmation pairs. Those fresh pairs independently reduce the
median from 78.79 to 70.75 seconds (10.20%). Their force counters record
two eligible plans and 1,269 products in every candidate. The coronene pair order is
reversed for the second repetition. These measurements establish this workload's
improvement rather than universal speedups across hardware, sizes or initial guesses.

Every complete call passes native energy/orbital-gradient verification without recovery.
Across independently converged calls, maximum energy spreads are
`1.14e-12` Hartree for the organic and
`9.09e-13` for coronene. At identical
accepted orbitals, separately rebuilt full forces agree with the original density
products within `4.09e-14` and `2.42e-13` Hartree/Bohr, respectively. Both accelerated
plans are exercised in each replay; their force times are 28.93 → 24.46 seconds and
63.48 → 49.70 seconds. This same-state check separates contraction error from the
force differences between separate SCF stopping points, which reach `2.20e-8` and
`4.49e-7` Hartree/Bohr in this panel.

The implementation passes 137 focused checks: 88 CPU lifecycle/provenance checks and
49 opt-in CUDA checks, including N/O/S/F/Cl organics, B3LYP and VV10, open shells,
LDA/GGA/MGGA contractions, stale tags, capacity/stream fallbacks, full forces, an
energy finite difference, exact cache invalidation and private dispatch restoration.
Focused core Ruff checks pass; inherited unrelated style findings remain outside this
scope. Every candidate audits the 535-file engine manifest, 18 Python additions/overrides
and 12 native-library hashes. The runtime Python sources match `cfb3c7c`; the earlier
and final timing sources have identical executable ASTs, with only an SCF docstring
addition. No native code or compiler flags changed in this follow-up.
A final baseline audit also confirms its 535-file manifest, 16 Python overrides and
12 native libraries, with the overridden Python sources matching `4707ff1`.

Retained raw results, states, source proofs, scripts, checks and setup/failed attempts
remain outside Git under `/scratch/failip/gpu4pyscf-benchmarks/` in
`grid-reuse-force-native-20261018-v1` through `v3`,
`grid-reuse-force-large-20261018-v1`, `grid-reuse-force-replay-20261018-v1`,
`grid-reuse-force-confirm-20261018-v1` and `grid-reuse-host-check-20261018-v1`.
Validation data remain held out. The local `assessment.json` retains every sample,
recipe/source checks, full gradients and the independently assessed force replays.

## AO tile occupancy diagnostic

The follow-up on `bbef205` measures whether whole-tile screening is promising in
the AO blocks that remain after native whole-block AO selection. It changes no
engine products, grids or screening. The CPU-only
[`ao_tile_sparsity.py`](ao_tile_sparsity.py) counts tile maxima at several amplitude
cutoffs, jointly across derivative components. Partial tiles contribute only their
valid sizes to the estimated Gram work; output percentages aggregate counts.
These estimates describe structurally skippable products under a hypothetical
cutoff, not measured speedups or numerical-error bounds.

The first sample consists of three retained ordinary-integration blocks from the
public 681-AO organic above, with 316, 438 and 526 active AOs and 4,096 grid points
each. It preserves their original AO/grid order and includes values and first
derivatives. At a hypothetical `1e-12` cutoff:

| AO × grid tile | Negligible tiles | Estimated Gram work skippable |
| --- | ---: | ---: |
| 16 × 16 | 4.62% | 7.39% |
| 16 × 32 | 3.58% | 5.45% |
| 32 × 32 | 2.63% | 3.01% |
| 64 × 64 | 3.79% | 2.04% |

Although 41.70% of individual joint-component entries are below that cutoff,
few complete tiles are negligible. Exact-zero tiles offer no savings. At `1e-10`,
32 × 32 tiles still expose only 3.83% estimated work savings. Using the native
AO-selection threshold as a tile cutoff does not establish equivalent accuracy:
they are different screening decisions.

A second diagnostic uses the matching native installation on user-authorized
GPU 6 and CPU affinity 24–27, without a competing calculation on that GPU. It
samples evenly spaced block IDs separately in native SCF and full-force grid
orders: eight per order for each of acetamide, pyridine, DMSO, acetyl fluoride and
acetyl chloride, plus 24 per order for the 681-AO organic. All 128 blocks include
values, first and second derivatives. Settings remain ωB97M-V/def2-tzvpd and
ordinary grid level 5. Padded AO rows are removed; zero-weight grid points remain
in their original positions and are not used to increase AO sparsity.

For the 681-AO organic, estimated skippable Gram work at `1e-12` is:

| Tile | Sampled SCF grids, values + first derivatives | Sampled force grids, all ten components |
| --- | ---: | ---: |
| 16 × 16 | 1.97% | 5.27% |
| 16 × 32 | 1.43% | 4.42% |
| 32 × 32 | 0.64% | 1.95% |
| 64 × 64 | 0.20% | 1.01% |

The five smaller organics expose 0.30–1.40% on SCF grids and 0.91–2.60% on
force grids at 32 × 32. None has exact-zero joint-component tiles. These are
selected-block diagnostics, rather than whole-grid averages or force-kernel
operation counts. The original three shape-selected blocks overstate the
opportunity compared with the broader sample. No validation data are used.

The current ordering therefore does not justify a general sparse-kernel rewrite
from this evidence. Measuring spatial AO/grid repacking would be the next
sparsity experiment; repacking cost and numerical accuracy would need separate
validation before an engine change. Nine CPU tests cover known sparse patterns,
partial tiles, derivative unions, count aggregation, invalid fields and explicit
Gram-product enumeration. Focused Ruff checks pass. All 128 archived maxima files
are hash-checked, and all 15,360 tile masks are independently reconstructed from
their retained 16 × 16 maxima.

Retained captures remain under `ordinary-grid-profile-20261018-v1`; the expanded
scan, source/native audit, per-block maxima, geometries and log are under
`ao-tile-sparsity-20261019-v2`, both in the existing remote benchmark root. The
cancelled `v1` controller never ran a GPU job. Local raw inputs and assessments
are retained outside Git in `gpu4pyscf-tile-evidence-20261019`.
