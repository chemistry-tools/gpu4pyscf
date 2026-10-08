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
