# Mixed precision benchmark development

Read [RUNNING.md](RUNNING.md) before running experiments. The [README](README.md) explains
scope; [PERFORMANCE.md](PERFORMANCE.md) owns measured results and their version limitations.
[NATIVE_BUILD.md](NATIVE_BUILD.md) covers ordinary wheel installation with matching native
libraries, complete compiler prerequisites and the RTX 3080 build flags.

- `gpu_performance.py`: fixed-method SCF, four-CPU enforcement, profiling, precision scheduling,
  original float64 energy/gradient verification and fallback. `protocol.json` owns scientific defaults.
- `gpu4pyscf/dft/mixed_precision.py`: per-instance SCF precision, float64 verification and fallback.
- `gpu4pyscf/tools/ase_interface.py`: explicit ASE precision option, native derivatives and cache lifecycle.
- `ase_workflow.py`: generic external-input energy/force/optimization/Hessian comparisons.
- `compare_ase.py`, `test_compare_ase.py`: recipe/source/resource guards and matched-geometry report comparisons.
- `test_ase_precision.py`, `test_ase_gpu.py`: lifecycle/fallback and real-force checks.
- `test_semilocal_precision.py`: private LDA/GGA grid products, B3LYP mixed-atom
  RKS/UKS energy and force checks, warm geometry changes and force finite differences.
- `test_gga_density.py`: adaptive float64 RKS GGA density products, fractional
  occupations and tagged-orbital identity. Sparse AO blocks can use one density/AO
  product instead of four occupied-orbital projections.
- `gpu4pyscf/df/_jk_contraction.py`, `test_df_exchange.py`: symmetric float64
  exchange updates, general-output accumulation, alias and stream guards, complete
  mixed-atom J/K comparisons and verified RKS/UKS SCF checks.
- `gpu4pyscf/df/_mixed_jk.py`: per-instance early float32 exchange for native
  density-fitted global-hybrid GGAs, sharing a bounded adaptive grid schedule and
  returning to native float64 before verification and derivatives.
- `vv10_experiment.py`: scoped VV10 adapters and CuPy pair-kernel/compiler comparisons. Adapted
  from upstream v1.8.1 `gpu4pyscf/lib/gdft/vv10.cu`, under the repository's Apache-2.0 license.
- `grid_experiment.py`: scoped grid matrix products and single-GPU task-dispatch experiments.
- `test_experiments.py`: CPU-only resource/protocol checks and opt-in CUDA kernel checks.
- `force_benchmark.py`, `test_force_response.py`: exact density-contraction force comparisons,
  full grid/auxiliary response and finite differences on synthetic molecules.
- `force_experiment.py`: fail-closed compatibility adapter for the reviewed contraction sites
  in installed-release gradients; keeps the installed grid loops and native calls intact.
- `gpu4pyscf/grad/rks.py`: density contraction before AO derivative-matrix formation in full
  RKS XC response and the shared RKS/UKS VV10 response. Full-response RKS streams density
  and orbital derivatives in one AO pass; real exactly symmetric LDA/GGA/MGGA densities reuse
  block-local density/AO products. GGA retains only the density/AO zeroth-component
  product and reuses it in the derivative contraction. Forces remain float64.
- `test_native_grid_response.py`: native density/product comparisons and LDA/GGA/MGGA
  complete-force comparisons plus energy finite differences. Set `DFT_REFERENCE_GRADIENT`
  to a retained matching native baseline source for the complete-force comparisons.
- `native_force_benchmark.py`: complete native force comparisons with a retained matching
  DF gradient source, identical rebuilt buffers and fresh float64 verification of saved SCF states.
- `test_native_df_response.py`: full/long/short-range DF derivative checks, nonsymmetric
  densities, Cartesian/spherical hybrid forces and energy finite differences. Set
  `DFT_REFERENCE_DF_GRADIENT` to a matching native baseline `df/grad/rhf.py`.
- `gpu4pyscf/dft/_vv10_grad.py`: one float64 pair traversal for U/W/E and contracted coordinate
  response, with atom-ordered labels and deterministic reductions. No SCF field cache.
- `vv10_response_experiment.py`, `test_vv10_response.py`: native field/response comparisons,
  independent coordinate finite differences, partial tiles, signed weights and CUDA streams.
- `gpu4pyscf/dft/_tau_syrk.py`, `test_tau_syrk.py`: signed symmetric float64 tau products,
  stream/error restoration, early float32 dispatch and organic SCF/force/finite-difference checks.
- `gpu4pyscf/dft/_tau_triangle.py`, `test_tau_triangle.py`: direct signed float64 triangular
  products for measured medium blocks on compute capability 8.6, compatibility/alias guards,
  partial tiles, streams, unchanged early float32 dispatch and mixed-atom derivative checks.
- `native_tau_benchmark.py`, `_organic_cases.py`: fresh ASE energy/force comparisons at the
  pinned recipe; generic mixed-atom organics and a fixed larger organic geometry.

Use physical GPU 1 and at most four logical CPUs for direct performance comparisons,
including builds. Run timed comparisons sequentially under the shared benchmark lock.
Independent correctness checks may use the other host GPUs, with four separate logical
CPUs per GPU and no competing workload on the selected device. Finish those checks before
timed comparisons so host contention does not affect their results.
Keep benchmark outputs and environments outside the checkout;
never overwrite or delete earlier evidence. Preserve unrelated branches and working-tree changes.

Keep validation datasets held out from tuning and implementation decisions. Designated development
test data and synthetic molecules may be used; do not copy private inputs or results into fixtures.
Every numerical optimization also needs mixed-atom organic checks, including nitrogen, oxygen,
sulfur and halogen chemistry. `test_tau_syrk.py` includes acetamide, pyridine, DMSO, acetyl fluoride
and acetyl chloride from ASE's built-in geometries. Exercise the accelerated path explicitly in
small-molecule correctness tests and include a larger generic organic that reaches its size guard
in performance comparisons. Hydrocarbon-only benchmarks do not satisfy this coverage.

Preserve geometries, atom order, total charge, multiplicity, functional, basis, grids, tolerance,
cycle budget and full nonlocal correlation when comparing performance. Report approximate-only
experiments separately. An accepted mixed result requires full float64 energy and orbital-gradient
checks. Energy-only validation does not cover forces, Hessians or transition-state validation.

The fork's master and the older wheel share a version string but different code. Check actual
import paths and source/native hashes; use `--require-gpu-source` when testing checkout code.
Never use mismatched Python sources/native libraries or generalize results beyond measured cases.
The force adapter applies only the new contraction algebra, atom reduction and optional fused
VV10 response to the pinned release's own gradient functions. `load_extensions(..., forces=True)`
loads the new pure Python/CuPy kernel module; `--vv10-fused` compares it against density-first
forces. The adapter records source hashes and restores both functions on exit;
this isolated compatibility check does not validate a native build of the entire checkout.
Run the focused Ruff and pytest commands in RUNNING.md; CUDA tests require the designated host.

The numerical kernels now live in `gpu4pyscf/dft/_mixed_grid.py` and `_mixed_vv10.py`; benchmark
adapters share them. Private source geometries, identifiers and results must stay outside Git.
[ASE.md](ASE.md) owns calculator usage and comparison instructions.
