# Mixed precision benchmark development

Read [RUNNING.md](RUNNING.md) before running experiments. The [README](README.md) explains
scope; [PERFORMANCE.md](PERFORMANCE.md) owns measured results and their version limitations.

- `gpu_performance.py`: fixed-method SCF, four-CPU enforcement, profiling, precision scheduling,
  original float64 energy/gradient verification and fallback. `protocol.json` owns scientific defaults.
- `gpu4pyscf/dft/mixed_precision.py`: per-instance SCF precision, float64 verification and fallback.
- `gpu4pyscf/tools/ase_interface.py`: explicit ASE precision option, native derivatives and cache lifecycle.
- `ase_workflow.py`: generic external-input energy/force/optimization/Hessian comparisons.
- `compare_ase.py`, `test_compare_ase.py`: recipe/source/resource guards and matched-geometry report comparisons.
- `test_ase_precision.py`, `test_ase_gpu.py`: lifecycle/fallback and real-force checks.
- `vv10_experiment.py`: scoped VV10 adapters and CuPy pair-kernel/compiler comparisons. Adapted
  from upstream v1.8.1 `gpu4pyscf/lib/gdft/vv10.cu`, under the repository's Apache-2.0 license.
- `grid_experiment.py`: scoped grid matrix products and single-GPU task-dispatch experiments.
- `test_experiments.py`: CPU-only resource/protocol checks and opt-in CUDA kernel checks.

Use physical GPU 1 only and at most four logical CPUs, including builds and tests. Run all
computational jobs sequentially. Keep benchmark outputs and environments outside the checkout;
never overwrite or delete earlier evidence. Preserve unrelated branches and working-tree changes.

Preserve geometries, atom order, total charge, multiplicity, functional, basis, grids, tolerance,
cycle budget and full nonlocal correlation when comparing performance. Report approximate-only
experiments separately. An accepted mixed result requires full float64 energy and orbital-gradient
checks. Energy-only validation does not cover forces, Hessians or transition-state validation.

The fork's master and the older wheel share a version string but different code. Check actual
import paths and source/native hashes; use `--require-gpu-source` when testing checkout code.
Never use mismatched Python sources/native libraries or generalize results beyond measured cases.
Run the focused Ruff and pytest commands in RUNNING.md; CUDA tests require the designated host.

The numerical kernels now live in `gpu4pyscf/dft/_mixed_grid.py` and `_mixed_vv10.py`; benchmark
adapters share them. Private source geometries, identifiers and results must stay outside Git.
[ASE.md](ASE.md) owns calculator usage and comparison instructions.
