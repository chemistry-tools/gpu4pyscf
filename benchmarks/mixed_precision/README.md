# Mixed precision DFT experiments

Measure and accelerate fixed-geometry ωB97M-V/def2-TZVPD calculations on a single NVIDIA GPU.
The experiments use float32 arithmetic in the expensive early grid and VV10 pair products,
then check the final energy and orbital gradient with the original float64 implementation.
A failed check triggers float64 reconvergence within the original cycle budget.

The harness, scoped CuPy kernel adapters and pinned [protocol](protocol.json) are self-contained.
The ASE calculator defaults to `precision="auto"`, selecting verified mixed precision for
supported GPU VV10 DFT. Its explicit `precision="mixed"` and `precision="float64"` options
control the SCF policy. The native build, full-response forces, geometry reuse and independent
energy finite differences are checked on synthetic development molecules; validation data
remains held out.

See [measured results](PERFORMANCE.md), [reproduction instructions](RUNNING.md) and
[development guidance](AGENTS.md).

Use the [ASE integration guide](ASE.md) for energies, forces, optimizer steps and workflow comparisons.
