# Mixed precision DFT experiments

Measure and accelerate fixed-geometry ωB97M-V/def2-TZVPD calculations on a single NVIDIA GPU.
The experiments use float32 arithmetic in the expensive early grid and VV10 pair products,
then check the final energy and orbital gradient with the original float64 implementation.
A failed check triggers float64 reconvergence within the original cycle budget.

The harness, scoped CuPy kernel adapters and pinned [protocol](protocol.json) are self-contained.
Default GPU4PySCF behavior is unchanged. These are energy experiments; wider chemistry and
analytical derivative validation are still needed before making the policy a library default.

See [measured results](PERFORMANCE.md), [reproduction instructions](RUNNING.md) and
[development guidance](AGENTS.md).
