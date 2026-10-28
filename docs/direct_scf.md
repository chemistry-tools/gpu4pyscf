# Native direct-integral regrouping

The direct SCF path evaluates screened electron-repulsion integrals as needed,
without retaining a three-index density-fitting tensor. Conservative primitive
regrouping can reduce the exchange kernel's contracted inner loops while
reconstructing the same original orbital basis. Integral screening tolerances,
float64 arithmetic, range separation and the Coulomb engine remain unchanged.

Set `mf.direct_scf_regroup = True` before SCF or constructing an ASE calculator.
The default is `False`. The qualification guard currently requires native
restricted singlet ωB97M-V/def2-TZVPD, spherical orbitals, H/C/N/O/F/S/Cl atoms,
one visible compute-capability 8.6 GPU and no density fitting. Other methods,
devices, elements, Cartesian bases and low-memory variants keep the original
representation. This option does not switch between direct and fitted SCF.

```python
from gpu4pyscf import dft
from gpu4pyscf.tools.ase_interface import PySCF

mf = dft.RKS(mol, xc="wb97m-v")  # mol uses spherical def2-tzvpd
mf.direct_scf_regroup = True
atoms.calc = PySCF(method=mf, precision="float64", grid_response=True)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The setting belongs to the SCF instance and persists through its usual scanner
and geometry reset. Ground-state analytic forces select the matching
representation through the same optimizer cache. Changing the flag, molecule
object or screening tolerance rebuilds an affected cache entry before use.
Set the flag to `False` for an original-representation control. After changing
the underlying method in an existing ASE calculator, call `calc.reset()` to
invalidate ASE's stored properties.

Qualified ground-state forces evaluate exchange as `hyb * full + (alpha-hyb) * LR`,
using the functional's short-range fraction `hyb` and long-range fraction `alpha`.
This equivalent decomposition uses fewer quadrature roots than the original
full/SR derivative passes, and its LR optimizer can reuse the SCF exchange cache.
Other methods retain their original force decomposition. Screening thresholds,
precision, density and grid response are unchanged.

Matching native builds also contract the generic J/K force raising/lowering terms
before applying primitive exponents and center offsets. The fourth-center force
is reconstructed after contraction using translation symmetry. The same per-instance
qualification selects this float64 rearrangement; block layouts, screening and
workspaces stay unchanged. Older native libraries fall back to their original
force entrypoint. Rebuild the native package to use this additional optimization.
Its complete SCF/force qualification is recorded separately in the measured results.

The precision option is independent: regrouping does not introduce float32
exchange. Qualification passes complete float64 SCF and physical-Fock checks
on five mixed-element organics and a 681-orbital development case, full forces
including same-state original-representation controls, and an independent energy
finite difference on acetyl fluoride. It does not
qualify Hessians, excited-state derivatives or transition states. More aggressive
primitive regrouping has failed strict numerical reconstruction checks and is
not exposed by this control.

See [reproduction instructions](../benchmarks/mixed_precision/RUNNING.md#direct-versus-fitted-jk)
for the fixed-density and complete SCF/force checks, and
[measured results](../benchmarks/mixed_precision/PERFORMANCE.md#direct-exchange-profile-and-primitive-grouping)
for the scope of the component timings. A matching native installation is
required. The original CUDA entrypoint remains available; the arithmetic optimization
adds an optional entrypoint with the same argument layout.

## Experimental bounded integral reuse

An additional opt-in prototype retains compact, contracted integral groups for
repeated direct exchange builds:

```python
mf.direct_scf_regroup = True
mf.direct_scf_cache_bytes = 256 * 1024**2
```

For a GPU with enough memory to retain every selected exchange integral group:

```python
mf.direct_scf_regroup = True
mf.direct_scf_cache_all = True
```

This explicit toggle overrides the byte budget and includes fully primitive
groups. Before allocating integral arrays, it estimates their complete storage
and checks free device memory, reserving the larger of 256 MiB or 5% of device
capacity for workspace. Insufficient space raises `MemoryError` with the required
and available sizes; allocation failures also raise, without spilling or silently
switching to a partial cache. A missing native fill entrypoint raises in this mode.
Reports expose planned bytes and whether every group is resident. Turn it off
with `mf.direct_scf_cache_all = False` to return to the configured bounded budget.
Here “every” means the native geometry-screened exchange pair groups; it does not
add derivative integrals, density-fitting tensors or other SCF workspaces.

The default budget is zero and the full-VRAM toggle defaults to `False`.
Reuse shares regrouping's method,
element and device qualification and requires the new `RYS_fill_int4c2e_k` native
entrypoint. Older native libraries use ordinary direct evaluation in bounded mode.
The cached
path additionally requires one real symmetric density on the default CUDA
stream; unsupported calls use the original kernel.

The budget bounds the stored GPU arrays for one exchange optimizer, including
its index metadata. Existing direct-SCF queues and other SCF workspaces are
separate. Distinct optimizers can each own a cache. No integral files or CPU
backing store are created. Whole angular/primitive groups are admitted by their
actual rectangular storage size, favouring high primitive contraction per stored
value. Groups outside the budget are recomputed normally; allocation misses also
return to the direct kernel.

Cached values use float64 and contain the weighted range-separated operator.
Construction covers every pair in an admitted group, independently of the current
density; each contraction reapplies the original shell symmetry and density
screening at the unchanged cutoff. A new density therefore cannot expose missing
cached quartets. Geometry, basis, tolerance, operator, device and stream identity
participate in invalidation. Reset and rebuild release the optimizer's cache.
Ordinary integral reuse does not cache derivative integrals for forces.

This is experimental: a cold cache must be built before reuse can help. Compare
the build plus roughly five density updates, rather than only hot contractions,
before enabling it for a workload. The matching native build, growing/signed
density tests, mixed-element CPU-oracle comparisons and complete SCF/force checks
are required; an admission estimate alone is not a performance result.
