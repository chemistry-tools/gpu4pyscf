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
required; the Python integration does not change its CUDA library ABI.
