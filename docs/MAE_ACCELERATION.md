# MAE acceleration plan

The objective is less time to a stated uncertainty in directional magnetic
energy, with separate bounds for implementation error, sampling error and
physical-model sensitivity. Full self-consistent SIESTA remains the numerical
reference. Frozen-potential and reduced-subspace results must identify their
approximations explicitly.

## Starting point

REMAG already has persistent device SCF, sparse spinor assembly, eight solver
lanes, eigenbasis warm starts and geometry reuse between selected orientation
calculations. The [FeCo profile](../benchmarks/evidence/mae-profile-20261025-20261025T193930Z/profile-analysis.json)
places assembly/eigensolving at about 70–72% of the electronic cycle and H0
setup at 25–26 seconds per direction. Force finalization is short but uses
large derivative arrays. These diagnostic timings are not a speedup claim.

The previous overlap-factor cache passed numerical checks but had no repeatable
end-to-end gain (v137). It stays deferred unless a new workload changes that
tradeoff. Mixed precision is also deferred.

## Implementation sequence

| Workstream | Change and acceptance condition |
| --- | --- |
| Electronic-only full SCF | Skip force/stress consumers for fixed geometry while preserving accepted-density energy, entropy, populations and requested electronic exports. Compare directly with pristine SIESTA and the ordinary REMAG path. Extend existing orientation reuse only after electronic and geometry lifetimes are separated. |
| Magnetic force theorem | Retain a converged reference potential; rotate its exchange field relative to the fixed lattice/SOC operator and solve each direction without another SCF cycle. Keep electron number, smearing and energy convention explicit. Bound approximation error against selected full-SCF directional differences before using it for screening. |
| Dense-k evaluation | Stream k points through bounded solver storage, with an eigenvalues-only path where eigenvectors are not needed. Use shared meshes, verified weights, consistent chemical potentials and stable directional energy/free-energy reductions. Check chunk-size invariance and direct full-mesh agreement. |
| Symmetry and BZ integration | Use only magnetic symmetries valid for the specified SOC/moment configuration. Compare reduced meshes and improved integration/adaptive refinement with unreduced uniform controls; maintain equivalent integration accuracy between directions. |
| Torque and angular dependence | Implement angular derivatives with finite-difference checks. Fit the crystal's appropriate angular invariants: a uniaxial shortcut does not establish cubic FeCo anisotropy. Add direction constraints where unconstrained moments drift, accounting explicitly for constraint energy. |
| Reduced-subspace SOC / Wannier | Evaluate smaller SOC subspaces and interpolation only after the direct frozen-potential calculation works. Converge the retained bands/energy window and validate degeneracies, avoided crossings and symmetry against direct diagonalization. Integrate only with measured benefit. |

The force-theorem approximation evaluates relativistic band-energy changes
using a common reference potential; it is distinct from a fully self-consistent
total-energy difference. See the [force-theorem analysis](https://doi.org/10.1016/0304-8853(95)00936-1)
and [FLEUR's implementation guidance](https://www.flapw.de/master/documentation/Calculations/forceTheorem/).
Quantum ESPRESSO's [`lforcet` option](https://www.quantum-espresso.org/Doc/INPUT_PW.html#lforcet)
also provides an example of rotating a retained collinear density for subsequent
noncollinear calculations. These are references for method design, not evidence
of REMAG correctness or cross-package equivalence.

## Qualification and delivery

Start with FePt x/z and cubic B2 FeCo [001]/[110]/[111], retaining exact input,
pseudopotential and executable identities. Check final moment directions and
both total/free directional differences at the stated electronic smearing.
Retain the existing property-specific numerical targets. Do not trade accuracy
for speed silently or count an optional unvisited refinement as a failed run.

Separate approximate-method bias from k/smearing convergence; then test basis,
confinement, grid, geometry and pseudopotential sensitivity. Cross-package
agreement additionally requires matching XC, relativistic treatment, valence/
semicore, geometry, occupations and constraints. Agreement is evidence, not a
substitute for these convergence studies.

Use focused atomic commits and the existing result/ledger format, preserving
failed attempts. Update the README only for usable capabilities. GPU stages
require a current budget and run serially through `scripts/run_modal.py`; this
plan does not authorize additional allocations.
