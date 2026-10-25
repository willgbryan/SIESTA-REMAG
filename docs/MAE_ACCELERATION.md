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

The first implementation milestone is [energy-only full SCF](RUNNING.md#energy-only-fixed-geometry-calculations).
Its recorded FePt/FeCo checks pass, including default force-path regression;
FeCo avoids 6.33 GB of orbital derivatives. It does not establish a repeatable
runtime gain or physical convergence. The next implementation work is retained
potential/SOC evaluation and bounded dense-k solves; the workstreams below
remain active.

The frozen-potential arithmetic now has local native checks against independent
complex-matrix algebra and high-precision sums. It separates the exchange part
of all eight sparse spinor components from the pinned upstream offsite SOC
operator, rotates only exchange, and preserves the fixed lattice SOC. Paired,
compensated reductions retain both raw band energy/free-energy differences and
the fixed-electron-number grand-potential correction; a charge residual still
fails the unchanged electron-count gate. These are internal building blocks,
not an available or GPU-qualified force-theorem workflow.

An experimental evaluator now borrows the reference before the SCF owner
releases its H/S and SOC storage. Its reference is the final generator
Hamiltonian whose spectrum produced the accepted density; an identity rotation
is recorded against that same SCF spectrum. H rebuilt from the output density
is a distinct reference and must not be substituted silently. One dense solver
workspace serves every k point. Optional eigenvector solves retain only band
spin expectations, evaluated with the original nonorthogonal overlap, rather
than the whole eigenvector bank. This code awaits CUDA build/runtime validation;
the local suite verifies the shared arithmetic, not the GPU executor.

For the next qualification stage only, build with
`-DREMAG_BUILD_FROZEN_POTENTIAL=ON` and set `REMAG_FROZEN_ROTATIONS` to a file of
relative `w x y z` quaternions, one per line. Identity is evaluated first.
`REMAG_FROZEN_SPIN_EXPECTATIONS=0` selects eigenvalues-only for a matched control;
the default includes spin expectations. The probe uses the existing FD solver,
exports raw spectra/occupations and writes `remag-frozen-potential.json` without
replacing any SCF result. Its explicit downloads, host paired reductions and raw
output time are qualification overhead, not a production speedup measurement.
Unsupported reference layouts fail explicitly. Full-SCF anchors, final moment
directions, eigenvalues-only agreement, reference bias and force-theorem bias
must pass review before this becomes a screening capability. The default build
does not include the probe.

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
