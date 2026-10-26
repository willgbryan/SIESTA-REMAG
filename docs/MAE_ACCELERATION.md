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
spin expectations, evaluated with the original nonorthogonal overlap's lower
Hermitian triangle, as used by the eigensolver, rather than the whole eigenvector
bank. CUDA runtime checks now pass for the sampled FePt 64-k and FeCo 4096-k
cases. This establishes the recorded identity-spectrum, occupation, moment and
full-SCF regression checks; it does not independently validate every rotated
Hamiltonian or qualify the approximation as a replacement for full SCF.

For experimental qualification only, build with
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

The [25 October qualification batch](../benchmarks/evidence/mae-frozen-potential-20261025T210823Z/REPORT.md)
completed six of seven calculations within its 20-minute allowance. All 18
executed checks passed; FePt's eigenvalues-only control passed, while FeCo's
control was unvisited when the budget reserve was reached. The current method
misses the 5 micro-eV diagnostic target: FePt's free-energy bias is 37.8 micro-eV,
and FeCo's directional biases reach 96.4 micro-eV and reverse the [110]/[111]
ordering. These are fixed-setting, 300 K electronic-smearing results, not
physically converged MAE. The probe remains experimental and disabled by
default; the README's usable capability remains energy-only full SCF.

The [reference-direction diagnostic](../benchmarks/evidence/mae-frozen-reference-20261025T221016Z/REPORT.md)
subsequently completed all five calculations and passed all 20 remote checks,
42 independent selected-k matrix captures, and FeCo's eigenvalues-only control.
The latter agrees with vector solves within 5.36e-8 eV per eigenvalue and
3.79e-11 eV in directional band/free differences. This closes the missing
implementation control without qualifying the approximation.

FeCo's frozen [110]−[001] free-energy difference changes from +105.007912 to
−106.512299 micro-eV/f.u. between [001] and [110] references, while fresh
full SCF gives +8.561012. The reference spread is 211.520211 micro-eV/f.u.;
the [111]−[001] spread is 131.031144. These are unchanged 300 K electronic
smearing, 16³-mesh results, not physical convergence. Every reference predicts
itself lowest among its sampled rotations. Passing matrix checks does not
make this method suitable for screening.

The next useful test is a common SOC-free, spin-polarized reference retaining
scalar-relativistic terms, followed by the same fixed-lattice-SOC rotations and
full-SCF anchors. This is an untested hypothesis for reducing the measured
reference dependence. Resolve the E/F bias before throughput optimization or
interpolation, retaining the same energy, free-energy and direction gates.

The optional diagnostic `REMAG_FROZEN_MATRIX_SAMPLES=1` retains the exact
generator H, overlap, fixed lattice SOC and rotated H at the first, middle and
last mesh indices (duplicates removed). It uses one additional buffer capped
at 64 MiB, retains both assembled triangles, counts its downloads and performs
no additional eigensolves. Run `python scripts/check_frozen_matrices.py <run>`
with the development dependencies to compare these captures with independent
NumPy SU(2) multiplication and SciPy generalized eigensolves. The check also
quantifies whether rotating before taking the solver's lower Hermitian triangle
changes the spectrum, and tests rotation invariance after removing SOC. This
selected-k diagnostic has local positive/negative tests and now passes its
sampled CUDA runtime checks: matrix rotation error is at most 6.47e-15 Ry and
spectral error at most 4.18e-10 eV. It cannot replace the full-mesh, full-SCF or physical-accuracy
checks, and terminal HSX output is not a substitute for this generator capture.

The completed follow-up has five serial calculations: FePt z with frozen x,
FeCo [001] with frozen [110]/[111], its eigenvalues-only control, and FeCo
[110] and [111] references each rotated to the other two directions. This
isolates implementation error and reference sensitivity with unchanged inputs
and historical pristine-SIESTA evidence; it adds no full CPU-SCF timing runs.
The single H100 allocation is released, with zero account containers. Its
conservative app lifetime was 856.537442 seconds, including image building,
within the newly approved 900-second allowance. The diagnostic completed before
its fixed deadline; there was no restart or follow-on allocation. Raw evidence
and method failures are retained. Repeated comparable performance measurements
remain open; no useful or upstream-GPU speedup is established.

Local angular analysis now fits explicitly selected cubic sixth-order or axial
quartic invariants, with separate energy estimators and units. It rejects
underdetermined samples, reports residuals and coefficient error amplification,
and distinguishes symmetry-equivalent controls from additional angular data.
The accompanying torque is the derivative of the fitted polynomial, checked
against finite rotations; it is not yet an electronic torque implementation.
The axial model assumes no azimuthal dependence and is not a general tetragonal
model. The 45-degree shortcut in the [original torque paper](https://doi.org/10.1103/PhysRevB.54.61)
depends on that angular form and must not be applied to cubic FeCo.

The [local FeCo interpolation](../benchmarks/evidence/mae-frozen-potential-20261025T210823Z/angular-analysis.json)
uses the recorded final moments and keeps total/free and SCF/frozen energies
separate. Three directions determine a constant and two cubic coefficients
exactly, leaving no independent test of omitted angular terms. Proposed [210]
and [211] holdouts and the [100]/[001] cubic equivalence control remain unrun
and were not part of the completed five-calculation diagnostic budget.

## SOC-free common reference (local implementation, runtime pending)

The experimental build now accepts `REMAG_FROZEN_SOC_FREE=1` with the frozen
rotation probe, GPU initialization and fixed-geometry energy-only execution.
It retains the existing scalar nonlocal Hamiltonian and full lattice SOC
operator, but binds a zero SOC operator throughout the reference SCF, including
initialization and energy contractions. Frozen targets rotate this converged
exchange field and restore lattice SOC exactly once. The ordinary full-SOC path
is unchanged by default. Final Hamiltonian recomputation is rejected in this
mode; requested electronic exports describe the SOC-free SCF reference.

An additional, explicitly counted SOC-free identity solve checks the reference
spectrum before SOC-restored targets are evaluated. Reports and selected-k
captures distinguish reference and target SOC scales; ordinary regression
comparison rejects different SCF SOC treatments. The independent matrix oracle
uses the declared scales and retains the existing tolerances. Local arithmetic,
negative controls and metadata gates pass; CUDA compilation and runtime/physical
qualification remain pending. This is not yet a production screening option or
a demonstrated improvement in MAE accuracy.

Changing the input to ordinary noncollinear SIESTA or scaling its semilocal SO
potential can regenerate different projectors (see pinned `atom.F:read_vps`
and `Docs/tex/sections/Options/Pseudopotentials.tex`). Those are not exact oracles
for removing only the assembled SOC operator with the original scalar operator
held fixed. The next diagnostic therefore combines full-mesh identity, selected
exact-Hamiltonian checks, unchanged full-SCF/pristine anchors and sensitivity to
two independently converged SOC-free reference directions. Method error remains
separate from implementation agreement and k/smearing convergence.

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
