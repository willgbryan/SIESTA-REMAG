# Density-history residency preflight

Density history still consumes the terminal host density matrix. Removing that
download requires a process-lifetime GPU owner plus migration of the next
geometry's initialization consumers. Moving only the saved copies would leave
the electronic round trip intact. This work follows the conditional E-export
candidate; none of the preflight code changes the running v84 implementation.

## Pinned reference behavior

The executable specification is upstream commit
`e486d12067b96ff688179f0496d0ec21b6fae0ab`. The
[consumer audit](../results/preflight-energy-exports-v83/density-history-consumer-audit.json)
records source hashes and measured boundary traffic.

* `state_init.F:706–716` rewrites **every** saved matrix at each geometry:
  supercell translation first, then destructive restriction to the new sparse
  pattern. Once an interaction is removed, a later reappearance must start at
  zero. Keeping immutable original matrices and remapping only at use is wrong.
* `restructSpData2D.F90` uses row/column matching, zero for missing columns and
  the last occurrence of duplicate old columns. All 1/2/4/8 spin components must
  follow the same map.
* `Fstack.T90` orders entries oldest to newest. At capacity, push evicts the
  oldest. `post_scf_work.F:195–208` stores the converged density with its geometry;
  force-constant/phonon handling can keep only the original reference.
* `m_new_dm.F90:1477–1578` derives coefficients from coordinates and accumulates
  saved matrices in stack order. Default CG uses depth one unless
  `DM.AllowExtrapolation` is true. The current GPU relaxation corpus therefore
  does not establish depth-three history equivalence.
* Initialization spin/Mulliken output, charge normalization, the SOC energy term
  in `setup_H0.F`, initial grid density/potential and the sparse-state seed still
  read host electronic arrays. These consumers must move together with history.

## Retained numerical evidence

[Status and evidence](../results/preflight-density-history-v84/status.json):

* 16 exact sparse-remap snapshots cover aliasing of input/output, duplicate
  columns, empty/new rows, removal/reappearance and all four spin layouts.
* Six default DIIS/SVD coefficient fixtures use the literal pinned routine and
  pristine linked SVD implementation. Independent NumPy least squares uses the
  upstream `rcond=1e-6`; maximum coefficient difference is `2.510e-14` (rounded
  upward).
* The expanded [24 coefficient-mode fixtures](../results/preflight-density-history-v84/coefficient-modes.json)
  additionally compare the literal original extrapolation, inverse DIIS and
  experimental first-difference SVD routines with independent matrix algebra.
  All four methods pass the six fixed fixtures; maximum difference is
  `5.707e-14` (rounded upward). This gives 56 unique primitive checks including
  remapping and supercell translation, without double-counting the six default
  SVD fixtures. Ill-conditioned boundary cases and failure paths remain open.
  The experimental routine's actual coefficient sum is not always one (for
  example `1.71577` for the three-entry random fixture); these checks preserve
  its actual behavior instead of imposing a generic DIIS formula.
* 16 exact single-matrix supercell checks cover expansion, contraction, mixed
  axis changes and no change, including translated columns and zeroed values.
* 32 linked stack diagnostics exercise five pushes into depth three, retain
  geometry markers 3/4/5, then apply the `state_init` remapping call order.
  All 20 unchanged-supercell or independently owned-pattern controls pass
  exactly. The 12 changed-supercell/shared-pattern cases expose the issue below.

These are local CPU reference checks. They do not establish GPU execution,
scientific equivalence of a future history owner, or any speedup.

Five [full local trajectory preflights](../results/preflight-density-history-v84/relax-qualification.json)
explicitly enable `DM.AllowExtrapolation` and depth three. Collinear Fe2 and
sampled-k non-collinear Fe2 pass both SCF convergence and the unchanged
`0.01 eV/Ang` final force criterion, each through nine geometries and six uses
of depth-three history. They take 97 and 96 SCF iterations respectively.
The displaced Gamma NC case fails its first SCF, and Gamma FePt SOC fails at
geometry 18. Sampled-k FePt SOC completes all 21 electronic cycles but its
final maximum force is `1.05033 eV/Ang`; it is not a converged relaxation.
All three unsuccessful relaxation attempts are retained and excluded from
relaxation qualification. A later converged SOC reference is recorded below;
its GPU history equivalence remains untested.

## Shared sparsity issue

`Pair` and sparse-matrix constructors retain references, and restructuring gives
each history matrix the same target sparsity object. `correct_supercell_SpD`
mutates that object's column indices. Iterating over history can therefore
translate already-translated columns again.

In the [retained linked diagnostic](../results/preflight-density-history-v84/history-stack-diagnosis.json),
the tested expansion and mixed-axis changes terminate on the second entry with
`Mismatch in sparsity pattern and nsc values`. Contraction returns but differs
from independent coordinate remapping. Separate pattern ownership eliminates
these failures for the same inputs. The diagnostic reproduces the relevant
class ownership and call order; it is not a full material trajectory exhibiting
the issue. Upstream remains unmodified.

A GPU map should derive each transition from unmodified old topology and apply
it once to every stored density. Qualification must explicitly address this
reference discrepancy; it must not report agreement with the failing shared
reference or silently discard history on a supercell change.

## Required implementation and acceptance

The device owner must outlive per-SCF cycle resources, retain ordered geometry
and density entries with bounded eviction, destructively advance every entry to
the new topology, and publish the extrapolated density directly to initialization
and sparse-state consumers. Requested DM/HSX/TSDE/NetCDF output must remain a
separate, reason-tagged export. Epoch, spin/shape, orbital distribution and
convergence validity require explicit checks; invalid calls must not partially
commit a new history entry or overwrite usable state.

Numerical acceptance needs full relaxation with actual depth greater than one,
nontrivial coefficients, all supported spin layouts, sparse interaction
removal/reappearance and auxiliary-cell changes. Owner checks must cover zero
depth, rollover, reset, repeated geometry and stale epochs. Transfer accounting
must demonstrate removal of terminal history D download and next-geometry D
upload, and independent whole-process memory/I/O observation remains required
for the overall residency claim. Learned SCF remains deferred.

## Isolated device owner qualified (v85)

`native/density_history.cu` now implements a persistent, bounded device owner.
It retains oldest-to-newest density and coordinate entries, evicts the oldest
at capacity, derives a supercell transition from the unmodified old topology,
and destructively remaps every entry before publishing the new topology. All
1/2/4/8 spin layouts use the same map, including last-duplicate-wins semantics.
Device epoch/distribution/convergence checks precede commits; invalid operations
leave retained state and caller output unchanged. Reset requires an increasing
run generation. The allocation bound, 32-entry maximum depth, odd auxiliary-cell
axes and fixed atom/orbital distribution are explicit restrictions. Capacity
beyond the allocation bound is rejected; a growth handoff is not implemented.

All four coordinate coefficient methods are implemented in
`native/density_history_math.hpp`. Its caller-owned workspace avoids large
thread-local matrices. The default method uses a symmetric Jacobi decomposition
with the upstream `1e-6` singular-value cutoff. Compensated re-projection and
subsequent eigenvector refinement repair a precision loss found near that
cutoff. The retained 60-digit Decimal diagnostic confirms the original candidate
error; no material scientific gate was changed. The local
[60-check arithmetic report](../results/preflight-density-history-v85/math.json)
covers the 24 actual upstream fixtures, invalid arithmetic/configurations and
12 scaled symmetric systems through order 33. These are CPU executions of the
shared arithmetic; they do not establish CUDA or material equivalence.

The isolated CUDA check is built as `remag_density_history_check`. Its operations
are designed to enqueue only kernels, with staging and diagnostic downloads
outside the tested sequence. Tests exercise direct and captured execution,
all coefficient modes, five pushes into depth three, cell expansion/contraction,
destructive removal/reappearance, reset, stale ownership, invalid topology,
nonfinite inputs and unconverged/duplicate pushes. Raw snapshots are retained
for each operation. The [qualified GPU batch](../results/20261007T055038Z/qualification-summary.json)
passes 104 sequences and 1,528 operations, including 328 rejected calls. Its 52
captured sequences contain 5,972 kernel nodes and zero non-kernel nodes. Maximum
coefficient error against the pinned fixtures is `1.999e-14` (rounded upward).
Raw sequence replay, source/archive verification and device release all pass.
This is isolated component qualification, not a full memory/I/O observer or a
material result. The owner is deliberately not linked into production SIESTA yet: the
Fortran handoff, initialization consumers, output gating and removal of the
actual terminal D download/next-geometry upload still need implementation and
full material qualification. v84 remains the latest qualified material baseline.

## Initial normalization and Hamiltonian components qualified (v86)

`native/initial_density.cu` implements upstream first-density normalization,
including its absolute-charge first-mixing decision and zeroing of the initial
energy density. All density spin components remain on device; invalid inputs
leave outputs unchanged. `ScfCyclePlan::initialize_hamiltonian` builds the initial
Hamiltonian from that normalized device density through the existing grid,
Hartree/PBE, seed and SOC producers before launching the electronic cycle.
The method rejects repeated initialization and initialization after launch.

[Run 20261007T060934Z](../results/20261007T060934Z/qualification-summary.json)
passes 784 direct/captured, alias/non-alias normalization checks against 160
pristine reference fixtures, including 144 invalid-input checks. Captured
normalization operations contain three kernels and no non-kernel nodes.
Twenty-four composed initialization/SCF fixtures cover all supported spin
layouts and first-mixing choices. Their matrices, occupations, scalars,
diagnostics and control decisions pass unchanged numerical gates. Source
provenance and raw-array rechecks pass locally; the sole A100 allocation is
released. This probe used driver 580.95.05, unlike v84/v85's 610.57.04.

These components are not connected to production SIESTA's geometry handoff.
The initialization is numerically checked before the captured SCF region;
the graph-node check does not independently observe all initialization memory
traffic. Initial population consumers, topology-only grid preparation, retained
history ownership, capacity growth and requested-output gating remain work.
No production transfer removal or new end-to-end speedup is claimed by v86.

## Production GPU initialization qualified for base materials (v87)

Run `20261007T063020Z` connects initial normalization and the initial Hamiltonian
to the production device cycle. Geometry-only grid and H/S preparation stage
topology and invariant inputs without reading host electronic H/grid buffers.
All nine base materials and both MAE comparisons pass independent raw-output
checks; final forces, density, populations and terminal accounting also pass.
There are 381 SCF iterations over 17 geometries with zero explicit steady-cycle
host/device transfers. The single GPU allocation is released.

Relative to the first v84 candidate for each base case, explicit initialization
traffic falls by 83,139,304 upload bytes, 27,178,296 download bytes and 153 copies.
The new initialization diagnostic adds 32 bytes after each electronic cycle
(544 bytes across the suite). Initial density upload, initial population host
consumers and atomic/Harris/geometry staging remain. This is not zero total
initialization traffic or complete independent memory/I/O observation.

The integral-force auditor now requires zero initial grid exports only when
batch metadata and native owner records establish GPU initialization, with a
separate initialization audit prerequisite. All 54 legacy v84 records retain
their original nonzero transfer expectations and still pass. Original failed
audits remain in the evidence folder. The comprehensive TSDE/NetCDF audit still
reports missing coverage: this batch ran the nine base cases only. Those output
variants need execution with GPU initialization before that combination is
qualified. Reused CPU references on different hardware do not establish a new
speedup. See the [scope and evidence](../results/20261007T063020Z/qualification-summary.json).

The production history handoff remains absent. Its capacity-growth prerequisite
is qualified below; initial populations and the geometry handoff remain.

## Device history allocation growth qualified (v88)

`DensityHistoryPlan::reserve` increases the allocation bound while preserving
all retained densities, coordinates, active topology, generation, epoch and
last-push guards. It copies with device kernels into larger strides, synchronizes
the owner stream, and then retires the old allocations. There is no electronic
or diagnostic readback in this method. Existing graphs referencing old storage
must be retired before growth; actual growth during capture is rejected.
Smaller positive requests are no-ops. Invalid bounds leave the owner unchanged.
An optional padded diagnostic stride allows exact before/after comparisons.

[Run 20261007T070741Z](../results/20261007T070741Z/qualification-summary.json)
passes 132 sequences and 2,300 operations, including 472 rejected operations.
Twenty-eight sequences exercise growth and its rejection, with 244 reserve
requests. Checks cover empty owners, live history, rollover at depths one and
three, zero depth, every supported spin layout, supercell remapping, reset,
invalid capacity and graph capture after growth. All retained state snapshots
reproduce in a local raw-array replay. The 68 captured sequences have 8,136
kernel nodes and zero non-kernel nodes; growth itself runs outside capture.
Source verification covers 316 hashes. The sole A100 allocation is stopped.

This qualifies the component, not production D transfer removal or complete
memory observation. Remaining integration must move both initial population
consumers and history extrapolation to the retained device density, push only
after final force acceptance, and replace host-D equality checks with explicit
device ownership. Requested output remains a separate export. Depth-greater-than-one
material trajectories, including a converged SOC relaxation, are still required.

## Initial device owner probe and converged SOC reference (v89 preflight)

The new initial-state owner stages unnormalized D once, computes requested
initial spin/Mulliken output on the GPU, and lends that same device allocation
to SCF after the output window closes. The standalone probe passes 16 cases
over all four spin layouts and all output request masks, including 264 rejected
ownership/error paths. Raw-array replay confirms exact D preservation after
poisoning its host source, requested population arithmetic and cached exports.
The [component evidence](../results/20261007T073134Z-interrupted-beff36f1f2654a26a1bbc9c61d68a051/component-qualification.json)
belongs to an interrupted Modal attempt. Its 238 source hashes and 239 working
runtime files verify. Evidence is not merged between attempts. The fresh v89
attempt subsequently stopped on an initial-population validator defect after
12 successful collinear comparisons. Its NC printer correctly exported derived
views, while the validator demanded an unused raw total. The separate
[raw-output diagnosis](../results/20261007T073134Z/validator-failure-diagnosis.json)
passes all 13 completed calculations' unchanged scientific value gates; original
validity and ledger records remain untouched. Spin-specific required-view checks
repair that defect and reject omission of any actually consumed table. The old
allocation is released and full corrected run `20261007T081045Z` is active.
Production density history and terminal D export removal remain unimplemented.

A [pristine sampled-k FePt SOC reference](../results/preflight-soc-history-relax/fept2_soc_z-mesh400-step002/relaxation-reference-verification.json)
now converges through seven geometries and 80 SCF iterations. It exercises
depth-three history four times with nontrivial coefficients. Compared with the
earlier unsuccessful reference, the mesh cutoff is increased from 100 to 400 Ry
and the maximum CG displacement is limited to 0.02 Ang. The initial displacement,
pseudopotentials, spin-orbit mode and electronic convergence thresholds remain
the same. Final maximum force is 0.00452725 eV/Ang, below the unchanged
0.01 eV/Ang criterion. Source/binary/input hashes, complete outputs and an
independent trajectory audit are retained. This establishes a local reference
for the future GPU history handoff, not GPU equivalence or a physical convergence
study.

## Production handoff and first trajectory diagnosis (v91)

The v90 initial owner/output implementation completed the full 18-case repeated
qualification in `20261007T081045Z`; its allocation is released. The production
history bridge now connects coordinate remapping, extrapolation, initial
populations, SCF and force-accepted history retention without a host D round
trip. Requested D output is separately gated.

The first v91 material case in `20261007T091252Z` completed nine collinear
relaxation geometries and 95 SCF iterations, but its original result is invalid
because the validator misinterpreted SVD rank. Default constrained extrapolation
uses a `(history length + 1)` system. Ranks three and four are valid for history
lengths two and three, respectively. The corrected metadata check distinguishes
all four methods, exact singleton reuse and fallback semantics; numerical
tolerances are unchanged.

A separate [raw diagnosis](../results/20261007T091252Z/history-rank-diagnosis.json)
reproduces the original failure and passes final scientific values, every
geometry's forces/stress/moments/charges, all initial populations and all printed
extrapolation coefficients. The latter agree exactly, with six nontrivial
depth-three uses. Initial D uploads occur only for the first geometry; terminal
D downloads and all explicit in-cycle copies are zero. Original results and
ledger remain untouched. This is one diagnosed trajectory, not full production
qualification or a speedup claim.

The original allocation is released. Corrected full retry `20261007T094904Z`
uses the identical numerical implementation and a new single GPU allocation.
The next terminal-spectrum/H-export candidate remains isolated pending this
history qualification. CPU geometry/ionic work and complete independent
memory/I/O observation remain open.
