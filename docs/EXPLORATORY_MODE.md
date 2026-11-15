# Exploratory research mode

The owner identified exploratory calculations as critical to the Astra research
environment and selected a **1–5 minute** turnaround target on 14 November 2026.
This is a product target, not a measured service guarantee or a new numerical
acceptance threshold. The research loop needs inexpensive observations that
help choose the next experiment, alongside slower numerical verification.

## Existing measurements

The [retrospective latency audit](../tests/evidence/research/fept-exploratory-latency-audit-2026-11-14.json)
reuses completed REMAG measurements. It launches no calculation. Exact source,
executable, runtime environment, basis, confinement, grid and occupation settings
match; only the k mesh changes. Each row includes a scalar seed and three
independent SOC directions.

| DZP mesh | Experiment time | E offset from DZP 40³ | F offset from DZP 40³ |
| --- | ---: | ---: | ---: |
| 6³ | 72 s | −1.184027 meV | −1.194182 meV |
| 8³ | 97 s | +0.801630 meV | +0.544368 meV |
| 12³ | 228 s | −0.089223 meV | +0.061810 meV |
| 16³ | 462 s | +0.366139 meV | +0.236002 meV |

These execution times exclude queueing, allocation startup, campaign-level
spectral postprocessing and retrieval. The product target includes the full
submission-to-usable-result interval; that interval remains unmeasured here.
Measurements span allocations and do not establish a paired hardware speedup.
The 40³ comparator is mesh-qualified only within its original DZP settings;
its remaining numerical controls and physical accuracy are unqualified.

The 12³ reference happens to be closer than 16³ while taking less time. This
nonmonotonic behavior prevents selecting a universally reliable mesh from one
reference point. All three sub-five-minute rows exceed the original 0.05 meV
criterion for at least one observable. These offsets are observed differences,
not statistical uncertainty bounds or response-error estimates. No exploratory
default is selected by this audit.

## Operating contract to implement

- Cached inspection and exact structural edits provide immediate reasoning
  context; they must remain distinguishable from a newly measured result.
- Exploratory REMAG results are provisional and source-bound. Each carries its
  actual numerical settings, measured latency and the calibration scope for the
  particular observable. Unknown error stays unknown.
- Choose the calculation to answer the question: an electronic or moment query
  need not automatically request three directional SOC energies. A scalar-only
  result cannot supply anisotropy or replace a requested SOC diagnostic.
- Calibrate response direction, relative changes and candidate ordering on
  matched structural interventions. Agreement at one unstrained structure cannot
  establish these properties. A signal unresolved by the measured sensitivity
  should prompt refinement, not an asserted mechanism or ranking.
- Expensive verification runs asynchronously for selected claims and candidates.
  Preserve the existing 0.05 meV E/F, two-refinement and transverse criteria;
  exploratory labeling never converts a failed check into a pass.
- Report queue/startup/execution/postprocessing latency separately. With exactly
  one GPU, an already-running long calculation can exceed the interactive target;
  expose that queue state and prioritize short work before the next long launch.
  Do not interrupt healthy work implicitly or create another GPU or warm pool.

## Next bounded work

The [completed 62-orbital common-mesh case](../tests/evidence/research/fept-basis24-r3-complete-review-2026-11-14.json)
took 72.7 minutes and leaves the full basis ladder unqualified. Its app is stopped.
Use the existing results to preregister an observable-specific exploration
calibration and a latency comparison of supported REMAG accelerated paths. The
current host-controlled configuration disables device-SCF, multiple solver lanes
and solver warm starts; their benefit on this workload remains unmeasured.
Preserve one executable within scientific convergence comparisons, and compare
changed execution paths against saved REMAG outputs rather than starting new
pristine SIESTA references.

No new GPU job, prediction target, budget reservation or stage-cap change is
authorized by this document. Remaining basis-stage funds and existing numerical
priorities still bound new convergence/calibration work. Preserve sealed models,
holdouts and failures. Native and magnetic-branch acceptance remain required for
any result represented as a completed measurement.

## First bounded execution-path check

The [registered REMAG-only check](../tests/evidence/research/fept-basis-accel-plan-r1-2026-11-14.json)
reuses the completed 62-orbital, 24³, 0.0025 Ry result as its saved baseline.
The scalar input bytes, executable, geometry, pseudopotentials and numerical
stopping criteria remain identical. It selects the integrated device-SCF,
vector cache, eight solver workspaces, warm starts and final-observable path
already documented in [Running](RUNNING.md). GPU radial-table generation stays
disabled, as required by the stronger-basis limitation recorded there.
Historical portability evidence supplies the configuration, not a speedup or
agreement claim for this different executable and dense input.

This single attempt has a $5 reservation ceiling inside the basis stage,
a 30-minute app deadline and 25-minute native deadline. The bound is a deliberate
feasibility cutoff, not an estimated runtime. It has no fallback, retry or
pristine reference. Native acceptance, branches, actual basis/radial identity,
transverse symmetry and overlap health are required. All eight scalar/SOC E/F
values and both anisotropies must agree with saved REMAG results within
0.0010001 meV; this is separate from the unchanged convergence criteria.

The check records submission-to-collection and submission-to-reviewed-result,
including recovery and review overhead. Individual queue/startup phases remain
unseparated. A passing result supports planning consistent-runtime radial/mesh
comparisons; it cannot qualify their numerical stability, transfer the failed
basis ladder, calibrate structural response accuracy or establish a universal
exploratory default. An unsuccessful run is retained and diagnosed before
any further attempt. After its reservation, $65 remains in the basis subcap;
the separate $50 final-control cap cannot fund this work.

## Execution-path attempt: scalar recovered, SOC unmeasured

The [saved-data diagnosis](../tests/evidence/research/fept-basis-accel-r1-diagnosis-2026-11-14.json)
records a harness failure after the scalar stage completed in 122.24 seconds.
The Hamiltonian-export audit misclassified `%block PAO.Basis` as an additional
analysis request. The corrected policy recognizes this basis-definition block
while continuing to reject unqualified analysis blocks and consumers.
Reprocessing the saved files locally passes the runtime and native scalar
checks; E/F differ from the saved host-controlled REMAG scalar by only
−0.000008332/−0.000008291 meV. The scalar input bytes and executable are identical.
The 62-orbital shell identity and magnetic branch pass; native radial arrays
differ by at most 1.01e-12. Scalar overlap is positive at all 7,224 native k
points, with maximum condition number 626,017. These are scalar health checks,
not basis completeness or anisotropy validation.

No SOC stage ran, and no completed scalar calculation was rerun. The original
failed report and evaluation are retained. Its 190.02-second
submission-to-reviewed-result field measures return of a failed attempt; it is
not a usable four-stage result or demonstration of the exploratory target.
The scalar timing is a cross-allocation observation, not a paired speedup.
The app stopped at 01:37:50 UTC on 15 November. No retry or follow-on GPU is
queued. Scientific convergence and observable-specific exploratory calibration
remain open under the existing budget and criteria.

The [next registered controls](../tests/evidence/research/fept-radial-mesh-plan-r4-2026-11-14.json)
use the saved host-controlled REMAG runtime for two radial points and one 28³
mesh recheck. They are bounded background numerical checks, not interactive
requests or measurements of the 1–5 minute target. Their maximum reservation
is the remaining $65 basis envelope, with explicit review between cases and
no automatic queue. The incomplete integrated-path probe cannot supply a SOC
anchor for mixing runtimes. Existing exploratory calibration evidence and
scientific failures remain unchanged.
