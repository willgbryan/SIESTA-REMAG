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

Finish and review the already-running 62-orbital common-mesh REMAG calculation.
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
