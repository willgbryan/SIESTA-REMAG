# FePt qualification priorities

Planning estimate, 14 November 2026. Mesh and basis checks take priority over
new physics-guided predictions. This proposal does not authorize additional
spending, alter the original $100 ledger, or launch compute.

The owner subsequently authorized **up to $250 additional**, one GPU at a time,
starting with the $50 mesh stage. The [authorization record](../tests/evidence/research/fept-authorization-r2-2026-11-14.json)
preserves the original reservations; the [36³/40³ execution plan](../tests/evidence/research/fept-mesh-r3-plan-2026-11-14.json)
allocates $20 and $30 respectively. The cumulative authorized ceiling is $350.
The remaining sections retain the proposal's estimates and scientific priorities.

The [first local TZP radial step](../tests/evidence/research/fept-radial-r2-first-step-2026-11-14.json)
is complete: reducing the energy shift from 0.02 to 0.01 Ry changes E/F
anisotropy by −0.100/−0.094 meV at 16³, exceeding the original threshold.
Native and magnetic-branch checks pass. Both overlap matrices are positive at
every native sampled k point, but the worst condition number grows from about
13,093 to 125,457. This is a conditioning diagnostic, not a basis-completeness
claim. Further radial and explicit-basis results remain pending.

The [second radial step](../tests/evidence/research/fept-radial-r2-second-step-2026-11-14.json),
0.01→0.005 Ry, changes E/F by +0.077/+0.066 meV: both fail and reverse the
first step's trend. All four native stages, magnetic-branch and transverse
checks pass. Overlap remains positive at every sampled point, with worst
condition number about 105,469. The registered 0.0025 Ry point is running;
no new prediction targets are scheduled.

## Completed 36³ mesh and local radial checkpoint

The [36³ result](../tests/evidence/research/fept-mesh-r3-k36-review-2026-11-14.json)
passes the original mesh criterion at the existing DZP/0.02 Ry settings:
28→32 and 32→36 both change E/F by less than 0.05 meV. The new step is
+0.035141/+0.006256 meV. Native, transverse and magnetic-branch checks pass;
the earlier 24→28 failure remains recorded. All 15 omitted native binary
artifacts were recovered and hash-verified. The 40³ confirmation started
12 seconds after the 36³ app stopped, on one A100. This mesh result does not
transfer to a different basis or orbital range.

The [local TZP radial ladder](../tests/evidence/research/fept-radial-r2-evaluation-2026-11-14.json)
finished without passing: its final 0.005→0.0025 Ry step changes E/F by
−0.077/−0.063 meV. All three radial steps exceed tolerance, with alternating
signs. All native, branch and transverse checks pass. The
[overlap diagnostics](../tests/evidence/research/fept-radial-r2-overlap-2026-11-14.json)
remain positive on every sampled k point; the final worst condition number is
about 298,404. The registered basis-size comparison selected 0.0025 Ry by its
preregistered rule and has started locally, retaining the failed radial gate.

The full $50 mesh-stage envelope is now reserved, bringing cumulative
reservations to $146.67 against the $350 authorized ceiling. The $150 basis
and $50 final-control sublimits remain unused; actual billing is unreconciled.

The [explicit TZP replication](../tests/evidence/research/fept-basis-r2-replication-2026-11-14.json)
passes at 16³ and 0.0025 Ry: all eight scalar/SOC E/F comparisons have zero
difference at native printed precision, with native and branch checks passing.
HSX, KP and both species' basis XML files are byte-identical for all four
stages, so the existing overlap diagnostic is reused by hash. This establishes
the explicit 42-orbital baseline, not a basis refinement. The 48-orbital
enrichment is running locally; the failed radial gate remains unchanged.

The [42→48-orbital enrichment](../tests/evidence/research/fept-basis-r2-first-enrichment-2026-11-14.json)
is complete and fails the E/F criterion: +0.352/+0.300 meV at fixed 16³ and
0.0025 Ry. Native, branch and transverse checks pass, and all shared radial
functions remain identical in every scalar/SOC stage. The overlap is positive
on every sampled k point, with worst condition number about 512,195. Runtime
was 2,708.98 seconds on the pinned local CPU. The 62-orbital case is running;
neither basis-size nor radial convergence is qualified.

## Qualification target

Qualify numerical stability of E[100]−E[001] and F[100]−F[001] for the fixed
FePt geometry, pseudopotentials, functional and stated electronic temperature.
Keep the existing 0.05 meV/cell threshold for both quantities and 0.01 meV/cell
x/y threshold. The mesh gate still requires two successive passing refinements.
These are per-control stability criteria, not a rigorous combined error bound
or a certificate of experimental accuracy, equilibrium structure, ground-state
magnetism or zero-temperature convergence.

The [completed evidence](FEPT_CONVERGENCE.md) remains unchanged: 24→28 fails,
28→32 passes individually, TZP and increased radial extent change both E and F
beyond tolerance, and the 150 K F control misses the threshold by 0.002 meV/cell.

## Ordered work

1. Continue the reference at 36³, then 40³ with identical settings and runtime.
   A passing 32→36 step would complete the existing two-step gate together with
   28→32; 40³ provides further confirmation. If a step fails, retain it and
   conditionally consider 44³/48³ within the approved allocation. Reassess
   oscillations or resource overruns rather than changing the threshold.
   Preserve scalar plus independent SOC z/x/y stages and magnetic-branch checks.
2. Screen basis and radial extent on the existing local CPU at 16³. Use TZP
   radial shifts 0.02, 0.01, 0.005 and, conditionally, 0.0025 Ry. Add an explicitly
   specified basis with more polarization functions and a further enrichment.
   Native shell/orbital/radius output and overlap health must verify each actual
   basis. The wrapper supports presets only through TZP; a validated, hash-bound
   explicit PAO basis interface is required before larger bases run. These
   low-mesh results select candidates; they do not qualify the basis.
3. Confirm basis-size and radial-extent ladders at a dense common mesh, changing
   one control at a time. Require two successive changes within the same E/F
   threshold for each proposed ladder. Recheck the mesh sequence with the
   selected enlarged basis/radial settings: a DZP mesh pass cannot carry over
   to a different basis. Use one executable within each comparison; qualify
   CPU/REMAG agreement for any basis transferred between them.
4. At the candidate final settings, repeat tight SCF, real-space grid and
   occupation-temperature checks, including a combined tighter-settings case
   to expose coupled errors. A zero-electronic-temperature claim requires a
   coupled mesh/temperature sequence, for example 300, 150 and 75 K; a single
   150 K comparison is insufficient. Retain a temperature-qualified scope otherwise.
5. Revisit response convergence only after the reference gates pass. Preserve
   all sealed models and outcomes. Repeat independent volume/shape responses
   at qualified settings before adding new prediction targets.

Commit detailed inputs and acceptance records before new outcomes. Basis
selection and mesh extensions depend on observed failures; this is a campaign
envelope, not a claim that all gates will pass.

## Runtime and cost estimate

The measured four-stage REMAG 32³ reference took 3,559.47 seconds including
runner overhead. Cubic k-point scaling estimates 1.41 hours at 36³, 1.93 hours
at 40³, 2.57 hours at 44³ and 3.34 hours at 48³ before extra SCF/collection
contingency. These are estimates, not benchmarks. Local 16³ DZP and TZP cases
took 12.4 and 23.0 minutes respectively; that CPU ratio is not a measured GPU
basis-scaling factor.

The current allocation is one A100-80GB, eight CPU cores and 96 GiB RAM.
[Modal list prices](https://modal.com/pricing), checked 14 November 2026, imply
about $3.64/hour for those requested resources at standard Function rates:
3600 × (0.000694 + 8 × 0.0000131 + 96 × 0.00000222). Higher actual CPU/memory
usage, provisioning, collection and storage may add cost. No credits or discounts
are assumed. Retain the existing conservative $10/hour reservation rate.

| Stage | Proposed additional reservation envelope |
| --- | ---: |
| 36³/40³ reference and checkpoint | $50 |
| Basis/radial confirmation and mesh rechecks | $150 |
| Final SCF/grid/occupation and combined checks | $50 |
| Total proposed additional cap | $250 |

For planning, roughly 15–25 allocated GPU hours imply approximately $60–100
in billed compute at the listed configuration, with low-mesh basis screening
on the existing local CPU. This is provisional: larger explicit bases, SCF
changes or repeated mesh failures could exceed it. The $250 cap funds a bounded
qualification attempt, not guaranteed qualification. Re-estimate after the first
$50 stage and local basis timings; stop before any approved cap. All-CPU work
could avoid extra cloud charges but would need longer local runtime and its own
dense-mesh timing and runtime controls.

The original ledger retains $96.67 reserved and $3.33 unreserved; actual billing
is unreconciled. Full approval would increase the cumulative authorized ceiling
to $350. Do not reset the ledger or refund reservations automatically. Before
launch, implement and test an append-only authorization extension and bounded
longer jobs: current code caps ledgers at $100 and campaigns at two hours.
Preserve the one-GPU guard, zero automatic retries, receipts and automatic shutdown.
