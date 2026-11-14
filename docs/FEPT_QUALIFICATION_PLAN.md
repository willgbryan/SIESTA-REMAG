# FePt qualification priorities

Planning estimate, 14 November 2026. Mesh and basis checks take priority over
new physics-guided predictions. This proposal does not authorize additional
spending, alter the original $100 ledger, or launch compute.

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
