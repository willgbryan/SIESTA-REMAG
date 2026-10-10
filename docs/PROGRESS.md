# Scope and engineering status

**Source-release cleanup, 9 October:** the initial Git tree includes source,
tests, licensing notices, contributor/run instructions, CPU-only CI, the solver
regression fixture and 26 selected existing evidence files. Raw results, builds,
downloaded upstream, withdrawn Gold–water inputs and generated graphics remain
local and ignored. The public README is concise; the previous README is
preserved in [development notes](DEVELOPMENT_NOTES.md). All 177 recorded v178
implementation files remain unchanged. The local suite passes 116 tests after
updating the evidence-fetch test double for the current SDK interface; the
source distribution passes 115 with the expected missing-upstream patch-test
skip. Source and wheel packaging succeed with license notices included. No
GPU calculation was launched for this cleanup. Historical `results/` links in
these development notes refer to the separate local archive; selected public
records live in [benchmarks/evidence](../benchmarks/evidence/README.md).

**New benchmark requested 9 October:** the predeclared v180 easy-convergence
suite is running sequentially on one A100-80GB. It compares the current v178
binary at default tile settings with screened upstream GPU native/ELSI routes
on Fe54 collinear, Fe16 noncollinear, Fe3C collinear, and a Fe2B full-SOC x/z
workflow. Each has a completed pristine CPU reference (47–89 SCF iterations)
and an existing passing candidate result. Inputs and tolerances are unchanged.
Three fresh repetitions follow solver screening; the full-suite geometric mean
and total-time ratio remain withheld until every workflow passes direct upstream
checks. Failed trials remain in the ledger. The
[frozen plan](../build/easy-magnetic-suite-v180/experiment.json) and
[run identity](../build/easy-magnetic-suite-v180/run-status.json) record scope.

The requested [social-chart bundle](../artifacts/social-charts-20261009/remag-social-charts.zip)
uses Project 14's #FF0860 pink, PP Fraktion Sans and a white editorial layout.
Its three PNG/SVG charts use already-qualified v127/v143/v178 data, with exact
source records and captions included. They do not yet include the v180 suite or
claim a global average across all SIESTA workloads.

The user revised the goal on 8 October: **deliver end-to-end speedup through
GPU-first execution and purpose-built GPU data structures, using upstream
SIESTA directly for numerical validation.** Noncollinear, fully relativistic
magnetic systems remain the primary workload. Equivalent inputs, scientific
agreement and honest end-to-end timings remain required; exhaustive residency
certification and expanding preparation/audit infrastructure do not.

The upstream GPU comparison is complete and independently rechecked. The
focused nonparallel SOC comparison also passes. Current implementation targets
the measured end-to-end bottlenecks, including initialization after the recent
SCF improvements, with integration governed by measured
end-to-end gains and upstream agreement. The v114 NVBit observer is deferred and must not launch under this
priority. Existing evidence remains available, but the older staged audit chains
below are historical preparation, not mandatory delivery gates. Learned SCF
remains deferred while the conventional performance baseline is established.

**Delivery checkpoint, 9 October 2026:** the revised conventional performance
baseline is qualified. Matched upstream-GPU comparisons cover collinear,
noncollinear and full off-site SOC workloads; upstream checks cover requested
observables, nonparallel spins, directional MAE and relaxation. The v178
optimization is integrated and the v179 ternary-magnet regression is complete.
Both allocations are released. Existing ledgers, raw outputs, source identities,
timing repetitions and unsuccessful experiments are retained for the future
writeup; no benchmark or instrumentation was added solely for graphics.

This completes the revised baseline milestone, not every possible extension of
the original program. Fe4N's difficult noncollinear settings remain unqualified;
broader hardware, materials and production convergence studies need their own
measurements. CPU setup/ionic work and requested output exports remain, and
exhaustive zero-transfer/zero-I/O certification is deferred. Learned SCF and the
site/writeup have not been started.

**Latest integrated incremental change:** v178 complex FP64 matrix rotation
improves FePt128 SOC z by **1.06520×** over v175 (203.275 → 190.832 s medians,
three matched pairs on one A100-SXM4-80GB). All six z runs, the large x run and
direct large-cell MAE pass pristine CPU comparisons. It remains opt-in because
the small workload does not establish a gain. This is separate from the
upstream-GPU ratios below; full data and limits are in the v178 entry.

**Latest verified upstream GPU comparison:** the integrated v142 FePt16 full
off-site SOC path measures **58.2746×**, with three-pair medians of
**573.311 s upstream and 9.838 s REMAG**, on one A100-SXM4-80GB. This is the
[v143 direct comparison](../results/upstream-gpu-refresh-v143-20261009T101214Z/qualification.json).
It measures the integrated v142 implementation directly; separately measured
incremental gains are not multiplied into the result. The earlier
[v117 18.4347× result](../results/upstream-gpu-refresh-v117-20261008T180328Z/qualification.json)
remains a historical comparison at its recorded stopping criteria.
The retained v126, v150 and v163 FePt128 z results pass the completed
pristine CPU reference (v133). The measured capacity change and Hermitian
rotation are integrated, qualifying the current 128-atom, 2,304-state SOC z
case with its recorded settings. The later v167 direct comparison also qualifies
current full-SOC x/z and larger-cell MAE: 0.138045 eV upstream versus 0.138046 eV
REMAG, a 1 micro-eV difference within the unchanged 5 micro-eV gate. The v172
matched larger-cell z comparison now passes direct local science/export checks:
**10,060.055 s upstream GPU versus 233.256 s REMAG, a provisional 43.1289×**
single-pair speedup. This is separate from the repeated small-cell result.

**Completed comparison refresh (v143):**
`upstream-gpu-refresh-v143-20261009T101214Z` compared the integrated v142
implementation directly with the established native ELPA two-stage/block-64/
one-thread upstream configuration. Both used the current FePt16 x/z inputs at
DM 1e-9 / H 1e-8 eV stopping criteria and the retained pristine CPU references.
All 17 calculations, six direct scientific/export comparisons, 11 MAE checks
and both upstream GPU-execution diagnostics pass. Maximum direct saved-DM
error is **1.785e-8** against the unchanged 1e-7 gate. Direct MAE is
**0.056828 eV** for every repeated contender at printed precision, within the
unchanged 5e-6 eV gate; final moment directions pass the existing checks.

| Fresh x/z pair | Upstream GPU (s) | REMAG (s) |
| --- | ---: | ---: |
| 1 | 578.316413 | 10.054099 |
| 2 | 573.311136 | 9.744279 |
| 3 | 572.345336 | 9.838099 |
| Median | 573.311136 | 9.838099 |

The pair median is calculated from complete pair totals. Upstream uses 106 z
and 82 x iterations throughout; REMAG uses 82/85/89 z and 81/79/80 x iterations.
This is end-to-end time to equivalent converged solutions, including process
startup, initialization, final observables and requested exports; build,
provisioning, evidence transfer and instrumented diagnostic timings are excluded.
Sampled GPU memory peaks are 557 MiB upstream and 2,545 MiB REMAG. The v117
stopping criteria were looser; its historical timings are not reused in this
ratio. This SZP/100 Ry magnetic regression is not a production convergence
study or a performance claim for other materials or sizes.

The [ledger](../results/upstream-gpu-refresh-v143-20261009T101214Z/ledger.json),
[raw timing summary](../results/upstream-gpu-refresh-v143-20261009T101214Z/performance.json),
source archive/identities and existing stage profile retain material for the
future writeup; no runs or instrumentation were added solely for visuals.
At completion, all 177 production source hashes matched the v142 integration
record. App `ap-TiZSC2eL0KPBhJAsTTYGF7` is stopped with no containers;
[allocation release](../results/upstream-gpu-refresh-v143-20261009T101214Z/allocation-release.json)
was verified at 10:54:24 UTC on 9 October. Function call
`fc-01M4G2CAJZA3QPH6V0JWJ8457V` and launcher log
`/tmp/remag-upstream-gpu-refresh-v143.log` identify the completed run.

## Larger pristine SOC reference complete (v133)

`soc-local-reference-v133-20261009T071147Z` completed at **11:48:47 UTC on
9 October**, converging in **107 SCF iterations** and 16,618.628 s on the local
CPU. The pinned pristine executable and unchanged FePt128 z input satisfy the
DM 1e-9 / H 1e-8 eV protocol. All existing scientific and raw export comparisons
against the retained v126 GPU result pass. Maximum saved-DM error is
**9.4811e-9** against the unchanged 1e-7 gate; eigenvalue error is 7.2e-6 eV,
RHO error 5.9605e-8, and HSX-H error 3.3073e-9 in the existing export units.
Energy, forces, stress and magnetic moments agree at their reported precision.
The [completed reference summary](../results/soc-local-reference-v133-20261009T071147Z/summary.json)
and [direct comparisons](../results/soc-local-reference-v133-20261009T071147Z/comparison.json)
establish upstream agreement for that retained 128-atom, 2,304-state full-SOC
z calculation. The CPU and GPU hardware differ: **no timing ratio is claimed**.
This does not establish the x direction, larger-cell MAE, or a matched upstream
GPU performance comparison.

The current run `soc-current-profile-v150-20261009T114825Z` also completes
and passes a **direct comparison with v133**, including the requested raw
RHO/DRHO/HSX/DM arrays. It takes 117 SCF iterations. Maximum saved-DM error is
**8.0717e-9** against the unchanged 1e-7 gate; RHO error is 1.1921e-7 and
eigenvalue error is 7.2e-6 eV. Energy, forces, stress and magnetic moments agree
at reported precision. Its output explicitly reports off-site SOC and fully
relativistic generation for both pseudopotentials.

The exact measured two-line capacity change is integrated into
`native/scf_cycle.cu`: up to 2,048 spatial orbitals and 4,096 combined spinor
states. **The largest directly qualified case is 2,304 states**, not every
material or dimension up to that allocation ceiling. Numerical algorithms and
all inputs/tolerances are unchanged. Smaller supported shapes follow precisely
the same path. [Direct upstream comparison](../results/soc-current-profile-v150-20261009T114825Z/upstream-comparison.json)
and [measured integration](../results/soc-current-profile-v150-20261009T114825Z/production-integration.json)
record the raw results, exact source and all 177 production hashes.

The instrumented run takes **319.524 s** overall, with **308.300 s** in the
device cycle. Assembly/eigensolves take **277.958 s (90.16% of the cycle)**;
density/normalization takes 21.406 s, grid/Hartree/XC 4.708 s and Hamiltonian
construction 3.024 s. Device-cycle workspace is 9,132,023,384 bytes. These
[stage measurements](../results/soc-current-profile-v150-20261009T114825Z/stage-profile.json)
identify the larger-scale bottleneck; they are not an untraced speedup or a
comparison with the local CPU's wall time. Larger x/MAE and matched upstream-GPU
timings remain outstanding. The next performance work should address whole
matrix movement and rotations in the block eigensolver at this size.

App `ap-D97g3qMSfXrtAlijU5xroY` is stopped with no containers;
[release](../results/soc-current-profile-v150-20261009T114825Z/allocation-release.json)
was confirmed at 11:54:17 UTC. The local CPU reference is also complete. Older
entries below describing v133 as running refer to historical measurement times.

## Tiled two-sided matrix rotation (v151 screen; integrated after v152)

`fused-matrix-rotation-v151-20261009T121114Z` tests a staged eigensolver change
against the exact integrated v150 executable/library on one A100-SXM4-80GB.
Each 256-thread CTA computes a 32-by-32 matrix tile, combining permutation,
right rotation and conjugate left rotation. Two padded shared tiles use
33,792 bytes for complex FP64; the full permutation and intermediate product
are no longer written to global memory. Local eigensolves, vector rotation,
warm-basis reuse, stopping rules and sorting are unchanged. The candidate is
opt-in with `REMAG_BLOCK_FUSED_MATRIX=1` and requires width 32. This screen
was staged in `build/fused-matrix-rotation-v151`; the exact measured source
was subsequently integrated after the successful v152 qualification below.

All 36 existing generalized solver checks, both changing-H trajectories,
four small FePt calculations and both small directional-energy checks pass.
The small x/z pair takes **11.0193 s accepted versus 10.6176 s candidate**.
The larger 128-atom, 2,304-state SOC z screen measures:

| Implementation | Wall time (s) | SCF iterations | Device cycle per iteration (s) |
| --- | ---: | ---: | ---: |
| Integrated v150 | 294.844379 | 100 | 2.803374 |
| Tiled matrix rotation | 278.982729 | 103 | 2.563429 |

This is a **1.05686× provisional single-pair ratio**, with matched untraced
settings. Repeated performance and production integration were not yet qualified
at this screen; the subsequent v152 result below governs integration.
Both runs pass a direct local comparison with the completed pristine v133 CPU
reference, including forces, stress, moments, eigenvalues and raw RHO/DRHO/HSX/DM.
Maximum saved-DM errors are 9.4999e-9 accepted and 9.4003e-9 candidate against
the unchanged 1e-7 gate. Both use 9,132,023,264 bytes of device-cycle workspace.
Larger-cell x/MAE and a direct upstream-GPU timing remain unqualified.

The [ledger](../results/fused-matrix-rotation-v151-20261009T121114Z/ledger.json),
[screen summary](../results/fused-matrix-rotation-v151-20261009T121114Z/summary.json),
[direct upstream comparisons](../results/fused-matrix-rotation-v151-20261009T121114Z/upstream-comparison.json),
raw outputs and measured source archive retain the existing work for the future
writeup. Automatic approval rejected uploading the local CPU outputs to Modal;
no upload occurred. GPU outputs were downloaded and compared locally instead.
At screen completion all 177 production source hashes still matched v150.
The subsequent v152 repetitions and existing magnetic/relaxation regressions
qualified opt-in integration; the original screen evidence remains unchanged.

App `ap-XaMckNPOYNxqhqMlWEiA1Q` is stopped with no containers;
[allocation release](../results/fused-matrix-rotation-v151-20261009T121114Z/allocation-release.json)
was confirmed at **12:24:15 UTC on 9 October**. Function call
`fc-01M4G967JMA97E9S4F8NXWXWW7` and launcher log
`/tmp/remag-fused-matrix-rotation-v151.log` identify the completed screen.

## Repeated tiled matrix rotation (v152, integrated opt-in)

`fused-matrix-qualification-v152-20261009T123001Z` reused the exact v151
candidate and v150 control binaries on one A100-SXM4-80GB (driver 610.57.04).
Three fresh small x/z pairs and three large z pairs alternated contender order.
All 18 timing calculations pass the existing science checks. The four existing
regressions also pass: nonparallel full SOC, Fe54 collinear, Fe16 noncollinear
and Fe2 relaxation. The z/x/z geometry-reuse sweep passes in 11.5673 seconds,
with 80/73/75 iterations. Nine small-pair MAE checks and two sweep MAE checks
pass. The 36 generalized and two changing-H solver checks from v151 were reused
because the binary hashes were unchanged, rather than rerun.

| FePt128 SOC z repeat | Accepted v150 (s) | Tiled candidate (s) | SCF iterations, both |
| --- | ---: | ---: | ---: |
| 1 | 322.320822 | 302.076487 | 114 |
| 2 | 317.580806 | 295.669121 | 110 |
| 3 | 315.906098 | 292.806042 | 109 |
| Median | 317.580806 | 295.669121 | — |

All three large pairs are faster: **1.07411× incremental speedup**, with means
318.602575 versus 296.850550 seconds. Device-cycle totals are retained in each
result: accepted 308.475471/304.739998/302.655984 seconds, candidate
288.697676/282.200985/279.430769 seconds. This is end-to-end timing including
initialization and requested final outputs, with profiling disabled; build,
provisioning and evidence downloads are excluded. It compares REMAG versions,
not upstream GPU SIESTA, and must not be multiplied into the v143 result.

All six large calculations also pass a direct local comparison with pristine
upstream CPU v133, including convergence, forces, stress, moments, eigenvalues
and raw RHO/DRHO/HSX/DM. Maximum saved-DM error is **8.0837e-9** against the
unchanged 1e-7 gate; eigenvalue error is 7.2e-6 eV, RHO error 1.1921e-7 and
HSX-H error 3.3037e-9 in the existing export units. Energy, forces, stress and
moments agree at reported precision. CPU and GPU hardware differ, so these
checks establish scientific agreement only.

**The small case has no gain:** FePt16 x/z pair medians are **9.547632 seconds
accepted and 9.750268 seconds candidate**, with only one of three candidate
pairs faster. Individual accepted totals are 9.536489/9.874298/9.547632 seconds;
candidate totals are 9.851208/9.750268/9.642698 seconds. Both means and medians
are retained. The kernel is therefore integrated as an explicit opt-in through
`--fused-matrix-rotation` / `REMAG_BLOCK_FUSED_MATRIX=1`; the default remains off.
The ordinary launcher requires width 32, device SCF and vector cache. No
untested size threshold is introduced. The exact measured native source is
promoted, with only non-numerical launcher and result-metadata wiring added
locally; Python compilation and CLI help checks pass. Upstream remains pristine
at the pinned commit.

The [ledger](../results/fused-matrix-qualification-v152-20261009T123001Z/ledger.json),
[complete repeated summary](../results/fused-matrix-qualification-v152-20261009T123001Z/summary.json),
[direct upstream checks](../results/fused-matrix-qualification-v152-20261009T123001Z/upstream-comparison.json)
and [measured source/integration](../results/fused-matrix-qualification-v152-20261009T123001Z/production-integration.json)
retain raw repetitions, settings, source identities and scope for later figures.
The original remote summary is preserved separately; no measurements were added
solely for graphics. The CPU reference stayed local and returned GPU arrays
were compared locally.

App `ap-C9KDbhA78nCNnD2LaAoyef` is stopped with no containers;
[allocation release](../results/fused-matrix-qualification-v152-20261009T123001Z/allocation-release.json)
was confirmed at **13:06:44 UTC on 9 October**. Call
`fc-01M4GA8KHEBYX01GD0AN8KZQJR` and launcher log
`/tmp/remag-fused-matrix-qualification-v152.log` identify the completed run.
Large-cell x/MAE and matched upstream GPU performance remain unqualified.

## Large x output and Fe4N failure (v155, completed)

`magnetic-coverage-v155-20261009T131951Z` ran the existing staged inputs
sequentially on one A100-80GB, reusing the exact v152 numerical binaries.
The large FePt128 full-SOC x calculation completed in **300.641298 seconds**
and **112 SCF iterations**, retaining a final moment of **[186.13288, 0, 0] muB**.
It used 283.459768 seconds of device-cycle time and 9,132,023,264 bytes of
workspace. The x CPU reference is still running, so direct x science/export
agreement and large-cell MAE remain **unqualified**. Already verified v152 z
outputs will supply the other GPU direction; z is not rerun.

Both Fe4N40 candidates returned an unconverged device-cycle state, then aborted
when final electronic export correctly refused it. Collinear elapsed time was
16.481992 seconds and noncollinear 30.354337 seconds. These are **failed-run
timings, not performance results**. Deferred SCF reporting leaves the parsed
iteration count at zero and the FDF log unflushed; source tracing confirms
that the device cycle executed before the final-export rejection. The error
must not be presented as an initialization failure. The already-retained
terminal scalar history needs inspection to identify the limiting residual.
Both retain `REMAG_BLOCK_FUSED_MATRIX=0`, consistent with the smaller-case
performance policy. The collinear CPU reference converged at 166 iterations;
the noncollinear CPU reference is still running. No Fe4N capability is promoted.

Only staged input files and benchmark code were uploaded. GPU raw arrays were
returned with the existing downloader, and all CPU reference outputs stay
local. The [ledger](../results/magnetic-coverage-v155-20261009T131951Z/ledger.json),
[local comparison status](../results/magnetic-coverage-v155-20261009T131951Z/upstream-comparison.json),
[raw recovery](../results/magnetic-coverage-v155-20261009T131951Z/raw-recovery.json)
and source archive retain completed and failed work. CPU and GPU hardware
differ, so no performance ratio is inferred.

App `ap-VBkD4GQpx0a4ouFkJ40TLe` is stopped with no containers;
[release](../results/magnetic-coverage-v155-20261009T131951Z/allocation-release.json)
was confirmed at **13:26:18 UTC on 9 October**. Call
`fc-01M4GD3VF4SZDB80NQ0YKS3BBM`, image `im-0HXuHCPvBOHcWjkDc8mv3s` and log
`/tmp/remag-magnetic-coverage-v155.log` identify this completed run.

## Fe4N convergence diagnosis (v156/v157, completed)

The v156 same-input collinear repeat exposes the last eight scalar records
already retained after the device cycle. No numerical kernel, stopping rule,
SCF transfer or successful-run logging changed; production remains untouched.
The cycle reaches **200 iterations**, with final density residual
**0.0720177** (target 1e-9) and Hamiltonian residual **0.0124522 Ry**
(target 1e-8 eV). This is a substantial nonconvergence discrepancy, not a
near-threshold miss. The upstream CPU reference converges in 166 iterations.
The [terminal residuals](../results/fe4n-nonconvergence-v156-20261009T133314Z/nonconvergence.json)
retain the actual evidence. The diagnostic app `ap-uL5B9aGlNA5UxFB0SKH6fr`
[released](../results/fe4n-nonconvergence-v156-20261009T133314Z/allocation-release.json)
at 13:35:37 UTC. Its measured final image is `im-tuBDpxasrAeZ1SHrH6WocY`.

The isolated check selects the existing cuSOLVER local-eigensolver fallback
(`REMAG_BLOCK_SHARED_JACOBI=0`) on the identical Fe4N collinear input. All other
runtime options and stopping criteria stay unchanged. This is a direct
investigation of observed nonconvergence; no speedup is claimed. It uses the
same one-GPU launcher, after confirmed release, with a 300-second function
limit and 240-second material limit. Log:
`/tmp/remag-fe4n-solver-isolation-v157-retry.log`.

The corrected collinear fallback run
`fe4n-solver-isolation-v157-20261009T133732Z` converged in **196 iterations**
and passes a [direct pristine CPU comparison](../results/fe4n-solver-isolation-v157-20261009T133732Z/upstream-comparison.json).
Energy, forces, stress, moments, densities, eigenvalues and saved matrices all
pass the existing gates; maximum saved-DM error is **1.3655e-8** against 1e-7.
Its 14.987948-second wall time is retained as a standalone measurement only.
The CPU and GPU machines differ, so no speedup is claimed. This qualifies the
**cuSOLVER local-solver route** for the collinear Fe4N fixture; the optimized
shared local solver remains unqualified on this material. The successful
fallback narrows the discrepancy to the local eigensolver or its effect on
the SCF trajectory, without yet proving a kernel arithmetic bug.

App `ap-aSSWZfoiBVdiXPXCHHUW0H` is stopped and its
[allocation released](../results/fe4n-solver-isolation-v157-20261009T133732Z/allocation-release.json)
at 13:38:59 UTC. The subsequent noncollinear fallback run
`fe4n-solver-isolation-v157-20261009T134008Z` still failed to converge by
200 iterations: final dD **0.00785484** and dH **0.00555145 Ry**. Its
[terminal residuals](../results/fe4n-solver-isolation-v157-20261009T134008Z/nonconvergence.json)
show that local solver choice alone does not explain both spin cases.
Noncollinear Fe4N remains unqualified. The next investigation should inspect
SCF numerical equivalence and transverse-spin handling; no specific arithmetic
bug is established yet. Its failed elapsed time is not a performance result.
The source/settings, failure history and successful collinear raw outputs are
retained for the eventual writeup.

Noncollinear app `ap-4yiVoAqwOSYljh3xYPsI2W` is stopped with no containers;
[release](../results/fe4n-solver-isolation-v157-20261009T134008Z/allocation-release.json)
was confirmed at 13:41:03 UTC. Log:
`/tmp/remag-fe4n-solver-isolation-v157-noncollinear.log`. The sole remaining
live reference is the large FePt x CPU calculation (session 19112). All 177
production source hashes still match the accepted v152 integration; failure
reporting changes remain staged.

Two setup failures are retained separately: the first v156 invocation selected
an intermediate image layer without a required runner source file; the first
v157 invocation had an incorrectly rewritten reference-run prefix. Neither
started a material calculation. Both allocations were confirmed released
before corrected retries. The v155 image imports the inherited runner from
`/opt/remag/siesta_remag`; its newly copied runner at `/opt/remag/src` was unused.
Its per-case matrix mode remains recorded as an invocation setting, while the
missing field in each result's runtime telemetry is left unknown. v156 fixes
the runner destination. These harness corrections do not change numerical
binaries in the v155 measurements.

## Fe4N cold-solver isolation (v158, completed)

Static inspection of sparse spin assembly, density contraction, PBE spin
rotation and the all-spin Pulay path has not established a sign/layout error.
`fe4n-cold-solver-v158-20261009T134700Z` therefore checks the same noncollinear
Fe4N input with the existing cuSOLVER local solver, disabling only eigenbasis
warm starts (`REMAG_SCF_WARM_START=0`). Physics, SCF tolerances and the
200-iteration input limit remain unchanged. The host diagnostic additionally
prints the first eight already-downloaded scalar records after the cycle,
allowing comparison with the existing CPU log. No extra SCF copies or disk
writes are introduced. Diagnostic source changes remain staged.

Disabling warm starts did not restore convergence: the cycle reached 200
iterations, with final dD **0.000912621** and dH **0.000232250 Ry**. Its elapsed
time is a failed-run measurement only. The first eight total energies differ
from the printed CPU trajectory by at most **1.39e-6 eV**; density residual
differences are below **4.31e-7**, within the CPU log's six-decimal rounding.
These early scalars make a gross initial-state discrepancy less likely but do
not establish whole-trajectory equivalence or the cause of nonconvergence.
The [trajectory comparison](../results/fe4n-cold-solver-v158-20261009T134700Z/trajectory-comparison.json)
uses the CPU build's CODATA2018 unit convention. All raw logs, source and
settings are retained. No failed calculation is included in a speedup claim.

App `ap-gjfnGFsInL3oCkICYW0ZJ2`; call `fc-01M4GENJ8AA667JF47F8Y6BF33`;
final image `im-zjcIGIVBSF44iqJap9ok7r`; log
`/tmp/remag-fe4n-cold-solver-v158.log`. The app is stopped with no containers;
[release](../results/fe4n-cold-solver-v158-20261009T134700Z/allocation-release.json)
was confirmed at **13:48:59 UTC on 9 October**. Production is unchanged.
The next direct check uses pristine upstream's native ELPA GPU route on the
identical NC input to investigate platform/solver sensitivity. It is a bounded
convergence diagnosis, without new instrumentation or a performance claim.

## Fe4N upstream GPU diagnosis (v159, timed out)

`fe4n-upstream-gpu-v159-20261009T135727Z` runs the unchanged noncollinear Fe4N
input with the pinned pristine GPU executable, using native ELPA two-stage,
block size 64 and eight one-thread MPI ranks sharing **one GPU**. This is a
single convergence diagnosis with a 1,800-second case limit and 1,920-second
function limit. The original 200-iteration maximum, tight stopping criteria,
physics and requested outputs are unchanged. Existing helpers record the
effective solver settings; no new GPU activity instrumentation is added, so
actual offload in this invocation remains unverified and no speedup is claimed.
Both v154 CPU spin
cases reach the same printed total energy, -27,332.526917 eV, and rotated
78.94518-muB moments; their different SCF trajectories alone are not a bug.

App `ap-zChWajERdiwQO0a6pwGsq9`; call `fc-01M4GF8PJQF2MPMMAPS9R0T9AV`;
final image `im-r4zyhWhIuLtPxNsYfIO1e6`; log
`/tmp/remag-fe4n-upstream-gpu-v159.log`. The previous GPU allocation was
confirmed released and the normal account-wide guard passed before launch.
The run **timed out at 171 SCF iterations after 1,801.382126 seconds**, before
the original 200-iteration limit. Its last printed residuals are dD **0.032602**
and dH **0.111474 eV**, far above the unchanged stopping criteria. Effective
solver, MPI/OpenMP and convergence settings pass the existing checks. No
converged final energy or electronic exports are available for qualification;
the elapsed time is not a speedup measurement. This does not prove failure at
the full 200-iteration cap, isolate a REMAG arithmetic bug, or qualify NC Fe4N.

The [failure analysis](../results/fe4n-upstream-gpu-v159-20261009T135727Z/failure-analysis.json)
retains the last eight records and early CPU-trajectory differences; the raw
logs retain all 171 iterations. The original remote summary, settings, source,
ledger and failure remain available for the future writeup. The app is stopped
with no containers, and [release](../results/fe4n-upstream-gpu-v159-20261009T135727Z/allocation-release.json)
was confirmed at **14:28:11 UTC on 9 October**. No automatic retry or mixing/
stopping-rule changes are planned. NC Fe4N remains an explicit coverage limit;
the next GPU work returns to the measured full-SOC eigensolver bottleneck.

## Hermitian tile reuse (v160, large solver failure)

`build/hermitian-matrix-rotation-v160` contains a small extension of the accepted
v152 matrix rotation. Each lower off-diagonal tile computes its two-sided
rotation once; the conjugate partner is written through the same padded shared
tile so both global stores remain coalesced. Diagonal-tile arithmetic, vector
rotation, local eigensolvers, FP64, workspace sizes and stopping criteria are
unchanged. This targets duplicate work within the previously measured dominant
assembly/eigensolver stage. It can change roundoff and the SCF trajectory;
neither correctness nor a performance gain is established yet.
The launch enumerates the packed lower-triangular tile count directly. A local
integer-ownership check covers every batch size 1 through 128 (up to 4,096
states), with exactly one writer for each tile including conjugate partners;
this is an index check, not CUDA arithmetic validation.

The existing v151 screen is reused with the exact v152 binary as control, and
both contenders enable tiled matrix rotation. It retains the existing real/
complex generalized and seeded checks, small x/z SOC/MAE comparisons and one
large z pair, followed by direct local comparison with pristine v133 outputs.
There is no new test framework or instrumentation. All three CUDA targets,
36 existing generalized checks and two changing-H trajectories pass. Both small
x/z contenders pass scientific/export checks and direct MAE. The large control
completes in **282.212 s / 101 iterations** and passes a direct local pristine
CPU comparison, including saved-DM error **9.5077e-9** against the unchanged
1e-7 gate. The large candidate aborts in **19.541 s**, before any accepted SCF
iteration. That failed elapsed time is not a performance result, and no
performance or production integration is qualified.

The captured [failure](../results/hermitian-matrix-rotation-v160-20261009T143016Z/failure-analysis.json)
is **global eigensolver sweep exhaustion**: the first 2,304-state problem uses
all 20 allowed sweeps, ending at relative off-diagonal norm **1.5748e-9** versus
the unchanged **2e-13** requirement. Every local eigensolve succeeded. The
retained history is still decreasing; this is distinct from the Fe4N SCF
nonconvergence. Failed H/S inputs and the solver history were already captured
by the existing error path. Small successes do not qualify the larger case.
All 177 production hashes still match v152; the candidate is not integrated.

Run `hermitian-matrix-rotation-v160-20261009T143016Z` started through the normal
single-allocation guard after confirmed v159 release. App
`ap-cFPU62xY7h6roMBHMuU5DT`; call `fc-01M4GH4T1VF8EJN047HCHKDT9K`; final image
`im-aN9qhq5HsDY5YM8Y6cukVa`; log `/tmp/remag-hermitian-matrix-v160.log`.
The app is stopped with no containers; [release](../results/hermitian-matrix-rotation-v160-20261009T143016Z/allocation-release.json)
was confirmed at **14:39:17 UTC on 9 October**. The returned raw arrays, direct
control comparison, original remote summary and failure remain retained.

## Captured Hermitian solver replay (v161, completed)

The first attempt `hermitian-solver-replay-v161-20261009T144527Z` completed its
LAPACK reference, then failed in the diagnostic wrapper: its adapter rejected
2,304 states against an old 2,048-state capacity guard before executing the
solver, and the reporter incorrectly tried to consume unavailable arrays.
The failed attempt and source are preserved. Its allocation released at
14:50:37 UTC. Only the staged adapter/reporting were corrected; the numerical
solver was unchanged, and the hash-verified completed reference was reused.

The corrected run `hermitian-solver-replay-v161-20261009T145306Z` replays the
exact captured H/S, retaining FP64 and the **2e-13** stopping tolerance:

| Route | Maximum sweeps | Actual sweeps | Result |
| --- | ---: | ---: | --- |
| Existing unfused rotation | 20 | 19 | Pass |
| Hermitian tile reuse | 20 | 20 | Sweep exhaustion |
| Hermitian tile reuse | 32 | 22 | Pass |

The 32-sweep case reaches relative off-diagonal **7.256e-16**. Its relative
residual is **3.295e-14**, overlap-orthogonality error **1.058e-11**, and maximum
LAPACK eigenvalue error **1.213e-11**; all pass the existing 1e-10 / 1e-9 / 1e-10
gates. The [ledger](../results/hermitian-solver-replay-v161-20261009T145306Z/ledger.json)
retains every sweep residual and native status. This diagnoses the observed
failure but does not qualify a full material calculation or performance.
Production remains unchanged. The next staged material screen gives the same
kernel 32-sweep headroom without changing accuracy or SCF stopping criteria.

App `ap-q6rZ0qHGqFq7XkFYc3HbvX`; call `fc-01M4GJEKD1C4N5882PJK3DQ02M`;
final image `im-uF5ytCKB9Cn8TsGb0YpDDg`; log
`/tmp/remag-hermitian-replay-v161-retry.log`. The app is stopped with no
containers; [release](../results/hermitian-solver-replay-v161-20261009T145306Z/allocation-release.json)
was confirmed at **14:58:09 UTC on 9 October**. Captured inputs, reference,
source archive, full solver arrays and both attempts remain retained.

## Hermitian rotation with sweep headroom (v162, successful screen)

`hermitian-headroom-v162-20261009T150025Z` tests the full FePt128 SOC z material
with the exact v160 Hermitian tile kernel and one additional staged line:
`BlockJacobiPlan` permits **32 sweeps instead of 20**, still at **2e-13**.
The v161 captured matrix required 22 sweeps. SCF inputs, 200-iteration maximum,
accuracy gates and requested scientific outputs remain unchanged. All 177
production hashes still match accepted v152; no candidate is promoted.
The existing v160 generalized/seeded and small SOC x/z/MAE checks are reused.

Both calculations complete on one A100-SXM4-80GB, driver 580.95.05:

| Contender | End-to-end seconds | SCF iterations | Device-cycle seconds | Cycle seconds/iteration |
| --- | ---: | ---: | ---: | ---: |
| Accepted v152 tiled rotation | 287.909043 | 106 | 275.676764 | 2.600724 |
| Hermitian tile reuse, 32-sweep headroom | 218.718070 | 101 | 206.277655 | 2.042353 |

This is a **1.31635× provisional incremental single-pair speedup** over v152,
with candidate first and accepted second. Both enable tiled matrix rotation;
tracing/profiling are off. It includes initialization, SCF, final observables
and requested exports, excluding builds, provisioning and evidence retrieval.
The cycle improvement also persists per iteration. Both use 9,132,023,264 bytes
of cycle workspace. Repeated performance and production integration remain
unqualified; this is not a comparison with upstream GPU SIESTA.

Both pass [direct pristine v133 CPU and pairwise comparisons](../results/hermitian-headroom-v162-20261009T150025Z/upstream-comparison.json),
including convergence settings, forces, stress, moments, eigenvalues and raw
RHO/DRHO/HSX/DM. Maximum saved-DM errors are **9.4977e-9 accepted** and
**9.4772e-9 candidate**, below the unchanged 1e-7 gate. Energy, forces, stress
and moments agree at reported precision; eigenvalue error is 7.2e-6 eV and
RHO error 5.9605e-8. The CPU/GPU hardware differ; CPU timings are not used for
a speedup. Large-cell x/MAE remains pending its separate CPU reference.

The [ledger](../results/hermitian-headroom-v162-20261009T150025Z/ledger.json),
[summary](../results/hermitian-headroom-v162-20261009T150025Z/summary.json),
raw outputs/hashes and exact source archive retain this work for later figures.
App `ap-czpqetUtWRJgqagzVM4Uzq`; call `fc-01M4GJW0NRN50PMAW4TH06TXX7`;
final image `im-oVUTkjrBe7bqZsLL9D5UHL`; log
`/tmp/remag-hermitian-headroom-v162.log`. The app is stopped with no containers;
[release](../results/hermitian-headroom-v162-20261009T150025Z/allocation-release.json)
was confirmed at **15:10:02 UTC on 9 October**.

## Repeated Hermitian rotation qualification (v163, integrated opt-in)

`hermitian-qualification-v163-20261009T151110Z` reuses the exact v162 candidate
and accepted v152 binaries on one **A100-80GB PCIe, driver 580.95.05**. Both
contenders enable tiled rotation. All 18 timing calculations, four existing
magnetic/relaxation regressions, and the three-direction z/x/z sweep pass.
The unchanged Hermitian kernel reuses its 36 generalized and two seeded checks;
v161 separately diagnosed the required 32-sweep headroom at fixed 2e-13 accuracy.

| FePt128 SOC z repeat | Accepted v152 (s) | Hermitian candidate (s) | SCF accepted / candidate |
| --- | ---: | ---: | ---: |
| 1 | 326.893128 | 268.742337 | 108 / 117 |
| 2 | 334.612714 | 268.647429 | 112 / 116 |
| 3 | 331.808198 | 264.676656 | 111 / 114 |
| Median | 331.808198 | 268.647429 | — |

All three large pairs are faster: **1.2351065× qualified incremental speedup**
over v152, with means **331.104680 / 267.355474 seconds**. Contender order
alternates accepted-first, candidate-first, accepted-first. Timings include
initialization, SCF, final observables and requested exports; builds, provisioning
and evidence retrieval are excluded. Tracing and stage profiling are disabled.
This is a matched REMAG-version comparison, not an upstream-GPU ratio, and must
not be multiplied into v143's 58.27× measurement on a different allocation.

All six large calculations pass [direct pristine v133 CPU comparison](../results/hermitian-qualification-v163-20261009T151110Z/upstream-comparison.json),
including effective stopping criteria, forces, stress, moments, eigenvalues and
raw RHO/DRHO/HSX/DM. Maximum saved-DM error is **8.0868e-9** against the existing
1e-7 gate; RHO error is 1.1921e-7 and eigenvalue error 7.2e-6 eV. Forces, stress,
energies and magnetic moments agree at reported precision. Every early-returned
raw array/result hash was reconciled with the final artifact before qualification.
The CPU runs are numerical references only; their different hardware does not
supply a timing ratio. Large x/MAE and upstream-GPU timing remain outstanding.

The small FePt16 x/z case has **no gain**: pair medians **8.195307 s accepted**
and **8.294830 s candidate**, means 8.195961 / 8.364228 s, and only one of
three candidate pairs faster. Accepted totals are 8.190709 / 8.201866 / 8.195307 s;
candidate totals are 8.602501 / 8.195354 / 8.294830 s. All nine small-pair MAE
checks pass. The z/x/z sweep passes in 8.6457 seconds with 80/73/79 iterations.
The existing nonparallel full-SOC, Fe54 collinear, Fe16 noncollinear and Fe2
relaxation regressions pass. These remain regression settings, not production
basis, mesh or k-point convergence studies.

The exact measured `native/block_jacobi.cu` and `native/scf_cycle.cu` are now
integrated. The former computes each lower off-diagonal tile once and fills its
conjugate partner with coalesced writes through existing shared memory. The
latter changes only the caller maximum from 20 to 32 sweeps; tolerance and early
exit remain **2e-13**, and SCF/material settings are unchanged. No persistent
workspace is added. The rotation remains opt-in through the existing
`--fused-matrix-rotation` / `REMAG_BLOCK_FUSED_MATRIX=1`; its default stays off.
All 177 production hashes are recorded, with exactly these two changed files.
The measured image's unchanged diagnostic-only `check_seeded.cu` and separate
CMake target remain outside the production source list; that existing exception
was checked explicitly. Pinned upstream remains clean and unmodified.

[Integration and complete timing data](../results/hermitian-qualification-v163-20261009T151110Z/production-integration.json),
[ledger](../results/hermitian-qualification-v163-20261009T151110Z/ledger.json),
source archive, 125 requested raw output files and hashes preserve the work for
the future writeup. Original remote and partial local reports are retained.
App `ap-BWI6kKYFfYSmNnHm7n89OP`; call `fc-01M4GKFPDCNTGM21FT1H5DD59M`;
final image `im-cpolkVzFvxwCxzbugVxIGn`; numerical image
`im-oVUTkjrBe7bqZsLL9D5UHL`; log `/tmp/remag-hermitian-qualification-v163.log`.
The app is stopped with no containers; [release](../results/hermitian-qualification-v163-20261009T151110Z/allocation-release.json)
was confirmed at **15:45:12 UTC on 9 October**.

## Large-case solver profile (v164, completed)

`large-solver-profile-v164-20261009T155157Z` profiles the integrated v163
solver on FePt128 SOC z using the existing device-clock method, with the two
old matrix-rotation boundaries combined around the fused kernel (12 stages).
It completes **111 SCF iterations** in **220.807964 s**, with a **209.635400 s**
device cycle on an A100-SXM4-80GB, driver 580.95.05. Instrumentation and different
hardware exclude this timing from a comparison with the untraced PCIe v163 run.

The [direct pristine CPU comparison](../results/large-solver-profile-v164-20261009T155157Z/upstream-comparison.json)
passes all scientific, export and convergence gates; saved-DM error is
**8.0972e-9** against 1e-7. All eight lane counters reconcile with **888 solves,
2,972 block sweeps and 424,996 paired rounds**. Coarse assembly/eigensolves take
**180.901562 s (86.29% of the cycle)**; density/normalization takes 20.272786 s.

| Solver stage | Share of summed overlapping lane intervals |
| --- | ---: |
| Local eigensolves | 29.94% |
| Vector rotation | 24.71% |
| Hermitian matrix rotation | 16.64% |
| Pair permutation/gather | 13.45% |
| Warm-basis QR | 6.36% |
| Generalized transformation | 4.51% |
| Overlap factorization | 0.67% |

These elapsed lane intervals include contention and marker overhead and are
**not wall-time partitions**. Warmup is excluded, and no numerical decision uses
the counters. [Raw lane data](../results/large-solver-profile-v164-20261009T155157Z/stage-profile.json)
and [analysis](../results/large-solver-profile-v164-20261009T155157Z/analysis.json)
retain every stage/call count for future figures. The low overlap-factor share
keeps the earlier cache experiment deferred. Vector rotation is a material cost
at this size and motivates the next focused row-tile experiment.

Production retains all 177 v163 hashes; timing markers are not promoted.
App `ap-qdAzLFMXNUKC4PbQdDlbO6`; call `fc-01M4GNTBZH97QYFTDSX0EVNG48`;
final image `im-hU4X9qfUpGtzfROofi8DZu`; log
`/tmp/remag-large-solver-profile-v164.log`. The app is stopped with no containers;
[release](../results/large-solver-profile-v164-20261009T155157Z/allocation-release.json)
was confirmed at **15:56:31 UTC on 9 October**. Raw requested outputs, source
archive, original remote summary and direct comparison are retained.

## Wider vector row tile (v165, completed; not promoted)

`vector-row-tile-v165-20261009T160705Z` shared each 32-state local rotation
across 64 rather than 32 eigenvector rows: 512 threads and 49,152 bytes of
complex shared memory instead of 256 threads and 32,768 bytes. It reduced row
tile count and repeated rotation loads, preserving each output's arithmetic
and adding no persistent storage. The partial row group exits after the common
load barrier. Production retains the previous 32-row implementation.

All 36 generalized and two seeded checks, both small x/z/MAE comparisons, and
both large direct pristine CPU comparisons pass. Maximum large saved-DM error
is 8.084e-9 against the unchanged 1e-7 gate. All requested raw outputs are
retained with hashes. The large matched screen on one A100-SXM4-80GB,
driver 610.57.04, does **not** show an end-to-end gain:

| Contender | Wall time (s) | SCF iterations | Cycle time per iteration (s) |
| --- | ---: | ---: | ---: |
| Accepted v163 | 231.443597 | 110 | 1.981577 |
| Wider row tile | 240.872876 | 115 | 1.976802 |

Cycle time per iteration differs by only about 0.24%; total candidate time is
4.07% worse. Small x/z totals are 9.741639 and 9.539538 s, respectively, from
one pair only. This screen does not justify repetitions or integration.
No speedup is qualified. Workspace remains 9,132,023,264 bytes for the large
case; all 177 production hashes still match v163.

[Summary](../results/vector-row-tile-v165-20261009T160705Z/summary.json),
[ledger](../results/vector-row-tile-v165-20261009T160705Z/ledger.json),
[direct upstream comparison](../results/vector-row-tile-v165-20261009T160705Z/upstream-comparison.json)
and measured source archive preserve the result. App
`ap-wFkJWoLxBU4IHmqfYYZAmd` stopped with no containers; release was confirmed
at **16:18:23 UTC on 9 October**. Final image `im-iEPJK3TlzD4mP8224TamcB`,
call `fc-01M4GPP27K9BAPQ2VVS2D97SDD`, log `/tmp/remag-vector-row-tile-v165.log`.

## Fused local gather (v166, completed; deferred)

`fused-local-gather-v166-20261009T162247Z` loaded mapped pair blocks directly
into the shared eigensolver, removing the standalone gather and intermediate
local input write/read. It retained the full original-block failure capture,
local lower-triangle reconstruction, real diagonals, all Jacobi arithmetic,
32-row vector rotation, matrix rotation, workspace and stopping criteria.
Production remains v163; this candidate is not integrated.

All 36 generalized and two seeded checks, small x/z/MAE comparisons and both
large direct pristine CPU comparisons pass. Maximum saved-DM error is below
1e-8 against the unchanged 1e-7 gate. The single large pair on one
A100-SXM4-80GB with driver 580.95.05 gives:

| Contender | Wall time (s) | SCF iterations | Cycle time per iteration (s) |
| --- | ---: | ---: | ---: |
| Accepted v163 | 222.940522 | 112 | 1.889236 |
| Fused gather | 218.007234 | 108 | 1.914796 |

Total time is 2.21% lower with four fewer SCF iterations, while cycle time per
iteration is 1.35% higher. The screen does not establish a convincing kernel
benefit or repeatable end-to-end gain. Repetitions are deferred in favor of
large directional coverage of the current implementation. Small x/z pair
totals are 10.013735 and 9.610495 s, also single observations with differing
iteration counts. No performance gain is qualified. Large workspace remains
9,132,023,264 bytes; all 177 production hashes still match v163.

[Summary](../results/fused-local-gather-v166-20261009T162247Z/summary.json),
[ledger](../results/fused-local-gather-v166-20261009T162247Z/ledger.json) and
[direct CPU checks](../results/fused-local-gather-v166-20261009T162247Z/upstream-comparison.json)
retain raw outcomes and exact source. App `ap-oNaKGQ0p36LFOVmSb6PzwY` stopped
with no containers; release was confirmed at **16:33:14 UTC on 9 October**.
Final image `im-GG7psRB5AwEuRTkVCtO9KC`, call `fc-01M4GQJT3B41D72870WK6SS345`,
log `/tmp/remag-fused-local-gather-v166.log`.

## Current production large x direction (v167, x/z and MAE qualified)

`large-soc-x-v167-20261009T163613Z` completed the existing 128-atom,
2,304-state full-SOC x input with the exact v163 production binaries.
It converged in **119 SCF iterations and 244.897231 s**, with 230.658834 s
in device cycles, 9,132,023,264 bytes of workspace and final moment
`[186.13288, -0.0, 0.0]` μB. The input and pseudopotential hashes
match the completed pristine v153 CPU x reference. Numerical sources, stopping
criteria and requested outputs are unchanged.

The pristine v153 x reference completed at **17:30:26 UTC on 9 October**:
109 SCF iterations, 17,118.008 seconds on the local CPU. Both current x/z
results pass direct pristine scientific and raw-export comparisons. Saved-DM
errors are **1.10801e-8 x / 8.08152e-9 z**, below the unchanged 1e-7 gate.
Forces, stress and moments agree at printed precision; maximum eigenvalue
error is 7.2e-6 eV. Direct **E(x)-E(z)** is **0.138045 eV upstream versus
0.138046 eV REMAG**, a **1.0 micro-eV** difference within the unchanged
5 micro-eV gate. The orientation-family and final-axis checks pass for all
four calculations. The z partner is the same v163 binary's `candidate-repeat-1` result. The
allocation is an A100-SXM4-80GB, driver 580.95.05. No speedup is claimed across
allocations or against the local CPU.

The [direct CPU/MAE report](../results/large-soc-x-v167-20261009T163613Z/upstream-comparison.json)
and [capability qualification](../results/large-soc-x-v167-20261009T163613Z/capability-qualification.json)
record this scope and all 177 unchanged production hashes. This qualifies the
recorded SZP/100 Ry 2,304-state regression, not a production convergence study
or every size up to the 4,096-state allocation ceiling. The retained older
v155 x result and its v152 z partner also pass pristine CPU/MAE checks. The
v155 batch remains failed overall because its two Fe4N calculations failed;
those failures are not reclassified by the successful FePt comparison.

[Summary](../results/large-soc-x-v167-20261009T163613Z/summary.json),
[ledger](../results/large-soc-x-v167-20261009T163613Z/ledger.json) and
[prior GPU comparison](../results/large-soc-x-v167-20261009T163613Z/previous-gpu-comparison.json)
retain timings, exact source and raw requested outputs. App
`ap-UuSvI0m2M55tbvnupQqBqP` stopped with no containers; release was confirmed
at **16:41:09 UTC on 9 October**. Final image `im-quDmdRflwfmmqJOsP8QQFL`,
call `fc-01M4GRBDXJ9QJ9ANM87XPD515H`, log `/tmp/remag-large-soc-x-v167.log`.

## Sourced carbide reference (v168, CPU complete)

`carbide-local-reference-v168-20261009T164908Z` completed both cementite
inputs with the pinned pristine executable on one CPU thread, sequentially,
under the existing 1,800-second per-case bound and DM 1e-9 / H 1e-8 eV gates.
Collinear converged in 68 iterations (84.589341 s) with moment
`[0, 0, 22.8668]` μB; noncollinear in 72 (136.226954 s) with moment
`[8.08463, 8.08463, 19.80323]` μB. These concurrent local CPU timings are
not a GPU performance baseline. Direct GPU agreement remains pending.
This addresses the original spec's missing carbide coverage; no Modal GPU is
allocated and no new synthetic or observer framework is involved.

The published 300 K synthetic Fe3C structure is AMCSD 0019113 from Wood et al.,
J. Appl. Cryst. 37, 82–90 (2004), DOI
[10.1107/S0021889803024695](https://doi.org/10.1107/S0021889803024695).
The [mirrored raw CIF](https://larixite.seescience.org/cifs/19113) and its retrieved
page are retained with hashes. The Pnma cell lengths are 5.081, 6.754 and
4.515 Ang; expansion of all eight provided operations gives Fe-site
multiplicities 8 and 4 plus 4 C sites, 16 distinct atoms total. Minimum periodic
separation is 1.89616 Ang. Ferromagnetic Fe initialization is 2.5 μB; the NC
case uses theta 30 and phi 45 degrees without constraining the final direction.

The existing `Fe_nc.psf` and `C.psf` are CA-generated and nonrelativistic; runtime
PBE, SZP, 100 Ry and a 2x2x2 k mesh follow the current regression convention.
This is an implementation comparison, not a production convergence study or
full-SOC qualification. Production source is unchanged. No science or speedup
is qualified before completed upstream and GPU comparisons.

Launcher session `70488`, PID `72143`, log
`/tmp/remag-carbide-reference-v168.log`; source and input records are under
`build/carbide-reference-v168` and the result directory. The requested nice
level was denied by the sandbox; the command continued with one thread at its
inherited priority. The existing large FePt x CPU reference remains live.

## Carbide GPU comparison (v169, qualified)

`carbide-gpu-v169-20261009T170851Z` passes direct pristine v168 CPU comparison
for both sourced cementite inputs using exact v163 production binaries. Energy,
forces, stress and moments agree at printed precision. Maximum saved-DM error
is 1.53454e-8 (collinear) and 1.58558e-8 (NC), below the unchanged 1e-7 gate;
maximum RHO error is 5.96046e-8 against 1e-5. Requested raw exports and the
unchanged DM 1e-9 / H 1e-8 eV convergence settings pass for both calculations.

| Case | CPU SCF iterations | GPU SCF iterations | GPU wall time (s) | GPU workspace (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Fe3C16 collinear | 68 | 66 | 3.452005 | 136,822,288 |
| Fe3C16 noncollinear | 72 | 67 | 3.548934 | 256,731,248 |

The GPU is an A100-SXM4-80GB, driver 610.57.04. CPU and GPU hardware differ;
**no speedup is claimed**. The existing small-case path uses shared local
Jacobi, warm eigenbasis reuse and eight solver lanes, with fused matrix
rotation off. There are no numerical source changes or additional kernel
checks; all 177 production hashes still match v163. This qualifies the
recorded collinear and uniformly tilted NC carbide regressions, not full SOC,
nonparallel carbide textures or production materials accuracy.

[Direct upstream comparison](../results/carbide-gpu-v169-20261009T170851Z/upstream-comparison.json),
[capability record](../results/carbide-gpu-v169-20261009T170851Z/capability-qualification.json)
and [ledger](../results/carbide-gpu-v169-20261009T170851Z/ledger.json) retain the
scientific results, timing data, exact source and 18 hashed raw output files.
App `ap-mPFtA6G6Be3nW0Ofuvpbjd` stopped with no containers; release was confirmed
at **17:09:41 UTC on 9 October**. Final image `im-qYtDUBjmApiAG5297ng39s`,
call `fc-01M4GT75VY79HPHDQESWYXX2J3`, log `/tmp/remag-carbide-gpu-v169.log`.

## Large x reference (v153, running)

The missing pristine large x-direction CPU reference is running locally as
`soc-local-x-reference-v153-20261009T123117Z`, using the same pinned pristine
binary as v133, eight threads and a six-hour timeout. The existing corpus x
input has the same orientation-family and pseudopotential hashes as the
completed z input; only magnetic initialization and the descriptive name
differ. It retains DM 1e-9 / H 1e-8 eV and all requested outputs. Its input
SHA256 is `2f405c86ce8a0a624f78c86ef8ee8e3ae2be8ae5489eb689c49e13e808316c0a`.
This calculation supplies the missing upstream reference for large-cell MAE;
GPU x/MAE agreement and final moment directions are not yet verified.

CPU launcher session: `19112`; PID: `58490`; log:
`/tmp/remag-soc-local-x-reference-v153.log`. The saved
[reference metadata](../results/soc-local-x-reference-v153-20261009T123117Z/metadata.json)
and input retain provenance. This CPU calculation uses no additional Modal GPU.

## Full large upstream comparison (v170 launcher failure; v172 correction)

The v132 short screens established that the native two-stage/block-64/eight-rank/
one-thread layout made the best observed progress for FePt128, but none produced
a completed upstream reference. `large-upstream-comparison-v170` reuses that
layout and the exact v163 numerical binaries for one complete matched z pair
on a single A100-80GB allocation. The full upstream process has an 18,000-second
limit; REMAG has 900 seconds. Original physical inputs, DM 1e-9 / H 1e-8 eV
stopping criteria, required outputs and the 200-iteration cap are unchanged.
No new numerical implementation is introduced.

The existing GPU kernel observer first checks one upstream SCF iteration with
eight ranks sharing the one GPU. Only that diagnostic uses `MaxSCFIterations=1`
and `SCF.MustConverge=false` to exit normally and retain the kernel evidence;
it cannot qualify science or performance. Both full calculations require the
original convergence settings and run without tracing. The full pair starts
only if the diagnostic establishes actual GPU execution and effective settings.
Direct scientific/export checks use the existing comparators, with the already
completed pristine local v133 z reference checked after outputs return.

One completed pair can supply a provisional measured ratio, not the three-pair
qualification used for the small-cell headline, and cannot qualify large MAE.
Individual timings, settings, source identity, raw outputs and failures are
retained through the existing ledger and evidence workflow. This fills a missing
performance baseline; it is not an additional run for the planned visuals.

The v170 observer completed in 91.656 seconds and verified numerical GPU
execution on all eight ranks (149,482 total kernels). The candidate then aborted
before SCF in 6.595 seconds because the new comparison harness inherited the
upstream-only `REMAG_GPU=0` flag. Successful v163 calculations leave that flag
unset. Every other recorded candidate runtime field matches v163. The full
upstream calculation never started; no scientific result or speedup is claimed.
The original evidence and
[failure diagnosis](../results/large-upstream-comparison-v170-20261009T171936Z/failure-diagnosis.json)
are retained. The v170 allocation was confirmed stopped with no containers at
17:22:46 UTC on 9 October.

The v172 correction removes only that inherited flag from the candidate
environment. Numerical binaries, scientific inputs, tolerances and time bounds
are unchanged. It launched only after v170's release was confirmed. Its existing
one-SCF observer passed in 119.269 seconds, then the untraced REMAG calculation
completed in **233.255639 s / 110 SCF**. Upstream completed in
**10,060.055475 s / 122 SCF** on the same A100-SXM4-80GB, driver 580.95.05.
The matched end-to-end ratio is **43.1289×**, provisional because it is one pair.
It includes initialization, SCF, final observables and requested exports;
the diagnostic, provisioning, build and artifact transfer are excluded.
This measures time to equivalent converged solutions with differing iteration
counts. It is not repeated timing qualification or a directional-pair timing.

Both GPU contenders pass direct local comparisons with pristine v133, and the
local GPU-pair comparison also passes. Maximum saved-DM errors are **1.61916e-8**
for upstream GPU versus CPU, **8.09571e-9** for REMAG versus CPU, and
**1.61882e-8** for the GPU pair, against the unchanged 1e-7 gate. All requested
raw exports pass. The first local upstream HSX download contained a malformed
record despite matching the expected length; re-fetching the retained artifact
produced a valid file and resolved the export failure without rerunning compute.
The failed download, failed local checks and both hashes remain in the
[recovery record](../results/large-upstream-comparison-v172-20261009T172335Z/upstream-HSX-recovery.json).

The [local comparisons](../results/large-upstream-comparison-v172-20261009T172335Z/upstream-comparison.json),
[timing summary](../results/large-upstream-comparison-v172-20261009T172335Z/summary.json),
ledger, source archive and 18 hashed raw outputs retain the completed attempt
`1bcfb1e7320a48eebeda5e35b9cd7280`. The app is stopped with no containers;
release was confirmed at **20:20:25 UTC on 9 October**. No additional full
upstream repetitions were launched. The separate v153 x/MAE comparison is also
complete, as recorded above.

## Fully relativistic boride (v171/v173, qualified)

`boride-local-reference-v171-20261009T172616Z` runs pristine upstream on a
published tetragonal Fe2B cell, with eight Fe and four B atoms. The structure
uses the calculated parameters in Tables 2 and 3 of
[Kumar et al., J. Phys. Chem. Solids 109 (2017) 18–25](https://doi.org/10.1016/j.jpcs.2017.05.012):
I4/mcm, a = 5.077 Ang, c = 4.254 Ang, Fe 8h (0.1673, 0.6673, 0) and B 4a
(0, 0, 0.25). Both tables were inspected in the original PDF. The exact
tabulated values are retained; they imply a shortest Fe-Fe distance of
2.402415 Ang, slightly different from the paper's prose value of 2.411 Ang.

The Fe and B files are official PseudoDojo standard PBE PSML pseudopotentials
from repository `ONCVPSP-PBE-FR`, pinned at
`777dce9786644b83e6c3686f1a24e06c21f141d5`. Both declare Dirac relativity,
PBE exchange/correlation, and scalar-relativistic, spin-orbit and lj projector
sets. Fe includes 16 valence electrons; B includes three. The completed upstream
output confirms **spin-orbit+offsite**, fully relativistic generation for both
species, and consumption of 34 Fe / 14 B lj projectors from the PSML files.
The GPU outputs confirm the same interpretation; the direct comparisons below
establish numerical agreement.

Two cold x/z inputs reuse the existing tight SOC regression protocol: SZP,
100 Ry, 2x2x2 sampled k mesh, 1000 K electronic smearing, DM 1e-9 / H 1e-8 eV,
and at most 200 SCF iterations. Initial Fe moments are 2 muB; B starts at zero.
Directions are initialized, not constrained. Final axes and direct MAE must be
checked after convergence. These coarse settings are implementation regression
inputs, not a production convergence study or an experimental MAE prediction.

The unchanged pinned pristine CPU executable completed both cases sequentially
with one CPU thread: z takes **47 SCF / 100.144 s**, x **75 SCF / 146.609 s**.
Final moments are `[0, 0, 16.88417]` and `[16.88366, 0, 0]` muB, respectively.
The pristine reference difference E(x)-E(z) is **-0.001622 eV** at printed
precision. The basis has 140 spatial orbitals / 280 spinor states and eight
k points. Both completion and unchanged convergence-setting checks pass.
This used no GPU and overlapped the local FePt x reference, so its wall times
are not a performance baseline. Source PDF, PSML files, hashes, exact inputs and
raw reference outputs are retained through the existing results workflow.
The [completed CPU summary](../results/boride-local-reference-v171-20261009T172616Z/summary.json)
and [experiment record](../build/boride-reference-v171/experiment.json) identify
the retained `/tmp/remag-boride-reference-v171.log`; launcher session 20371 has exited.
The v173 GPU run reused the exact v163 production image after v172's confirmed
release, with no additional kernel checks or profiling. It completed z in
**4.063440 s / 47 SCF** and x in **4.362689 s / 87 SCF**, with 392,241,632 bytes
of retained workspace. Both direct pristine comparisons pass, including energy,
forces, stress, moments, densities and requested matrices. Maximum saved-DM error
is **1.27908e-8** against 1e-7. Final moment directions pass the existing check;
**E(x)-E(z) = -0.001622 eV** for both CPU and GPU at printed precision, with zero
printed difference against the unchanged 5e-6 eV MAE gate. Eighteen raw output
files and the complete source/settings evidence are retained in the
[qualification](../results/boride-gpu-v173-20261009T202117Z/capability-qualification.json).
This qualifies the stated full-SOC boride regression, not a production convergence
study or an upstream-GPU speedup. The app is stopped with no containers;
release was confirmed at **20:22:27 UTC** before the next trial.

## FP64 vector tiles (v174 screen; v175 integrated opt-in)

The v164 profile places vector rotation at **24.71% of summed overlapping
solver-lane intervals**, making it a material target after the previous row-tile
and fused-gather screens did not establish a useful gain. The staged
`build/fp64-vector-tiles-v174/block_jacobi.cu` changes only the complex vector
rotation. It gathers each 32x32 complex panel and local rotation into separate
real/imaginary FP64 shared-memory planes, with a 36-double column stride.
Sixteen warps own disjoint 8x8 result tiles and assemble complex multiplication
from four double-precision matrix products per k fragment. The buffers are
reused for coalesced interleaved-complex output after a block-wide barrier.

[CUDA 12.6's documented double WMMA path](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#double-precision)
uses double operands/accumulators, 8x8x4 fragments and round-to-nearest-even on
SM80 or newer hardware. This candidate does not convert to reduced precision.
It **does change accumulation grouping**, so existing residual, orthogonality,
scientific and MAE gates must pass before any performance interpretation.
Fragment origins are 32-byte aligned, all warp lanes participate, and barriers
separate shared input reads from output reuse. The launch has 512 threads and
36,864 shared bytes, versus 256 threads and 32,768 bytes in the existing complex
rotation. No persistent workspace is added. Occupancy and packing costs may
offset arithmetic gains; no speedup is assumed.

The new setting `REMAG_BLOCK_FP64_VECTORS=1` is opt-in and off by default;
real-valued rotation stays on the existing kernel.
The screen reuses the existing generalized and seeded checks for this concrete
numerical change, small full-SOC x/z and MAE checks, and one large matched pair
with direct local pristine CPU verification. There are no new synthetic suites
or timing probes. Repetitions are justified only if the screen shows a meaningful
end-to-end gain.

At the screen, all **177 production hashes still matched v163**. CUDA compilation, all 36 existing
generalized-solver checks and both seeded trajectories pass. The
[staged experiment](../build/fp64-vector-tiles-v174/experiment.json) retains exact
source hashes and the measured motivation for later interpretation and visuals.
Run `fp64-vector-tiles-v174-20261009T202641Z` completed on one A100-SXM4-80GB.
The 128-atom z case takes **219.245773 s / 101 SCF** with accepted v163 and
**204.644354 s / 103 SCF** with the candidate: **1.071350x provisional incremental
speedup**, or 6.66% less total time. Device-cycle time per iteration falls from
2.026699 to 1.843411 s. Small z takes 5.899 to 5.710 s (79 versus 94 SCF),
and small x 5.290 to 5.089 s (79 SCF each). These are single-pair screens,
not repeated performance qualifications or new upstream-GPU ratios.

Direct pristine CPU comparisons pass for both large results, with maximum
saved-DM error 9.47135e-9 against 1e-7. All small scientific/export checks and
the direct small MAE comparison pass; E(x)-E(z) remains 0.056828 eV at printed
precision. Five malformed local matrix downloads were preserved and re-read
from the completed run using serial block requests; the existing parsers and
scientific comparisons then pass. One intervening Modal storage HTTP 500 is
recorded. No calculation was rerun for this transport failure. The
[raw results and comparisons](../results/fp64-vector-tiles-v174-20261009T202641Z/upstream-comparison.json),
[download recovery](../results/fp64-vector-tiles-v174-20261009T202641Z/matrix-recovery.json)
and all 54 output-file hashes are retained. The app stopped with no containers;
release was confirmed at **20:37:36 UTC**.

The completed `fp64-vector-qualification-v175-20261009T205055Z` reuses the exact
v174 numerical image and kernel checks. Its four existing magnetic/relaxation
regressions, z/x/z sweep, three small x/z pairs, three large z pairs and one large
x calculation all pass their scientific checks. Contender order alternates
between repetitions on one A100-SXM4-80GB:

| Large z repeat | Accepted v163 (s) | FP64 vectors (s) | SCF accepted / candidate |
| --- | ---: | ---: | ---: |
| 1 | 245.832141 | 219.048211 | 113 / 110 |
| 2 | 234.029433 | 220.394612 | 107 / 111 |
| 3 | 237.266447 | 221.176020 | 109 / 111 |

Median time falls **237.266447 → 220.394612 s**, a **1.07655285× qualified
incremental speedup over v163**; means fall 239.042673 → 220.206281 s and all
three pairs are faster. All six large z outputs pass direct pristine v133 CPU
and convergence checks. Large x completes in **218.723 s / 112 SCF** and passes
direct v153 checks. Maximum saved-DM error across these comparisons is
**1.13150e-8** against 1e-7. Direct **E(x)-E(z) = 0.138045 eV upstream versus
0.138046 eV candidate**, a 1.00001e-6 eV difference inside the unchanged 5e-6
gate; final moment directions pass. The z partner is candidate repeat 1.

The completed v175 small pairs pass their scientific/export/MAE checks, but do
not establish an incremental performance gain: accepted pair times are
11.420171, 11.326068 and 12.372217 s; candidate times are 11.741296, 11.939027
and 11.318223 s. The candidate median is 2.81% higher and only one of three
pairs is faster. The flag therefore stays off by default, with no automatic
size threshold or broader hardware/material claim.

The exact measured kernel is now integrated with
`--fp64-vector-rotation` / `REMAG_BLOCK_FP64_VECTORS=1`. The launcher requires
32-state blocks, device SCF and vector cache; the qualified large configuration
also uses Hermitian matrix rotation. Python syntax, CLI option exposure and the
12 existing local regression/runner tests pass. Numerical source is byte-identical
to the measured image; CLI forwarding, guards and telemetry were added locally.
The pinned upstream checkout remains clean. No upstream-GPU ratio is inferred
by multiplying this incremental result.

[Direct upstream/MAE checks](../results/fp64-vector-qualification-v175-20261009T205055Z/upstream-comparison.json),
[timings and science summary](../results/fp64-vector-qualification-v175-20261009T205055Z/summary.json),
[integration/source identity](../results/fp64-vector-qualification-v175-20261009T205055Z/production-integration.json)
and 218 hashed output files retain the benchmark evidence. One malformed local
DM download was preserved and re-read serially; the existing comparison then
passed. Its [recovery record](../results/fp64-vector-qualification-v175-20261009T205055Z/matrix-recovery.json)
does not establish the transport fault's cause. No calculation was rerun.
App `ap-pjiaV4b6gpyhmpZfvtUV0Z` is stopped with no containers;
release was confirmed at **21:23:42 UTC** on 9 October.

## FP64 matrix tiles (v176 screen; v178 integrated opt-in)

`build/fp64-matrix-tiles-v176/block_jacobi.cu` retains the measured
complex matrix-rotation candidate motivated by v164's 16.64% share of summed
overlapping solver-lane intervals. It computes both U_row^H P(A) U_col products
with FP64 WMMA, retaining the intermediate in shared memory and preserving the
lower-triangular/Hermitian-partner layout. The opt-in requires fused 32-state
rotation; it adds no persistent workspace or host boundary. Its
[source and design record](../build/fp64-matrix-tiles-v176/experiment.json) are
retained. CUDA compilation and the existing generalized/seeded checks pass.
Completed run `fp64-matrix-tiles-v176-20261009T213621Z` compares it against the
qualified v175 kernel. Both contenders retain FP64 vector rotation and identical
solver settings; only the candidate enables FP64 matrix rotation. Small x/z
science, requested exports and MAE pass. The single large pair measures
**212.628 → 188.763 s**, with **111 → 101 SCF iterations**; measured cycle time
per iteration falls **1.77216 → 1.70838 s**. Workspace stays 9,132,023,264 bytes.
This is a provisional incremental screen, not a qualified or upstream-GPU
speedup. Both calculations pass direct local pristine-CPU science, exports and
convergence checks, with maximum saved-DM error **9.494e-9** against 1e-7.
All 30 raw binary outputs are retained. Five initial local HSX/DM copies were
malformed despite passing remote checks. Three recovered on re-read; the two
large control files required strictly sized server-padded reads after a second
malformed copy. The failed copies and hashes are retained; the transfer fault's
root cause remains unknown. No GPU calculation was repeated. The existing
artifact downloader now requires complete server-padded blocks, keeps SDK
digest checks, and never synthesizes a missing tail. [Recovery record](../results/fp64-matrix-tiles-v176-20261009T213621Z/matrix-recovery.json).
The allocation released at **21:48:07 UTC** on 9 October. After direct upstream
verification completed, `fp64-matrix-qualification-v178-20261009T215709Z` launched
the existing repeated qualification on one A100-80GB: unchanged v176 numerical
libraries, existing magnetic/relaxation and directional-sweep regressions,
three small x/z and large z matched pairs, and one large x calculation for MAE.
Only the candidate enables the matrix flag. Small performance is reported
separately. Direct large CPU/MAE checks and the existing timing acceptance rule
now pass, and the exact measured matrix kernel is integrated.
The existing four magnetic/relaxation regressions and same-process z/x/z sweep
pass. All three completed small pairs pass science/exports/MAE; pair times are
11.011169/11.249782/10.827161 s accepted and
10.904200/11.024580/11.134910 s candidate. The candidate median is slightly
higher (11.024580 versus 11.011169 s), so these repetitions establish no small-cell
gain. All three large repetitions now pass the remote checks: accepted
204.135488/203.006729/203.275279 s (101/101/101 SCF iterations), candidate
190.832317/189.679258/192.261136 s (100/101/104). Medians are
**203.275279 → 190.832317 s**, a qualified **1.065204×** incremental gain;
means also improve and all three pairs are faster. The large x calculation
completes in 191.815 s / 109 SCF iterations. All six z runs and this x run pass
direct local pristine-CPU science, requested-export and convergence checks.
Maximum direct saved-DM error is **9.521e-9**, below 1e-7. Direct MAE, using the
preselected first z candidate, is **0.138045 eV** for both upstream and REMAG at
printed precision, within the unchanged 5 micro-eV gate.

The production integration changes only the measured native kernel, launcher
wiring and runtime-setting telemetry. `--fp64-matrix-rotation` requires device
SCF, vector cache, width 32 and fused matrix rotation. Both measured contenders
also enable FP64 vector rotation. The new flag stays **off by default**, with
no automatic size threshold. Four 32×36 FP64 shared planes use 36,864 bytes per
512-thread block; persistent large-case workspace remains **9,132,023,264 B**.
Precision, stopping tolerance and maximum sweeps are unchanged.

The existing 12 regression-parser tests, Python compilation and CLI-help check
pass; pristine upstream remains clean at the locked commit. The existing 36
generalized-solver and two seeded checks are reused from the identical v176
numerical libraries. [Integration and source identities](../results/fp64-matrix-qualification-v178-20261009T215709Z/production-integration.json),
[individual timing ledger](../results/fp64-matrix-qualification-v178-20261009T215709Z/ledger.json),
[direct CPU/MAE checks](../results/fp64-matrix-qualification-v178-20261009T215709Z/upstream-comparison.json)
and 218 hashed binary outputs retain the evidence for reproduction and visuals.
Five completed comparisons were reused after matching the final result, raw
output and effective-FDF hashes. One truncated container-exec JSON copy is
preserved under `failed-downloads`; the strict volume reader recovered it
without repeating computation. No new upstream-GPU ratio is claimed, and this
factor is not multiplied into earlier ratios. The app is stopped with no
containers, confirmed at **22:28:15 UTC** on 9 October.

## Ternary candidate magnet (v177/v179 qualified)

Fe5PB2 adds a 32-atom, three-species candidate-magnet case with two inequivalent
Fe sites. The fixed I4/mcm structure is the GGA-SP row of Table I in
[Werwinski et al., Phys. Rev. B 98, 214431 (2018)](https://doi.org/10.1103/PhysRevB.98.214431):
a = 5.456 Å, c = 10.296 Å, Fe1 16l (0.170, 0.670, 0.139), Fe2 4c (0,0,0),
P 4a (0,0,0.25), and B 8h (0.381,0.881,0). The original PDF table was visually
checked and retained with the inputs. Fourfold rotations, horizontal reflection
and body centering expand the sites to Fe20P4B8 without duplicate positions.

The official PseudoDojo fully relativistic PBE Fe/B files are reused from v171;
the P file comes from the same pinned repository commit. All three PSML files
declare Dirac relativity and contain scalar, spin-orbit and lj projector sets.
`candidate-magnet-local-reference-v177-20261009T214755Z` completes pristine
upstream x/z references sequentially on one local CPU thread, with no GPU
allocation. The z reference takes 1057.965 s / 88 SCF iterations and x takes
1530.803 s / 128. These local CPU times are not GPU speedup baselines.
The existing tight full-SOC settings, requested outputs and numerical gates are
reused. These SZP/100-Ry/2³-k/1000-K inputs are implementation regressions, not
converged predictions of physical MAE. Directions are initialized, not
constrained; final axes pass the existing comparison. [Structure, settings and source hashes](../build/candidate-magnet-reference-v177/experiment.json)
are retained for reproduction and the future writeup.
The existing two-direction material harness is reused in
`candidate-magnet-gpu-v179-20261009T222951Z`, launched only after the sole v178
GPU allocation released. It uses the qualified v175 binary with fused-matrix
and FP64 matrix/vector tile flags off, following the defaults for an unmeasured
smaller material. Both directions pass direct pristine-CPU science, raw exports
and convergence checks. CPU and GPU stdout confirm `spin-orbit+offsite` with
eight spin components; the fully relativistic pseudopotential content is recorded.

| Direction | REMAG end-to-end (s) | SCF CPU / GPU | Final total moment (Bohr magnetons) |
| --- | ---: | ---: | --- |
| z | 12.796271 | 88 / 84 | (0, 0, 36.66203) |
| x | 14.304233 | 128 / 112 | (36.66379, 0, 0) |

Forces, stress, eigenvalues, atom moments/charges, densities and saved matrices
pass. Maximum saved-DM error is **9.642e-9**, below 1e-7. Direct **E(x)−E(z) is
−0.000048 eV** in both implementations at printed precision, within the unchanged
5 micro-eV gate; both final magnetization directions pass. These measurements
qualify numerical coverage, not a prediction of physical anisotropy or an
upstream-GPU speedup. The GPU was one A100-SXM4-80GB, driver 580.95.05.
[Capability and limits](../results/candidate-magnet-gpu-v179-20261009T222951Z/capability-qualification.json),
[direct comparisons](../results/candidate-magnet-gpu-v179-20261009T222951Z/upstream-comparison.json),
[timing ledger](../results/candidate-magnet-gpu-v179-20261009T222951Z/ledger.json),
source archive, exact settings and 20 hashed raw binary outputs are retained.
No additional component tests or profiling were introduced. App
`ap-mDZAh8aaHQRPUZQx9Xmz9H` is stopped with no containers; release was confirmed
at **22:34:51 UTC** on 9 October.

## Current solver profile (v148, diagnostic only)

After the no-gain v147 thread-count screen, the existing 13-stage eigensolver
profiler was reapplied to the integrated v144 solver. Its local-eigensolve marker
was moved before the shared/cuSOLVER branch so it measures the current shared
kernel. `eigensolver-profile-v148-20261009T113727Z` passes the existing upstream
science/export gates and the existing profile consistency checks at 80 SCF
iterations. Local eigensolves account for **55.36% of summed overlapping lane
intervals**, followed by generalized transformation 7.63%, vector rotation
7.62%, warm-basis QR 6.65%, matrix-column products 6.19% and matrix-row products
5.30%. These are diagnostic lane intervals including marker overhead, **not a
wall-time partition or speedup**. All [raw stage data](../results/eigensolver-profile-v148-20261009T113727Z/stage-profile.json)
and exact instrumented source are retained. No profile code is promoted;
the app is stopped with no containers, verified at 11:38:07 UTC.

The evidence motivated `build/fused-local-pivot-v149`: each local Jacobi
row-pass owner publishes its analytic pivot entries directly, removing one block
synchronization per cyclic round. The 512-thread layout and numerical settings
are retained. Run `fused-local-pivot-v149-20261009T114034Z` passes all 36 existing
generalized checks, both seeded trajectories and both FePt z screens. Device
time per iteration is **0.0218072 → 0.0218187 s**, with no useful improvement.
Wall time **5.6973 → 5.5014 s** includes 87 versus 85 SCF iterations and is not
claimed as a gain. Repeated timings and broader regressions were skipped.
The change is not integrated; [screen evidence](../results/fused-local-pivot-v149-20261009T114034Z/ledger.json)
is retained. The app is stopped with no containers, verified at 11:43:42 UTC.

## Smaller shared eigensolver block (v147, not integrated)

The current v144 profile puts 0.83276 s of a 1.78061 s device cycle in matrix
assembly/eigensolves. `build/compact-shared-jacobi-v147` tested 256 threads per
32-state local eigensolve instead of 512, assigning each half warp a rotation
pair and two rows per thread. FP64 arithmetic, rotation order, sorting, stopping
criteria and capacity were unchanged; the norm reduction grouping changed.

Run `compact-shared-jacobi-v147-20261009T113207Z` passes all 36 existing real/
complex generalized solver checks, both eight-step seeded trajectories, and
both upstream-verified FePt z screens. Each material run takes 80 SCF iterations.
Wall time is **5.601494 → 5.599862 s** and device-cycle time **1.769068 →
1.772121 s**. There is no useful performance gain; the screen correctly stops
before repeated timing and broader regression work. The candidate is not
integrated. These single diagnostic screens are not a new speedup claim.
The [summary](../results/compact-shared-jacobi-v147-20261009T113207Z/summary.json),
[ledger](../results/compact-shared-jacobi-v147-20261009T113207Z/ledger.json)
and measured source archive preserve the attempt. The app is stopped with no
containers; release was verified at 11:35:58 UTC. Production remains v144.

## Nitride coverage (v145/v146 unqualified)

The existing 16-atom Fe8N8 collinear/noncollinear inputs extend the chemical
coverage beyond Fe/FePt. Both pinned Fe_nc and N pseudopotentials identify
nonrelativistic CA generation; the calculations use PBE and are numerical
regressions, not SOC qualification or physical convergence studies. No candidate
numerical source changed from integrated v144.

Run `nitride-qualification-v145-20261009T111200Z` used fresh pristine upstream
CPU references with matched DM 1e-9 / H 1e-8 eV stopping criteria. The collinear
CPU run hit its 200-iteration cap and was rejected. The noncollinear CPU run
converged at 152 iterations. The first GPU calculation passed science and
exports at 151 iterations (4.793 s); the diagnostic profile converged at 154
iterations but reversed every magnetic moment and failed the component and
matrix gates. The CPU and first GPU moment is (0.11140, 0, 0.19296) muB; the
profile moment is its negative. Equal moment magnitude and nearly equal energy
do not qualify the requested magnetic state. No nitride performance ratio or
corpus promotion is claimed. The [raw ledger](../results/nitride-qualification-v145-20261009T111200Z/ledger.json)
and [failure analysis](../results/nitride-qualification-v145-20261009T111200Z/failure-analysis.json)
retain this result. The app is stopped with no containers, confirmed at
11:18:34 UTC. An earlier setup-only attempt used an incorrect CPU executable
path; its log and confirmed release are retained in the v145 build directory.

The bounded follow-up `nitride-damped-v146-20261009T112037Z` reduced conventional
Pulay mixing weight from 0.05 to 0.02 and allowed 400 iterations on both
contenders. Physics, DM/H stopping criteria, scientific and export tolerances
were unchanged. Both pristine CPU references converged (212 collinear and 278
noncollinear iterations); both first GPU calculations also converged (201 and
179), but selected the opposite final spin direction from their respective
CPU references. Collinear moments are -0.22281 versus +0.22281 muB; noncollinear
moments are (0.11140, 0, 0.19296) versus its negative. Total moment discrepancies
are 0.44562/0.38592 muB and saved-DM discrepancies 0.0067104/0.0058114, far above
the unchanged gates. Subsequent repeats/profiles were skipped after failure.

This is an observed state-selection limitation on weak-moment, unconstrained
Fe8N8. A spin reversal explains the leading component differences; it is not
accepted as numerical equivalence or proof that no other difference exists.
There is no nitride speedup claim, corpus promotion, or candidate numerical
change. Further blind mixing sweeps and post-hoc spin alignment are not planned.
Nitride coverage remains outstanding. The [follow-up ledger](../results/nitride-damped-v146-20261009T112037Z/ledger.json),
[failure analysis](../results/nitride-damped-v146-20261009T112037Z/failure-analysis.json)
and raw reference/candidate outputs preserve both attempts for later work.
App `ap-jUTGlH9Ma0xudrOco5Jzs9` is stopped with no containers;
[release evidence](../results/nitride-damped-v146-20261009T112037Z/allocation-release.json)
confirms no remaining GPU allocation. The larger local FePt128 CPU reference
remains in progress; its completion is still required for larger-cell qualification.

## Sourced Fe4N nitride inputs (v154, CPU references complete)

The weak-moment Fe8N8 failures above remain unqualified. The next nitride
input uses bulk gamma-prime Fe4N: a cubic 3.795 Ang cell with Fe at its corner
and face centers and N at its body center, as described in
[Zhang et al., Scientific Reports 5, 10602](https://www.nature.com/articles/srep10602).
The Wyckoff positions are also given in the introduction of
[the tetrametal-nitride exchange study](https://arxiv.org/abs/1601.01522).
These sources establish a magnetic nitride target; they do not validate this
regression's approximate numerical settings or its eventual magnetic state.

`build/fe4n-regression-v154` contains two staged 40-atom (2x2x2) inputs,
collinear and noncollinear. The latter seeds both transverse components
(theta 30 degrees, phi 45 degrees). Both retain the existing Fe/FeN PBE,
SZP/100 Ry, 2x2x2 k-mesh, 300 K, DM 1e-9 / H 1e-8 eV protocol, 200-iteration
limit and final scientific outputs. The existing upstream Fe_nc/N PSFs are
CA-generated and nonrelativistic. This is implementation regression coverage,
not a production Fe4N prediction or a full-SOC qualification. The generator,
source links, complete inputs and pseudopotential hashes are retained.

Pristine CPU references completed locally as
`fe4n-local-reference-v154-20261009T125055Z`, sequentially on **one CPU thread**,
with a 1,800-second limit per case. The pinned executable is the same as the
completed v133 reference. These CPU wall times are for reference bookkeeping,
not a performance baseline: the larger FePt x reference is running concurrently.
CPU outputs stay local. The collinear reference converged in 166 iterations
with a final cell moment of 78.94518 muB. The noncollinear reference converged
in 185 iterations. Both satisfy the unchanged tight stopping criteria. The
v155 GPU cases returned nonconverged states, so GPU qualification and corpus
promotion remain pending.

Launcher session: `12295`; PID: `59433`; log:
`/tmp/remag-fe4n-local-reference-v154.log`. Use the existing upstream/candidate
science and export checks when the references are ready; the v152 GPU
allocation has released. Do not overlap GPU allocations. The production corpus remains unchanged.

## Shared-spin grid potential (v144, integrated)

The v143 diagnostic passes upstream science/exports and measures an
80-iteration device cycle of 1.9596 s: assembly/eigensolves 0.8302 s,
Hamiltonian construction 0.4456 s, grid/Hartree/XC 0.4189 s and sparse
density/normalization 0.0883 s. These are instrumented stage timings, not
end-to-end performance ratios. They motivate `build/grid-potential-spins-v144`:
one block shares orbital-pair lookup and basis loads across active spin
components when adding the grid potential to H. The integrated kernel preserves
per-spin sub-point arithmetic and SOC component duplication, uses at most
2 KiB of block shared memory, and adds no persistent allocation or transfer.
It is opt-in (`REMAG_GRID_POTENTIAL_SPINS=1`, ordinary launcher
`--grid-potential-spins`), with the existing kernel retained as default and for
cluster sizes/layouts outside the specialization. The same sub-point arithmetic
is retained for each spin; sparse atomic arrival order can differ.

Run `grid-potential-spins-v144-20261009T105702Z` passes 16 existing periodic-grid
and captured-replay checks, 16 FePt calculations (including two diagnostic
profiles), four existing magnetic/relaxation regressions and a z/x/z sweep.
All 11 MAE checks pass. Maximum ordinary FePt saved-DM error against the retained
CPU references is 7.672e-9, below the unchanged 1e-7 gate. The three x/z pairs are:

| Repeat | Accepted v142 (s) | Shared-spin potential (s) |
| --- | ---: | ---: |
| 1 | 10.050225 | 9.537879 |
| 2 | 10.042981 | 9.644816 |
| 3 | 10.149839 | 9.440622 |

Medians are **10.050225 → 9.537879 s**, a **1.053717× incremental gain**;
all three pairs and the mean improve. Both diagnostic z profiles take 80 SCF
iterations: the Hamiltonian stage falls **0.444822 → 0.266232 s** and the device
cycle **1.957047 → 1.780613 s**. Other measured stage costs are nearly unchanged.
These instrumented timings explain the change; the gain above uses untraced
end-to-end repetitions. Device-cycle workspace, staging and exports are
unchanged in the paired profiles. This is not a new upstream-GPU ratio and is
not multiplied into v143's direct result.

The measured `native/grid.cu` is integrated byte-for-byte. The ordinary launcher
flag and runtime-setting recorder were added after measurement and syntax/
argument forwarding checked; no numerical code changed after the run.
[Integration record](../results/grid-potential-spins-v144-20261009T105702Z/production-integration.json)
retains all 177 production source hashes, measured source identity, raw paired
results and profile links for the future writeup. The app
`ap-RDjfndoN0w4SNQM1NGJaOF` is stopped with no containers;
[release](../results/grid-potential-spins-v144-20261009T105702Z/allocation-release.json)
was verified at 11:01:22 UTC. The larger FePt128 CPU reference remains in
progress; this run does not extend capacity or larger-cell qualification.

**Earlier verified comparison (v109), 8 October:** FePt16 full off-site SOC has a
6.0278× end-to-end gain over tuned pristine native ELPA GPU SIESTA on one
A100-SXM4-80GB. All 42 comparisons and nine repeated MAE checks reproduce from
raw outputs, together with four GPU-execution diagnostics and all timing and
solver-selection decisions. [Independent recheck](../results/upstream-gpu-performance-v109-20261008T033242Z/performance-recheck.json).

| Three-repeat median | Upstream GPU (s) | REMAG (s) | Speedup |
| --- | ---: | ---: | ---: |
| FePt16 SOC z | 278.507 | 47.545 | 5.858× |
| FePt16 SOC x | 267.080 | 42.769 | 6.245× |
| Complete x/z pair | 545.232 | 90.453 | 6.028× |

The winning baseline is native ELPA two-stage, block 64, one thread, selected
from 18 tuning trials and three fresh paired repeats of both shortlisted
flavors. The pair row uses the median of pair totals, not the sum of directional
medians. Timings include process startup, initialization, SCF, final observables
and requested exports; build, provisioning and evidence transfer are excluded.
Iteration counts vary, so this is time to an equivalent converged solution,
not a fixed-iteration kernel speedup. Direct MAE is 0.056828 eV for all repeated
contenders at printed precision, within the unchanged 5e-6 eV tolerance.
This is a small SZP/100 Ry regression, not a physical convergence study or a
claim for other materials, sizes or spin layouts. The completed app is stopped
with no remaining allocation; [release evidence](../results/upstream-gpu-performance-v109-20261008T033242Z/allocation-release.json).

Across the six untraced REMAG repeats, the median electronic-cycle share is
87.07% for z and 86.60% for x; geometry initialization accounts for 8.57% and
8.76%. These existing end-to-end profiles put the next performance investigation
inside SCF. The subsequent stage profile below identifies matrix assembly and
eigensolves as 93.54% of the measured electronic cycle.
Historical CPU speedups below remain a secondary baseline. The single-GPU
limit and the revised performance objective above continue to govern work.

The user withdrew Gold–water and made fully relativistic, spinful noncollinear
magnetic systems the primary target. The comparison now centers on FePt with
complex spinors and off-site SOC, followed by larger magnetic cells and general
nonparallel moment configurations. The retained v103 FePt outputs confirm eight
spin components, off-site SOC and fully relativistic pseudopotentials for both
species. This establishes the treatment used by those small cases, not broad
material or size coverage. Larger finite-k-point SOC cells, canted moments,
forces/stress, directional MAE, requested exports and repeated geometries remain
the decisive correctness/residency tests. Collinear and scalar-relativistic cases
remain regression and secondary performance checks; speedups must be measured
for each workload. Gold–water inputs and CPU diagnostics are retained as unused
provenance; no further Gold–water benchmark work is planned.

## Larger full-SOC capacity experiment (v125, upstream incomplete)

The prepared FePt128 case has 1,152 spatial orbitals and 2,304 spinor states,
which exceeds the production device-SCF solver's 2,048-state guard. Its matrix,
pair schedule and reduction buffers are dynamically allocated; the inspected
integer products fit at a 4,096-state bound. The isolated candidate in
`build/soc-capacity-v125` changes only the two native capacity checks, retaining
the 2,048 spatial-orbital bound and all numerical algorithms/tolerances.
The larger device-SCF capacity remains staged pending upstream scientific qualification.

Run `soc-capacity-v125-20261008T210159Z`, app
`ap-ZzTvEfZ1w7LdAIuMuNF6QX`, tests one FePt128 z direction with the v124b
DM 1e-9 / H 1e-8 eV stopping protocol, eight solver workspaces and retained
warm-start bases. The candidate has a 600-second limit; if it completes,
pristine native ELPA two-stage/block-64/four-thread GPU SIESTA runs sequentially
with a 1,200-second limit and the existing science/export gates. Diagnostic
stage profiling and upstream kernel counts exclude these timings from a
speedup claim. FePt128 CPU reference, x direction, MAE and repeated untraced
performance are not yet qualified. This is one bounded GPU allocation.

The candidate completed in 533.226 seconds and 109 SCF iterations; effective
stopping criteria pass. Its electronic cycle took 514.793 s, including
467.074 s in assembly/eigensolves (90.73%), 37.247 s in density construction,
4.838 s in Hamiltonian work and 4.505 s in grid/Hartree/XC. The cycle workspace
is 9,014,576,216 bytes; sampled total GPU memory was about 12.2 GiB. Completion
alone does not establish upstream agreement. The sequential upstream GPU
comparison timed out after 1,200.176 seconds and three completed SCF iterations.
Its effective two-stage/block-64/four-thread settings pass, but the incomplete
process has no complete GPU-execution census or final scientific outputs. Its
first reported iteration spent 304.284 of 308.425 s in `compute_dm`; this is not
a qualified end-to-end timing. No capacity change has been promoted and no
larger-cell speedup is claimed. The app is stopped with no containers and the
GPU allocation is released. [Retained outcome](../results/soc-capacity-v125-20261008T210159Z/summary.json).

This measured bottleneck motivates the staged `build/paired-block-layout-v126`
solver: retain both matrix axes in each Jacobi round's pair ordering, rotate
contiguous column panels and disjoint strided row panels directly, and restore
canonical order at sweep boundaries. It reduces full-matrix rearrangements
from six per round to two plus three boundary passes, preserving the pair
schedule, local eigensolver, warm-basis representation and convergence test.
The strided row outputs use independent, nonoverlapping elements, as in the
[NVIDIA banded-output GEMM example](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/).
A targeted CPU algebra/indexing check passes ten real/complex combinations
with widths 32/64, including multiple round schedules. CUDA compilation passes.
Run `paired-block-layout-v126-20261008T213300Z`, app
`ap-UMJO067huvho3B2mjdOVhx`, completed and released its GPU. All 14 FePt16
calculations, six direct science/export comparisons, six CPU MAE checks and
three direct MAE checks pass. Three fresh x/z pair medians are
**13.104287 → 12.885871 s**, a **1.01695× incremental gain** on one A100-80GB
PCIe. The initial single-direction screen was slightly slower; all three
complete paired repetitions were faster. Nonparallel full-SOC Fe3, collinear
Fe54, noncollinear Fe16 and all nine Fe2 relaxation geometries pass against
their existing upstream references. The six candidate repeats and four
regressions retain zero explicitly accounted SCF host/device copies.

The FePt128 z control/candidate pair completed in **539.626 / 391.865 s**,
a provisional **1.37707× incremental gain** despite 99/103 SCF iterations.
Direct observables and saved matrices agree, including a maximum DM error of
5.68e-9 against the unchanged 1e-7 gate. This single pair has no completed
upstream reference, x direction or MAE qualification; it establishes neither a
larger-cell upstream speedup nor production capacity qualification.

The follow-up `soc-large-cpu-v131-20261008T234151Z` ran the identical
FePt128 z input through the existing local pristine upstream CPU build, with
eight BLAS threads. It timed out after 1,800.243 seconds and 11 SCF iterations;
the result is invalid and cannot qualify candidate agreement or performance.
Its convergence-settings check passed, which does not mean SCF converged.
The pinned source remains clean, and the upstream executable target was
rebuilt/up-to-date before this run.
The existing v126 candidate's requested RHO, DRHO, HSX and DM outputs have been
recovered for direct comparison. This uses no Modal GPU allocation and is a
correctness-reference attempt, not a new performance baseline. The larger
capacity guard remains staged pending agreement; x/MAE qualification and
larger upstream-GPU timing are still outstanding.

The bounded follow-up in `build/soc-upstream-layout-v132` screened three native
ELPA layouts on the same FePt128 input: two-stage/block-64 with one MPI rank
and four threads, and one-/two-stage/block-64 with eight one-thread MPI ranks
sharing one GPU. The earlier traced large reference sampled only 2.42% GPU
utilization; its incomplete timings cannot determine the best untraced layout.
Each new screen allows three SCF iterations and at most 180 seconds. Screens
are diagnostic only, not converged calculations. Only a layout whose conservative
150-iteration projection fits 1,800 seconds proceeds to a full unchanged-input
reference, followed by the current candidate if upstream completes. All
contenders run sequentially. Original inputs, individual timings, solver settings,
raw results and measured source are retained for later comparisons and figures.
No larger capacity promotion or speedup follows from a short screen.

Run `soc-upstream-layout-v132-20261009T065928Z` completed with all three screens
timing out. The one-rank two-stage layout reported zero complete SCF iterations
in 180.119 s; eight-rank two-stage reported one in 181.406 s; eight-rank
one-stage reported zero in 181.412 s. Effective solver settings and unchanged
convergence thresholds passed for all three, but none is a converged result.
The eight-rank two-stage first iteration spent 86.434 s in `compute_dm` (97.83%
of that iteration). Mean sampled GPU utilization was 2.62%, 21.03% and 71.87%,
respectively; utilization alone is not useful-solution throughput. MPI CPU/RSS
samples cover the launcher rather than every rank and must not be presented as
total CPU utilization/memory. The conditional full reference and candidate were
not launched. The app `ap-9QsmOIbcNYBMhsBD7OKzyL` is stopped, with no containers
and its allocation released. [Screen data](../results/soc-upstream-layout-v132-20261009T065928Z/screen.json)
and [decision](../results/soc-upstream-layout-v132-20261009T065928Z/decision.json)
retain the raw results and limits for subsequent analysis and figures.

A 9 October source review confirms that pinned upstream
`Src/diag_option.F90` explicitly sets `ParallelOverK = .false.` for both native
ELPA flavors (lines 263–279). Setting `Diag.ParallelOverK` therefore cannot
add k-point concurrency to these GPU routes; no extra screen was launched for
that ineffective setting. This is a capability limit of this pinned interface,
not a claim that the tested solver layouts are globally optimal.

The unchanged local CPU reference is now running as
`soc-local-reference-v133-20261009T071147Z`, using the same pinned executable,
identical physical input and eight BLAS threads, with a six-hour limit. This
uses no Modal allocation. Its launcher is
`build/soc-local-reference-v133/run_reference.py`, log
`/tmp/remag-soc-local-reference-v133.log`, and initial local session is 8685
(launcher PID 40024). If it converges, the launcher compares its observables and
requested arrays directly with the retained v126 FePt128 z candidate using the
existing tolerances. That would establish only this directional scientific
comparison; it cannot qualify a GPU speedup, the x direction, MAE or the entire
expanded capacity. Do not restart it merely because its SCF output is buffered.

The exact measured paired-layout solver is integrated into `native/block_jacobi.cu`.
The whole device-SCF guard remains at 2,048 spinor states (1,024 spatial
orbitals for NC/SOC); the expanded whole-cycle capacity remains staged. The
lower-level solver's measured 4,096-state allocation guard is retained, without
a claim that every size is qualified. GPU material coverage uses block width 32
with the batched Jacobi local backend; the algebra/indexing check alone does not
qualify width 64 on GPU. This corrects the earlier width-64 statement: all 20
saved GPU results explicitly record width 32/Jacobi. Four existing
local integration/accounting tests pass. [Integration record](../results/paired-block-layout-v126-20261008T213300Z/production-integration.json)
retains the exact source identity, measured gains and scope. No later incremental
ratio is multiplied into the v117 upstream-GPU result.

## Fused paired-vector rotation (v134, integrated)

`build/fused-vector-rotation-v134/block_jacobi.cu` replaces the 32-wide path's
separate vector permutation and BLAS rotation with one shared-memory kernel.
Each block gathers 32 rows of the paired columns, reuses the local 32-by-32
rotation and writes four output columns per thread. Existing V/scratch buffers
alternate, and the odd round count allows restoration directly into V at the
sweep boundary. This removes one full vector rearrangement per round and one
full device copy per sweep without additional storage or host/device traffic.
The matrix rotations, local eigensolver and convergence tolerance are unchanged.
The vector accumulation order changes, so residuals, orthogonality and warm-basis
trajectories require numerical checks before material timing.

Run `fused-vector-rotation-v134-20261009T071946Z` completed on one A100-SXM4-80GB.
All 36 existing block checks, both eight-step seeded trajectories and all 20
material calculations pass. Three fresh FePt16 x/z pairs pass science,
requested exports and direct MAE, with median pair times **13.336350 →
13.031773 s (1.02337× incremental gain)**. All three paired repetitions are
faster, although the separate initial screening pair was slower. Nonparallel
Fe3 SOC, Fe54 collinear, Fe16 noncollinear and Fe2 relaxation regressions pass.
These are end-to-end timings with differing SCF iteration counts.

The single FePt128 z pair passes direct science/export checks and improves
**353.982954 → 320.742949 s (provisional 1.10363×)**, with 103 and 101 SCF
iterations. Device-cycle times are 336.020 and 303.133 s; persistent cycle
workspace is unchanged at 9,014,649,824 bytes. Saved-DM maximum error is
6.55e-9 against the unchanged 1e-7 gate. This compares two REMAG versions with
the same staged larger guard; it does not establish upstream agreement, x/MAE
coverage or a repeated larger-case performance result. The v133 local upstream
CPU reference continues separately without a Modal allocation.

The exact tested `block_jacobi.cu` is integrated. All other production native
sources match the measured candidate except for the retained production SCF
capacity guard, which remains at 2,048 spinor states. Width 64 keeps its prior
implementation. App `ap-gADTTpWhnwiTewGEInpiqu` is stopped with no app containers;
allocation release was confirmed at 07:37:36 UTC on 9 October. The
[integration record](../results/fused-vector-rotation-v134-20261009T071946Z/production-integration.json),
[individual timings and comparisons](../results/fused-vector-rotation-v134-20261009T071946Z/summary.json)
and measured source archive retain the data for later figures. No new
upstream-GPU ratio is claimed or formed by multiplying incremental gains.

## Magnetic-orientation reuse (v135, staged)

`build/magnetic-orientation-v135` adds an explicit fixed-geometry sweep in one
SIESTA process. `REMAG.SpinRotations` supplies absolute quaternions (w, x, y, z)
relative to the initial spin configuration. The first direction rotates the
atomic density; later directions apply the relative rotation to the last
converged GPU density. The full complex operation is U D U-dagger, including
all eight SOC density components. The crystal geometry and SOC Hamiltonian are
not rotated. This changes the initial guess; it does not constrain the final
magnetization, so direct moment-direction and MAE checks remain necessary.

Requested outputs use separate `.oNNN` system labels. The staged workflow
requires serial NC/SOC device SCF, GPU history, fixed geometry and last-density
reuse without extrapolation; per-direction NetCDF remains unsupported. It
reuses density and process initialization, while geometry/basis setup and
eigensolver workspaces are still rebuilt between directions. Production source
is unchanged by this staged feature.

The C++/CUDA and Fortran changes compile. Run
`magnetic-orientation-v135-20261009T075154Z` passed all ten targeted rotation
checks (maximum error 8.88e-16 against the full complex matrix oracle). Its
z/x/z material sweep reported normal completion, with 79/130/124 SCF iterations.
Both reused directions have zero initial-density upload bytes. Energy, forces,
real-space densities and both MAE comparisons agree, but the return-to-z saved
DM differs by **5.953e-7**, exceeding the unchanged **1e-7** gate. This feature
is not qualified or promoted.

The original result collector also failed because Modal Volume does not support
hard links. Already-produced arrays were recovered without rerunning the
calculation, and [observable reanalysis](../results/magnetic-orientation-v135-20261009T075154Z/recovered-observables.json)
preserves the saved-DM failure. The subprocess return code and external wall
time were not saved before that collector failure, so they remain unknown and
the complete run gate is unqualified. The native timer records 19.813 s for
the diagnostic sweep: geometry initialization is 6.676 s, including Setup_H0
4.390 s, while device cycles total 10.042 s. These are diagnostic stage times,
not a speedup claim. App `ap-fpP1UrCrlHHwZpSvTi1JAC` is stopped and released.

The corrected v135b collector copies requested output views and saves execution
status before analysis. Both fresh upstream CPU references pass at matched DM
**1e-11** / H **1e-10 eV** stopping criteria; scientific comparison tolerances
are unchanged. Run `magnetic-orientation-v135b-20261009T080756Z` then passes the
ten rotation checks, but aborts in the second direction when final electronic
export requires convergence. The first z direction converges after 94 iterations
and its observed scientific/export values agree. The complete sweep fails:
return code -6, measured external duration 13.447712164 s, no timing repetitions,
no qualified speedup. The second direction's actual iteration count and residuals
were not flushed before abort and remain unknown; the collector's parsed zero
is not a measurement of zero SCF work. The third direction did not start.

The [ledger](../results/magnetic-orientation-v135b-20261009T080756Z/ledger.json),
execution status, source archive, both CPU references and failure logs are
retained. In particular, the empty FDF log after abort cannot confirm effective
settings even though the requested inputs are available. Tighter stopping did
not qualify retained-density rotation, so this feature remains staged. App is
`ap-NcKq4hDtSmOfx6DOvW243r`; launcher log is
`/tmp/remag-magnetic-orientation-v135b.log`. The app is stopped with no app
containers; allocation release was confirmed at 08:14:16 UTC on 9 October.
The earlier build-only attempt used the wrong CMake executable target and is
also stopped, with its log retained. The local v133 large SOC CPU reference
continues independently.

## Fixed-geometry magnetic sweeps (v136, integrated)

The measured v135 setup cost motivated retaining sparse topology, overlap,
H0/mesh setup and GPU pair/projector/H0 owners across explicit magnetic
directions. Each direction starts from independent atomic occupations and a
fresh SCF cycle. The crystal/SOC frame stays fixed; only the initial spin density
is rotated. The unqualified retained-density guess from v135 is rejected by
the integrated interface, which requires `DM.AllowReuse false`.

Run `magnetic-geometry-reuse-v136-20261009T082001Z` measures a **1.31107×
incremental end-to-end gain** on one A100-SXM4-80GB. Three fresh paired medians
are **12.825832 s** for separate FePt16 z/x calculations and **9.782714 s** for
one cached sweep. All three pairs are faster. All 18 directional calculations,
their scientific/export checks and seven sweep MAE checks pass. Three additional
[cold-pair MAE checks](../results/magnetic-geometry-reuse-v136-20261009T082001Z/cold-pair-mae.json)
also pass from the retained outputs, with zero MAE error at printed precision.
Maximum saved-DM
error is **7.6731e-9**, below the unchanged **1e-7** gate. Inputs use the existing
matched DM 1e-9 / H 1e-8 eV atomic-start references. No tolerance was loosened.
The [individual repeats](../results/magnetic-geometry-reuse-v136-20261009T082001Z/summary.json),
ledger, exact input/source identities and raw results are retained for later
figures. This compares REMAG workflows; it is not a new upstream-GPU ratio.

The same run's diagnostic three-direction screen records geometry initialization
falling from **5.702 to 2.588 s**, including Setup_H0 **3.654 to 1.209 s**.
Those single screen observations explain the change but are not the repeated
timing result. After the first direction, pair geometry, projector geometry and
nonlocal H0 record zero new device allocation and zero setup uploads. Their
existing buffers remain allocated and are rebound only after completed SCF and
force leases. This is not a claim of zero total setup traffic or zero memory use.

The [integrated regression](../results/magnetic-geometry-regression-v136-20261009T082632Z/summary.json)
passes the FePt z/x/z sweep, ordinary nonparallel Fe3 full-SOC, collinear Fe54,
noncollinear Fe16 and the nine-geometry Fe2 relaxation. Two additional
identity-rotation sweeps exercise retained geometry for nonparallel Fe3 SOC and
Fe16 NC against their existing upstream references. Every local native file,
the Fortran module and the integration patch match the qualified source record;
the standard patch reproduces the measured Fortran seams. The existing patch
test passes. Both benchmark allocations are stopped and released; the final
release was confirmed at 08:28:41 UTC on 9 October.

Use `REMAG.ReuseOrientationGeometry true` with a `REMAG.SpinRotations` block
of absolute quaternion rows `(w,x,y,z)` and `DM.AllowReuse false`. The current
scope is serial fixed-geometry NC/full-SOC device SCF with the production force,
initialization and history paths. Outputs use `.o001`, `.o002`, etc. No spin
constraint is applied; final moment directions must still be verified. Each
direction still stages its atomic density and rebuilds eigensolver workspaces.
Per-direction NetCDF, geometry-changing sweeps and retained-density rotation are
unsupported. Production capacity is unchanged. These are regression inputs,
not production MAE convergence studies.

## Shared-memory local eigensolver (v142, integrated)

The existing v138 lane profile put 59.28% of its overlapping eigensolver
intervals in the small local solves. The current v141 profile still makes SCF
the largest end-to-end component. The new opt-in implementation therefore keeps
each 32-state real or complex Hermitian local problem in one GPU thread block:
padded column-major shared storage holds the matrix and eigenvectors throughout
disjoint cyclic Jacobi rotations. Arithmetic remains FP64, with machine-epsilon
relative Frobenius stopping, up to 100 local sweeps and stable eigenvalue ordering.
The outer generalized eigensolver, SCF tolerances, occupations and normalization
are unchanged. Existing cuSOLVER paths remain available.

Run `shared-local-eigensolver-v142-20261009T095832Z` passes all 36 existing
generalized-eigenproblem checks and both eight-step seeded trajectories,
including real/complex problems, padding, ignored upper triangles and bad-overlap
failure/recovery. Maximum checked residual, orthogonality error and eigenvalue
error are 7.09e-14, 3.63e-12 and 2.51e-12 respectively. All 14 ordinary FePt
comparisons, four magnetic/relaxation regressions, the three-direction z/x/z
sweep and 11 MAE comparisons pass. Maximum saved DM error is 6.22e-8 against
the unchanged 1e-7 tolerance.

| Sequential x/z pair | Accepted v141 (s) | Shared local solver (s) |
| --- | ---: | ---: |
| 1 | 11.511639 | 11.236511 |
| 2 | 11.061929 | 9.643438 |
| 3 | 11.057816 | 9.630004 |
| Median | **11.061929** | **9.643438** |

All three pairs improve, with **1.14709× incremental speedup** on one
A100-SXM4-80GB. This is time to a converged solution, including setup and outputs;
SCF counts vary, particularly for x. The raw ledger retains every count and
timing, including the slower first candidate pair. Median ordinary electronic
cycle time is 2.2649 → 1.8970 seconds. Cycle workspace falls by 2,565,760 bytes,
from 479,862,376 to 477,296,616 bytes. These results do not establish a new
upstream-GPU speedup, and must not be multiplied into the historical ratios.

Enable `REMAG_BLOCK_SHARED_JACOBI=1` with `REMAG_BLOCK_WIDTH=32` and
`REMAG_BLOCK_LOCAL_SOLVER=jacobi`, or add `--shared-local-solver` to an ordinary
Modal launch with those block settings. The default remains the cuSOLVER local
solver. Production whole-SCF capacity remains 2,048 spinor states. The new kernel
and calling implementation match the measured archive exactly; the standalone
seeded diagnostic target is not added to production CMake. The ordinary launcher
exposes the option and the existing result reader records it.

[Integration and source identity](../results/shared-local-eigensolver-v142-20261009T095832Z/production-integration.json)
retain 177 production hashes, numerical errors, memory and cycle measurements,
with [raw paired results](../results/shared-local-eigensolver-v142-20261009T095832Z/summary.json),
inputs, settings, profiles, source archive and scientific ledgers available for
the later writeup. No benchmark was added solely for graphics. The completed
app is stopped with no containers, and its GPU allocation was confirmed released
at 10:04:08 UTC on 9 October. The larger v133 upstream CPU reference remains live.

## GPU initial mesh and reference fields (v141, integrated)

The existing setup profile motivated integrating the previously staged v107
mesh producer into current v140 production. Compact orbital/basis geometry now
produces stored FP32 mesh values, the atomic reference density, and its initial
Hartree energy on the GPU. SCF borrows these fields directly; a device kernel
binds the Hartree energy. The same immutable fields survive fixed-geometry
magnetic direction changes, while relaxation rebuilds them. Upstream mesh
arithmetic, occupation cutoffs and scientific/convergence tolerances are unchanged.

The corrected run `initial-mesh-integration-v141b-20261009T094041Z` passes
14 ordinary FePt comparisons, four existing magnetic/relaxation regressions,
and a three-direction z/x/z sweep. All scientific and requested-output checks
pass, including 11 MAE comparisons. The maximum saved DM difference across the
ordinary calculations is 6.79e-8, below the unchanged 1e-7 tolerance.

| Sequential x/z pair | Accepted v140 (s) | GPU mesh (s) |
| --- | ---: | ---: |
| 1 | 10.958769 | 10.144988 |
| 2 | 11.178393 | 10.554090 |
| 3 | 11.064520 | 10.254101 |
| Median | **11.064520** | **10.254101** |

All three pairs improve: **1.07903× incremental speedup** on one
A100-SXM4-80GB. These are times to converged solutions, including setup and
requested outputs; SCF iteration counts vary and are retained in the ledger.
This is not a new comparison against upstream GPU SIESTA, and incremental
ratios must not be multiplied into historical upstream comparisons.

Across six ordinary x/z repetitions per role, median `PHION` time falls from
0.3865 to 0.0915 seconds; the new GPU producer takes 0.008 seconds. Median
`Setup_H0` falls from 1.0355 to 0.7885 seconds. These inclusive routine timers
overlap and must not be added. CPU mesh topology preparation remains. FePt
avoids the 21,277,184-byte mesh-value upload and borrows its 497,664-byte atomic
reference density, with 75,568 bytes of new compact setup staging. The producer
reports 22,348,168 bytes of device allocations plus 1,014,784 bytes of Hartree
workspace; those are owner sizes, not net additional memory.

The initial run `initial-mesh-integration-v141-20261009T093742Z` completed with
passing numerical outputs but was correctly marked invalid: the existing
reader omitted the new 75,568-byte setup owner from its transfer total.
Its original result is preserved with `reader-correction.json`. The corrected
run uses identical compiled numerical binaries. Three existing reader tests
pass. No additional observer or residency-certification work was introduced.

The implementation remains opt-in through `REMAG_INITIAL_MESH=1`, or
`--gpu-initial-mesh` in the ordinary Modal launcher, alongside the existing
production device-SCF, GPU initialization, orbital-basis and final-force paths.
It requires serial periodic meshes with stored FP32 orbitals and FP64 grids;
unsupported initialization settings are rejected. Existing capacity and physics
limits remain. Explicitly accounted SCF copies remain zero; library transfers,
managed/mapped traffic and complete filesystem activity remain unmeasured.

The exact measured runtime and six generated Fortran integration units match
the promoted sources. The [integration record](../results/initial-mesh-integration-v141b-20261009T094041Z/production-integration.json)
retains 176 source hashes, available profiles, capability limits and links to
the [raw repetitions and summary](../results/initial-mesh-integration-v141b-20261009T094041Z/summary.json).
Inputs, scientific ledgers, source archive, hardware/settings, failed attempt
and raw outputs remain available for the future writeup. Both apps are stopped;
the corrected run's GPU allocation was confirmed released at 09:47:05 UTC on
9 October. The larger v133 upstream CPU reference continues independently.

## Shared orbital-pair density contractions (v140, integrated)

The measured density/normalization cost motivated a compact representation of
distinct orbital pairs. Geometry preparation now records one row/column pair
and a map for its periodic copies. The GPU contracts occupation-weighted
density and energy-density once per unique pair, then applies each image's
phase and scatters into upstream's existing spin-component layout. Four-component
NC symmetrization is applied after the phase, as before. The phase and spin
arithmetic move outside the state reduction, changing rounding without changing
cutoffs, normalization or convergence criteria. Private cycle scratch is reused
serially across k/spin jobs; there is no dense n-by-n density bank or added
steady-state host/device transfer. Inputs with no repeated pairs retain the
existing kernel.

Run `unique-density-pairs-v140-20261009T091807Z` passes the existing 24 density
checks, including 48 captured replays, real/complex vectors, four/eight spin
components, finite-k phases and normalization failure/recovery. Maximum captured
matrix error is 2.78e-17 (rounded upward). All 14 FePt16 calculations and nine
MAE comparisons pass the unchanged upstream science and requested-export gates.
Three fresh complete x/z pairs, with alternating contender order, measure:

| Repeat | Accepted REMAG (s) | Unique-pair candidate (s) |
| --- | ---: | ---: |
| 1 | 11.264640 | 10.468355 |
| 2 | 11.365361 | 10.756391 |
| 3 | 12.066302 | 10.644180 |
| Median | 11.365361 | 10.644180 |

This is a **1.06775× incremental end-to-end gain** on one A100-SXM4-80GB; all
three pairs improve. SCF iteration counts differ, so the result is time to an
equivalent converged solution, not a fixed-iteration comparison. The initial z
screen's electronic cycle falls from 2.7393 to 2.1893 s despite 78 / 80 iterations.
No new upstream-GPU speedup is inferred or obtained by multiplying earlier gains.

FePt contracts **20,736 unique pairs instead of 217,840 image entries** per
k/spin job. Additional cycle scratch is **2,654,208 bytes (2.53 MiB)** and the
geometry maps add **1,037,248 setup-upload bytes**. Those maps persist with the
existing density geometry. The separate nonparallel full-SOC Fe3, Fe54 collinear,
Fe16 noncollinear and nine-geometry Fe2 relaxation regressions pass. The maximum
saved-DM error across material checks is 5.86e-8 against the unchanged 1e-7 gate.
All 11 candidate calculations retain zero explicitly accounted SCF copies;
unobserved library, managed-memory and filesystem activity remain unknown.

The exact tested density source, header, included kernels and SCF integration
are promoted. Grouping is enabled by default when repeated pairs exist;
`REMAG_DENSITY_UNIQUE_PAIRS=0` selects the previous path. Legacy density consumers
continue using their original accumulation kernel. The production capacity guard
is unchanged. Other spin layouts pass numerical regressions but do not acquire
new speedup claims from this FePt measurement.

The inherited SIESTA executable is byte-identical to the v136 qualified build.
The unchanged orientation template remains covered by that integration's module
proof; the only differing compiled diagnostic source is the separate density
checker, which is not linked into the production library. App
`ap-nXCXRSilO1aZV51Nrjik93` is stopped with no containers and its allocation was
confirmed released at 09:22:20 UTC on 9 October.
[Measured comparisons](../results/unique-density-pairs-v140-20261009T091807Z/summary.json)
and [production integration](../results/unique-density-pairs-v140-20261009T091807Z/production-integration.json)
retain individual timings, source identity, layout counts, memory, scientific
checks and all raw benchmark artifacts for the eventual writeup.

## Smaller local eigensolver blocks (v139, not integrated)

The v138 profile motivated a staged 16-state paired layout with a fused
32-row vector-rotation tile. It reduces each local eigensolve's size while
increasing the number of paired rounds. Partial row tiles are bounded, and
the existing 32/64-state implementations, scientific tolerances and whole-cycle
capacity remain unchanged. The experiment is isolated in
`build/small-block-layout-v139`.

Run `small-block-layout-v139-20261009T090543Z` passes all 36 existing generalized
solver checks and both eight-step real/complex eigenbasis-reuse trajectories.
Maximum generalized residual is 7.36e-14 and orthogonality error is 3.91e-12
(rounded upward). Both FePt16 z screening calculations pass unchanged upstream
science, convergence and export gates, with maximum saved-DM error below 7.67e-9.
On one A100-SXM4-80GB, the current 32-state layout takes **6.301 s**, while the
16-state candidate takes **7.623 s**. Electronic-cycle totals are nearly equal:
**2.6717 / 2.6733 s**, with 75 / 78 SCF iterations. Per-iteration time is slightly
lower for the candidate, but the full calculation does not improve.

The candidate is **not integrated**. The planned screen gate skips the repeated
x/z pairs and broader regressions; this single screen does not qualify a
repeated performance ratio or width-16 MAE/general support. The production
source still matches all 165 files in the v136 integration record. App
`ap-2cAWLdjFOX2gKGR4LrEmSB` is stopped with no app containers; its allocation was
confirmed released at 09:08:28 UTC on 9 October.
[Screen and numerical data](../results/small-block-layout-v139-20261009T090543Z/summary.json)
and [integration decision](../results/small-block-layout-v139-20261009T090543Z/integration-decision.json)
retain the individual timings, source identity and scope. The summary's false
regression flag means the optional regression batch was not run, not that a
material regression failed.

After this screen, the next measured bottleneck was density construction. FePt16 has
217,840 stored sparse entries but at most 20,736 distinct orbital (row,column)
pairs. The current kernel repeats occupation-weighted state contractions for
periodic copies, applying their phase inside the state sum. A compact unique-pair
representation could share those contractions and apply each image's phase
afterward. The changed arithmetic order requires the existing upstream
science/export/MAE checks and matched end-to-end timings; this is a hypothesis,
not a speedup claim. The subsequent v140 implementation and measurement are above.

## Eigensolver stage profile before unique-pair density (v138, diagnostic only)

Run `eigensolver-profile-v138-20261009T085919Z` profiles the accepted v136
implementation with temporary device-clock boundaries in an isolated build.
The identical FePt16 z input passes the retained pristine upstream CPU
scientific, convergence and requested-export gates. It completes 75 SCF
iterations in 5.978 s, with a 2.699 s electronic cycle. These timings include
instrumentation and are not a new speedup measurement. Production is unchanged.

The coarse profile places assembly/eigensolves at **1.073 s (39.76% of the
cycle)**, density/normalization at 0.650 s, Hamiltonian work at 0.417 s and
grid/Hartree/XC at 0.393 s. Older profiles reporting about 90% eigensolver share
do not describe the profiled small-cell implementation.

Within the solver, local 32-state batched eigensolves account for **59.28% of
summed lane intervals**; vector rotations account for 7.56%, generalized
transformation 6.80%, warm-basis QR 5.93%, matrix-column/row rotations 5.43% /
4.83%, and pair permutation/gather 3.75%. Physical overlap factorization is
only **2.04%**, consistent with v137's lack of a repeatable end-to-end benefit.
These are overlapping elapsed intervals across eight concurrent lanes, not
additive wall-time shares or isolated kernel execution times. They include
scheduling/contention and marker overhead. Warmup is excluded. All counters
reconcile with 600 solves, 1,289 block sweeps and 21,913 paired rounds.

The next optimization target is useful work and overhead in the local batched
eigensolves; density and grid work also remain material to end-to-end time.
Numerical algorithms and stopping tolerances were unchanged for this profile.
One build-only syntax failure was corrected before any material calculation;
its log and confirmed release are retained in the stage directory. The completed
app `ap-FjoHhN7eYcwlaVfq7l3qfX` is stopped with no app containers and its GPU
allocation released at 09:00:00 UTC on 9 October.
[Raw lane/stage data](../results/eigensolver-profile-v138-20261009T085919Z/stage-profile.json),
[analysis](../results/eigensolver-profile-v138-20261009T085919Z/analysis.json),
the scientific ledger, exact source archive, inputs and hardware metadata are
retained for subsequent optimization and the future writeup. No benchmark or
instrumentation was added solely to create visuals.

## Immutable overlap-factor cache (v137, not integrated)

The staged `build/overlap-factor-cache-v137` experiment caches each appended
k-point/spin job's Cholesky factor of the fixed overlap matrix on the GPU.
Separate jobs retain separate factors even when they share a solver workspace;
new SCF plans invalidate all caches. A device conditional publishes a factor
only after successful factorization and skips later factorizations. Hamiltonian
updates, eigensolver tolerances and production capacity remain unchanged.

Run `overlap-factor-cache-v137-20261009T084046Z` passes all 36 ordinary solver
checks, two eight-step real/complex fixed-S changing-H trajectories (including
nonfinite-H failure/recovery), all FePt science/export/MAE comparisons and the
four existing magnetic/relaxation regressions. It adds **10,616,864 bytes** to
the FePt16 cycle workspace, from 477,208,168 to 487,825,032 bytes.

The three fresh x/z pair times are:

| Repeat | Current REMAG (s) | Cached overlap (s) |
| --- | ---: | ---: |
| 1 | 12.198506 | 12.304239 |
| 2 | 13.426621 | 12.194781 |
| 3 | 12.581587 | 14.468985 |

The median favors caching by 1.02254×, but only one pair is faster and the
mean worsens from **12.735572 to 12.989335 s** (0.98046×). The matched 80-iteration
z screen's device-cycle times are also close: 2.795186 versus 2.781831 s.
This is insufficient evidence of a repeatable end-to-end gain. The
[integration decision](../results/overlap-factor-cache-v137-20261009T084046Z/integration-decision.json)
keeps the experiment staged and the qualified v136 production source unchanged.
The original summary's `performance_qualified` identifies completed comparable
timings, not a decision to promote the optimization. No upstream speedup or
production improvement is claimed. Individual timings, settings, scientific
checks, memory and measured source are retained for the future writeup.

The first image build compiled CUDA but failed to discover a newly generated
check target in the same Make invocation; separate build calls corrected it.
Both attempts are stopped. The completed GPU allocation release was confirmed
at 08:48:10 UTC on 9 October. The existing v133 FePt128 CPU reference continues;
this experiment did not launch another large-cell calculation.

## Collinear and noncollinear GPU measurements (v127/v127b)

`spin-layout-performance-v127-20261008T220559Z`, app
`ap-gxG43r3SPMdWVQOpt8z6bc`, runs the integrated production sources on one
A100-80GB, after confirming v126 released. The image restores the production
SCF capacity guard, and checks every native source against the current tree.
Fe54 collinear and sampled-k Fe16 noncollinear reuse their existing pinned
upstream CPU references and matched DM 1e-9 / H 1e-8 eV inputs. Fe54's initial
production calculation passes the science/export gates in 9.620 s.

The same allocation screens native ELPA two-stage/block-64 with one/four
threads and one-stage/block-64 with four threads. Collinear Gamma additionally
screens both ELSI/ELPA flavors with eight MPI ranks and one thread each on the
same single GPU; eight ranks satisfy the two-spin communicator requirement.
Pinned ELSI rejects noncollinear physics, so it is excluded for Fe16. Existing
kernel-count diagnostics establish actual upstream GPU computation separately
from three fresh untraced repeats of each fastest passing native/ELSI setting.
All calculations run sequentially. Screening is bounded to 240 s per process.
The Fe54 native one-thread trial timed out and is excluded. Both four-thread
native flavors passed (112.849 s two-stage, 149.734 s one-stage), as did ELSI
with eight ranks (204.669 s one-stage, 79.340 s two-stage). The selected native
and ELSI settings each passed the separate GPU-execution diagnostic.

**Fe54 collinear is now performance-qualified:** three fresh untraced median
times are **79.626778 s ELSI → 9.511503 s REMAG (8.37163×)**. Native ELPA's
median is 110.914700 s (11.6611× relative to the same REMAG measurements), so
ELSI is the primary denominator. All six direct scientific/export comparisons
pass, with maximum saved-DM error 5.10e-8 against the unchanged 1e-7 gate.
This is a bounded settings screen on the same A100-80GB PCIe allocation,
not a claim of globally optimal ELPA settings. Both route families qualify.
[Complete Fe54 result](../results/spin-layout-performance-v127-20261008T220559Z-interrupted-c7b6eddcebea4576b95627dfced200ef/summary.json).

Modal preempted the container during Fe16's first upstream trial, after SCF
iteration 59. Fe16 had no completed upstream result. The automatic restart was
stopped to avoid rerunning the already-complete Fe54 case; the original app is
stopped with no containers. All 316 compact evidence files (9.85 MB) from the
original attempt are recovered locally and its large arrays remain in the same
volume folder. The saved Fe54 summary and metadata match the copies read before
preemption exactly. The overall attempt remains incomplete; only its completed
Fe54 segment is qualified. [Recovery and scope](../results/spin-layout-performance-v127-20261008T220559Z-interrupted-c7b6eddcebea4576b95627dfced200ef/recovery.json).

After confirmed release, `spin-layout-performance-v127b-20261008T224249Z`, app
`ap-OyDl5MbdcPGBCNReqR5CH1`, continues only Fe16 with the same production sources,
physical input and numerical gates. Its process limit is 360 s, and it refuses
automatic repetition if preempted again. All 11 calculations completed and
passed. Native two-stage took 251.275 s with one thread and 178.301 s with four;
one-stage/four-thread took 177.527 s and was selected for fresh repeats. Its
separate GPU-execution diagnostic passes.

**Fe16 noncollinear is performance-qualified at 29.73287×:** three fresh
untraced medians are **177.813891 s upstream → 5.980381 s REMAG** on one
A100-80GB PCIe. All three direct science/export checks pass, including maximum
saved-DM error 5.08e-8 against the unchanged 1e-7 gate. All Fe16 performance
data come from this one completed attempt, without merging timings from the
interrupted allocation. The launcher lost connectivity after the calculations;
the completed result was retrieved from the original function call, without
rerunning any workload. The run marker is complete, and the app is stopped
with no remaining allocation. [Results](../results/spin-layout-performance-v127b-20261008T224249Z/summary.json)
and [release evidence](../results/spin-layout-performance-v127b-20261008T224249Z/allocation-release.json).
These collinear/NC cases have no directional MAE pair; SOC MAE remains covered
by the preceding independent FePt evidence. The bounded settings screen does
not establish globally optimal upstream tuning.

## Atomic final-force reduction experiment (v128, not promoted)

The completed v126 Fe2 relaxation timer attributes 3.274 of 9.248 s to
`remag_atomic_force` across nine geometries. That timer includes derivative
generation, allocation/capture and GPU contraction, so it does not isolate
atomic-add cost. The contraction currently sends every entry's nine stress
components into the same nine global addresses. The candidate in
`build/atomic-force-reduction-v128` reduces stress within each warp before
global accumulation, and also reduces force when all warp entries address one
atom. Mixed-atom warps retain the existing force atomics. Inactive/tail entries
contribute zero without reading their fields; per-entry physical arithmetic,
FP64 precision and sticky errors are retained. No new persistent storage or
host/device boundary is added. Production is unchanged by this candidate.

The experiment reuses the existing masked/unmasked atomic-force checks
for the specific changed reduction and inactive-lane risks, then measures three
same-allocation Fe2 relaxation pairs and existing collinear, NC, nonparallel SOC
and FePt x/z science/export/MAE regressions. Run
`atomic-force-reduction-v128-20261008T231846Z` completed after v127b's confirmed
allocation release. All 154 unmasked and 180 masked existing checks pass, as do
all 11 material calculations, three direct relaxation comparisons, five
magnetic regressions and FePt MAE. The three Fe2 relaxation pair medians are
**10.645141 s accepted / 10.647935 s candidate (0.999738×)**: no demonstrated
end-to-end gain. The median atomic-force timer changes only from 3.286 to
3.275 s across 18 calls. The timing comparison is scientifically qualified,
but the optimization is not promoted. Production `native/atomic_grid_force.cu`
remains unchanged. The app is stopped and the allocation released.
[Completed result](../results/atomic-force-reduction-v128-20261008T231846Z/summary.json).

The diagnostic `atomic-force-profile-v129-20261008T232554Z` adds host timers
around the original force stages, with no new CUDA operations or changed math.
Both Fe2 relaxation and FePt16 SOC z still pass their existing upstream
science/export gates. Across Fe2's 18 calls, allocation of basis arrays takes
0.02225 s, problem staging plus the existing stream synchronization takes
3.22388 s, and the wait after launching the atomic graph takes 0.02057 s.
Graph construction/instantiation/launch and cleanup together take about 0.005 s.
The 0.402–0.404 s waits occur on the first term of geometries 2–9; geometry 1
and FePt's single geometry do not show them. [Diagnostic evidence](../results/atomic-force-profile-v129-20261008T232554Z/ledger.json).
This locates a synchronization cost, not a 3.2-second atomic kernel or isolated
32-byte-copy cost. Earlier nonlocal force graphs are queued on the same stream.
The diagnostic app is stopped with its allocation released.

## Parallel nonlocal sparse-row gather (v130, integrated)

The earlier nonlocal-force gather assigned one thread per sparse row and
serially scanned all entries of each row. The candidate in
`build/nonlocal-row-gather-v130` assigns a CUDA block per row and distributes
independent entries across its threads. Each entry retains the original
duplicate-column and spin summation order, SOC last-entry semantics and error
checks. Row spans are already validated as disjoint. No persistent storage,
transfer or graph node is added. The experiment tests whether the earlier
queued gather explains the measured wait.

After v129's confirmed release, run
`nonlocal-row-gather-v130-20261008T233135Z`, app
`ap-leY0NdktMz9nxMRrlX8gyU`, completed on one A100-SXM4-80GB. All 734 existing
compact nonlocal checks pass, covering duplicate entries, invalid lookups and
unused fields. All 11 material calculations, three direct relaxation pairs,
five magnetic regressions and FePt MAE pass under the unchanged gates.

Three fresh Fe2 relaxation medians are **10.415229 → 7.042971 s**, a
**1.478812× incremental end-to-end gain**. All three pairs improve. The median
timer around atomic-force completion falls from 3.281 to 0.237 s, although
the atomic kernel itself is unchanged; this supports the earlier queued
nonlocal gather as the source of most of the measured wait. The candidate
calculations retain zero explicitly accounted SCF host/device copies.

The exact measured `native/nonlocal_compact.cu` is now integrated. All 161
current native files match the previously measured production manifest plus
this one change, and the compiled source in the returned archive matches the
integrated file. The app is stopped and its allocation released.
[Integration record](../results/nonlocal-row-gather-v130-20261008T233135Z/production-integration.json)
includes scope and source identity. This is an incremental relaxation
comparison, not a new upstream-GPU ratio or production convergence study.

## Earlier baseline attempts

The following preparation and interrupted attempts are historical. The completed
FePt16 comparison above supersedes their pending-performance status; failures
remain retained under their original inputs and settings.

The unconstrained canted FePt seed has a pristine CPU preflight: it converges
in 426 iterations with the existing mixing settings and accuracy tolerances,
after raising its iteration ceiling from 200 to 1,000. The failed 200-iteration
attempt is retained. Its moments relax close to z; this exercises nonparallel
initialization and transverse spin components, not a stable canted ground state.
The 16- and 128-atom canted inputs are prepared but unqualified. All 41 pre-existing
input/manifest pairs remain unchanged, and the 12 targeted regression tests pass.
See [CPU evidence](../results/preflight-upstream-gpu-baseline-v109/canted-cpu-qualification.json).

The pristine upstream GPU build is complete with native ELPA and ELSI, with all
2,884 pinned source files unchanged. The completed
[kernel-census diagnostic](../results/upstream-gpu-census-v109-20261008T014616Z/census-recheck.json)
reproduces every original result from the executed source: 12 of 14 native runs
have complete GPU-execution proof and 10 pass the original science/export gates.
The observer's return, exit and no-compute controls pass. The earlier overflowing
full-API trace remains retained as failed evidence. The new census establishes
GPU computation, not residency or performance. Its app is stopped with no
containers, and all 329 MB of raw evidence are local.

Both native ELPA flavors pass the two-atom finite-k and Gamma SOC x/z cases,
including four direct MAE checks. FePt16 one-stage fails x-axis agreement and
two-stage fails z-axis agreement. The latter differs by 0.00015 μB in total
moment against a 0.0001 limit, and exceeds the density and HSX H gates. An
[additional saved-DM audit](../results/upstream-gpu-census-v109-20261008T014616Z/additional-export-check.json)
passes all eight small cases but finds FePt16 coefficient differences of
1.11e-6 to 8.28e-6, exceeding the existing TSDE/NetCDF DM tolerance of 1e-7.
The larger directional pair remains unqualified. Both canted native routes
reach their 900-second process limits, after 277 and 367 iterations, while the
fresh Modal CPU oracle converges in 360. These remain incomplete timeouts; a
follow-up needs a longer bounded runtime. No failed result was reclassified.

The pinned SIESTA ELSI interface explicitly rejects noncollinear and spin–orbit
calculations in `compute_dm.F:142`; its documentation specifies collinear spin
support. Native ELPA is therefore the applicable upstream GPU baseline for SOC.
The earlier ELSI rank plan omitted the spinor factor of two, and is retained as
superseded evidence in the [applicability record](../results/preflight-upstream-gpu-baseline-v109/elsi-kpoint-layout.json).
A CPU-only runtime guard check with two ranks initially stopped on an idle rank
because block 32 exceeds the small cell's distribution needs. The corrected
block-8 check reaches and confirms the explicit SOC rejection without changing
the binary. Both CPU-only apps have stopped and released their containers.
This scope concerns the pinned SIESTA
interface, not other releases or the ELSI library itself.

The locked REMAG candidate has compiled beside pristine upstream, with all 161
native/patch files byte-identical to the accepted implementation. The matched
FePt16 benchmark is staged. Its current input protocol tightens DM/H convergence
to 1e-8 and 1e-7 eV equally for all contenders; existing scientific comparison
limits remain fixed. The additional saved-DM check uses the established 1e-7
coefficient tolerance across every SOC component. Effective convergence criteria
must also match the FDF runtime logs. The earlier tenfold-tighter
[CPU preflight](../results/preflight-upstream-gpu-baseline-v109/matched-cpu-preflight-20261008T021638Z/qualification.json)
converges in 74/57 z/x iterations with correct final directions. The current,
hundredfold-tighter [CPU preflight](../results/preflight-upstream-gpu-baseline-v109/matched-cpu-preflight-20261008T025347Z/qualification.json)
also passes, in 78/71 z/x iterations with correct final directions. Its effective
FDF convergence settings match the protocol, and input comparison confirms only
the two intended stopping thresholds changed. Fresh same-allocation CPU/GPU
agreement is still required.

After candidate qualification, the timing harness tests both native ELPA flavors,
block sizes 32/64/128 and thread counts 1/4/8, then makes three fresh repetitions
of each complete x/z pair in the shortlist. One identical solver configuration
must qualify both directions and direct MAE; selection uses the lowest median
paired time. Complementary passes from different configurations cannot form a
qualified directional baseline. See [preparation](../results/preflight-upstream-gpu-baseline-v109/performance-preparation.json).
No speedup over pristine upstream GPU SIESTA is established yet.

The separate [v114 address-observer preparation](../results/preflight-memory-access-observer-v114/launch-preparation.json)
is deferred by the revised performance priority and its launcher is disabled.
It has never compiled or executed on GPU and is not a prerequisite for optimization.

The first matched run `upstream-gpu-performance-v109-20261008T030256Z` was stopped
to correct a verifier error before selecting the performance baseline. Both
initial REMAG FePt16 x/z comparisons pass against fresh same-allocation
CPU references, including all saved DM components (maximum coefficient errors
2.57e-8 and 6.77e-8 for x/z, against 1e-7). The direct MAE check passes with equal
printed directional differences. The one-thread native ELPA trial passes;
the four-thread trial passes science and exports but was incorrectly rejected
because the verifier read SIESTA's MPI-times-OpenMP count as the MPI rank count.
The corrected reader checks the explicit rank header, actual requested thread
count and their product separately. The real four-thread header passes, and six
wrong or ambiguous controls are rejected. Original records remain intact; the
first app is confirmed stopped with no containers. A fresh guarded comparison
is running as `ap-kvtvfzKfUP4H1dGJZMYihn`, run
`upstream-gpu-performance-v109-20261008T033242Z`, with no reused timings or CPU references.
Modal preempted this run after both initial candidate directions and direct MAE
passed, together with six native ELPA tuning trials. The interrupted attempt
`539af6b7bc8b4f4b84b64041699bf9f5` is preserved separately, including 186 compact
evidence files recovered locally. Its original running marker is retained; it
is not a completed or performance-qualified result. The same app restarted as
attempt `1b57cf7b1ebc4880a72782734e09c290` on a new A100-SXM4 allocation. Both
initial candidate directions pass against fresh CPU references. The entire
timing batch is being rebuilt on that allocation; results from the interrupted
PCIe allocation are not combined with it. By 06:53 UTC, all 18 native tuning
trials and all four shortlisted GPU-execution diagnostics pass. The shortlist
is native one-stage/block 128/four threads and two-stage/block 64/one thread.
Both initial candidate directions and the first candidate repeat pass; the
three paired timing repetitions and independent completed-run recheck remain
open. The 06:49 lifecycle check confirms this app's single container is still
the sole active allocation. These intermediate results are not a final speedup.
These initial candidate results do not establish upstream GPU performance.
The completed-run rechecker will verify source/artifact hashes and reconstruct
scientific, export, MAE, offload and timing decisions. It uses bounded temporary
storage for the large HSX/DM files; original raw artifacts remain on Modal.

A separate [tighter canted CPU preflight](../results/preflight-upstream-gpu-baseline-v109/extended-cpu-preflight-20261008T030957Z/qualification.json)
converges in 550 iterations with the same 1e-8 DM / 1e-7 eV H thresholds and
unchanged mixing. The final Fe/Pt moment angle is approximately 0.000118 degrees,
again nearly collinear. This prepares a longer bounded GPU follow-up but does
not qualify a stable canted state. The canted and 128-atom input checks confirm
that only the two stopping thresholds change; all physical settings remain fixed.

The 128-atom CPU setup completes with 1,152 spatial orbitals, a 2,304-dimensional
spinor problem and a 72×72×96 mesh. Full SCF/GPU and broader canted-state
qualification remain open.

## Focused nonparallel SOC comparison

Three [linear Fe3 CPU references](../results/preflight-nonparallel-soc-v112/qualification.json)
now retain clearly nonparallel moments after convergence with full off-site SOC.
They cover Gamma with xz and rotated xyz moment components, plus two sampled
k points with xyz components. The runs converge in 273, 108 and 164 iterations
at the same 1e-8 DM / 1e-7 eV H thresholds. Edge-to-central moment angles remain
about 97.82 degrees. All eight saved H/DM banks are nonzero, and raw observables,
forces/stress, eigenvalues, geometry, density fields and effective settings have
been independently rechecked. A matched zero-SOC control completes but correctly
fails full-SOC coverage: H banks 5/6 fall from about 0.00409 Ry to numerical zero.
Six negative moment controls reject parallel, antiparallel, zero, malformed and
nonfinite configurations. The first preflight report's mistaken requirement for
a separately printed SOC energy is preserved; the recheck accepts the upstream
off-site `Eso(nil)` convention already used by the main scientific comparator.

These are frozen correctness inputs, not a GPU qualification or speed result.
Their fixed geometry comes from pinned upstream's linear Fe3 example, using the
existing fully relativistic PBE Fe potential. The
[published noncollinear Fe3 study](https://doi.org/10.1088/2515-7639/aae7db)
motivates this choice, but uses different numerical settings. The small periodic
PBE/SZP fixtures have large residual forces (about 13 eV/Ang), so they establish
neither relaxed structures nor isolated-cluster or ground-state convergence.
The GPU follow-up must use fresh CPU/native/REMAG runs and the existing science,
export and offload gates. FePt remains the primary performance comparison.

The [Fe3 GPU follow-up preparation](../results/preflight-nonparallel-soc-v112/gpu-preparation.json)
now passes 31 local controls, including frozen FDF/pseudopotential/restart
rejection, completed nonparallel state checks, replay of an accepted device
cycle and rejection of bad or incomplete cycle accounting. The sequential
single-GPU harness will run fresh CPU, REMAG and pristine native GPU calculations
for all three inputs, reusing the qualified FePt solver configuration for
correctness only. It checks every contender's final nonparallel full-SOC state,
all scientific/export gates, native offload and complete-cycle explicit copies.
The launch guard binds 244 prepared source/input files. The independent
rechecker verifies executed sources, completed attempts, binary identity and
both allocations' release evidence. Full raw
outputs are retained. The [first GPU attempt](../results/nonparallel-soc-v112-20261008T161840Z/interruption-summary.json)
was cancelled after all six Gamma calculations converged; its allocation is
released and the cancellation cause is not established. All completed energy,
force, stress, moment and full-SOC/nonparallel-state checks pass. Saved density
matrix coefficients fail the unchanged 1e-7 limit: CPU versus REMAG errors are
4.18e-7 (xz) and 1.51e-7 (xyz), while CPU versus upstream GPU errors are 2.65e-7
and 1.14e-7. Thus neither Gamma input is fully qualified, and the sampled-k input
has no completed result. The original ledger and running attempt marker are
preserved as interrupted evidence.

The [focused follow-up](../results/nonparallel-soc-v112b-20261008T164432Z/qualification.json)
passes all three inputs and all nine CPU/REMAG/native comparisons with the same
binaries and physical inputs, using equally tighter SCF stopping criteria
(DM 1e-9, H 1e-8 eV) for every contender. Scientific comparison tolerances are
unchanged. [Direct local replay](../results/nonparallel-soc-v112b-20261008T164432Z/raw-comparison.json)
reproduces all scientific/export, nonparallel full-SOC state and effective
convergence checks. The largest saved-DM coefficient error is 2.955e-8 versus
the unchanged 1e-7 limit. The candidate converges in 511/417/333 iterations;
all three cases retain nonparallel moments and eight-component SOC matrices.
The allocation is released. Native timings are traced for execution evidence,
so this establishes correctness coverage, not a Fe3 speedup or ground-state study.

## SCF performance work

The existing FePt16 profile places about 87% of end-to-end time in the device
SCF cycle. An optional seven-stage timer is isolated in
`build/scf-performance-v115`; the accepted native implementation remains intact.
It records serialized device timestamps around the existing stage boundaries,
exports 120 bytes after SCF, and is disabled by default. Its diagnostic times
include graph scheduling and instrumentation overhead and cannot qualify a
speedup. A single-GPU run compares unprofiled and profiled FePt16-z against the
retained, verified CPU oracle using the existing scientific/export checks.
The timing code compiled. A remote Python path error in the first launcher was
corrected before any material run; its stopped/released app and log are retained
in `results/scf-profile-v115-startup-failure`. The corrected run completed as
`scf-profile-v115-20261008T170210Z` and its allocation is released.

The original profiled result was rejected solely because terminal accounting
omitted the new 120-byte diagnostic export. The original result is preserved;
the reader now recognizes that exact optional structure and still rejects
missing timing metadata or an incorrect byte total. [Local reanalysis](../results/scf-profile-v115-20261008T170210Z/profile-reanalysis.json)
passes the scientific comparison against the verified CPU oracle, the direct
unprofiled/profiled scientific and raw-export comparisons, and the corrected
terminal accounting. The ten existing related tests pass. Instrumented runs
are explicitly excluded from performance qualification.

| Serial SCF stage | Instrumented seconds | Share of cycle |
| --- | ---: | ---: |
| Matrix assembly and eigensolver | 41.067 | 93.54% |
| Density, grid, Hartree and XC | 1.200 | 2.73% |
| Density matrices and normalization | 0.600 | 1.37% |
| Scalar reductions | 0.542 | 1.23% |
| Hamiltonian updates | 0.383 | 0.87% |
| Convergence and mixing | 0.088 | 0.20% |
| Occupations | 0.016 | 0.04% |

The 69-iteration device cycle takes 43.901 s; the seven stage totals cover
43.895 s. This is one diagnostic FePt16-z observation, not a new speedup.
Its eight independent k-point eigensystems currently run sequentially in every
iteration. The next implementation target is a bounded pool of persistent GPU
workspaces for those independent matrix assembly/solve jobs, keeping the common
occupation calculation and density accumulation order intact. Measure the
unprofiled end-to-end result against the accepted implementation and upstream
SIESTA before promoting it. Initialization/observer work remains deferred.

The workspace pool was developed in `build/scf-solver-lanes-v116` and the
qualified native sources are now integrated in `native/`. It supports 1/2/4/8 persistent solver
workspaces, capped by the number of independent jobs. Each owns its dense H/S,
library scratch, solver state and spectral output. A device join reports the
first failure in job order before the common occupation calculation; density
accumulation remains ordered. Multiple workspaces require the full device
vector bank. The default remains one workspace.

The first v116 run compiled and completed its accepted and one-workspace
FePt16-z calculations. Both pass all numerical/export checks, but the new result
was rejected by an exact metadata-dictionary comparison when `solver_lanes` was
added. Original results remain in
`results/scf-solver-lanes-v116-20261008T172545Z`; its allocation is released.
The existing reader now validates the requested/effective workspace count
alongside the unchanged vector-bank shapes. Saved-result replay accepts both
formats and rejects mismatched counts. The corrected run reuses the same native
image and verifies its binary/source identities.

The corrected [v116 performance run](../results/scf-solver-lanes-v116-20261008T173132Z/summary.json)
passes all 17 calculations and six direct MAE checks. Eight workspaces were
selected from the 1/2/4/8 screen, then compared in three fresh x/z paired repeats
with alternating role order. On the same A100-80GB PCIe allocation, median pair
time falls from 89.326 s for the accepted library to 24.437 s for the candidate:
**3.6553× incremental end-to-end speedup**. The three control/candidate pair times
are 89.326/24.437, 91.790/25.549 and 84.193/23.730 s. No trace or stage timer was
enabled. Direct MAE error is zero at printed precision in all six comparisons.

Median cycle time per SCF iteration falls from 0.6357 to 0.1307 s; iteration
counts also vary, so the qualified number remains end-to-end time to an
equivalent converged solution. Persistent workspace increases from 367,543,664
to 466,573,672 bytes (about 94.44 MiB extra). The allocation is released. This
comparison is against the preceding REMAG implementation, not a fresh upstream
GPU measurement. Do not multiply it by the earlier 6.03× upstream-GPU result:
that was measured on a different allocation and A100 SXM variant.

The [focused v116 regression](../results/scf-solver-lanes-regression-v116-20261008T174227Z/ledger.json)
passes the genuinely nonparallel sampled-k Fe3 full-SOC case, including all
numerical/export checks, eight matrix banks and final 97.82-degree edge-central
moment angles. Fe54 passes the scientific observable gates but its saved DM
coefficient error is 3.77e-7 against 1e-7. Direct replay of all three preceding
accepted Fe54 repeats also fails that coefficient gate (1.85e-7 to 4.49e-7), so
this discrepancy predates the workspace pool. The original results remain
unchanged, and the allocation is released. A focused follow-up uses fresh
upstream/candidate Fe54 runs with equally tighter SCF stopping criteria,
DM 1e-9 and H 1e-8 eV, while retaining all scientific comparison thresholds.
The [follow-up](../results/scf-solver-lanes-regression-v116b-20261008T174638Z/qualification.json)
passes all three remaining cases. Fe54's saved-DM error falls to 3.024e-9 versus
the unchanged 1e-7 limit. Sampled noncollinear Fe16 and all nine Fe2 relaxation
geometries pass on their original input/stopping settings. Effective workspace
counts are two for Fe54's real collinear spin blocks, eight for Fe16 and Fe2,
and two for the previously passed nonparallel Fe3 SOC case.

The exact tested `scf_cycle.cu`, `scf_cycle.hpp` and `scf_cycle_state.cpp` are
promoted to `native/`. The main Modal entry point accepts `--solver-lanes 8`
alongside `--device-scf --vector-cache`; standalone candidate execution uses
`REMAG_SCF_SOLVER_LANES=8`. Counts 1/2/4/8 are supported and capped by available
jobs; the default remains one. CPU references and cache-disabled controls use
one workspace. Optional stage profiling remains disabled for performance runs.
Broader material/size performance remains unmeasured. The direct upstream-GPU
refresh below now qualifies the integrated pool; the 3.6553× increment remains
a separate same-allocation comparison against preceding REMAG.

The direct upstream-GPU refresh completed as
`upstream-gpu-refresh-v117-20261008T180328Z`, app
`ap-KGKT9AniPQDOlnbZLTE7Wq`, call `fc-01M4EAYEQQ1BWDWDDS0D42M2XW`.
The local launcher log is `/tmp/remag-upstream-gpu-refresh-v117.log`.
It reuses the qualified v116 image/binaries and previously selected native
ELPA two-stage/block-64/one-thread configuration. On one A100-SXM4-80GB
(UUID `GPU-addc9607-ff7a-25e7-65c8-6e5dab3b2ba8`), three fresh paired repeats give
upstream x/z totals 494.506, 495.392 and 495.223 s, versus REMAG totals 27.282,
26.864 and 26.566 s. The median pair time falls **495.223 → 26.864 s**, a qualified
**18.4347× end-to-end speedup over upstream GPU SIESTA**. Direction-specific
median ratios are 18.10× (z) and 18.81× (x). This includes initialization, SCF,
final observables and requested exports, and excludes build/provisioning and
evidence transfer. Iteration counts differ, so this is time to an equivalent
converged result, not an equal-iteration microbenchmark.

All 17 scientific/export rows, six direct comparisons and 11 MAE comparisons
pass. MAE errors are zero at printed precision; separate x/z native diagnostics
confirm GPU execution. Traced/profiled runs are excluded from timing. The run
completed in one uninterrupted attempt. See the existing
[qualification](../results/upstream-gpu-refresh-v117-20261008T180328Z/qualification.json),
[ledger](../results/upstream-gpu-refresh-v117-20261008T180328Z/ledger.json) and
[direct comparisons](../results/upstream-gpu-refresh-v117-20261008T180328Z/direct-comparisons.json).
The upstream setting was selected in v109; this refresh does not claim global
optimality on the new allocation, broader-material performance, or a production
convergence study.

The eight-workspace diagnostic (65 iterations) takes 8.485 s in SCF:
assembly/eigensolves 5.817 s (68.55%), density/grid/Hartree/XC 1.130 s (13.31%),
density matrices/normalization 0.564 s (6.65%), scalar reductions 0.510 s (6.00%),
Hamiltonian updates 0.361 s (4.26%), mixing/convergence 0.084 s (0.98%), and
occupations 0.015 s (0.17%). This is instrumentation for choosing further work,
not an additional speedup result. The spectrum stage remains the largest SCF
cost. An experimental job-major solver bank is now staged in
`build/scf-batched-eigensolver-v118`, with independent device convergence and
failure state per job. Its local eigensolves and rotations combine the tile
banks from independent k points in one library batch; converged jobs freeze
while remaining jobs continue. Generalized reduction/back-transformation and
density accumulation preserve the existing arithmetic/order. A smaller final
wave has its own fixed-size plan. No new SCF host transfers are introduced by
the code; this is not a measured residency certification.

The v118 experiment completed without a performance improvement; it is not
promoted to production. Its single-GPU harness reused the qualified v116 image
and retained that exact library as the eight-lane control. v117's allocation
release was confirmed before starting it. A first image build stopped before
GPU work because its new test target was requested before cached CMake
regeneration; app `ap-xBlvSypfTp8brI4MuYvo1I` is stopped/released. The corrected
run is `scf-batched-eigensolver-v118-20261008T184324Z`, app
`ap-Z135IU2ogcaRsJyiWxNpv5`; both build logs and its allocation-release record
are saved with the [screen results](../results/scf-batched-eigensolver-v118-20261008T184324Z/summary.json).
No allocations overlapped.

The focused real/complex check passes independent convergence freeze,
Cholesky-failure isolation and graph replay. All five FePt16-z material screens
pass the existing upstream scientific/export/convergence gates. On one
A100-SXM4-80GB, the accepted eight-lane solver takes 16.805 s (72 iterations).
Batches of 1/2/4/8 jobs take 50.225/30.223/22.292/18.221 s respectively. The best
batch needs fewer SCF iterations (66) but more cycle time per iteration:
0.17616 s versus 0.13034 s accepted. Its roughly 28.2 MiB workspace saving does
not compensate for the measured slowdown. These are screening observations,
not a repeated timing qualification or a broader claim about batching. Because
no candidate improves the screen, paired x/z repeats and additional regression
runs were skipped. The batched path has no MAE qualification and remains only
in the experimental build directory; the accepted production pool retains its
qualified 18.4347× direct upstream-GPU result. Further work should target a
measured cost within the existing pool rather than promote this slower layout.


The conventional eigenbasis-reuse change developed in
`build/scf-seeded-eigensolver-v119` is integrated as opt-in `--solver-warm-start`
(`REMAG_SCF_WARM_START=1`). It targets the prior pool's dominant eigensolve stage. Each
physical k-point/spin job keeps its complete padded orthogonal eigenbasis on
the GPU; sharing a solver lane across jobs does not share the saved basis.
Before reuse, Cholesky QR reorthogonalizes the saved basis, then the solver
transforms the current generalized-reduced Hamiltonian into that basis and
runs the same FP64 block-Jacobi convergence criterion. Successful solves update
the saved basis; failed solves invalidate it, and geometry reconstruction
starts fresh. The SCF mixer, requested outputs and scientific tolerances are
unchanged. This is conventional eigensolver reuse, not learned SCF.

The existing nested-graph checker passes a short real/complex changing-H
trajectory including padded states, failed Cholesky and recovery. Warm problems
use two Jacobi sweeps instead of four in that diagnostic. The matched FePt run
`scf-seeded-eigensolver-v119-20261008T190153Z` qualifies a **1.5675× incremental
end-to-end speedup** over the accepted eight-workspace implementation on one
A100-SXM4-80GB. Median x/z pair times are **25.732 → 16.416 s**; paired totals are
28.742/16.614, 25.732/16.416 and 24.502/16.398 s. All 14 scientific/export rows,
three direct paired comparisons and their MAE checks pass; direct MAE error is
zero at printed precision. Median cycle time per iteration falls from 0.13020
to 0.05737 s. Persistent workspace increases by about 10.1 MiB. These are fresh
same-allocation measurements; do not multiply this ratio by the separately
measured 18.4347× upstream-GPU result. See the existing
[summary](../results/scf-seeded-eigensolver-v119-20261008T190153Z/summary.json).
The performance app `ap-lJWzjV8BGvsPa7NLdaAVB0` is stopped/released.

Focused regression `scf-seeded-regression-v119-20261008T190844Z` passes
nonparallel full-SOC Fe3 and the already-tight Fe54 collinear comparison.
Fe16 noncollinear passes observable, Hamiltonian and real-space-density checks
but fails saved-DM coefficients: 5.6681e-7 versus the unchanged 1e-7 gate at its
original DM=1e-6, H=1e-5 eV SCF stopping settings. The original failure is retained
and the run stopped before relaxation and profiling. Its app
`ap-hPbKePQaFJIR17LJ6OFH4n` is stopped/released.
The completed [v119b follow-up](../results/scf-seeded-regression-v119b-20261008T191317Z/qualification.json)
passes all three remaining checks. A fresh upstream CPU/candidate Fe16 pair at
matched DM=1e-9 and H=1e-8 eV stopping criteria reduces the saved-DM error to
6.107e-10, below the unchanged 1e-7 gate. This resolves that matched tighter
protocol; the original loose settings remain unqualified for warm start. Fe2
relaxation passes all nine geometries (saved-DM error 6.2993e-8), and FePt-z passes
the optional diagnostic stage profile. App `ap-ksNTKaUjN1LoB3bz0RAMuS` is
stopped/released. All five promoted native source hashes match the executed
candidate. CLI forwarding, runtime mode validation and legacy result parsing
pass local checks (29 saved candidate results and 10 existing regression tests).
The [integration record](../results/scf-seeded-regression-v119b-20261008T191317Z/production-integration.json)
records exact source identity and qualified scope. Warm start remains off by
default. The v117 direct upstream-GPU comparison predates the subsequent
clustered-density kernel; its measured configuration remains archived.

The [new diagnostic profile](../results/scf-seeded-regression-v119b-20261008T191317Z/stage-profile.json)
contains 75 iterations and 4.182 s of SCF cycle time: assembly/eigensolve takes
1.106 s (26.4%), density/grid/Hartree/XC 1.309 s (31.3%), density/normalization
0.649 s, scalar reductions 0.581 s, Hamiltonian 0.417 s, convergence/mixing
0.097 s and occupations 0.017 s. Eigensolve share has fallen from 68.6%; the
next optimization should follow the new density/grid bottleneck and measured
setup costs. Profiled timings are diagnostic, not the performance qualification.


The next measured experiment is staged in `build/grid-cluster-density-v120`.
It keeps all eight sub-grid density accumulators in registers, sharing each
orbital-pair lookup and density-matrix read across a SIESTA nsm=2 cluster.
Pair assignment and the FP64 per-point reduction tree are unchanged; other
cluster sizes retain the existing path. No extra persistent storage or SCF
host transfer is added. The completed single-GPU FePt comparison
`grid-cluster-density-v120-20261008T193307Z` measures **1.10057× incremental
end-to-end speedup** on one A100-80GB PCIe: median complete x/z pair time is
18.3596 s for accepted warm-start REMAG and 16.6819 s for clustered density.
The three pair totals are 18.5260/16.4230, 18.1411/16.7540 and 18.3596/16.6819 s.
All 16 calculations, six direct science/export comparisons, six CPU MAE checks
and three direct MAE checks pass (zero direct MAE error at printed precision).
The optional profiles are excluded from performance timings. Median SCF time
per iteration falls from 57.35 to 44.08 ms. The diagnostic density/grid stage
falls from 17.719 to 5.236 ms per iteration; other stages remain similar. This
is an incremental measurement, not a new upstream-GPU claim. See the
[summary](../results/grid-cluster-density-v120-20261008T193307Z/summary.json).
The performance app `ap-WWsl3qXUZ7ZT1OhI01kggT` is stopped/released.

The completed focused regression
`grid-cluster-regression-v120-20261008T193841Z` passes nonparallel full-SOC Fe3,
tight collinear Fe54, tight noncollinear Fe16 and original Fe2 relaxation with
all nine geometries. Saved-DM errors are respectively 2.489e-8, 4.714e-9,
1.132e-9 and 7.482e-8, all below the unchanged 1e-7 gate. Full-SOC and final
nonparallel moment evidence pass. Established pristine upstream CPU oracles
and identical inputs are reused; no comparison tolerance changes were made.
The native grid source is now promoted with its exact executed hash. All six
recently changed native files and the two runtime readers match the measured
candidate. See the [integration record](../results/grid-cluster-regression-v120-20261008T193841Z/production-integration.json).
Both apps are stopped/released; no GPU remains allocated.

The new profile still assigns 14.85 ms/iteration to assembly/eigensolve,
8.77 ms to density/normalization and 7.88 ms to scalar reductions. Inspection
of `native/scf_scalars.cu` identifies a concrete next target: charge reduction
assigns each sparse row to one thread, traversing that row serially; a FePt
spinor calculation has too few spatial rows to occupy the GPU well. Parallel
sparse-row dot products motivated the following measured experiment.
Setup/geometry costs also remain significant.


The sparse-row charge change was developed in `build/scf-row-charge-v121`.
A CUDA block now cooperates across each row's stored entries, instead of
assigning the whole row to one thread. It retains four charge partials per
block, still excludes sparse padding and preserves the SOC offdiagonal
averaging. The row partial count is capped at 1024 blocks; there are no new
SCF transfers, allocation calls or host decisions. Parallel summation changes
roundoff, so the existing scalar checker is reused against pristine upstream
energy/charge/residual routines before the existing FePt science/MAE/timing
comparison. The completed performance and regression results below now qualify
its production integration.
Two packaging attempts stopped before any material calculation: the cached
image first lacked the existing Python checker, and the old checker then
omitted CUDA library-search flags when linking its pristine reference wrapper.
Both apps are confirmed stopped/released, and failure logs remain in their
result directories (`194523Z` and `194805Z`). The staged wrapper now retains the
upstream linker search flags and initializes/finalizes MPI for this upstream
build. The existing CPU reference check runs in the CPU image-build step before
GPU allocation. The corrected launch log is
`/tmp/remag-scf-row-charge-v121-final.log`.


The corrected run `scf-row-charge-v121-20261008T195053Z` completed and
qualifies **1.07177× incremental end-to-end speedup** over accepted v120 on
one A100-SXM4-80GB. The three complete x/z pairs are 16.5445/15.8565,
17.9612/15.5466 and 16.9945/15.8595 s; median times are **16.9945 → 15.8565 s**.
All 16 material/diagnostic calculations, six direct science/export comparisons,
six CPU MAE checks and three direct MAE checks pass. The direct MAE differences
are zero at printed precision. The existing scalar checker passes all 39
scenarios, including sparse padding, all spin layouts and failure/recovery;
maximum normal-case scalar error is 1.066e-14.

Median SCF time per iteration falls from 45.46 to 38.58 ms. Separate optional
profiles show scalar reductions falling from 7.783 to 0.579 ms per iteration;
the other non-eigensolver stages are nearly unchanged. Persistent workspace
increases by only 4,576 bytes in FePt16. These diagnostic timings are excluded
from end-to-end qualification. This is a fresh same-allocation comparison against
REMAG, not a combined or refreshed upstream-GPU speedup. See the
[summary](../results/scf-row-charge-v121-20261008T195053Z/summary.json).
Performance app `ap-0XVk3zXHQo4JXSoQE9GBHz` is stopped/released.
The completed `scf-row-charge-regression-v121-20261008T195606Z` passes all four
focused cases against established pristine CPU oracles and unchanged gates.
Saved-DM errors for nonparallel full-SOC Fe3, tight Fe54, tight Fe16 and original
Fe2 relaxation are 1.9113e-8, 4.8610e-9, 5.8680e-10 and 6.5016e-8 respectively,
all below 1e-7. Nonparallel moment/full-SOC evidence and all nine relaxation
geometries pass. The native scalar source is promoted with its exact executed
hash; all seven recently changed native files match this measured candidate.
See the [integration record](../results/scf-row-charge-regression-v121-20261008T195606Z/production-integration.json).
Regression app `ap-28axYjDok6TxbgTNrw1KPa` is stopped/released; no GPU remains
allocated.

The latest diagnostic SCF shares are assembly/eigensolver 40.7%, density and
normalization 23.8%, Hamiltonian 15.2%, density/grid/Hartree/XC 14.3%, convergence
and mixing 3.5%, scalar reductions 1.6%, and occupations 0.6%. For one unprofiled
FePt-z repetition, SCF takes 2.469 s of the 8.028 s complete process; the existing
state-initialization timer has 2.050 s exclusive time. The next choice should
weigh initialization/data preparation against the remaining eigensolver and
density costs. These measurements do not yet isolate individual initialization
boundaries or establish performance on other spin layouts and larger systems.

The completed initialization profile
`setup-profile-v122-20261008T200319Z` passes the existing FePt-z scientific and
export gates. Its diagnostic process takes 8.771 s, including 1.449 s building
the overlap radial table, 0.175 s for the kinetic table, 0.064 s for harmonic
expansion and 0.483 s staging the two-center tables. The GPU electronic cycle
takes 2.554 s across 69 iterations. These diagnostic timings motivate a setup
optimization; they are not a new performance comparison. The profiling app is
stopped and released.

The `build/radial-batch-v123` candidate batches the pinned radial
Fourier algorithm with FP64 cuFFT, including the direct small-q integration and
endpoint convention. The existing CPU phase factors, interpolation and table
compression remain. Explicit radial inputs are uploaded once per table and the
transform bank is downloaded once for compression; no SCF transfer is added.
The experiment retains the exact v121 executable/library as its timing control
and runs existing FePt science/export/MAE checks, three complete directional
pairs if screening improves, then the focused magnetic regressions on the same
single allocation. The completed measurement and integration below qualify this
change as an opt-in production path.

Run `radial-batch-v123-20261008T201716Z` measures **1.14164× incremental
end-to-end speedup** over v121 REMAG on one A100-SXM4-80GB. Median complete
FePt x/z pair times are **15.5194 → 13.5940 s**. All 14 FePt calculations,
six direct scientific/export comparisons, six CPU MAE comparisons and three
direct MAE comparisons pass; direct MAE errors are zero at printed precision.
Iteration counts vary; median SCF seconds per iteration are nearly unchanged
(37.85 versus 37.56 ms). Median inclusive state-initialization time falls from
2.4565 to 1.3035 s. No refreshed or multiplied upstream GPU claim is made.

The same allocation then passes nonparallel full-SOC Fe3, tight collinear Fe54,
tight noncollinear Fe16 and all nine Fe2 relaxation geometries against the
existing pristine CPU outputs. Maximum saved-DM errors are respectively
1.7282e-8, 5.0170e-8, 4.9314e-8 and 7.9634e-8, below the unchanged 1e-7 gate.
The native sources and generated Fortran source are promoted with exact measured
hashes. The standard integration seam and transfer-accounting tests pass (4/4);
the CLI forwards all 91 parameters consistently. The new
`--gpu-radial-tables` / `REMAG_RADIAL_BATCH=1` setting requires production
integral forces and defaults off. See the
[integration record](../results/radial-batch-v123-20261008T201716Z/production-integration.json).

FePt setup adds 1,132,152 explicit upload bytes and 84,164,800 download bytes
across three table batches. The bank is transient; compressed tables retain
their existing persistent GPU layout. SCF explicit copies remain zero in all
six candidate performance repeats and all four regression cases. CPU phase
adjustment, spline construction, table compression and geometry consumers remain;
this is not fully resident
initialization. The next setup investigation can reduce the downloaded bank or
move its remaining consumers, using the measured 1.30 s state-initialization
cost to judge its value against remaining SCF work. Simpler spin layouts still
need matched upstream GPU performance measurements. App
`ap-s3bNuEK19PLWW7kLYJ7kXP` is confirmed stopped with no containers or allocation.

The GPU compression experiment in `build/radial-compact-v124` moves phase
adjustment, radial division and first-match proportionality compression onto
the device. Eight warps search canonical functions independently and choose the
earliest matching index, preserving the upstream order and absolute tolerance.
Only compact values and maps return to the remaining CPU spline/angular
construction. The full FePt transform-bank download falls from 84,164,800 to
2,599,592 bytes, a 96.91% reduction.

The initial run `radial-compact-v124-20261008T203248Z` is **unqualified** and
unpromoted. All ten completed calculations pass against upstream, but the second
direct FePt-z pair has a saved-DM error of 1.06722e-7 against the unchanged 1e-7
gate. The control/candidate errors against CPU are 6.1117e-8 and 4.7645e-8;
their overlap matrices differ by only 1.04e-17 and table counts match. This
suggests sensitivity to the SCF stopping point, rather than demonstrating a
compression defect or resolving the failure. The original evidence and failed
qualification remain intact. Its app is stopped/released.

The completed follow-up `radial-compact-v124b-20261008T204611Z` uses the identical
numerical binaries with fresh pinned upstream CPU references and equally
tightened FePt stopping criteria (DM 1e-9, H 1e-8 eV). CPU reference construction
ran in the image build before GPU allocation. All 14 FePt calculations, six direct
science/export comparisons, six CPU MAE checks and three direct MAE checks pass.
The largest direct saved-DM error is 5.8168e-9; direct MAE errors are zero at
printed precision. Comparison gates are unchanged. The original looser-input
failure remains unqualified; this result applies to the stated tighter protocol.

On one A100-80GB PCIe, the three complete accepted/candidate pair times are
13.7325/13.4257, 13.5176/13.3262 and 13.5336/13.1305 s. The median improves from
**13.5336 to 13.3262 s (1.01557×)**, with every pair faster. This is a modest
incremental gain, not a new upstream GPU comparison. Median inclusive
state-initialization time falls from 1.0365 to 0.751 s. The explicit setup
download falls by 96.91%; requested outputs and explicit SCF copy counts remain
unchanged.

The same allocation passes nonparallel full-SOC Fe3, collinear Fe54,
noncollinear Fe16 and nine Fe2 relaxation geometries against their existing
CPU oracles. Saved-DM errors are 4.2451e-8, 5.3802e-8, 5.1072e-8 and 6.7933e-8,
all below 1e-7. The native and generated Fortran sources are promoted with exact
measured hashes; all four existing integration/accounting tests pass. The
existing opt-in `--gpu-radial-tables` setting selects the compact implementation.
See the [integration record](../results/radial-compact-v124b-20261008T204611Z/production-integration.json).
CPU splines, angular construction and geometry consumers remain; the downloaded
compact tables are still uploaded to the existing persistent force owner.
App `ap-rKsozp5Ca3YF31zKD7YniV` is stopped with no remaining containers or GPU
allocation. Larger SOC support and matched simpler-spin performance remain
outstanding; further changes should follow their measured costs.


The extended magnetic harness and its independent raw rechecker are prepared.
Thirteen prerequisite controls pass, including rejection of incomplete attempts,
changed binaries/settings, a mismatched native solver winner and an unreleased
allocation. The rechecker regenerates the magnetic inputs from archived code,
replays scientific/export/offload gates and directly checks all three CPU/native/
REMAG MAE pairings for FePt128. Large matrix exports are checked two calculations
at a time. No extended GPU run has launched or qualified yet.

An initial core/neutral-field GPU shadow is staged in
`build/atomic-fields-shadow-v110`. It prepares the existing compact atomic owner
before SCF, evaluates both fields on device, and retains full raw CPU/GPU arrays
for independent comparison. CPU fields remain authoritative. The early owner is
checked for reuse and kept through force completion. Three changed Fortran units
compile, three C++ units pass syntax checks, eight targeted regression checks
pass, and ten raw-auditor controls pass. All 363 runtime source files are frozen;
the 44-stage material audit, raw-output recovery and prerequisite-checked launcher
are prepared. The launcher correctly reports the unrun v107/v108 qualifications
as pending. CUDA integration and GPU/material validation remain pending behind
the current magnetic comparison and those qualifications. This staged change
removes no production uploads yet and establishes neither strict residency nor
a speedup.

A production successor is in development in `build/atomic-fields-production-v111`.
It skips CPU core/neutral-field generation and leases device-generated fields to
SCF, retaining them through force completion. It introduces no field/status
uploads or status readbacks; failed device generation propagates nonfinite
fields into the existing SCF validation. Optional host NaN poisoning checks that
the removed CPU producers are not still required. A requested VNA export makes
one explicit final download through the original SIESTA writer.

The staged host integration compiles (three Fortran units and six distinct C++
units with strict syntax checks), and 25 lifecycle/accounting reader controls pass. The VNA reader
handles binary output and NetCDF's scalar `GRID/Vna` in Ry, with gates fixed
before GPU execution. Twenty-one targeted checks pass, including corruption,
units, ordering, precision and prior density-output behavior. Fresh pristine
CPU binary exports pass for Fe collinear, Fe noncollinear and FePt SOC; all 39
existing corpus inputs/manifests remain byte-identical. Six VNA output variants
are prepared. Eighteen additional controls bind their full physical inputs to
retained qualified parents and reject changes to coordinates, spins, k points,
pseudopotentials, convergence or basis. The 39-case material verifier, raw-output
recovery, 45-stage independent audit and final qualifier are prepared.

The real GPU owner probe is also prepared: eight scenarios exercise analytic
fields, invalid radial data, host poisoning and 212 expected lifetime/error
rejections. Its reader passes 23 synthetic rejection controls; native GPU
execution is pending. Preparation tightened the exact SCF-convergence flag and
requires completed forces before any VNA export. All 373 runtime files are
frozen, with 31 changed/new files relative to v110. The launcher correctly stops
before Modal because v110 qualification has not run. Actual NetCDF execution,
CUDA integration and all v111 GPU/material validation remain pending. This
successor is prepared conditionally and is not promoted. The accepted runtime
and the running magnetic comparison are unchanged.

The next S/T setup component is isolated in `build/initial-pair-matrix-v113`.
A [fresh CPU reference and independent replay](../results/preflight-initial-pair-matrix-v113/preparation.json)
pass 104 cases using literal pinned overlap/kinetic scatter arithmetic and the
atomic-density diagonal correction, backed by 18,500 pristine `get_matel`
queries. Cases cover strict cutoffs, repeated images, unsorted/duplicate sparse
columns, inactive padding, optional gradients and 1/2/4/8 density banks. Eight
sensitivity controls catch omitted/partial SOC seed correction, incorrect
cutoffs and altered scatter/add behavior. The proposed device reduction and
atomic correction are implemented as an isolated component, with 456 prepared
capture/replay and failure scenarios. CUDA compilation, execution, composition
with the device radial evaluator and all production/material qualification are
pending. No production transfer has been removed by this preparation.

The composed v113 probe now derives cutoffs/distances and selection on device,
evaluates selected S/T queries using the existing radial kernel, then assembles
the sparse matrices and corrects atomic D on the same stream. The staged radial
source preserves the accepted evaluator and adds an entry point for selected
queries; the accepted source is unchanged. Its
[134 CPU composition references](../results/preflight-initial-pair-matrix-v113/composition-preparation.json)
recheck against retained pristine tables and literal Fortran output. The 286
prepared GPU scenarios include s/spd/high/subset tables, all spin layouts,
131,073-query loop coverage, graph replay with device-rescaled tables, and
strictly excluded queries with poisoned unused tables. Intermediate radial
values, cutoff sums and distances are not uploaded by this composed adapter.
These are prepared tests; CUDA compilation and all GPU execution remain pending.

The component's [guarded launch preparation](../results/preflight-initial-pair-matrix-v113/launch-preparation.json)
passes 36 local controls and binds 46 source/reference files. The image compiles
before GPU allocation; its two component checks run sequentially in one bounded
GPU call. The real launch command stops before Modal while independently checked
nonparallel SOC qualification and its allocation release are missing. A separate
raw rechecker binds the completed attempt, app release, invocation, source archive,
library and prerequisite, then replays every numerical/status/accounting gate.
Its completed-run path has not executed yet. This preparation does not change the
running FePt benchmark or promote the component into the accepted runtime.

The [accepted v103 profile and consumer inventory](../results/preflight-initial-pair-matrix-v113/profile-and-consumers.json)
puts S/T setup at 0.64–1.34% for the small SOC cases and 7.55% for Fe54 collinear.
This makes S/T a residency dependency rather than evidence of a large SOC timing
gain. Device overlap must become available before `new_DM`; atomic/user seeds
need the diagonal correction on GPU before initial populations, while loaded
and history density must bypass it. Requested S/gradS output and randomized
initial states require explicit support or guards. Magnetic comparison and
nonparallel SOC qualification keep priority; the accepted runtime is unchanged.

## Position against the specification — 7 October 2026

The conventional GPU electronic calculation is validated on the supported Fe/FePt
regression suite. The current work extends GPU ownership into initialization and
geometry boundaries. The strict zero-transfer/zero-I/O milestone remains open;
explicit copy counters do not establish complete memory and filesystem coverage.
The 33 cases below are input/output variants, not 33 distinct materials.

| Spec area | Current position | Remaining work |
| --- | --- | --- |
| Baseline, profiles and optimization ledger (§6, §20–23) | Reproducible pinned upstream comparisons, retained evidence and repeated end-to-end GPU measurements | Complete independent memory/I/O observation and broader CPU/hardware baselines |
| Core SCF, layouts, solver and grid (§7–11, §14) | Device-controlled SCF, mixing, convergence, density, potentials and electronic solve pass the current corpus | Remaining setup producers and strict residency qualification |
| Magnetism and required outputs (§4–5, §13) | Collinear, noncollinear, SOC, direct directional MAE, forces, stress, moments and requested exports pass current regressions | Broader physics/settings and production convergence studies |
| Repeated geometries (§5, §7) | Electronic density history remains on device across supported relaxation trajectories | Remaining CPU geometry setup and ionic updates |
| Initial mesh and atomic fields (§7, §14) | Mesh/density/Hartree material shadow qualified in v106b; direct production handoff is compiled and staged | Production substitution and initial core/neutral fields |
| Representative materials (§3) | Accepted full path includes Fe/FePt and sizes through 54 atoms | 100–300 atoms, borides, carbides, nitrides and more complex magnets |
| K-point throughput and solver/precision experiments (§11–12, §15) | Sampled k-point workloads supported | Concurrent batching/streams, alternative solvers and mixed-precision studies; extra GPUs require user permission |
| Magnetic workflows (§5, §16) | Directional energy differences and electronic exports verified | Orientation-state reuse, anisotropy fitting and downstream exchange workflow validation |
| Learned SCF (§17–19) | Deferred by the user's residency-first priority | Begin only after the conventional resident baseline qualifies |

## Initial mesh shadow qualified: v106b

[Run 20261007T211716Z](../results/20261007T211716Z/qualification-summary.json)
passes all 33 material/output comparisons, two directional MAE checks and 43
independent audit stages against fresh CPU references from the same allocation.
Its 107 geometries contain 1,514 SCF iterations. The GPU-generated orbital mesh,
atomic reference density and initial Hartree fields agree with the pristine CPU
initialization. All 492 requested raw output files were recovered; the original
launcher exited successfully and Modal reports the app stopped with no containers.
An audit snapshot initially omitted `mulliken.F`; supplying the identical pinned
source fixed the audit path without changing a verifier or scientific gate. The
initial audit logs are retained. This qualifies the initialization shadow only;
production mesh substitution and strict residency remain unqualified. It is not
a performance measurement.

## Current qualification and matched performance: v103 compact GPU queries

[Run 20261007T184850Z](../results/20261007T184850Z/qualification-summary.json)
passes all 99 material comparisons, six directional MAE checks and 43 independent
audit stages across 33 supported input/output cases with three cold GPU
repetitions. Its 321 geometries contain 4,599 SCF iterations with zero explicitly
recorded complete-cycle H2D/D2H bytes or copy calls. All 984 requested raw outputs
were recovered with matching completed-attempt markers. Source/component
integrity verifies 251/1,755 hashes, and 253 archived runtime files match the
frozen implementation. The original launcher exited zero and Modal confirms
the app stopped with no remaining containers.

[Matched timings](../results/20261007T184850Z/PERFORMANCE.md) on one
A100-SXM4-80GB measure 6.11× for Fe16 NC, 7.32× for FePt SOC history relaxation,
14.16× for Fe54, 2.84× for Fe2 history relaxation and 3.35× for NC Fe2 history
relaxation against fresh serial upstream CPU references. Setup, SCF, forces and
requested outputs are included; cloud provisioning/builds and evidence transfer
are excluded. GPU medians use three cold repeats; CPU has one measurement per
case. Iteration counts differ in Fe54 and some history cases. This qualifies the
combined implementation, not the incremental effect of compact queries or a
comparison with optimized multicore CPU. Strict memory/I/O residency remains
unqualified, and CPU setup/ionic consumers remain.

## Promoted implementation: v102 compact GPU nonlocal queries

[Run 20261007T180248Z](../results/20261007T180248Z/qualification-summary.json)
passes all 33 input/output cases, two directional MAE checks and 42 independent
audit stages. Its 107 geometries contain 1,530 SCF iterations with zero explicitly
recorded complete-cycle H2D/D2H bytes or copy calls. All requested scientific
outputs pass; 492 raw files were recovered from the same completed attempt.
Source/component integrity verifies 251/1,755 hashes, with 252 runtime files
matched to the frozen implementation. The launcher exited successfully and the
single GPU allocation is released.

Production nonlocal queries now expand on GPU from compact neighbor/projector
metadata. Across this suite, 61,996,360 staged bytes replace 1,335,365,728 bytes
of expanded query uploads: a 95.36% reduction for that setup representation.
H0 matrices and integral derivatives remain resident through SCF and forces;
the existing four-byte setup status readback per geometry is unchanged.
The [qualified implementation is promoted](../results/preflight-nonlocal-query-integration-v102/promotion.json).
This run reused CPU references and measured one GPU repeat; it qualifies
scientific/output agreement, not a new speedup or strict residency.

The [current setup profile](../results/20261007T180248Z/SETUP_PROFILE.md)
and [remaining routine measurements](../results/preflight-nonlocal-query-integration-v102/remaining-profile-early.json)
identify orbital values on the mesh (`PHION`) and atomic-grid force work as
next investigation targets. CPU neighbor selection, kinetic/overlap setup and
ionic updates also remain. Fresh matched performance is qualified above.

The [orbital-mesh primitive](../results/preflight-orbital-mesh-v104/status.json)
is GPU-component qualified separately. Pristine upstream checks show that `PhiOnMesh/phiatm`
and force `all_phi` values differ at cutoff/angular boundaries. The new path
preserves the mesh-specific rules and FP32 storage while leaving existing force
arithmetic unchanged. Its local CPU rehearsal matches 15,977 numerical values,
but the strict bit gate remains failed for eight signed-zero differences;
that diagnostic is retained. All 44 CUDA checks now pass against the pristine
CPU oracle on the same Linux allocation, including exact stored FP32 bits and
changed-geometry replay. Material qualification remains pending.
The [consumer inventory](../results/preflight-orbital-mesh-v104/ownership-boundaries.json)
requires migrating atomic reference density and its initial Hartree-energy
consumer alongside the stored mesh values, avoiding a replacement download.

The [initial atomic-density and Hartree-energy handoff](../results/preflight-initial-mesh-density-v105/status.json)
is GPU-component qualified alongside that mesh primitive. Thirteen local checks against
pristine upstream `rhooda/reord/poison` fixtures pass, including empty rows and
invalid-data controls; the largest atomic-density difference is
8.88e-16. The device handoff writes the initial Hartree energy directly into
the grid configuration and leaves an invalid energy on failure. A single
allocation probe ran the 44 mesh-value checks and 30 density/Hartree
checks sequentially after v103's verified release. The first image build failed
because its two component CMake minima exceeded the image's installed version;
that failure is retained, both minima now match production's supported 3.20,
and the corrected [run 20261007T202010Z](../results/20261007T202010Z/component-verification.json)
passes all 74 checks and their independent retained-array recheck. All 38
archived source files match the frozen inputs. The allocation is released.
Real-material integration and production promotion remain pending. These
component checks do not yet establish the complete GPU producer/consumer chain.

The [early mesh owner and material-shadow integration](../results/preflight-initial-mesh-owner-v106/status.json)
has completed all 33 material calculations in a diagnostic after the component allocation's
verified release, with 354 source files frozen. It prepares the existing
compact orbital-basis owner during initialization, removes its Fortran module
dependency on `m_dhscf`, and reuses that owner for force derivatives later.
The new GPU chain generates mesh values, atomic reference density, initial
Hartree potential/energy/stress and a device Harris-energy handoff. Full raw
arrays are retained for comparison; mesh values and atomic density use upstream
CPU references, but the material Hartree reference currently uses the existing
GPU Hartree bridge. A separate pristine CPU Hartree comparison is still needed;
the preceding component run has that independent CPU oracle. This diagnostic is
explicitly excluded from speedup claims. Twelve Fortran compilation units,
three strict C++ syntax checks, 18 evidence-reader acceptance/rejection checks,
six existing transfer/attempt checks and 12 regression/patch checks pass locally. Full
real-material qualification has not passed; production grid/SCF consumers
still use their existing initialization until that evidence is available.

The diagnostic run is `20261007T202743Z` on the single authorized GPU.
Its live attempt checkpoint is consistent; 257 recorded source hashes pass and
259 archived runtime files match the frozen implementation. The local full-run
audit and qualifier are prepared for 43 stages, including complete retained
mesh arrays, scientific outputs, forces and MAE. Its live ledger passes 26 of
33 comparisons: six output variants fail exact overlap-matrix equality, and
three SOC variants fail the density-matrix tolerance (seven distinct cases).
Energy, force, stress and magnetic-moment checks pass. The reused CPU references
come from a different allocation; whether fresh references explain these
differences remains incompletely verified. Six fresh CPU controls printed passing
candidate comparisons and failing reused-reference comparisons, but the container
ended before their separate archive was saved. These observations do not qualify
the batch. Tolerances are unchanged. The original failed run is now recovered,
including all 492 requested raw outputs, and its allocation is released.

An [independent pristine CPU Hartree diagnostic](../results/20261007T202743Z/initial-mesh-pristine-hartree-verification.json)
recomputes all 107 material geometries from the retained upstream atomic density
and exact cell headers. Potential, energy and stress comparisons all pass, with
maximum absolute errors 4.71e-14, 3.24e-12 and 2.50e-15 respectively. The wrapper
first reproduced all seven retained upstream component fixtures exactly. This
closes the missing Hartree numerical comparison but does not erase the seven
scientific output failures or qualify production mesh substitution.

The [corrected v106b shadow](../results/preflight-initial-mesh-owner-v106b/status.json)
is running as `20261007T211716Z` on one GPU after the preceding allocation's release. All 33 cases will
use fresh CPU references on the same allocation. Its host Hartree bridge is
disabled specifically for the mesh-shadow CPU-reference mode, so upstream
`poison` supplies an independent initial potential/energy/stress reference while
device SCF and the GPU shadow retain their unchanged Hartree primitives. Twelve
Fortran and four C++ compilation checks, 18 full-array reader checks and 15 Python
regressions pass. All 354 execution sources are frozen; full material qualification
remains pending.

The next [production handoff](../results/preflight-initial-mesh-production-v107/status.json)
is staged separately in `build/initial-mesh-production-v107`. It skips CPU
mesh-value evaluation, atomic-density generation and the initial Hartree solve;
the grid contracts borrowed GPU mesh values, SCF borrows the GPU atomic density,
and the initial Hartree energy is bound by a device kernel. Producer errors
invalidate that energy and flow through the existing SCF grid validation,
without a new setup status or energy download. Ownership is checked through
accepted SCF and final-force completion before another geometry can replace it.
The diagnostic poisons and verifies the unused host arrays with NaNs.
Thirteen Fortran units and six C++ units compile locally; 18 existing Python
checks and 15 accounting acceptance/rejection checks pass. The changed CUDA unit
and six C++ units also compile with the actual CUDA toolchain in the existing
allocation, without executing GPU kernels. The poisoned full-material run and
production qualification are pending.
The v107 runtime is frozen with 358 hashed source files. Its launch helper now
requires corrected v106b qualification and allocation release, and its planned
material run also uses fresh CPU references. It remains unlaunched.

The next [initial core-density and neutral-potential component](../results/preflight-atomic-fields-v108/status.json)
is staged separately as v108. Twenty-two CPU field checks across eleven fixtures
match pristine upstream grid generation and reordering bit for bit. The ordered
CUDA deposition implementation, direct GPU radial-evaluation composition and
their test adapter compile and link. The portable CPU oracle also passes all
22 field comparisons using the pinned source manifest. The probe is frozen
with 20 source files and prepares 72 deposition plus 76 composition scenarios,
including device-only changed-table replay and invalid-input controls. GPU
execution remains pending after v107's qualification and allocation release.
The design preserves upstream accumulation order and preallocates its working storage.
Early atomic-state ownership and direct SCF consumption still need integration
and full material qualification. This component has no speedup or residency
qualification yet.

## Earlier matched performance: v98 production GPU nonlocal H0

[Run 20261007T145602Z](../results/20261007T145602Z/qualification-summary.json)
passes 99 material comparisons and six MAE checks across 33 cases and three
cold GPU repetitions. All 43 audit stages pass, including full requested-output
coverage and the retained nonlocal H0/integral device leases. The unused host
H0 matrices are not poisoned in this production timing run; the separate
poisoning control and complete matrix shadow remain its prerequisite evidence.
Source/component integrity verifies 249/1,755 hashes, with 250 runtime files
matched to the frozen implementation. All 984 raw binary outputs were recovered
from the same completed attempt. The launcher exited successfully and its GPU
allocation is released.

The 321 geometries contain 4,620 SCF iterations with zero explicitly recorded
complete-cycle H2D/D2H bytes and calls. H0 matrix transfers are zero. Its setup
status readback is four bytes per geometry, totaling 1,284 bytes before SCF.
[Fresh matched performance](../results/20261007T145602Z/PERFORMANCE.md) on one
A100-SXM4-80GB measures 5.07× for Fe16 NC, 11.88× for Fe54 and 4.84× for FePt
SOC history relaxation against serial upstream CPU. Fe54 iteration counts vary;
these are solution timings, and the earlier PCIe allocation cannot isolate
H0's incremental effect. CPU geometry/kinetic-overlap setup, ionic consumers,
broader corpus coverage and complete independent memory/I/O observation remain.

The [latest isolated observer calibration](../results/20261007T164248Z/capability-recheck.json)
passes four allocation/registration controls with all 39 sources verified, but
fails migration observation. Both late-init controls fail the actual open of
`/dev/nvidia-uvm-tools` with `ENXIO`; device-node metadata alone had suggested
access. [Raw evidence](../results/20261007T164248Z/profiling-device-open-evidence.json)
preserves control PIDs and trace hashes. Filesystem coverage controls also remain
unqualified. The allocation is released; the production performance checkpoint
is unchanged. The independent setup investigation follows below.

The [seven-case setup profile](../results/20261007T165703Z/SETUP_PROFILE.md)
is complete, with seven scientific comparisons and the Gamma SOC MAE check
passing. CPU geometry/staging dominates the measured setup of sampled-k and
relaxation cases: Fe2 history relaxation takes 6.213 s preparing geometry versus
0.767 s launching/waiting for the GPU graph; SOC history relaxation takes
7.397 s versus 1.402 s. Tiny Gamma SOC cases instead spend about 0.57 s in GPU
launch/wait versus 0.034 s preparing geometry. This motivated the compact query
migration qualified above. This diagnostic reuses CPU references and does not
qualify speedup or full-corpus coverage. The full-corpus gate failures remain
retained; the [limited profile check](../results/20261007T165703Z/profile-material-verification.json)
does not replace them. The single allocation is released.

A [staged FeN adapter](../results/preflight-nitride-corpus-v100/status.json)
reproduces the two converged pristine CPU inputs exactly and preserves all 39
existing generated cases. Promotion and GPU qualification remain pending.

The [compact-query GPU component](../results/20261007T174532Z/component-verification.json)
passes 22 direct/captured checks, including exact query ordering, displacement
bits, graph replay and rejection of invalid metadata. Nine archived sources
match the frozen inputs and the complete attempt. The first run is retained:
its large replay control failed because fixture writes on the default stream
were not ordered before the nonblocking graph stream. Ordering those fixture
writes fixes the check without changing the expansion kernel. The subsequent
production integration is qualified above; this component alone makes no
speedup or full-residency claim.

The [v102 integration record](../results/preflight-nonlocal-query-integration-v102/status.json)
retains the frozen runtime, local compilation and 30 targeted Python checks,
the completed regression and the promotion record. Its graph expands compact
records before the unchanged integral/H0 kernels, with no new status readback.

Relative to the original specification, the baseline harness and supported
conventional collinear/NC/SOC electronic path are established. The current
phase is completing GPU residency through setup and repeated workflows, and
expanding scientific coverage. Current-path 100–300 atom qualification,
borides/carbides/nitrides and complex magnets, orientation reuse, anisotropy
fitting and downstream exchange workflows remain. K-point batching/streams,
alternative solver experiments and mixed precision are later optimization
work. Learned SCF remains deferred until the conventional resident baseline
is demonstrated.

## Earlier qualification: v92 final spectra and conditional H export

Run [20261007T105658Z](../results/20261007T105658Z/qualification-summary.json)
passes all 99 candidate comparisons and six directional MAE checks across 33
cases and three GPU repetitions. All 42 independent audit stages pass,
including source/component integrity, per-geometry history, initial populations,
forces, density fields, requested DM/HSX/TSDE/NetCDF output and final spectra.
The final 984 raw binary outputs were recovered with matching complete attempt
markers before the independent audits. The archive verifies 243 recorded
source hashes and 1,755 component hashes against the frozen runtime.

All 321 geometries and 4,610 SCF iterations record zero explicit complete-cycle
H2D/D2H bytes or copy calls. Eigenvalues remain on GPU until exactly one final
analysis export per calculation; occupations no longer download. Generator H
exports now depend on actual supported consumers. All 18 no-matrix controls
pass with zero H/D/E downloads across 84 geometries and 1,028 SCF iterations,
while preserving their other requested scientific outputs. Complete independent
memory/I/O observation and CPU geometry/setup and ionic work still prevent a
full-residency claim.

[Fresh matched timings](../results/20261007T105658Z/PERFORMANCE.md) give Fe16 NC
5.00× (121.809 s CPU / 24.369 s GPU median), Fe54 13.43× (230.450 / 17.164 s)
and FePt SOC history relaxation 4.73× (134.471 / 28.410 s), against serial
upstream CPU on one A100 80GB PCIe. Fe54 and history cases can have different
iteration counts. The preceding run used an A100 SXM4; cross-run differences
do not isolate incremental optimization gains. The app is stopped, the
launcher exited successfully and no GPU container remains. See the
[completed run record](../results/preflight-terminal-spectrum-v92/live-run.json).

## Qualified component: v93 nonlocal H0 assembly

The next setup candidate is in `build/nonlocal-h0-v93`.
Its two-kernel implementation validates compact topology and gathers each
sparse H0 entry in upstream accumulation order, without floating-point atomics.
A value-only specialization shares the existing SOC arithmetic and does not
read force gradients. All 676 CPU arithmetic/topology rehearsals pass against
the 104 pristine fixtures; an independent dense-angular/CSR reconstruction
reproduces their outputs and failure statuses. The auditor rejects altered
matrix values, altered row mappings and missing scenarios. Valid disjoint CSR
rows may be stored in any order; row permutation and empty-row offset controls
preserve the original matrix values. A CUDA probe covers 1,352 direct/captured
scenarios. Another 480 CPU checks show the shared force arithmetic is bitwise
unchanged and the value-only specialization ignores poisoned gradients.

The component runs through the existing single GPU function, with a frozen
54-file execution snapshot and an audit binding source, binary and complete
attempt identities before independent numerical reconstruction. Ten synthetic
provenance checks pass, including rejection of altered archives, mixed attempts,
changed native binaries and incomplete execution. Only its Modal dispatch was
promoted after v92 qualification and allocation release. Run
[20261007T115723Z](../results/20261007T115723Z/qualification-summary.json)
passes all 1,352 direct/captured CUDA scenarios and the independent retained
input/output reconstruction. Maximum scalar/SOC assembly errors are
2.665e-15 and 3.663e-15 (rounded upward). All 54 execution-source hashes,
native binary identity and the complete attempt agree. The allocation is
released; production native code remains unchanged. Pre-SCF integral ownership,
retention through forces and material integration remain pending. This component
does not qualify a material speedup or strict memory/I/O residency. See the
[candidate evidence](../results/preflight-nonlocal-h0-v93/status.json).

## Earlier v91b production density history

Run [20261007T094904Z](../results/20261007T094904Z/qualification-summary.json)
passed 81 candidate comparisons and six directional MAE checks across 27
cases. All 41 audit stages passed, with 237 geometries and 3,567 SCF iterations
recording zero explicit complete-cycle copies. This qualified production GPU
density history, initial populations and conditional terminal D export.
An initial audit snapshot omitted a pristine reference file; copying its exact
hash-verified contents repaired the environment without changing a verifier or
numerical gate. The original failure and diagnosis remain preserved. Its
performance and pending export work are superseded by v92 above.

## Earlier v90 and initial v91 validation record

The following records the preceding checkpoint and retry observations; the
completed v91b qualification above supersedes their pending status.

Run [20261007T081045Z](../results/20261007T081045Z/qualification-summary.json)
passes all 54 material comparisons and six directional MAE checks across the
18-case corpus. Independent raw-output, component, source and transfer audits
pass: 238 source hashes, 1,755 component hashes, and 1,730 SCF iterations across
78 geometries with zero explicit in-cycle copies. Initial spin/Mulliken output
matches upstream for every geometry, and one initial device D allocation is
borrowed by SCF. The standalone owner probe covers 16 cases and 264 rejected
operations. Requested TSDE/NetCDF, forces and relaxation remain equivalent.

Fresh matched medians give Fe16 NC 6.51× and Fe54 17.03× against serial upstream
CPU on one A100-SXM4-80GB (driver 580.95.05). Fe54 uses different SCF iteration
counts. These are combined solution-time results, not isolated incremental
speedups; see the [performance report](../results/20261007T081045Z/PERFORMANCE.md).
The app is stopped, its launcher exited successfully and no GPU container
remains. Full residency is not qualified.

Production density history and conditional terminal D export were promoted
after v90 qualification. Run `20261007T091252Z` stopped after its first
converged nine-geometry history case because the metadata validator incorrectly
bounded constrained-SVD rank by the history length. The augmented system has
one additional dimension. The original failed result is preserved; a separate
raw diagnosis passes all final scientific values, all initial populations,
all nine geometries and the printed coefficients (exactly). Its 95 SCF
iterations record zero explicit copies, and every terminal D download is zero.
The allocation is released; 242 source and 1,755 component hashes verify.
Only the rank validator changed for retry `20261007T094904Z`, active under one
guarded allocation with the full 27 cases and three repetitions requested.
The correction passes 92 local tests. Full history qualification is pending.
The retry has completed all 81 candidate comparisons successfully. Its remote
attempt is sealed complete; archive recovery and full independent audits are
pending.
Its live checkpoint independently verifies all 242 recorded source hashes
and 243 archived runtime files against a frozen local snapshot. These are
partial run observations; the complete material, trajectory and output
audits still gate qualification and performance publication. A separate
[checkpoint audit](../results/preflight-history-rank-v91b/partial-history-verification.json)
independently passes the first six collinear/noncollinear trajectories:
54 geometries, 577 SCF iterations, exactly matching printed extrapolation
coefficients, zero reused-density uploads and zero terminal D downloads.
Per-geometry forces, stress, moments and charges pass their unchanged gates.
Requested binary output and full batch qualification remain pending.
A later [history checkpoint audit](../results/preflight-history-rank-v91b/completed-history-verification.json)
passes all 18 history comparisons: 150 geometries and 1,622 SCF iterations
across all three spin modes, with requested DM files both enabled and disabled.
It independently reproduces per-geometry science, history coefficients,
initial population output, ownership and explicit cycle accounting. It does
not replace the remaining full-batch and requested-binary-output audits.
H/spectral exports, CPU ionic/setup work and complete independent
memory/I/O observation remain. Earlier v87 initialized H/grid on the device
for the nine base cases, and v90 adds complete initial-population/output
coverage and fresh timing evidence.

The next terminal-export candidate is isolated in `build/terminal-spectrum-v92`.
It retains eigenvalues on the GPU through force evaluation and successive
geometries, drops obsolete geometry spectra, and exports only the final
eigenvalues at their existing analysis consumer. Occupations have no host
consumer in this mode; TD wavefunction requests remain explicitly unsupported
by the new mode. The existing EIG/printed output behavior is preserved.
It also records why a host Hamiltonian is required and omits its download
when no matrix-file, legacy-analysis or diagnostic consumer is enabled.
Six new controls disable matrix files across collinear, NC and SOC fixed and
relaxing geometries, while retaining their other scientific outputs. All 27
existing input files remain byte-identical. Local C++ syntax checks, 16 Fortran
units, the NetCDF configuration and 96 Python tests pass.
This candidate has not been promoted or run on a GPU; the active retry's
native source is unchanged. Review found and removed a circular audit
dependency: terminal cycle exports must be checked before integral forces,
then final spectrum consumption, then performance publication. All three
remain mandatory performance gates. The corrected dependency graph is
acyclic and the 96 local tests still pass. Qualification must reconcile every cycle, accepted force epoch,
one final spectrum copy and unchanged scientific output before any gain is
claimed.

The measured next setup target is `Setup_H0/nlefsm`, which occupies a median
9.110 seconds (about 46%) of the accepted Fe2 relaxation runtime, including
its geometry capture work. An isolated [nonlocal H0 oracle](../results/preflight-nonlocal-h0-v93/status.json)
now retains 104 scalar/SOC fixtures from literal pinned upstream assembly
and scatter blocks. Independent dense/angular contractions reproduce them
within 1.954e-14 for scalar H0 and 1.612e-14 for SOC. The fixtures preserve
empty and excluded rows, padding, both scalar/SOC split modes, l=0..5,
duplicate columns and repeated accumulation. Scalar H0 uses the first
duplicate CSR entry; complex SOC H0 adds to every duplicate. This differs
from the final-force density lookup and must survive the setup port.
The oracle is isolated in `build/nonlocal-h0-v93`; no GPU implementation,
production change or additional speedup is qualified by it.

## v86 GPU initialization components

Run `20261007T060934Z` qualifies initial density normalization and the composed
initial Hamiltonian/SCF path as isolated components: 784 normalization checks
(including 144 rejected invalid operations) and 24 composed cases covering
collinear, noncollinear and SOC layouts. Retained GPU arrays pass a local
numerical recheck against the pristine reference fixtures; 315 source hashes
match the executed archive and working sources. The A100 allocation is released.
Maximum density normalization error is 8.882e-16; composed D error is 9.548e-15
(rounded upward). See the [qualification evidence](../results/20261007T060934Z/qualification-summary.json).

The production SIESTA initialization/history handoff is still pending. These
checks establish neither removal of the production D round trip nor a new
material speedup. At that stage, v84 was the latest qualified end-to-end performance.

## v85 density-history component

The isolated GPU history owner is implemented and qualified in
`results/20261007T055038Z`: 104 direct/captured sequences, 1,528 operations,
328 rejected invalid calls, all 1/2/4/8 spin layouts, bounded eviction,
destructive sparse/supercell remapping and all four coordinate extrapolation
methods. Raw snapshots were independently replayed locally and match the
source archive. The only GPU allocation was released. This does not change
production transfer counts or establish new performance: the SIESTA adapter,
next-geometry electronic initialization and host-D/output consumers still need
migration. That component result did not replace the then-current v84 baseline.
See `docs/DENSITY_HISTORY_RESIDENCY.md` for limits and acceptance requirements.


### Historical v91-era scope snapshot

The following table and narrative retain the earlier development state. Their
pending items and priorities are historical; the current status is at the top
of this document. The 8 October user revision supersedes exhaustive zero-transfer
and zero-I/O certification as a prerequisite. Completion of the entire original
multi-stage specification is not claimed.

The current combined route is qualified by v91b above. Its input/source evidence
and all requested outputs are retained; the single allocation is released.
[Density history and next-geometry initialization](DENSITY_HISTORY_RESIDENCY.md)
now pass the production corpus. Hamiltonian/spectral consumers and CPU setup
remain the next migrations. The table below records earlier adapter milestones
and scopes.

| Area | Current status |
| --- | --- |
| Reference implementation | Pinned upstream 5.4.2, independent build, immutable source checks, clean input directories |
| Baseline profiling | Native hierarchical timers, exclusive component attribution, individual SCF timings, wall time, convergence/iteration counts, hardware and input provenance |
| Runtime policy | Single Modal A100; one container; no concurrent remote calls; no warm pool; no retry expansion; launch lock and active-app check |
| I/O policy | New corpus inputs disable iterative DM checkpoints and retain end-of-cycle artifacts; four local upstream policy comparisons and MAE pass; other I/O remains |
| Electronic solve | Integrated FP64 real/complex generalized cuSOLVER path with reusable allocations and explicit retained eigenvector slots |
| Density matrices | GPU sparse density and energy-density accumulation for serial real/complex/spinor paths; device accumulation across k points |
| Grid contractions | Integrated GPU density and local-potential assembly; resident basis/topology; full SOC block conventions; guarded reuse of normalized device density |
| Grid lifetime ownership | Explicit geometry epochs avoid repeated basis/topology scans; debug validation and 24 normal-mode pairs, six GPU controls and FePt16 MAE pass; Fe128 median 1.60× faster than hashing control |
| SCF boundary accounting | Explicit bridge copies recorded by phase/purpose; six qualified CUPTI traces additionally expose three extra scalar downloads per solve; migrations and mapped-host access remain unknown |
| Filesystem audit | Deferred diagnostics show zero selected file operations during SCF in five cases; ten upstream comparisons, five controls and MAE pass; mappings/observer/page-fault limits remain |
| Device charge normalization | Validated through Fe128 and FePt16, including 20 untraced pairs, five controls and MAE; no demonstrated incremental end-to-end speed benefit; four-byte status and D/E downloads remain |
| Device loop control | Coupled convergence/Pulay/history passes 75 fixtures and 441 records; the blocked solver passes 36 scenarios through size 1152. On driver 580.95.05 all 111 component memcpy/memset traces verify zero H2D/D2H in their full graph regions. Whole-cycle control is now integrated experimentally; Fe54 exposes intermittent local solver failures absent from those synthetic checks |
| Device occupations | Integrated retained spectrum/FD/DM path passes 16 upstream pairs, eight GPU controls, both MAE checks and 16 CUDA traces; E/Q uploads removed from density construction; temporary exports and scalar status remain |
| Spectral-array residency | Deferred E/Q export passes 18 upstream pairs, nine controls, both MAE checks and 36 traces: zero host/device E/Q array transfers inside 365 SCF iterations; 17 post-SCF exports; larger untraced batch passes 20 pairs, five controls and FePt16 MAE; Fe128 median 95.5 s (12.04× CPU) |
| Device H/S assembly | Validated in 27 upstream pairs, 18 controls, six MAE checks and 54 traces; zero dense H/S uploads in 351 candidate SCF iterations. Eight native numerical/ownership scenarios pass; sparse H still uploads |
| Device periodic Hartree | Reusable component and opt-in adapter pass the same 27-pair diagnostic, 19 isolated device scenarios and native ownership/error checks; temporary density/potential/result transfers remain |
| PBE mesh functional | Device component passes 34 full-GridXC numerical/capture scenarios, including legacy NC, small periodic grids, vacuum interface, energy/stress and device errors; the combined grid component also passes 32 scenarios. Full grid-stage SCF validation passes 18 upstream pairs, nine controls and four MAE checks |
| Resident density/potential grids | All 32 CUDA component checks, 18 upstream pairs, nine controls, four MAE checks and 36 complete traces pass. Density/potential arrays stay on device for 435 candidate iterations. Temporary 224-byte result export and sparse H/D/E transfers remain |
| Stable-Pulay mixing | All 24 isolated CUDA checks and three reciprocal probes pass; maximum upstream trajectory error 1.083e-14. Production sparse-state integration passes nine-case regression, including reused relaxation geometries and first-mix behavior |
| SCF scalar consumers | All 39 isolated CUDA checks pass, with maximum scalar error 1.07e-14. The device-controlled cycle has no explicit per-iteration scalar transfers; the legacy host-controlled adapter still exports scalar summaries/status |
| Hamiltonian seeds and SOC | All 52 device scenarios pass; matrices are exact, energy error ≤8.89e-15 Ry. Production integration passes collinear, non-collinear, SOC/MAE, force and relaxation regression |
| Sparse electronic ownership | H/Hold/D/Dold/E/Eold and Pulay history remain resident in the nine-case traced diagnostic; 37 owner and 92 separate convergence scenarios pass. Larger five-case untraced groups pass 20 upstream comparisons, five controls and repeated performance through Fe128 |
| Scientific regression | Energy, bands/eigenvalues, Fermi energy, all density components, total/atomic moments, atom charges, forces, stress, MAE differences, relaxation geometry |
| Final grid-force inputs | Retained D/V/Vaux shadow passes nine cases / 17 geometries and both MAE pairs; maximum force error 1.674e-13 Ry/Bohr |
| Production grid forces | GPU dfscf and device orbital derivatives pass nine materials, nine sequential CPU-derivative controls and four MAE checks; net Fe54 derivative upload reduced by 1.071 GB. Compact geometry staging and legacy electronic exports remain |
| Core/neutral forces | Authoritative GPU contractions and GPU radial derivatives share force/stress accumulation with dfscf; full production/control qualification passes. Compact tables/mappings stage before SCF |
| Kinetic/overlap forces | Authoritative GPU integration passes the current 18-case repeated corpus with retained device D/E; compact geometry staging and shared final publication remain explicit boundaries |
| Scalar/SOC nonlocal forces | Authoritative GPU integration passes the current repeated corpus; SOC projector and scalar/complex force primitives pass 168 and 398 GPU scenarios with independent reanalysis |
| Observable preservation | Upstream DM, RHO/DRHO, HSX, EIG, orbital-index and geometry files retained in the run Volume |
| Magnetic corpus | Iron size ladder, tilted non-collinear initialization, Gamma spinor/SOC cases, FePt SOC directions, displaced-atom forces and CG relaxation |
| Performance evidence | Current combined v84 route measures 1.76–2.43× on small cases, 5.72× Fe16 NC and 13.30× Fe54 against serial upstream CPU. GPU medians use three cold-process repeats; Fe54 iterations vary. Later history/initialization components have no new material timing qualification |
| Full GPU-resident SCF loop | **Production integration implemented; full qualification incomplete.** FP64 32-state batched Jacobi passes all nine cases and both MAE comparisons: 372 iterations with zero explicit transfers and selected file/mapping operations. Seven CUDA-traced cases verify zero copies over 272 iterations; Fe16 faults under CUPTI and Fe54 trace coverage is pending. Final GPU observables remain |
| CUDA activity trace | Schema-2 external correlation qualifies all candidate/control traces in the nine-case sparse-state batch. Historical schema-1 phase claims require requalification. Conditional-loop attribution remains an experimental gate; migrations and mapped-host access are outside coverage |
| Broader materials | **Pending.** Sourced boride, carbide, nitride and complex candidate fixtures with validated pseudopotentials |
| Orientation/state reuse | **Pending.** Current x/z comparisons are cold starts and verify final magnetization directions |
| K-point batching and streams | **Pending.** Current solver calls are sequential on one device |
| Precision experiments | **Pending.** FP64 compute; upstream's stored FP32 basis representation is retained |
| Learned SCF acceleration | **Deferred by user direction.** Requires the fully resident conventional baseline and its scientific/performance validation first |
| Production magnetic accuracy | **Unvalidated.** Regression cutoffs, basis, smearing and k meshes need physical convergence studies |

The first profiles placed `vmat` and `rhoofd` ahead of eigensolver work, motivating the resident grid implementation. That milestone passed Fe54 at 4.10× and Fe128 at 2.31× against serial upstream, with force, relaxation, non-collinear and SOC regressions. Those earlier single-run measurements and the unchanged scientific gates remain in the [historical validation record](VALIDATION.md).

The latest accepted [geometry-lifetime milestone](LIFETIME_VALIDATION.md) avoids repeated host scans of immutable basis/topology. Six debug pairs and 24 normal-mode pairs pass, including relaxation, non-collinear, SOC, Fe54 and Fe128. Fe128's median is 100.354 s (91.527–107.566), 13.985× versus the same-host serial CPU reference and 1.596× versus one hashing GPU control. Iteration counts differ and are disclosed; mean iteration time improves from 1.362 s in the control to 0.846–0.864 s. Strict residency remains unverified, with multiple gigabytes still crossing the device boundary during SCF.

The subsequent [normalization validation](NORMALIZATION_VALIDATION.md) passes all 16 upstream pairs, eight GPU controls and both Gamma/sampled-k MAE checks, with 27 native checks and 14 local tests. Every candidate SCF iteration normalizes on device, including all nine relaxation geometries. Complete syscall traces identify stdout/FDF writes. The [deferred-log validation](IO_VALIDATION.md) then passes ten upstream pairs, five controls and MAE, with zero selected SCF file operations in all five candidates and normalization additionally validated for Fe16 non-collinear. Traced speedups are never qualified. The larger untraced normalization batch and decomposed-solver capability probe have since completed, as detailed below.

The larger batch, `20261005T203343Z`, completed with all 20 upstream comparisons, five normalization controls and FePt16 MAE passing. Its Fe128 median is 98.783 s versus 1151.876 s for serial upstream (11.661×). The host-normalization GPU control is faster overall at 92.633 s and takes fewer iterations: normalization has not demonstrated an incremental end-to-end gain. Full timing ranges, iteration counts and errors are recorded in [normalization validation](NORMALIZATION_VALIDATION.md). All 20 source hashes verify. The GPU was released before the next diagnostic started building.

That diagnostic, `20261005T211820Z`, passes the known array/four-byte-copy CUPTI validation and all 25 isolated FD occupation scenarios, but SIESTA traces expose a shutdown-order error: CUDA is already unloading at the final synchronization. Coverage remains false. Explicit finalization before process destruction fixes this in the completed rerun below. The generic batched-eigensolver probe passes real/complex changing-input device loops at size 64, but capture fails at sizes 576/1152. The occupation component is not yet integrated into SIESTA. Source archive hashes for both earlier accepted normalization/I/O batches also pass the reusable `scripts/verify_evidence.py` audit.

The corrected [CUDA activity validation](CUDA_ACTIVITY_VALIDATION.md), `20261005T212723Z`, has now completed with six scientific comparisons, MAE and all six trace-coverage checks passing. All 27 source hashes verify and the GPU is released. Across the cases, observed uploads match explicit counters, but there are three additional downloads (16 bytes) per eigensolver call. These scalar boundaries were absent from the old bridge-only accounting. The ordinary/captured occupation kernel remains a validated component awaiting SCF integration; no fully resident loop is claimed.

The subsequent [occupation integration](OCCUPATION_VALIDATION.md), `20261005T221744Z`, passes 16 upstream comparisons, eight controls, two MAE checks and all 16 CUDA traces through Fe54. All 306 candidate SCF iterations compute occupations on device; 5,349 density consumers borrow retained E/Q without uploading either array. The 28 native checks, 27 isolated occupation scenarios, 26 local tests and all 28 source hashes pass. The GPU is released. Temporary eigenvalue/occupation/result downloads remain, and this change does not uniformly lower transfer volume. The next bounded step is to remove host band selection and defer spectral array exports until SCF ends, followed by larger untraced validation; full density/potential, mixing and force residency remain pending.

That next step now passes the [spectral residency gate](SPECTRAL_RESIDENCY_VALIDATION.md), `20261005T223820Z`: 18 upstream comparisons, nine controls, two MAE checks, 36 CUDA/filesystem coverage checks, 29 native scenarios and all 28 source hashes. Across 365 SCF iterations, E/Q stay on device and 17 explicit exports occur afterward. Selected SCF file operations are zero; mappings, library scalar transfers and other boundaries remain. Fe54 takes 61 CPU, 50 candidate and 70 control iterations, so timing/traffic differences cannot be attributed solely to this feature. The diagnostic GPU was released before the larger untraced batch `20261005T225818Z` launched; that batch passes all 20 upstream pairs, five controls, FePt16 MAE, 29 native checks, 27 FD scenarios and all 28 source hashes. Its 15 candidate processes keep E/Q resident for 733 SCF iterations and make 15 post-SCF exports. Fe128 median time is 95.498 s versus 1149.427 s for CPU (12.036×); spectral residency shows no uniform incremental gain over its GPU control. Full times, iterations and error bounds are recorded in the same validation document. The GPU was released before the matrix/Hartree diagnostic launched.

Three failures were found and retained: an eight-history Fe16 Pulay run that failed candidate SCF convergence, a cached CPU binary that raised SIGILL on a different Modal host, and a Fe54 total-moment difference above the regression gate at the initial SCF thresholds. The iron corpus now uses six Pulay history entries, portable image builds explicitly disable host-specific CPU optimization, and cells with at least 30 atoms use tenfold tighter SCF thresholds. Scientific comparison gates were not loosened. Failed runs are never reported as qualified speedups.

The second stage moves sparse density and energy-density construction onto the GPU, with retained eigenvectors and verified normalized-density reuse in the grid. Its final nine-case batch passed 36 upstream comparisons, nine direct old/new GPU ablations and two MAE checks. All 18 native numerical scenarios and 12 local tests passed. Three cold candidate repetitions per case give median speedups of 7.11× for Fe128, 4.21× for Fe16 non-collinear and 2.77–2.80× for FePt16 SOC against a single serial CPU reference. Direct same-host ablations show 2.95× improvement over the prior GPU path for Fe128 and 1.42–1.45× for the larger spinor cases. Tiny Gamma cases have little benefit. See [density-matrix validation](DENSITY_VALIDATION.md) for ranges, iteration counts, errors, transfer accounting and source provenance.

Fe128 density construction now occupies 6.828 s of the first candidate's 163.674 s, while the same-host GPU control spends 318.897 s in CPU `r-buildD`. The remaining largest routine totals are XC (36.478 s), density grid contraction (34.329 s), potential assembly (27.148 s) and final force contraction (19.238 s). Next, distinguish content-validation, transfer and device-compute costs inside the grid path, then prioritize these grid, XC and force boundaries. Full residency remains unfinished: sparse D/E outputs still return to CPU consumers, and total transfer volume is not universally reduced. The dense integer sparse-index lookup also needs scaling evaluation before broadening the sparse-system target. The final Modal app completed and its GPU container was released.

The [matrix/Hartree diagnostic](MATRIX_HARTREE_VALIDATION.md), `20261005T235359Z`, passes 27 upstream pairs, 18 direct controls, six MAE comparisons, all 54 traces, 38 native checks, 27 FD checks, 19 Hartree scenarios and all 34 source hashes. Device assembly replaces 2.392 GB of dense H/S uploads with 0.640 GB of sparse/geometry uploads across 351 candidate SCF iterations. Equal-iteration Fe16 NC steady uploads fall by about 69% against the CPU-matrix control. The Hartree adapter adds explicitly counted grid/result copies until its surrounding consumers migrate. No selected SCF file operations occur; mappings and scalar/library boundaries remain. The GPU was released before a larger untraced batch launched, which will also validate the new isolated PBE component.

The larger [matrix/Hartree performance batch](MATRIX_HARTREE_VALIDATION.md#larger-untraced-validation), `20261006T001018Z`, has completed: 25 upstream comparisons, ten controls, five FePt16 MAE checks and all 41 source hashes pass. Fe128's three-run median is 103.407 s versus 1259.268 s for serial CPU (12.178×). Matrix assembly improves equal-iteration spinor cases, but neither feature demonstrates a uniform incremental benefit. All 34 isolated PBE checks also pass. Its sole GPU was released before the resident-grid diagnostic launched.

The combined [resident grid milestone](RESIDENT_GRID_VALIDATION.md), `20261006T014308Z`, now passes 18 upstream comparisons, nine direct controls, four MAE checks, 36 complete CUDA/filesystem traces and all 57 source hashes. All 435 candidate SCF iterations retain density/potential arrays. The remaining grid result is 224 bytes per iteration; sparse H/D/E still transfer. CUDA cache reads occur during the first iteration, while later iterations have zero selected file operations. Larger untraced five-case batch `20261006T020310Z`, with three cold candidate repetitions and direct grid controls, is running sequentially on one GPU. Its new standalone Hamiltonian seed/SOC plan passes all 52 checks and 62 source hashes; the [next integration contract](SPARSE_STATE_INTEGRATION.md) connects that plan, scalar consumers and Pulay through explicit sparse-state ownership.

The opt-in sparse-state adapter is now implemented in `native/sparse_state.cu` and the disposable-source patch. It joins the qualified seed, scalar and Pulay plans to device H/D/E ownership, suppresses six density-builder host zero-fills, borrows H for diagonalization, and writes grid contributions directly into H. All 20 affected Fortran units compile; 26 local tests and ten coupled eight-step upstream CPU oracle sequences pass. A new checker links the production owner and exercises identities, ordering, host opacity, geometry changes, error propagation and Pulay reset. **CUDA compilation, ownership checks and full SCF qualification are still pending**; the currently running grid-performance image predates these edits. The draft retains an explicitly counted 144-byte scalar summary and four-byte mixer status, plus older scalar boundaries and final CPU force work. It does not establish the zero-transfer target.

A separate device-convergence component is implemented but unvalidated. The upstream oracle passes 34 sequences covering every nonempty conventional criterion combination, strict threshold edges, minimum iterations and reset. The CUDA plan preserves the upstream energy latch and emits device convergence/mixing/continuation/exhaustion flags with sticky errors until epoch reset. Its intended 92 ordinary/captured/error scenarios are queued as an isolated check in the next single-GPU batch; production SCF still uses the temporary scalar adapter and host convergence. The larger resident-grid batch is still active on Fe128's reference calculation.

Update, 6 October: the larger resident-grid batch `20261006T020310Z` completed successfully (20 upstream pairs, five direct controls, four MAE comparisons, 62 source hashes). [Its measured timings](RESIDENT_GRID_VALIDATION.md) include an 83.470 s Fe128 median, 18.65× the same-host serial reference and 1.47× the grid control; iteration differences are disclosed.

The sparse-state component now passes 37 GPU ownership scenarios and the isolated convergence component passes 92 scenarios. The Fe2 force pair passes after adapter corrections; full relaxation/spin/SOC qualification remains pending. All retained failures and fixes are detailed in [the integration record](SPARSE_STATE_INTEGRATION.md). A later known-copy calibration found that schema-1 CUDA timestamps can misassign operations to host phases. Historical phase-specific residency counts require requalification, while whole-process totals, explicit copy counters, scientific results and untraced timings remain valid. The current single-GPU diagnostic uses schema-2 external correlation; 36 local tests pass. Learned acceleration remains deferred.


The sparse-state diagnostic `20261006T034347Z` has now completed successfully: 18 upstream pairs, nine controls, four MAE comparisons, 36 correlation-based traces and 73 source hashes. [Validation details](SPARSE_STATE_VALIDATION.md) document 349 resident-array SCF iterations, zero steady uploads and selected file operations, and 347,128 remaining scalar download bytes. All nine relaxation cycles pass. The GPU was released before the isolated graph/small-eigensolver probe launched. Larger ownership/performance validation, scalar-free control and final observable residency remain unfinished.

The isolated small eigensolver now passes all 48 scenarios in `20261006T040015Z`. The first [generalized blocked prototype](BLOCK_JACOBI_EXPERIMENT.md), `20261006T042549Z`, compiles but is unqualified: real graph assembly fails, while complex n=18 solves accurately but conditional-body kernel correlations are missing. Independent SIESTA force checks pass. Workspace/capture diagnostics are being corrected without changing the production solver or gates. The larger batch `20261006T040628Z` retained passing Fe54/Fe16/FePt16-z repeated results before a local TLS disconnect; it is explicitly partial. FePt16 x/z and Fe128 are being rerun serially in `20261006T042922Z`. The launcher now detaches bounded remote work from local connectivity while still refusing overlapping apps or GPU allocations.

Recovery update: the stopped `20261006T040628Z` batch contains more completed remote work than its disconnected local log showed: all three FePt16 x candidates also finished. Recovered raw densities permit the unchanged comparator to qualify all three same-instance x/z MAE pairs (maximum difference 1.000004e-6 eV). Fifteen upstream comparisons and three complete controls pass. Its x control and Fe128 remain unfinished; [the partial scope](../results/20261006T040628Z/PARTIAL_SCOPE.md) and regenerated report label this explicitly.

The subsequent untraced batch `20261006T042922Z` completed with all 12 upstream comparisons, three sparse-state controls, four FePt16 MAE checks and 77 source hashes passing. Combined with the earlier complete Fe54/Fe16 groups, the [larger sparse-state qualification](SPARSE_STATE_VALIDATION.md) covers 20 upstream comparisons and five controls. Fe128 median is 55.773 s, versus 73.678 s for its GPU control and 1383.043 s for serial upstream; candidate iteration counts vary 75/105/79 versus 98 for both references. Final CPU observables occupy about 26.4 s, including about 20.5 s in the grid-force contraction. The allocation was released before the corrected blocked-solver/alternate-observer diagnostic started. Full scalar-free SCF and final-observable residency remain unfinished.

The [blocked eigensolver experiment](BLOCK_JACOBI_EXPERIMENT.md) now has a numerical component and initial SCF qualification. In `20261006T051126Z`, all 36 ordinary/nested scenarios pass both untraced and with CUPTI 12.6; CUPTI 12.8's device faults do not reproduce, but launch attribution remains incomplete. In `20261006T052229Z`, all 18 upstream comparisons, nine standard-GPU controls and four MAE checks pass: 11,278 blocked solves across 371 SCF iterations, including relaxation. The blocked path is slower and remains opt-in; four-byte status downloads and host control still remain. A separate observer follow-up records graph-executable IDs/flags and memset activity without relaxing attribution or scientific gates. All allocations remain sequential and limited to one GPU.

The device SCF control component now passes all 75 coupled convergence/Pulay/history scenarios on two host drivers (`20261006T061050Z` and `20261006T061806Z`), with 441 device records per run and maximum vector error 2.499e-16. Both runs verify 82 source and 309 component evidence hashes. Conditional profiling faults reproduce on driver 610.57.04 even without requesting kernel collection; numerical execution succeeds. Corrected offline known-work calibration passes on the retained driver-580 trace, but no solver/control trace is reclassified. Reports now use actual GPU/driver inventory instead of assuming every one-GPU A100-40GB request received that physical model. The next nine-case regression validates a stream-aware, capturable H/S assembly interface. Full device SCF integration and final observables remain open.

The capturable H/S interface now passes its [nine-case regression](../results/20261006T062841Z/REPORT.md): 46 native checks including eight captured matrix scenarios, both MAE comparisons, 18 trace-coverage checks and 84 source hashes. Maximum captured matrix error is 2.483e-16. Zero steady uploads are observed, while 353,456 scalar/library download bytes remain. Relaxation also records three steady filesystem calls reading one byte from `/proc/sys/vm/overcommit_memory`; this batch does not establish zero steady I/O. The next serial run validates a stream-aware density/vector/normalization interface and retries conditional profiling on a requested single A100-80GB. No additional concurrent GPU is authorized or allocated.

The [density and conditional-transfer batch](../results/20261006T065114Z/REPORT.md) now passes all 58 native checks, nine upstream comparisons, both MAE comparisons, 18 full SIESTA trace-coverage checks, 86 source hashes and 420 component evidence hashes. Twelve captured density scenarios cover 48 replays, normalization failures/recovery and maximum D/E error 2.776e-17. All 75 control and 36 blocked-solver traces independently qualify zero H2D/D2H inside their measured graph regions on driver 580.95.05; raw reanalysis reproduces 119 worker/calibration traces. Complete kernel activity and strict residency remain outside that component scope. Production still records 358,384 steady scalar/library download bytes and three steady relaxation filesystem calls. The next single-GPU batch validates capturable grid contractions and the energy-seed handoff before composing the complete electronic cycle.

The [grid and energy-handoff batch](../results/20261006T071319Z/REPORT.md) passes 68 native checks, nine upstream pairs, both MAE comparisons, 18 trace-coverage checks and 88 source hashes. Eight grid captures cover 24 changing-input replays with maximum error 2.221e-16. H/S, density and grid operations now have stream-aware device interfaces, and scalar energy seeds compose without an adapter export. Production retains 368,240 steady scalar/library download bytes over 398 iterations and three relaxation filesystem calls. The single GPU has been released. The next architectural step is explicit whole-cycle ownership and composition, including device error gates and frozen first-pass spectra; isolated capture success does not qualify a full resident SCF loop.

The first [composed-cycle diagnostic](../results/20261006T074753Z/REPORT.md) passes 28 synthetic scenarios and 80 completed device iterations, with maximum scalar error 1.497e-14. The graph joins every electronic producer through convergence/Pulay, preserves the sampled-k first spectrum, stops failed producers and remains inert after completion. Eight expected failures, the independent Fe2 force pair, 68 native checks, 92 source hashes and 64 component evidence hashes pass. Its transfer harness stops on a Python return-value error after four passing calibrations, so whole-cycle copy coverage remains unqualified. The corrected next serial batch adds larger coupled solves and bounded per-step Fermi/energy/grid diagnostics. Production SIESTA integration and final GPU observables remain open.

The [expanded composed-cycle batch](../results/20261006T075424Z/REPORT.md) passes 32 synthetic checks and 96 device iterations, including 33-state real and 66-state complex solves and complete bounded Fermi/normalization/seed/grid diagnostics. Maximum diagnostic error is 9.948e-14. All 92 source and 112 component evidence hashes verify; the independent Fe2 force regression and 68 native checks pass. Driver 610.57.04 triggers the established conditional-profiling skip, so transfer coverage remains unqualified. The next sequential batch adds selected-filesystem tracing around the entire cycle and retries the memcpy/memset observer. Production SIESTA integration remains unfinished.

The [composed-cycle residency batch](../results/20261006T080125Z/REPORT.md) completes all three 32-case suites, each with 96 accepted iterations and eight expected failures. The full graph executes every electronic producer, convergence and mixing without host iteration callbacks. Measured H2D/D2H and selected filesystem operations/mappings are zero for the entire cycle, including its first accepted iteration and terminal replay. Raw reanalysis reproduces 36 CUDA worker/calibration traces and 32 syscall traces; 92 source and 219 component hashes verify. Maximum scalar/diagnostic errors are 6.218e-14/1.386e-13. The single A100 80GB PCIe allocation has ended. This is synthetic executor qualification; production handoff, real-material trajectories using this executor, final GPU forces/stress and performance remain open.

The whole-device production handoff now passes Fe2 force pilot `20261006T083311Z` and eight of nine broad cases in `20261006T083554Z`, including relaxation and both MAE comparisons. Fe54 fails locally in its eigensolver, then passes retry `20261006T085932Z` without numerical changes. The reliability diagnostic `20261006T090823Z` reproduces failure after 36 accepted iterations, with local cuSOLVER info 22. Post-failure H/S retention permits a successful independent LAPACK solve (relative residual 6.386e-15); exact graph/warmup replay is under way. Scientific tolerances and the blocked convergence limit are unchanged. Independent production transfer/I/O tracing, solver reliability and final GPU forces/stress remain required before baseline qualification. CPU references can now be reused with exact input/pseudopotential/executable checks; comparisons across allocations never qualify speedup.


The 32-state batched-Jacobi backend now passes both captured Fe54 failures (64 exact replays), five fresh Fe54 runs, 36 solver checks and 32 whole-cycle checks in [20261006T093950Z](../results/20261006T093950Z/REPORT.md). The five runs have zero explicit cycle copies over 270 iterations and a 28.968 s median versus 38.943 s for one same-allocation GPU control, with differing iteration counts. The subsequent traced batch passes seven cases then faults on Fe16 under CUPTI; Fe54 is not reached and the batch remains unqualified. A filesystem-only rerun is isolating this fault. The new FP64 grid-force contraction passes its 36-case pinned CPU oracle and all 156 isolated GPU scenarios; a post-SCF shadow diagnostic is being prepared to compare actual material derivatives, sparse periodic indexing and spin conventions against unchanged `dfscf`. Final forces still use upstream CPU computation, and learned acceleration remains deferred.


The [filesystem-only rerun](../results/20261006T100735Z/REPORT.md) passes all nine magnetic cases and both MAE comparisons. Its 17 geometry cycles cover 372 accepted iterations with zero explicit host/device copies and zero selected filesystem operations or mappings. Fe16 succeeds without CUPTI on the same physical GPU model/driver version as the prior failed allocation; this narrows but does not establish causality. All 102 source and 66 component evidence hashes pass. The force component's 156 GPU checks agree with the pinned arithmetic to 4.441e-16. An opt-in post-SCF shadow now compiles and is being tested against actual upstream `dfscf`, while retaining the CPU result and disqualifying timings. Complete production CUDA-copy coverage and final force/analysis residency are still open.


The first [material force-shadow batch](../results/20261006T101802Z/REPORT.md) passes eight calculations / 16 geometry contractions and both MAE comparisons. GPU dfscf arithmetic agrees within 1.163e-13 Ry/Bohr (force) and 6.828e-15 Ry/Bohr³ (stress). Fe54 reaches the final-force stage but exceeds the diagnostic's 1 GiB derivative capacity, so that row remains invalid. The next bounded run checks a 4 GiB cap and owned host snapshots; no numerical tolerance changes. Draft auxiliary-potential/deformation-density getters prepare the subsequent final-state handoff and are not yet GPU-qualified.


Fe54 and the follow-up force/relaxation/SOC shadows now pass in [20261006T102710Z](../results/20261006T102710Z/REPORT.md), completing dfscf contraction coverage across the current corpus. The [CUPTI-only follow-up](../results/20261006T103553Z/REPORT.md) still faults on Fe54 despite omitting strace; larger-case copy coverage remains unqualified. Its auxiliary-potential/deformation fields pass all 32 GPU component checks. A device-to-device final-state snapshot and `--force-shadow-resident` consumer are now implemented and under validation. They aim to remove the diagnostic's electronic-input uploads while retaining the original CPU output for comparison; other final observable consumers and legacy array exports remain.


The [retained final-state batch](../results/20261006T104613Z/REPORT.md) now passes all nine magnetic cases, both MAE comparisons and 17 geometry force checks. GPU dfscf consumes retained device D/V/Vaux; the handoff also retains E and deformation density for later consumers. Maximum force/stress differences are 1.674e-13 Ry/Bohr and 5.496e-15 Ry/Bohr³. Eighteen retained final-grid output comparisons, 106 source hashes and 66 component evidence hashes verify. All 373 SCF iterations have zero explicit copies. This batch is untraced; broader CUDA-copy coverage remains blocked by the observed profiling faults. CPU force outputs remain authoritative, and legacy exports still serve unported consumers. Production force substitution and remaining final consumers are next; learned acceleration stays deferred.


The [authoritative grid-force batch](../results/20261006T110416Z/REPORT.md) passes all nine cases, nine sequential CPU-force controls, four directional MAE checks and 17 candidate force geometries. All 370 candidate iterations have zero explicit copies. The native GPU output now replaces the CPU dfscf contraction; derivative generation and the remaining final consumers still use the host. Fe16 dfscf falls from 4.504 to 0.459 s at the same 43 SCF iterations; Fe54 dfscf falls from 3.699 to 1.460 s, with 53/55 control/candidate iterations. These are preliminary single observations, followed by a repeated Fe16/Fe54 batch. All 107 source and 66 component hashes, 18 retained grid-field comparisons and nine production audits verify. The next core/neutral contraction oracle passes 27 pinned-Fortran fixtures to 1.777e-15; GPU validation and integration remain pending. Large-case profiling faults and final legacy exports remain open. Learned acceleration stays deferred.


The [repeated GPU-force batch](../results/20261006T111309Z/REPORT.md) passes ten candidates, two CPU-force controls and every direct candidate/control scientific comparison. Fe16 median is 44.396 s (44.319–44.595) versus 49.742 s, with 43 iterations throughout; dfscf is 0.793–0.924 versus 6.520 s. Fe54 median is 24.684 s (23.708–27.535) versus 29.842 s, but candidates take 43–59 iterations versus 60 for the control; dfscf is 2.564–2.847 versus 4.931 s. All 462 candidate SCF iterations have zero explicit transfers. Source/component/force/grid-field audits and raw-density scientific rechecks pass. The allocation is complete. The isolated core/neutral CUDA component and 154-scenario checker are drafted behind `--probe-atomic-grid-force`; GPU compilation/validation and production integration remain next. No learned SCF work has begun.


The [isolated core/neutral component](../results/20261006T113218Z/REPORT.md) passes 154 GPU scenarios to 6.662e-15 and independently reproduces all retained array comparisons; 112 source and 71 component hashes verify. Fe2 production dfscf and 18 retained grid-field checks also pass. The next material-shadow adapter compiles, preserves original CPU outputs, borrows retained electronic fields and enforces ordered core/neutral/dfscf consumers for every geometry. A nine-case single-GPU shadow batch is under validation; production core/neutral substitution, derivative staging and the other final consumers remain open.


The [core/neutral material batch](../results/20261006T114022Z/REPORT.md) passes all nine cases, both MAE pairs and 34 strict shadow contractions across 17 geometries. Maximum force/stress differences are 2.354e-13 Ry/Bohr and 6.354e-14 Ry/Bohr³. Final-state ownership enforces the expected core → neutral → dfscf sequence; all electronic fields come from the retained device snapshot. All 114 source and 71 component hashes, 154 isolated atomic scenarios, 18 final-grid field comparisons and nine inverse maps verify. Local raw-density reanalysis reproduces all eleven scientific comparisons. The 350 SCF iterations have zero explicit copies; this batch is untraced. CPU core/neutral outputs remain authoritative in the shadow, and the completed allocation is released. Next, share a production force/stress/status accumulator across the three grid terms and export only once; derivative staging and the other final consumers still require migration.


The [shared production grid-force batch](../results/20261006T121029Z/REPORT.md) passes nine candidates, nine CPU-core/neutral controls, nine direct ablations and four MAE checks; all 31 raw-density scientific rechecks reproduce. Core and neutral CPU contractions are bypassed, and 34 atomic terms across 17 geometries accumulate with dfscf in shared device force/stress/status buffers. Intermediate atomic exports and per-term files are absent; host mutations, invalid ownership and sticky errors are checked. All 42 coupled scenarios, 154 atomic scenarios, 18 final-grid comparisons, nine inverse maps, 116 source hashes and 73 component hashes pass. The 366 candidate electronic iterations have zero explicit copies; this batch is untraced. Fe54 still stages 1.31 GB, and FINAL_HF rises from 3.664 to 3.904 s; its lower total time includes 51 rather than 64 SCF iterations. Fe16 FINAL_HF is 1.602 versus 1.522 s at the same 43 iterations. The allocation is stopped. Final CPU grid recomputation, derivative generation/staging, integral/SOC/overlap and analysis consumers, legacy exports and larger-case CUDA observation remain next. Learned acceleration stays deferred.


## Final force-grid production qualification

[Run 20261006T125358Z](../results/20261006T125358Z/REPORT.md) passes all nine candidates, nine sequential CPU-final-grid controls, nine direct ablations and four MAE checks. All 31 raw scientific comparisons reproduce. It bypasses CPU density, Hartree, XC and potential-matrix recomputation in final forces, using the already-published 27-value cycle record and retained device electronic fields. The optional final H is assembled only for an actual consumer or shadow; all current production fixtures require no extra H export. The corrected preceding shadow establishes full-H/grid agreement. Scientific thresholds are unchanged.

Fe54 removes 11,035,008 H2D and 7,407,160 D2H bytes from the final-force phase. Its shared force/stress/status/topology output is 1,376 bytes, while derivative/setup uploads remain 1,311,389,084 bytes. Fe16 removes 16,006,656 H2D and 8,936,536 D2H bytes. Independent checks confirm the removed transfer categories and CPU profile calls. All 119 source and 73 component hashes, 154 atomic scenarios, 42 shared-force scenarios, 18 retained-grid checks, nine inverse maps, 18 force/cycle audits and nine final-grid audits pass. The 366 candidate iterations (735 with controls) have zero explicit copies; this is untraced.

Fe16 takes 42.304 versus 42.496 s with 43 iterations each; inclusive FINAL_HF is 1.866 versus 2.060 s. Fe54 takes 22.513 versus 23.424 s at 51 versus 54 iterations, with FINAL_HF 4.715 versus 5.150 s. These single observations do not establish repeated speedups, and Fe54 total time includes different SCF work. The allocation is stopped. Separate density-output recomputation, CPU derivative generation/staging, kinetic/nonlocal/SOC/overlap and analysis consumers, legacy exports and independent larger-case CUDA observation remain. Learned acceleration stays deferred.


## Density-output shadow qualification

[Run 20261006T133147Z](../results/20261006T133147Z/REPORT.md) passes all nine materials and both MAE pairs with retained RHO and spin-resolved DRHO. Across 17 geometry lifetimes it retains 34 optional fields; only the final geometry in each calculation exports its two requested arrays. Every FP64 shadow element agrees exactly with the original analysis path (18 full comparisons, maximum absolute difference zero). Independent checks reproduce the raw arrays, written FP32 files, upstream RHO/DRHO agreement, ownership and transfer accounting. All eleven scientific comparisons reproduce.

The 350 SCF iterations have zero explicit transfers, without independent tracing. All 120 source and 73 component hashes, 154 atomic checks, 42 shared-force checks, 18 retained-grid comparisons, nine inverse maps and nine force/cycle/final-grid audits pass. This diagnostic preserves original analysis outputs and does not qualify speed. Its allocation is stopped; the production bypass and sequential recomputation controls are under validation. CPU derivative generation/staging and the remaining integral/analysis consumers still prevent full residency.


## Requested density-output production qualification

[Run 20261006T134040Z](../results/20261006T134040Z/REPORT.md) passes all nine production candidates, nine sequential original-analysis controls, nine direct ablations and four Gamma/sampled-k MAE checks. All 31 scientific comparisons reproduce from raw stdout, eigenvalues and RHO/DRHO files. The separate density audit also verifies both requested files against upstream for all nine controls. The preceding shadow in 20261006T133147Z established exact FP64 equality for all 18 field arrays; this run makes retained GPU RHO and DRHO authoritative.

Across 17 candidate geometries, 34 field records retain both requested grids. Only each calculation's final geometry performs its two explicit output downloads. Raw counters show zero analysis uploads and no density/Hartree/potential/status round trips; profiles show no analysis rhoofd, Poisson, XC or potential-matrix recomputation. The unchanged SIESTA RHO/DRHO writers still perform the requested final I/O. Other grid outputs, partial-charge analysis and time-dependent density output are unsupported by this opt-in handoff. The current material corpus requests both fields; RHO-only, DRHO-only and neither-field material trajectories remain unvalidated.

Fe16 removes 7,443,456 analysis upload bytes. Its two requested downloads total 2,985,984 bytes, versus 1,866,328 bytes in the old path; the 1,119,656-byte increase replaces CPU DRHO formation with a GPU-produced requested field. Fe54 removes 4,887,648 upload bytes; its requested downloads total 5,038,848 bytes versus 3,779,224, an increase of 1,259,624 bytes. Each calculation has two analysis copies instead of five. The explicitly retained output buffers have the same sizes as those exports. These are measured consumer migrations, not a claim that every phase's download total decreased.

Single same-allocation observations: Fe16 analysis is 0.046 versus 0.169 s, with whole-run time 42.619 versus 42.792 s at 43 iterations each. Fe54 analysis is 0.050 versus 0.413 s, with whole-run time 21.300 versus 23.793 s at 48 versus 46 iterations. Inclusive density-analysis DHSCF is 0.027 versus 0.155 s (Fe16) and 0.036 versus 0.398 s (Fe54). Timers overlap, whole-run SCF work differs for Fe54, and these are not repeated performance estimates. The CPU reference is reused from a different allocation; CPU/GPU speedup is unqualified.

All 120 source and 73 component hashes, 154 atomic scenarios, 42 shared-force scenarios, 18 grid-field comparisons, nine inverse maps, 18 force/cycle/final-grid audits and nine requested-density audits pass. The 363 candidate electronic iterations (724 including controls) have zero explicit copies. This run is untraced; independent larger-case CUDA observation remains unresolved. The single GPU allocation is complete. CPU radial/orbital derivative generation and 1.31 GB Fe54 force staging, kinetic/nonlocal/SOC/overlap and other analysis consumers, legacy electronic exports and the four-byte force topology check remain. Learned acceleration stays deferred.


## Orbital derivative component qualification

[Run 20261006T142039Z](../results/20261006T142039Z/REPORT.md) passes all 81 isolated CUDA scenarios and independent retained-array audits. Actual pristine `all_phi` calls supply the oracle for 12,722 direct orbital queries through l=5 and 2,673 compact-geometry locations with nsp=1/8. Maximum numerical difference is zero for all 30 value/gradient/displacement comparisons, including two sticky-error scenarios. Direct query outputs agree bit-for-bit; four geometry cases differ only in the sign of zero for empty-radial gradients. Every captured graph has one kernel and no forbidden nodes. Host input staging and final probe exports are outside execution.

All 125 source hashes and 79 component provenance checks pass, and the 125 corresponding working runtime files match the accepted archive. The Fe2 production regression and its raw RHO/DRHO recheck pass at 25 iterations, along with the existing solver/atomic/shared-force/grid-field/ownership checks. Its electronic cycle has zero explicit copies; this run is untraced. The new component is not integrated into production and removes no material upload yet. The preceding 20261006T141449Z failure is retained: its harness expected host configuration rejection at stage 2 instead of the actual stage 3. The corrected run uses the same native library hash and unchanged tolerances.

The allocation is stopped. Next, implement geometry-epoch ownership of real basis tables and compact mappings, then validate full material derivative shadows before replacing CPU `all_phi` and the approximately 1.09 GB Fe54 dfscf upload. Core/neutral derivative staging (another approximately 220 MB), kinetic/nonlocal/SOC/overlap and other analysis consumers, legacy exports and independent larger-case CUDA observation remain. Learned acceleration stays deferred.


## Material orbital derivative shadow qualification

[Run 20261006T144142Z](../results/20261006T144142Z/REPORT.md) passes all nine materials, both MAE pairs and all eleven raw scientific rechecks. Across 17 geometries (nine from relaxation), the GPU comparison checks 250,926,088 orbital value/gradient/displacement elements with zero numerical difference, zero nonfinite values and complete shape-derived counts. This tests actual species tables, orbital order, periodic-image coordinates, subpoint layout and geometry-epoch reuse. CPU-generated derivatives still determine the force contraction in this shadow.

Fe16 compact staging is 6,245,152 bytes against 323,205,120 derivative bytes; Fe54 compact staging is 19,757,912 bytes against 1,090,817,280 derivative bytes. Both are uploaded in the shadow. The eventual production replacement would remove net 316,959,968 and 1,071,059,368 upload bytes respectively, but that removal is not yet implemented or qualified. Each geometry adds only a 72-byte comparison initialization and final diagnostic download; no complete derivative array is downloaded for validation. The material audit verifies recorded full comparisons, ownership and traffic, not an independent replay of all material derivative arrays.

All 130 source and 81 component hashes verify, along with 81 orbital cases, 28 comparator cases, the existing 40 solver/154 atomic/42 shared-force cases, 18 retained-grid checks, nine inverse-map fixtures and nine force/cycle/final-grid/density/material audits. All 371 SCF iterations have zero explicit copies; the run is untraced and its shadow timings are disqualified. The single allocation is stopped.

The next production gate is device-authoritative derivatives with the CPU loop and upload bypassed, plus same-allocation CPU-derivative controls across the full corpus. Before that gate, extend the comparator probe beyond 262,144 elements to exercise its capped-grid stride with injected failures, and test invalid owner lifetimes. Then migrate core/neutral derivatives and the remaining integral/analysis consumers. Learned acceleration remains deferred.


## Device-owned orbital derivatives qualified

[Run 20261006T150954Z](../results/20261006T150954Z/REPORT.md) passes all nine production materials, nine sequential CPU-derivative controls, nine direct comparisons and four MAE checks: 31 scientific comparisons independently reproduced with raw RHO/DRHO files. The 17 production geometries include all nine relaxation steps. `--orbital-basis` bypasses the CPU `all_phi` loop, its three derivative buffers and their uploads. A two-kernel graph generates FP64 values/gradients/displacements and consumes them directly in the force contraction. Its owner uses the existing final force status; no intermediate derivative/status download or shadow comparison record is added.

| Case | CPU derivative upload removed | Compact initialization upload | Net upload removed | Remaining force setup upload |
| --- | ---: | ---: | ---: | ---: |
| Fe16 non-collinear | 323,205,120 B | 6,245,152 B | 316,959,968 B | 65,426,148 B |
| Fe54 collinear | 1,090,817,280 B | 19,757,912 B | 1,071,059,368 B | 220,571,804 B |

Production records show zero CPU derivative storage and zero derivative upload, with unperformed comparison fields recorded as null. Compact basis/geometry staging occurs before each electronic cycle. The existing force/stress/status/topology export is unchanged (1,376 bytes for Fe54); this scoped figure excludes legacy electronic and buffered diagnostic downloads, which still total about 18.18 MB in Fe54's full post-SCF phase. Requested density output remains a separate explicit export.

All 81 exact-upstream orbital scenarios, 48 comparator scenarios and 41 isolated owner scenarios pass and reproduce from retained evidence. Comparator faults include 65,537 points (458,759 elements), exercising the capped launch's strided tail. Ownership checks cover missing/duplicate preparation, phase/mode/shape/stream changes, missing or duplicate generation/consumption, stale epochs, device failure and changed-geometry reuse. All 132 source and 205 component hashes verify, and all 132 working runtime files match the accepted archive. Existing 40 blocked-solver, 154 atomic and 42 shared-force scenarios, 18 grid-field comparisons, nine inverse maps, 18 cycle/force/final-grid/density audits and nine production orbital audits pass.

The 359 production SCF iterations (727 including controls) have zero explicit transfers. This batch is untraced; it does not close independent larger-case CUDA, managed/mapped-memory or filesystem coverage. Inclusive Fe16 dfscf falls from 0.596 to 0.083 s and FINAL_HF from 1.875 to 1.364 s, with both processes taking 43 SCF iterations (42.318 versus 41.774 s total). Fe54 dfscf falls from 1.956 to 0.204 s and FINAL_HF from 4.696 to 2.970 s; whole-run times are 23.336 versus 18.688 s at 53 versus 44 iterations, so that whole-run difference cannot be attributed solely to derivatives. These are single observations. Native compact-staging timers add 0.003/0.008 s for Fe16/Fe54 and exclude Fortran table packing; whole-run time includes it.

The one A100-SXM4-80GB allocation (driver 580.95.05) is stopped with no containers. Next migrate core/neutral radial evaluation and its roughly 220 MB Fe54 staging, then the kinetic/nonlocal/SOC/overlap and other analysis consumers of legacy electronic exports. Preserve the distinct upstream core and neutral cutoff/radius rules rather than treating both as the same radial call. The four-byte topology readback can be replaced by the grid owner's already-retained entry count. Independent larger-case CUDA observation remains open. Learned acceleration stays deferred.

The resulting Fe54 final-force profile has 1.795 s exclusive CPU `nlefsm` and 0.566 s exclusive `DHSCF4`. The core/neutral port targets the remaining large staging boundary; nonlocal/projector evaluation is the largest remaining CPU force routine in this measurement.


## Core/neutral radial component qualification

[Run 20261006T154544Z](../results/20261006T154544Z/REPORT.md) qualifies `native/atomic_basis.cu` in isolation: all 238 ordinary/captured/replay/error scenarios pass and reproduce from retained arrays. Its 160 numerical comparisons have zero absolute difference against the cloud pristine-library oracle; all 275,608 bit differences are signed zeros. An independent local replay of the 28 query and ten compact-geometry oracle inputs also passes, with maximum difference 4.372e-15. The fixtures contain 528,344 radial queries and 289,560 geometry locations (counted across core/neutral and direct/mesh modes), including 131,073-query and 131,080-location launches that exercise capped-grid striding. Every output is poisoned before each replay outside capture, and each captured evaluation contains one kernel with no forbidden nodes.

The component calls the arithmetic of `chcore_sub` for core density and `phiatm(is,0)` for neutral potential. Core mesh selection requires cutoff >1e-12 and unshifted radius <cutoff; neutral mesh selection requires only radius <cutoff. Both inner routines add 1e-20 to the radius, but neutral evaluation additionally subtracts 1e-12 from its cutoff. Selection masks are checked separately from numerical zeros. Fixtures cover zero/two/many radial samples, floating species with deliberately unused invalid tables, origins, radial nodes, cutoff neighbors and allowed interpolation-endpoint extrapolation. Compact geometry preserves full-image atoms separately from unit-cell force atoms; core indices stay clustered and neutral indices agree with original `reord` for nsm=1/2/3/4.

All 137 source hashes, 211 component hashes and corresponding working runtime files verify at qualification. Existing orbital/comparator/owner/solver/atomic/shared-force/field/map checks pass. The unchanged Fe2 production path and full RHO/DRHO scientific recheck pass at 25 iterations with zero explicit SCF copies. This allocation is stopped. The new component is not yet linked into the production force path and removes no material staging; no independent CUDA/filesystem trace or speedup is claimed.

The material adapter will stage compact `(full atom, mesh offset, cluster)` pairs and species tables before SCF, leaving radius selection and radial evaluation on the GPU. Compare the complete selection mask and all selected CPU-generated values/gradients/displacements before the production bypass. Bind both core/neutral consumers to geometry epochs and shared terminal status, with no new intermediate readback. Preserve clustered potentials versus sequential deformation-density indexing. CPU nongradient initialization remains an explicit initial-staging concern. After the full nine-case material shadow passes, bypass the post-SCF CPU radial loops and uploads and qualify sequential CPU-radial controls before claiming the roughly 220 MB Fe54 reduction.


## Material core/neutral derivative shadow qualified

[Run 20261006T163834Z](../results/20261006T163834Z/REPORT.md) passes all nine materials and both MAE pairs. All eleven scientific comparisons reproduce locally from raw stdout, eigenvalues and RHO/DRHO outputs. Across 17 geometries, including nine relaxation steps, the GPU comparison checks 16,490,336 selection locations, 87,667,268 numerical elements and 14,470,616 point/atom indices. Every value, gradient and selected displacement has zero numerical difference; selection masks and indices match exactly. The selected-point total is 7,235,308 across both terms and all geometries.

`--atomic-basis-shadow` stages compact species tables and `(full atom, mesh offset, cluster)` mappings once per geometry before SCF. The geometry owner checks phases, epochs, ordered consumers and shared status. Each core/neutral force graph contains generation, complete comparison and the existing CPU-derivative contraction. It preserves distinct clustered core and sequential neutral field indexing. Comparison covers values/gradients at every location, including inactive zeros, plus exact selected displacements and indices. CPU derivatives remain authoritative. Each required term exports only its 120-byte comparison summary at final force publication, after the existing terminal status; no new per-term scalar readback is introduced. Buffered JSON is written at shutdown. The shadow deliberately uploads full reference arrays and is excluded from performance claims.

| Case | Current core/neutral point and derivative upload | New compact geometry/table staging | Potential net upload removal after production bypass |
| --- | ---: | ---: | ---: |
| Fe16 non-collinear | 65,352,448 B | 2,322,540 B | 63,029,908 B |
| Fe54 collinear | 220,564,512 B | 6,994,004 B | 213,570,508 B |

The table excludes the two existing 32-byte force-problem uploads, which remain separate. These reductions are prospective: this shadow removes no CPU generation or upload. Fe54's shadow device allocation is 1,226,803,268 bytes, including actual and reference arrays. Production should retain the GPU selection mask and contract over the complete pair/subpoint domain, avoiding a selected-count download and host compaction. It must skip the CPU radial loops and point/derivative buffers, allocate only generated arrays, omit comparison transfers, and then pass same-allocation CPU-radial controls and direct scientific/MAE comparisons across the full corpus.

All 238 radial, 80 comparator and 71 owner scenarios pass and reproduce independently. Comparator probes include 131,073 locations, crossing the capped launch's strided tail, with mask/index/active/inactive/nonfinite/sticky faults. Ownership probes cover both partial-core modes, changed-geometry reuse, staged-input snapshots, phase/mode/shape/stream changes, missing/duplicate consumers and captured but unlaunched graphs. All 145 source hashes, 427 component hashes and 145 corresponding working runtime files verify. Existing 81 orbital, 48 orbital-comparator, 41 orbital-owner, 40 blocked-solver, 154 atomic-force, 42 shared-force and 156 grid-force checks pass; force/cycle/final-grid/density audits, 18 retained-grid comparisons and nine inverse maps also pass.

All 364 electronic iterations record zero explicit transfers. This batch is untraced; independent larger-case CUDA, managed/mapped-memory and complete filesystem coverage remain open. The single A100-SXM4-80GB allocation, driver 580.95.05, is stopped with no containers. The preceding [interrupted run](../results/20261006T161942Z/REPORT.md) preserves its passing component evidence and remote cancellation log; no material calculation ran there. The launcher now submits exactly one asynchronous function call and waits for its result, retaining the lock and active-allocation checks.

After the production radial bypass, migrate kinetic/nonlocal/SOC/overlap and other analysis consumers before removing legacy electronic exports. Replace the four-byte force-topology readback using the grid owner's retained entry count. Learned acceleration remains deferred.


## Device-owned core/neutral derivatives qualified

[Run 20261006T170248Z](../results/20261006T170248Z/REPORT.md) passes all nine production materials, nine sequential CPU-radial controls, nine direct comparisons and four directional MAE checks. All 31 scientific comparisons reproduce locally from raw stdout, eigenvalues and RHO/DRHO outputs. The 17 production geometries include all nine relaxation steps. `--atomic-basis` bypasses the post-SCF CPU `chcore_sub`/`phiatm(is,0)` loops and their point, atom, gradient and displacement buffers/uploads. The original nongradient initialization remains an explicit initial-geometry operation.

A two-kernel graph generates FP64 values, gradients, displacements, selection masks and field/atom indices, then contracts directly from these device arrays. It iterates the complete compact pair/subpoint domain and skips inactive entries on device. No host compaction or selected-count download is needed. Production records identify the full evaluated domain; selected/reference comparison fields are null. The geometry owner allocates only generated arrays and consumes the existing terminal force status. It performs no comparison initialization, derivative download or additional status readback. The qualified shadow remains available separately, and `--atomic-basis-ablation` restores CPU radial generation as a sequential control.

| Case | CPU point/derivative upload removed | Compact initialization upload | Net upload removed | Remaining force setup upload |
| --- | ---: | ---: | ---: | ---: |
| Fe16 non-collinear | 65,352,448 B | 2,322,540 B | 63,029,908 B | 73,700 B |
| Fe54 collinear | 220,564,512 B | 6,994,004 B | 213,570,508 B | 7,292 B |

The remaining force setup includes the two 32-byte atomic-force problem descriptions and existing dfscf geometry/contributions. Compact basis tables and mappings are staged before SCF and counted separately. The generated owner buffers occupy 183,034,988 bytes for Fe16 and 616,898,516 bytes for Fe54, including compact inputs. The larger full-domain device allocation replaces selected CPU buffers and transfers. Force/stress/status/topology exports remain 464 bytes for Fe16 and 1,376 bytes for Fe54. These scoped values exclude legacy electronic/diagnostic exports: Fe54's full post-SCF phase still downloads 18,184,208 bytes in this run, and requested final density exports remain a separate boundary.

All 238 radial, 80 comparator, 136 ownership and 180 masked-contraction scenarios pass and reproduce independently. Masked tests pad pristine Fortran fixtures across launch boundaries through 131,073 entries, retain invalid inactive data, and exercise all-inactive masks, malformed active masks/indices, nonfinite active derivatives, sticky status and captured replays. Production owner tests independently inspect generated fixture arrays, including changed geometry and staged-input snapshots. Shadow-only tests still reject captured but unlaunched comparisons; production relies on the caller's checked graph launch/synchronization and existing final status, with no added execution-receipt download.

All 146 source hashes, 627 component hashes and corresponding working runtime files verify. The existing orbital, comparator, owner, blocked-solver, atomic-force and shared-force checks pass, together with 18 cycle/force/final-grid/density audits, 18 retained-grid comparisons and nine inverse maps. All 364 primary SCF iterations (727 including controls) record zero explicit transfers. This untraced batch does not close independent larger-case CUDA, managed/mapped-memory or complete filesystem coverage. The one A100-SXM4-80GB allocation, driver 580.95.05, stopped with zero tasks and no remaining REMAG container.

Single same-allocation observations: Fe16 inclusive `DHSCF4` falls from 0.373 to 0.139 s and `FINAL_HF` from 1.506 to 1.257 s; whole-run time is 41.934 versus 42.225 s at 43 iterations each. Fe54 `DHSCF4` falls from 0.991 to 0.320 s and `FINAL_HF` from 3.005 to 2.440 s; whole-run time is 19.988 versus 20.310 s at 49 versus 48 iterations. Both compact-setup timers are 0.011 s and include Fortran packing and native staging. Timers overlap, measurements are unrepeated, smaller cases show mixed timings, and differing Fe54 SCF work prevents attributing its whole-run ratio solely to radial generation. The reused CPU-reference allocation does not qualify CPU/GPU speedup.

The remaining Fe54 final-force profile is dominated by CPU `nlefsm` at 1.798 s, with `kinefsm` and `overfsm` at 0.129 s each. Next remove the four-byte force-topology readback using the grid owner's retained entry count, then migrate integral/projector and other analysis consumers before deleting legacy electronic exports. Learned SCF acceleration remains deferred.


## Force-topology readback removed

[Run 20261006T173034Z](../results/20261006T173034Z/REPORT.md) passes all nine materials and both directional MAE comparisons. Independent raw stdout/eigenvalue/RHO/DRHO reanalysis reproduces all eleven scientific checks. The 17 force evaluations include nine relaxation geometries; all 360 electronic iterations record zero explicit host/device transfers.

`GridContractionView` now carries the entry count already retained in its geometry owner. Final force preparation checks the incoming count against that value after the existing epoch, fingerprint and exclusive-lease guards. It no longer downloads the last device endpoint merely to read the count. Eleven isolated native scenarios cover valid preparation, missing staging, wrong count/shape, stale epoch, changed endpoints/orbitals/basis/cell metadata, lease exclusion and changed-count restaging. All recover to a valid force begin and perform zero explicit post-SCF transfers. All intentional probe exports are confined to initialization density evaluations.

Raw phase counters independently reproduce exactly three final grid-output copies per geometry: terminal status, force and stress. The previous qualified production run had four. This removes 68 bytes in 17 copy calls across this batch. Fe16 force exports fall from 464 to 460 bytes and Fe54 from 1,376 to 1,372; force uploads remain 73,700 and 7,292 bytes respectively. No timing improvement is attributed to this small metadata change. The reused CPU reference and previous production run are from other allocations; there are no new same-allocation ablations in this batch.

All 148 source hashes, 661 component-evidence hashes and 148 matching working runtime files verify. Twenty independent verification reports pass, plus the raw scientific recheck. The radial/orbital/comparator/owner/masked-force/shared-force checks remain passing, along with 18 retained-grid comparisons, nine inverse maps and all cycle/force/final-grid/density audits. Local validation passes 75 tests. The single A100-SXM4-80GB allocation, driver 580.95.05, stopped with zero tasks; the final container inventory is empty.

This is an untraced explicit-copy qualification. Fe54 still exports 18,181,964 bytes over the full post-SCF phase, including legacy electronic matrices and iteration-dependent diagnostics; requested final density exports are separate. The four-byte per-force reduction does not explain other run-to-run diagnostic-volume differences. Independent larger-case CUDA observation, managed/mapped-memory coverage and complete I/O qualification remain open.

## Next integral primitive: pristine two-center oracle

The next target is shared S/T integral-and-gradient evaluation for kinetic, overlap and nonlocal projector consumers. [TWO_CENTER_RESIDENCY.md](TWO_CENTER_RESIDENCY.md) records the exact upstream semantics and migration sequence. The new CPU oracle builds actual reduced tables using the unmodified pinned registry, spline, Fourier and spherical-harmonic routines, then calls `matel_t%get_matel`. Eight fixtures retain 18,500 query results, including angular products through order 10, nonzero table offsets, origin/axis behavior and cutoff/spline boundaries. Independent checks reproduce raw table/output hashes, exact outside-cutoff zeros, 1,500 finite-difference gradients (maximum error 1.693e-9) and 1,936 reciprocity elements (maximum error 3.571e-11).

This oracle is a prepared numerical reference; the GPU two-center evaluator and material table/neighbor handoff are not implemented yet. Qualify the device evaluator first, then material shadows, production contractions and shared final accumulation, preserving the complex SOC conventions. Only delete electronic exports after their remaining force/analysis consumers have migrated. Learned SCF stays deferred.


## GPU two-center primitive qualified

[Run 20261006T175529Z](../results/20261006T175529Z/REPORT.md) qualifies the isolated FP64 S/T two-center evaluator in `native/two_center.{hpp,cu}`. All 79 GPU scenarios pass independent retained-array reanalysis. The verifier reconstructs all 384 executed input arrays and checks 2,872,688 output elements across 718,172 query positions, including expected-zero error cases. The 38 valid ordinary/strided/sticky comparisons have zero numerical difference from pristine `matel_t%get_matel` results at stored FP64 precision. This is numerical equality, not a claim about signed-zero bit patterns or performance.

The primitive preserves one-based registry/harmonic maps, reduced radial tables, coefficients and nested summation order, the positive-z 1e-12 near-origin replacement, inclusive cutoffs, exact spline arithmetic and spherical-harmonic product orders through 10. It handles asymmetric table bounds and nonzero bases. One kernel borrows staged tables and query arrays, writes values/gradients and maintains a sticky device status, without allocation, copies, synchronization or I/O inside the operation. Invalid queries produce defined zeros. The existing production orbital helper is unchanged.

Ordinary and captured execution run one and three replays with poisoned outputs before every replay. A 131,073-query fixture crosses the capped 1,024-block, 128-thread launch's strided tail. Captures contain one kernel and no forbidden nodes. Faults cover low/high query and harmonic indices, term bounds/order, unsupported angular indices, radial indices, cutoff/base descriptors, nonfinite coordinates/tables/coefficients, interpolation range, overflow, sticky status, empty expansions and invalid host configuration. Shape-derived allocations, staging, exports and poison coverage reconcile independently. These are isolated probe boundaries; no whole-process CUDA observer ran.

All eight fresh pristine CPU table fixtures also pass cutoff, raw-output hash, reciprocity and finite-difference checks. The one Fe2 force regression passes independent raw stdout/eigenvalue/RHO/DRHO comparison at 25 SCF iterations, with zero explicit transfers inside the electronic cycle. It uses the existing CPU final-force route: this batch does not qualify material integration of the new primitive. The unchanged common preflights and 40 blocked-solver scenarios pass; all 153 source hashes, 67 component-evidence hashes and 153 corresponding working runtime files verify. Eighteen retained-grid field checks pass. Local validation passes 75 tests. The single A100-SXM4-80GB allocation, driver 580.95.05, is stopped with zero tasks and an empty final container inventory.

`--probe-two-center` enables the standalone check. The new evaluator is compiled only into its check library, not linked into production force consumers. No new material speedup or export removal is claimed. Next stage actual main S/T tables once, collect post-SCF kinetic/nonlocal/overlap query geometry and complete pristine derivative references, and shadow device evaluation over every material/relaxation/SOC case. Then make derivatives authoritative and migrate their contractions to retained D/E and a force/stress/energy accumulator. Preserve outer strict neighbor cutoffs and periodic sparse multiplicities separately from the integral evaluator's inclusive cutoff. The scalar and complex off-site SOC nonlocal contractions require their own pristine arithmetic gates before production substitution. Learned SCF remains deferred.


## Material two-center derivative shadow qualified

[Run 20261006T182622Z](../results/20261006T182622Z/REPORT.md) passes all nine materials and both Gamma/sampled-k directional MAE comparisons. All eleven scientific comparisons reproduce from raw stdout, eigenvalues and requested RHO/DRHO outputs. Across 17 geometries, including nine relaxation steps, the complete GPU comparison covers 7,653,010 integral queries and 30,612,040 value/gradient elements with zero numerical difference and no nonfinite values. This is numerical equality at stored FP64 precision, not a signed-zero bitwise claim.

`--two-center-shadow` stages the actual immutable main S/T tables once, then captures complete query/reference batches from kinetic, scalar or off-site SOC nonlocal, and overlap force consumers. Each geometry enforces kinetic → nonlocal → overlap ordering; the appropriate S/T owner remains fixed across relaxation. Every nonempty consumer executes a two-kernel graph: integral evaluation followed by complete value/gradient comparison. All 51 material consumers execute; only a 56-byte comparison summary and four-byte status return per consumer. CPU integral results and force contractions remain authoritative.

Across the batch, table staging totals 21,376,936 bytes, diagnostic query/reference staging totals 489,795,700 bytes, and diagnostic downloads total 3,060 bytes. These intentional shadow costs are counted separately and excluded from performance qualification. The 360 electronic iterations have zero explicit host/device transfers. The batch is untraced; independent larger-case CUDA, managed/mapped-memory and complete I/O qualification remain open. This milestone does not remove legacy electronic downloads or establish full final-force residency.

All 79 evaluator scenarios, 52 full-comparator scenarios and 26 ownership scenarios pass independent retained-evidence checks. Comparator cases include launch-tail value/gradient faults, nonfinite inputs, overflow and sticky status. Owner cases cover missing or duplicate tables, invalid maps/dimensions, changed host buffers after staging, invalid phase/mode, consumer ordering, stale epochs, failed comparisons, empty batches and changed-geometry reuse. The material audit reconciles raw lifecycle records and shape-derived transfers; it does not independently replay every material derivative array.

All 161 source hashes, 774 component-evidence hashes and 161 matching working runtime files verify. The prior orbital/atomic/shared-force/solver checks, retained-grid comparisons, inverse maps and all cycle/force/final-grid/density audits pass. Local validation passes 75 tests. The A100-SXM4-80GB allocation (driver 580.95.05) is stopped with zero tasks and an empty container inventory. The preceding CPU image-build failure is retained in `results/two-center-build-v64`: a macOS-hidden `nlefsm.F`/`nlefsm.f` path mismatch was corrected before GPU execution.

Next retain compact geometry and contraction topology before SCF, make kinetic/nonlocal/overlap contractions consume the retained device D/E matrices, and accumulate their energy/force/stress results without per-consumer downloads. Qualify the scalar and complex SOC arithmetic separately, preserving sparse multiplicities and the exact upstream spin reconstruction, then validate the full material corpus before deleting legacy exports. Other analysis consumers and the remaining residency gates also remain. Learned SCF stays deferred.

## Kinetic/overlap contraction primitive qualified

[Run 20261006T190534Z](../results/20261006T190534Z/REPORT.md) passes all 202 isolated GPU scenarios with independent reconstruction of 1,352 executed input arrays. Maximum normal-output discrepancy from the literal pristine Fortran oracle is 4.122e-13. All 98 captures contain one kernel and no forbidden nodes; faults include sparse indexing, nonfinite data, accumulator overflow, sticky status and launch-tail errors. The primitive performs no allocation, host/device copies, synchronization or I/O inside its operation. All 166 source hashes, six recovered component hashes and 166 working runtime files verify; 75 local tests pass.

The completed probe was recovered after Modal preemption. Its automatic retry failed on the existing run directory, so this run contains no material result. The preceding run's unchanged Fe2 force path passes raw scientific reanalysis; its two original checker failures are retained and explained in the [pair-force qualification](PAIR_FORCE_RESIDENCY.md). Both allocations are stopped. Material contraction topology, retained D/E handoff, extended shared force accumulation and scalar/SOC nonlocal ports remain. No new production speedup or export reduction is claimed.

## Kinetic/overlap material handoff: partial evidence

`--pair-force-shadow` now connects actual orbital/atom query topology, GPU-generated S/T derivatives and retained device D/E. Its leases enforce kinetic → grid → overlap ordering and prevent premature analysis/reset. Sparse metadata is checked against the geometry that indexed the electronic matrices. All 88 handoff probes and 202 numerical primitive checks passed remotely; the primitive's independent reanalysis also passes. Existing shared/grid-force checks pass. Patched Fortran compiles locally and 78 tests pass.

[Run 20261006T192943Z](../results/20261006T192943Z/REPORT.md) completed Fe2 forces and relaxation, with two raw scientific comparisons reproduced and 20 pair contributions over ten geometries checked. Its worker disappeared before the remaining material cases. All 172 source hashes verify; 171 recorded files still match the working tree, with only `modal_app.py` intentionally changed to repair interruption handling. Full material/SOC/MAE qualification remains pending in run `20261006T195822Z`. The new attempt isolation has already survived a real automatic preemption/restart while preserving the old attempt separately. CPU pair outputs remain authoritative and no speedup or export removal is claimed.

## Kinetic/overlap retained-state material shadow qualified

[Run 20261006T195822Z](../results/20261006T195822Z/REPORT.md) passes all nine materials and both Gamma/sampled-k directional MAE comparisons. All eleven scientific checks reproduce from raw stdout, eigenvalues and requested RHO/DRHO. Across 17 geometries, including nine relaxation steps, 34 kinetic/overlap contributions cover 1,370,568 queries using retained device D/E and GPU-generated S/T derivatives. Maximum energy, force and stress discrepancies are 4.514e-11 Ry, 1.833e-15 Ry/Bohr and 2.554e-15 Ry/Bohr³. Energy uses the existing absolute-plus-relative gate; no gate changed.

All 88 retained-state handoff scenarios, 202 pair-contraction scenarios, 79 two-center scenarios, 52 comparison scenarios and 26 two-center owner scenarios pass independent audits. The shared-force/grid-owner checks pass, together with cycle, force, final-grid, density and 18 retained-field checks. All 173 archived source hashes and 957 component-evidence hashes verify; 173 working runtime files matched at recovery before beginning the SOC-projector addition. Local validation passes 78 tests.

All 350 electronic iterations record zero explicit host/device transfers. This run is untraced and does not close independent larger-case CUDA, managed/mapped-memory or complete I/O coverage. Pair diagnostics add 28,185,968 upload bytes and 7,520 output bytes after SCF, with zero D/E upload bytes. CPU pair outputs remain authoritative. No electronic export has been removed by this shadow and no new speedup is qualified.

The completed attempt ran on an A100 80GB PCIe with driver 580.95.05. Its app stopped with zero tasks and an empty container inventory. An earlier automatically interrupted attempt was preserved separately; all completed material evidence belongs to attempt `2ff40ef2fd78420386d1c6bea0643dda`, with no cross-attempt merge.

Next stage geometry/index data before SCF, move scalar and complex SOC nonlocal contractions to the retained electronic state, extend final accumulation across all force consumers, and remove intermediate reference uploads/readbacks and legacy electronic exports after their remaining consumers migrate. Learned acceleration remains deferred.

## SOC projector primitive and nonlocal contraction reference

The new `native/soc_projector.{hpp,cu}` primitive passes all 168 GPU scenarios in run `20261006T202253Z`. Independent retained-array reanalysis reproduces every normal output with zero numerical difference from pristine Fortran at stored FP64 precision, verifies all 144 mutated input arrays, and reconciles shape-derived staging, exports, poisoning and one-kernel captures. The 64 pristine angular/sign/split fixtures contain 263,680 queries; a separate angular-matrix contraction and full-operator finite differences also pass. This is isolated arithmetic, not material nonlocal-force integration or whole-process CUDA observation.

The next contraction reference, `scripts/check_nonlocal_force.py`, extracts literal scalar, SOC l=0 and angular-SOC sparse/energy/force blocks. All 54 fixtures pass independent NumPy gathers and contractions, with maximum discrepancy 1.137e-12. It preserves scalar duplicate accumulation, SOC last-entry semantics, all complex density signs, both energy channels and force/stress ordering. Empty/padded/periodic data, nonzero initial outputs, additive replay and the 131,073-query tail are retained. See [the migration contract and references](NONLOCAL_FORCE_RESIDENCY.md).

The SOC GPU batch's Fe2 force calculation passes independent raw stdout/eigenvalue/RHO/DRHO reanalysis at 25 SCF iterations, with zero explicit electronic-cycle transfers. All unchanged prerequisite and 40 blocked-solver checks pass. Source/component/working-runtime verification passes 178/667/178 hashes. The A100-SXM4-80GB allocation (driver 580.95.05) stopped with zero tasks and an empty container inventory. [The completed report](../results/20261006T202253Z/REPORT.md) records the full scope. The tested primitive is linked only into its check library; nonlocal density contraction, geometry ownership and production accumulation are the next implementation steps. Local validation remains 78 passing tests; pinned upstream is clean.

## Nonlocal contraction primitive qualified

[Run 20261006T205852Z](../results/20261006T205852Z/REPORT.md) passes all 398 isolated GPU scenarios and independent retained-input/output reanalysis. The 108 normal executions agree with literal pristine scalar, SOC l=0 and angular-SOC force blocks to a maximum 1.294e-12. All 189 captures contain one kernel and no forbidden nodes, including one/three additive executions. The independent audit reconstructs all 2,717 native input-array bindings and 289 mutated arrays; unchanged arrays are referenced by exact hashes rather than duplicated in the archive.

The primitive preserves scalar duplicate accumulation, SOC last-entry assignment, all eight density-component signs, separate scalar and four complex SOC energies, and original force/stress conventions. Faults cover invalid configuration/CSR/atom indices, nonfinite consumed inputs, deliberately unused inputs, overwritten SOC density entries, arithmetic overflow, energy/imaginary-energy/force/stress accumulator overflow, sticky status, and a 131,073-query strided tail. Invalid contributions do not accumulate; any nonzero status disqualifies the complete result. The operation itself performs no allocation, host/device copies, synchronization, scalar polling or I/O. Shape-derived probe boundaries reconcile independently; there is no whole-process CUDA trace.

The updated 54-fixture pristine oracle also retains omitted-final-query outputs for tail faults. Independent NumPy sparse gathers reproduce all original and omitted-tail outputs to 8.811e-13. The ordinary Fe2 device-controlled SCF calculation passes raw stdout/eigenvalue/RHO/DRHO reanalysis, with 25 iterations and zero explicit electronic-cycle transfers. This run uses the existing CPU final-force path; it does not qualify a new material nonlocal substitution. The unchanged common checks and all 40 blocked-solver scenarios pass. All 183 source hashes, 67 component-evidence hashes and 183 corresponding working runtime files verify. Local validation passes 78 tests. The single A100-SXM4-80GB allocation (driver 580.95.05) is stopped with zero tasks and no containers.

`--probe-nonlocal-force` builds and runs the isolated check library only. No production speedup, material-force replacement or electronic-export removal is claimed. Next connect actual projector/neighbor geometry and GPU overlap values to retained D, extend kinetic → nonlocal → grid → overlap ownership, and qualify the material shadow before making the contribution authoritative. Learned SCF stays deferred.

## Compact nonlocal material handoff qualified

[Run 20261006T213041Z](../results/20261006T213041Z/REPORT.md) passes all nine materials and both Gamma/sampled-k directional MAE comparisons. All eleven scientific checks reproduce from raw stdout, eigenvalues and requested RHO/DRHO. Across 17 geometries, including nine relaxation steps, the nonlocal shadow uses 6,282,442 GPU overlap records, 1,156 projector centers, 355,852 neighbor records and 16,840 compact projector blocks. Maximum full-vector energy, force and stress discrepancies are 6.007e-9 Ry, 1.020e-14 Ry/Bohr and 2.663e-14 Ry/Bohr³. The existing absolute-plus-relative energy gate and absolute force/stress gates are unchanged.

Actual geometry, sparse metadata and overlap queries are staged during the matrix-only H0 build, then explicitly bound after the grid epoch advances. Every query and compact descriptor must reproduce at final forces, and the sparse fingerprint must match the retained grid owner. The GPU gathers scalar duplicate weights in pristine order or selects the last SOC entry, evaluates angular operators directly from interleaved GPU overlaps, and contracts retained device D. Its lease enforces kinetic → nonlocal → grid → overlap. No density upload or pair-expanded electronic/operator storage is introduced.

All 204 handoff scenarios, 398 nonlocal arithmetic scenarios, 168 SOC projector scenarios and 202 kinetic/overlap scenarios pass independent audits after sharing the exact arithmetic helpers. The two-center, comparator and owner checks pass, as do the existing atomic/orbital/shared-force checks, retained-grid fields and cycle/final-grid/density audits. All 191 archived source hashes and 1,201 component-evidence hashes verify. The 191 working runtime files matched at recovery. The single A100-SXM4-80GB allocation, driver 580.95.05, stopped with zero tasks and no remaining containers.

All 355 electronic iterations record zero explicit host/device transfers. The run is untraced; independent larger-case CUDA, managed/mapped-memory and complete I/O coverage remain open. Nonlocal geometry staging totals 210,292,712 bytes before SCF. Its final diagnostic initial-output uploads and full-output downloads each total 4,848 bytes, with zero D upload. Nonlocal query metadata is no longer reuploaded by the two-center shadow, but full CPU reference derivatives are still diagnostic inputs. CPU nonlocal outputs remain authoritative; no production speedup or legacy electronic-export removal is claimed.

The independent mode audit identified an omitted `REMAG_NONLOCAL_FORCE_SHADOW` entry in the saved runtime-settings whitelist. The executed archive's launcher assignment, runner guard and native guard, together with the complete matching raw owner/transfer records, establish activation. The verifier retains these source hashes and labels this mode evidence explicitly. The original failed mode assertion is preserved; numerical evidence and original results were not changed. A subsequent reporting-only fix adds the missing setting for future runs, recorded in `runtime-telemetry-followup.json`; it is the only runtime-file change since qualification. The earlier two-warning CPU build failure is preserved separately. Patched Fortran compiles locally and 78 local tests passed.

Next qualify compact-kernel error/edge cases and geometry-owner negative probes, then make these contributions authoritative in the extended shared final accumulator. Kinetic/overlap geometry must also move before SCF. Remove comparison uploads, per-consumer readbacks and legacy electronic exports only after all remaining force/analysis consumers migrate. Learned SCF stays deferred.

## Compact nonlocal sparse-kernel qualification

[Completed run and evidence](../results/20261006T221200Z/REPORT.md).

The compact nonlocal kernel passes **734 GPU scenarios** and independent retained-input/output reanalysis in run **20261006T221200Z**. The new composed CPU reference extracts pristine sparse-density, scalar/SOC force and angular-projector routines literally. Its 75 fixtures cover scalar spin layouts 1/2/4/8, SOC angular orders 0–5, both energy splits, mixed blocks with actual projector offsets, duplicate and absent sparse entries, padded integral storage, excluded local rows, periodic/self atom identities, additive replay and a 131,073-neighbor strided tail. An independent angular-matrix/CSR calculation agrees with the CPU reference to 1.649e-12.

The independent GPU audit reconstructs all 5,572 input-array bindings across 398 variants. All 276 numerically compared executions pass, with maximum discrepancy 7.248e-13. Every one of the 336 captures contains exactly two kernels and no forbidden nodes. Scratch is poisoned before each replay outside capture. Faults cover invalid spans and indices, missing/first/other-row/padded lookup entries, extreme angular indices, nonfinite consumed and deliberately unused data, overwritten SOC entries, overflow, sticky status and tail faults. Structurally invalid inputs have no meaningful partial physics result: their checks require error status, safe completion and finite initialized outputs; every nonzero status disqualifies the full result. This is isolated graph/arithmetic evidence, not a whole-process CUDA trace or an exhaustive malformed-CSR proof.

Hardening initializes active scratch before lookup validation, verifies canonical last-entry mappings, checks the selected entry against its actual row span and validates angular order before computing block widths. Geometry capture also validates complete coordinates and neighbor/block descriptors before appending them. The valid material path is requalified below; geometry-owner negative and recovery probes remain necessary before production substitution.

All nine material cases and both Gamma/sampled-k directional MAE comparisons pass independent raw stdout, eigenvalue, RHO and DRHO reanalysis. Across 17 geometries, the compact shadow consumes 6,282,442 GPU overlap records with retained device D. Maximum material energy, force and stress differences are 2.399e-9 Ry, 1.263e-14 Ry/Bohr and 3.411e-14 Ry/Bohr³ under the unchanged gates. All 204 handoff, 398 standalone nonlocal, 168 SOC-projector and 202 kinetic/overlap checks pass, together with the two-center, atomic/orbital/shared-force and retained-field audits. All 194 archived source hashes, 1,207 component evidence hashes and 194 corresponding working runtime files verify. Local validation passes 78 tests. The runtime-settings record now explicitly includes the nonlocal mode, so no archived-source fallback is needed for this run.

All 364 SCF iterations record zero explicit host/device transfers. Strict residency remains unproven: independent larger-case CUDA, managed/mapped-memory and complete I/O coverage are still open. Nonlocal geometry stages 210,292,712 bytes before SCF; its diagnostic initial-output upload and full-output download each total 4,848 bytes, with zero D upload. CPU derivative references and CPU kinetic/nonlocal/overlap outputs remain authoritative. The run does not establish a new production speedup or remove legacy electronic exports. Its single A100-SXM4-80GB allocation (driver 580.95.05) is stopped with zero tasks and an empty container inventory.

Next complete geometry-owner failure/recovery checks, stage kinetic/overlap geometry before SCF, and extend the shared accumulator through kinetic → nonlocal → grid → overlap. Replace CPU contributions and remove diagnostic transfers together; migrate the remaining analysis consumers before deleting electronic exports. Fresh matched performance and full residency observation remain required. Learned SCF stays deferred.

## Nonlocal geometry owner qualified; shared pair staging next

[Run 20261006T232754Z](../results/20261006T232754Z/REPORT.md) passes nine materials, both directional MAE comparisons and all eleven raw scientific rechecks. Its 40 actual owner probes cover 45 geometries and 925 rejection/recovery operations across scalar and SOC storage, including empty queries, captured execution and two-geometry reuse. Independent sparse/matrix contractions agree with complete GPU outputs to 2.776e-17. Failed reference/status completion cannot be retried to clear a lease; stale epochs prevent application/publication and downstream consumers. Preparation rejects overlapping row spans and binds only after its preparation epoch advances.

All 734 compact, 398 standalone nonlocal, 168 SOC-projector, 202 pair and 204 retained-state handoff checks pass, with the existing two-center, atomic/orbital/shared-force, final-field and density audits. All 196 source and 1,368 component hashes verify. The 196 working runtime files matched before starting the next candidate; subsequent changes are recorded separately. All 367 SCF iterations have zero explicit transfers. Full residency and fresh performance remain unqualified; CPU final contributions, diagnostic references and legacy exports remain. The A100 stopped normally, with zero tasks and no containers verified before the next launch.

The next candidate, now under single-A100 validation, stages orbital-pair topology and coordinates during H0 and shares the device snapshot between kinetic and overlap. Both final traversals must reproduce it exactly. This removes per-consumer metadata uploads from those post-SCF paths, while retaining the existing diagnostic reference/output transfers. Based on the preceding corpus, geometry uploads should fall from 72,036,080 to 36,018,040 bytes and move before SCF; this is a projection pending GPU evidence. A six-scenario snapshot/recovery/stale-owner probe is included. Strict C++ checks, patched Fortran compilation and 78 local tests pass. Next extend the shared final accumulator and single publication through all force terms, then migrate remaining analysis consumers.

## Shared pair geometry qualified; authoritative integral accumulator candidate

Run **20261006T235525Z** passes all nine material comparisons and both directional MAE checks. Eleven scientific comparisons reproduce from raw stdout, eigenvalues and recovered RHO/DRHO, bound to the returned archive before use. Across 17 geometries and **373 SCF iterations**, explicit electronic-cycle accounting records zero H2D/D2H bytes and zero copy calls. Independent CUDA/managed-memory observation and complete filesystem coverage remain unqualified.

Kinetic and overlap now share geometry staged once before SCF. Measured geometry uploads total **36,018,040 bytes**, down from **72,036,080 bytes** in the preceding run for exactly matched geometry/query records: a **50% reduction** in this boundary. The final pair shadows still upload 8,064 diagnostic bytes and export 7,520 bytes. CPU derivative references and CPU kinetic/nonlocal/overlap outputs remain authoritative. This is a transfer reduction, not a newly qualified speedup or full final residency.

The six geometry scenarios revalidate all nine retained device snapshots, two-geometry recovery, 100 recovery rejection labels and stale lease behavior. All 200 source hashes and 1,393 component hashes pass. Independent two-center, 734 compact nonlocal, 398 nonlocal, 168 SOC, 202 pair, 204 handoff, atomic/orbital, shared-force and retained-field audits pass. Material energy/force/stress checks, device cycles, final grid and requested density checks also pass. Maximum pair discrepancies are 3.184e-11 Ry, 2.274e-15 Ry/Bohr and 2.721e-15 Ry/Bohr³ under unchanged gates.

The single A100-SXM4-80GB allocation, driver 580.95.05, is stopped with zero tasks and no remaining containers. The archived working-tree proof predates the next migration; `next-migration-status.json` identifies later changes. The next candidate connects actual kinetic, nonlocal, grid and overlap kernels to one accumulator, retains Hartree stress and publishes one terminal result. Strict C++ syntax checks, patched Fortran compilation and 78 tests pass; its CUDA execution and numerical equivalence are not yet verified. CPU ionic terms, legacy analysis exports, full residency observation and fresh matched performance remain open. Learned SCF is deferred.


## Authoritative integral-force pilot and reporting correction

[Run 20261007T002812Z](../results/20261007T002812Z/REPORT.md) completes the first Fe2 force calculation with the shared GPU kinetic/nonlocal/grid/overlap accumulator. Independent raw stdout, eigenvalue and RHO/DRHO reanalysis passes every unchanged scientific gate. The 25 SCF iterations record zero explicit copies; final integral publication is 204 bytes with no intermediate or derivative-reference export. This is one pilot, not a completed corpus or speedup qualification.

The harness then aborts because it expects an `epoch` absent from the native device-cycle schema. The next candidate publishes the real epoch, treats missing epoch evidence as an invalid result instead of a collector crash, and adds the previously omitted integral-transfer total to reconciliation. Forty new publication-owner probes cover spin layouts 1/2/4/8, partial core, two-geometry recovery, failed device status, nonfinite output, stale epochs and retry rejection. Local C++ checks and 79 tests pass. The failed app is stopped with zero tasks/containers; its replacement is launched serially, within the one-GPU cap.

Remaining migrations include final matrix consumers (charge/spin populations and relaxation density history), CPU ionic contributions, initialization staging and requested-output policy. Full independent residency/I/O observation and fresh matched performance remain open. No learned acceleration is enabled.


## Authoritative GPU integral-force path qualified

Run **20261007T004619Z** passes all nine material comparisons, both Gamma/sampled-k directional MAE comparisons, and eleven independent rechecks of raw stdout, eigenvalues and requested RHO/DRHO. The corpus covers **17 geometries and 366 SCF iterations**, including nine relaxation geometries. Every complete electronic cycle records zero explicit host/device bytes and copy calls. This is explicit accounting; independent larger-case CUDA, managed/mapped-memory and complete I/O observation remain open.

Kinetic, nonlocal (including complex SOC), grid and overlap contributions now accumulate on the GPU and publish once after all consumers finish. The corpus exports **5,052 bytes in 34 terminal copies**, with no intermediate force readbacks or derivative-reference uploads. Two-center tables, nonlocal geometry and shared pair geometry stage before SCF. CPU ionic terms, analysis/history and legacy electronic matrix exports remain; this is not a claim that the entire process is GPU-only.

All 40 publication-owner scenarios pass independent arithmetic, lifecycle and transfer audits, covering spin layouts 1/2/4/8, partial core, two-geometry recovery, device failure, nonfinite results, stale epochs and retry rejection. Component audits, final-grid and requested-density checks pass. All **207 source hashes and 1,514 component evidence hashes** verify. Archived working-tree agreement was recorded before the next candidate changed runtime files.

The first material-auditor attempt incorrectly classified initialization grid outputs as intermediate final-force copies. Its failure and verifier are retained under `audit-history/initial-grid-scope`. The corrected audit binds the actual initialization exports to grid and sparse-matrix shapes and raw grid counters: **24,781,792 bytes** remain before SCF. Four injected phase/volume/count faults are rejected. SCF and post-SCF intermediate-output gates and all numerical thresholds are unchanged.

The single A100-SXM4-80GB allocation (driver 580.95.05) completed and stopped with zero tasks and no containers. CPU references were reused from another allocation, so no new speedup is qualified. Fresh matched performance, especially the non-collinear device cycle, remains required. Learned acceleration is deferred.

The next candidate removes H(Dout), Dold and optional Eold downloads by publishing H(Dgen) directly to its final host destination. D/E compatibility exports remain until their analysis/history/requested-output consumers migrate. Local syntax/Fortran checks and 79 tests pass; compact exports still require GPU and requested-HSX validation.

[Completed evidence](../results/20261007T004619Z/REPORT.md).


## Compact terminal electronic exports qualified

All nine material cases, both directional MAE comparisons and eleven raw scientific rechecks pass. Across 17 geometries and 350 SCF iterations, explicit whole-cycle accounting records zero host/device bytes or copy calls. Independent CUDA, managed/mapped-memory and complete filesystem coverage remain open.

Publishing the Hamiltonian that generated the final density directly to its host destination removes redundant H(Dout) and Dold exports. Measured eliminated matrix volume is **37,308,608 bytes**. All nine requested HSX outputs pass the additional fixed matrix/geometry gates; the maximum H discrepancy is 4.030e-7 Ry (rounded upward). D/E compatibility exports, requested outputs, CPU ionic terms and initialization remain. Original scientific tolerances are unchanged.

All 209 source hashes and 1,514 component evidence hashes verify, together with component, integral-force, final-grid and final-density audits. Working-runtime agreement was recorded before the subsequent vector-cache and population changes. Those changes are not covered by this qualification. Incomplete-fetch audit attempts remain retained separately.

The single Modal A100 allocation is stopped with zero tasks and no containers. CPU references were reused from another allocation, so **no new speedup is qualified**. Latest Fe16 non-collinear timing is 45.230 s, including 39.759 s in the device cycle over 46 iterations; this is one observation. An optional retained eigenvector bank is the next performance experiment. Learned SCF remains deferred.

[Completed report](../results/20261007T011339Z/REPORT.md); [current performance scope](../results/20261007T011339Z/current-performance-scope.json).

## Retained eigenvectors and fresh matched performance qualified

Run **20261007T015902Z** uses fresh serial upstream references on the same
single A100 80GB PCIe allocation, three cold candidate repetitions and one
cache-disabled control for each of nine cases. All 36 material comparisons,
nine cache controls and eight directional MAE comparisons reproduce from raw
stdout, eigenvalues and RHO/DRHO under the unchanged gates. All 36 requested
HSX outputs pass; maximum H discrepancy is 1.455e-7 Ry (rounded upward).

Fe16 non-collinear falls from **127.227 s CPU to 24.202 s GPU median (5.26×)**,
with 43 SCF iterations in every run. Retaining eigenvectors lowers its GPU time
from **41.752 s to 24.202 s (1.73×; 42.0% less time)**; the three candidate
times span 24.189–24.204 s. Sampled-k FePt2 SOC gains another 1.37× from this
cache, about 27% less runtime. Gamma already retains all vector blocks, so this
toggle has no expected algorithmic effect there. Configured solve counts and
bank shapes are audited; no independent executed-kernel count is claimed.

Fe54 collinear records **240.504 s CPU to 17.366 s GPU median (13.85×)**, but
SCF counts differ: 61 CPU versus 53/45/50 GPU iterations. This is an end-to-end
ratio, not a per-iteration improvement or a cache gain. Fe2 forces and relaxation
record 2.35× and 2.01× CPU/GPU ratios. CPU and control timings each have one
sample; these comparisons do not cover an optimized multicore CPU baseline.
Provisioning, builds and artifact transfer are excluded from all run timings.

All 215 source hashes, 1,520 component evidence hashes and component/material
audits pass. Across 68 geometry executions and 1,454 SCF iterations, explicit
whole-cycle transfer accounting records zero bytes and calls. Independent
CUDA/managed/mapped-memory and full I/O observation remain unqualified. Final
analysis/history consumers, CPU ionic terms and requested exports remain.

The isolated population primitive passes all 184 GPU scenarios and independent
NumPy reconstruction (maximum error 2.221e-15, rounded upward). Its 131,073-row
fixture covers the partial last block but does not cross the 256-thread ×
1,024-block stride; a larger fixture remains necessary. The production owner
and Fortran consumers are drafted afterward, disabled by default and outside
this run's qualification. Strict C++ syntax and patched Fortran compile checks
pass; Modal wiring and CUDA/material owner validation remain pending. D/E
compatibility exports and derived spin/species arithmetic remain.

The allocation is stopped with zero tasks and no containers. Source agreement
with the working tree was recorded before the next population edits; frozen
archived parsers were used for every v80 audit. No v81 GPU has been launched.
Learned SCF remains deferred.

[Current performance report](../results/20261007T015902Z/PERFORMANCE.md);
[qualification](../results/20261007T015902Z/qualification-summary.json).

## Retained GPU population contractions qualified

Run **20261007T024355Z** passes all nine material calculations, both directional
MAE comparisons and eleven raw scientific rechecks under unchanged gates. Final
forces, grid/density outputs, all requested HSX matrices and terminal exports
also pass. All 221 source hashes, 1,713 component evidence hashes and 221 current
working runtime files verify. Local validation passes 79 tests and patched
Fortran/C++ compile checks.

The opt-in population route contracts retained GPU D/S into orbital, atomic and
total spin-charge populations. It retains topology/overlap before the private
cycle buffers are freed and merges population failure into the existing force
status before publication. Across 17 geometries, atom/problem staging is
**740 bytes**, requested population output is **20,400 bytes in 43 copies**,
and topology/overlap retention is **6,241,616 device-to-device bytes**. Repeated
final reads are cached. There is no additional population status download or
overlap upload. All **358 SCF iterations** record zero explicit host/device
bytes and copy calls.

All 198 primitive executions pass independent reconstruction and original
Fortran gates, with maximum independent error 2.221e-15 (rounded upward). The
new 262,145-row fixture crosses the actual capped-grid stride. All 48 actual
owner scenarios pass independent audits across 56 geometries and 544 rejection/
recovery operations, including poisoned original D/S/topology, failed status,
nonfinite results, stale epochs and cached repeated exports. Maximum owner
population/force reconstruction error is 5.552e-17 (rounded upward).

The single A100 80GB PCIe allocation is stopped with zero tasks and no
containers. CPU references were reused, so this run qualifies no new speedup;
the preceding fresh matched performance report remains current. The full
component archive and requested outputs are retained locally and bound by
hashes. Source-backed independent audits used the frozen executed package.

Derived spin-vector/species/partial-charge arithmetic, D history, D/E
compatibility exports, CPU ionic terms, initialization and complete independent
memory/I/O observation remain. A nine-fixture pristine-spnvec reference and two
unintegrated CUDA kernels are retained in
`results/preflight-population-observables-v82`; those kernels are not GPU-qualified.
The E-consumer source audit identifies optional TSDE/NetCDF exports that must
remain correct when unrequested downloads become conditional. No next GPU has
been launched. Learned SCF remains deferred.

[Population qualification](../results/20261007T024355Z/POPULATION.md);
[machine-readable evidence](../results/20261007T024355Z/qualification-summary.json).

## Derived population observables integrated; GPU qualification running

The v82 candidate extends the retained population owner with orbital and
atomic spin vectors, signed collinear spin, species-order cumulative totals,
net atomic charges and final spin/charge summaries. Species IDs and valences
are staged before SCF. Two device kernels run before the existing shared
force-status publication. Every terminal export validates its buffer size;
accepted repeated reads use the owner's cached result.

Final spin output, non-collinear/SOC Mulliken tables and Mulliken partial-charge
formatting now consume these GPU views. Collinear up/down orbital tables still
request their raw components. Initialization's spin printer remains upstream
because final electronic state does not exist at that point. D/E compatibility
exports are unchanged in this candidate.

All 79 local tests, strict C++ syntax checks and patched Fortran compilation
pass. The candidate is frozen in `/tmp/remag-v82-candidate-source.tar.gz` and a
single guarded Modal launch is active as app `ap-NCTTjPAxzT8qUaR1MWBwSj`, launcher
session 9275, log `/tmp/remag-population-v82.log`. The new GPU primitive checks
compare against pristine `spnvec`, including a true second grid stride; the
owner probe includes derived arithmetic failure joining the shared status.
Neither GPU numerical qualification nor new performance is claimed yet.

The previous fresh matched performance remains run `20261007T015902Z`. This
batch reuses its CPU references and cannot establish a new speedup. The full
residency goal remains active; conditional E output, D-history migration,
initialization/ionic terms and independent memory/I/O observation remain.


## Derived spin and charge observables qualified

Run **20261007T033144Z** passes all nine materials and both MAE comparisons;
all 11 scientific comparisons reproduce from raw outputs with unchanged gates.
Derived GPU arithmetic now supplies final spin printing, NC/SOC Mulliken tables
and Mulliken partial-charge output. Collinear up/down tables retain their
requested raw components. Initialization's spin printer is unchanged.

All 34 derived-output fixtures, 198 population executions and 56 owner scenarios
pass independent audits. The owner audit covers 64 geometries and 624 rejected
or recovered operations. Derived fixtures agree exactly with pristine spnvec;
the owner maximum discrepancy is 5.552e-17 (rounded upward). The final table
audit covers 2,882 values, labels and order. Force, grid, density, HSX and
terminal-export audits pass, with 225 source hashes and 1,749 component-evidence
hashes verified against the executed archive.

All **391 SCF iterations across 17 geometries** record zero explicit host/device
bytes or calls. Population staging is 1,580 bytes in 85 copies; requested
population output is 28,128 bytes in 85 copies, cached after the first read per
kind. The single A100 allocation is stopped with zero tasks and no containers.

No new speedup is qualified because CPU references were reused. Fe54 takes
76 iterations and 23.931 seconds here, so the next fresh performance batch must
retain repetitions and iteration counts. D/E compatibility exports, resident
history, initial H0 and ionic terms, and complete memory/I/O observation remain.
The E-consumer audit identifies 16,890,848 bytes of existing E downloads across
the preceding corpus's 17 geometries; omission must preserve requested TSDE and
NetCDF outputs. No E download has been removed by this change. Learned SCF
remains deferred.

[Qualified report](../results/20261007T033144Z/REPORT.md);
[evidence summary](../results/20261007T033144Z/qualification-summary.json).

### v83 — requested energy-density exports (validation running)

The candidate now records explicit terminal E export reasons: legacy forces
(bit 1), requested TSDE (bit 2), or NetCDF EDM at end of cycle (bit 4). With
GPU integral forces and no requested E output, the terminal E download is
omitted. The resident force owner keeps its device E; D/history exports are
still present. Cycle telemetry is schema 3, with shape-derived bytes and
copy counts checked against raw FDF requests.

Local preflight: 82 Python checks, strict C++ syntax, patched Fortran without
NetCDF, and real pristine CPU TSDE files for collinear, NC and SOC all pass.
The TSDE reader preserves eight SOC DM components and four EDM components.
New matrix gates were fixed before GPU execution: DM 1e-7, EDM 1e-6 Ry and
Ef 1e-5 eV, with exact sparse metadata. Existing scientific gates are unchanged.

One Modal app, `ap-PS5wswUkqLKEFZlpin0sUQ`, is building/running validation.
The image enables NetCDF for both pristine reference and candidate. The
batch requests nine base materials plus nine TSDE/NetCDF variants, fresh CPU
references and three GPU repeats per case. It has not yet qualified this
candidate or a new performance result. Live handoff: `/tmp/remag-v83-handoff.json`;
launcher log: `/tmp/remag-energy-v83.log`. Do not launch a second GPU job.

### v83 diagnostic outcome and v84 validation correction

Run `20261007T041440Z` stopped after the first NetCDF material candidate was
marked invalid. The process completed and converged, with both requested GPU
density exports present. The validator incorrectly required separate RHO/DRHO
files although upstream `CDF.Save` stores those fields in `GRID/Rho` and
`GRID/RhoDelta` inside `remag.nc`. The original invalid result, ledger and
failed-run report are preserved; no speedup or full residency is qualified.
The three TSDE repetitions passed before this failure. The GPU was released.

The corrected reader handles the actual upstream format, preserves spin axes
and precision, and rejects missing/unwritten fields. The retained NetCDF pair
passes every raw numerical gate: maximum density/DRHO discrepancy
1.637e-10 e/Bohr^3, DM 3.764e-10 and EDM 4.490e-10 Ry (rounded upward).
Existing tolerances remain unchanged. These are diagnostic comparisons, not
retroactive approval of the failed batch. Local checks: 83 pass. The native
CUDA/Fortran implementation has not changed; a fresh full v84 batch is next.

For the following history migration, pristine CPU reference preparation now
covers 16 destructive sparse-remap cases and six default DIIS/SVD coefficient
fixtures (maximum independent discrepancy 2.510e-14). No GPU history owner or
kernel is implemented yet. Evidence: `results/preflight-density-history-v84`.

The corrected v84 batch is live as run `20261007T044323Z`, app
`ap-dG5Ld7Br58FGlsPLYvH5Pu`, after verified release of v83. All 226 runtime
source hashes match the executed archive. It starts with the omitted-E case,
then exercises NetCDF/TSDE choices, all nine base materials and fresh repeated
performance. Live handoff: `/tmp/remag-v84-handoff.json`; no second GPU launch
is permitted while this call is active.

### Production GPU initialization qualified for the base corpus (v87)

The v84 matched-performance batch is complete and qualified; its performance
report remains current. The v85 history owner and v86 initialization components
subsequently passed isolated GPU checks. Production initialization now passes
all nine base materials, both MAE pairs, force/density/population checks and
explicit transfer audits in `20261007T063020Z`. Its 381 SCF iterations across
17 geometries have zero explicit host/device transfers. Initial H/grid round
trips are removed; about 110.3 MB of startup transfers are omitted across the
suite, while density/history and other setup boundaries remain.

This does not qualify full residency or TSDE/NetCDF variants with the new mode.
The comprehensive requested-output audit retains its missing-coverage failure.
The integral auditor's old startup-copy expectation was made conditional on the
verified native initialization mode; all 54 older records still pass unchanged
expectations. Source evidence, initial audit failures, the metadata-only saved
environment correction and remaining limits are retained in the
[qualification summary](../results/20261007T063020Z/qualification-summary.json).
The allocation is stopped. No new speedup is claimed from reused references.

Density-history capacity growth subsequently passed isolated GPU qualification
in `20261007T070741Z`: 132 sequences, 2,300 operations and 472 rejected operations,
with every retained state snapshot rechecked locally. Growth uses device kernels
at geometry boundaries and preserves all history entries; capture-time growth
is rejected. The single allocation is released. Production history handoff,
initial populations and requested D output gating remain; no actual production
history download has been removed by this isolated component change.
Learned SCF remains deferred.

The initial device density/population owner is implemented and its isolated
GPU probe passes 16 cases and 264 rejected ownership/error paths. Independent
raw replay confirms requested population arithmetic, cached exports and exact
device density preservation when the host source is poisoned. Modal interrupted
the first v89 attempt before material qualification; that component evidence is
preserved separately in `20261007T073134Z-interrupted-beff36f1f2654a26a1bbc9c61d68a051`.
The fresh attempt stopped after 12 successful collinear comparisons because
the initial-population validator wrongly required a raw total from the NC/SOC
printer, which uses derived orbital/atomic/species views. All 13 completed
calculations pass the unchanged scientific value gates in a separate raw-output
diagnosis; the NC initial table matches all 126 printed values exactly. The
original invalid result and ledger remain unchanged. Both attempts' 16 isolated
initial-owner cases replay successfully, and the 310 completed SCF iterations
record zero explicit copies. The v89 allocation is stopped.

The correction requires every consumed view for the corresponding spin layout;
no numerical tolerance changes. All 88 Python checks pass, including omission
tests for each required view and a cross-attempt evidence-recovery regression.
Native and Fortran sources remain identical to v89. Corrected run
`20261007T081045Z` is active under one new guarded GPU allocation, with the
previously rejected NC case first, all 18 cases, fresh CPU references and three
GPU repeats. No new speedup or complete production initial-owner qualification
is claimed yet. Performance publication additionally requires the new
initialization and initial-state audits.

The missing SOC history reference is now available locally: pristine sampled-k
FePt SOC with a 400 Ry mesh and 0.02 Ang maximum CG displacement converges across
seven geometries and 80 SCF iterations, including four nontrivial depth-three
history uses. Its final maximum force, 0.00452725 eV/Ang, meets the unchanged
0.01 eV/Ang gate. Full raw-output and source provenance checks pass in
`results/preflight-soc-history-relax/fept2_soc_z-mesh400-step002`.
This is a CPU reference for upcoming production history validation; it does not
qualify GPU history or remove the remaining terminal D download.

### Production density-history integration staged (v91, not GPU qualified)

The next candidate is isolated in `build/history-integration-v91`; the live
v90 source archive still matches all 239 working runtime files. The staged
Fortran hooks preserve upstream history depth/reuse policy, remap/extrapolate
through the device history owner, and retain converged D only after accepted
forces. Initial populations and SCF borrow that same extrapolated allocation.
The terminal D download is conditional on recorded file, diagnostic or legacy
analysis consumers; final-grid and density-output ownership no longer depend
on reading host D. H and spectral exports are still present.

Local validation passes 91 Python tests, strict C++ syntax checks and 16
Fortran compilation units. The NetCDF D/E policy also compiles with the
upstream options module rebuilt under CDF/NCDF_4. The new trajectory reader
replays all 9/9/7 geometries of the pristine collinear/NC/SOC history references,
including 6/6/4 nontrivial depth-three uses. These are local checks, not GPU
production qualification. Small geometry/status/coefficient transfers remain
explicit outside SCF. Staged source, hashes, logs and remaining work are saved
in `results/preflight-density-history-integration-v91`.

The v90 live log now reports all 54 material comparisons passing, including
Fe54. Official evidence return, raw-output reconciliation, independent audits
and allocation release are still pending; no new qualified performance claim
has been published.
