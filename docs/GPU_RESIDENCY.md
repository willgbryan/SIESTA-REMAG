# GPU residency: architecture and historical evidence

**Engineering priority, revised 8 October 2026:** GPU-first execution and
purpose-built GPU data structures must deliver an end-to-end speedup over
upstream GPU SIESTA. Validate numerical results directly against upstream SIESTA.
Keep state on device and remove costly transfers where this improves the
implementation, but do not require exhaustive residency certification or build
further observer/audit infrastructure as a prerequisite. The prepared v114 NVBit
diagnostic is deferred. The [project scope](../README.md#limitations) and
[contribution guidelines](../CONTRIBUTING.md) supersede older acceptance gates
in this historical record.

The following records the earlier residency target, implementation evidence and
remaining boundaries. It remains useful for choosing and measuring optimizations.
Learned SCF remains deferred while the conventional performance baseline is
established.

As clarified on 7 October, fully relativistic, spinful noncollinear magnetic
systems are the primary target for architecture, correctness and performance.
Exercise complex spinors with off-site SOC and fully relativistic
pseudopotentials, including general moment directions, finite k points,
forces/stress, directional MAE, requested exports and repeated geometries.
Use upstream GPU SIESTA as the main performance competitor and pristine CPU
SIESTA as the numerical oracle. Simpler spin layouts remain regressions and
secondary timing checks; gains on them are not assumed. Gold–water is withdrawn
from the active benchmark plan.

Converged nonparallel full-SOC coverage now passes three linear Fe3 inputs:
Gamma in two moment orientations and two sampled k points. The
[GPU follow-up](../results/nonparallel-soc-v112b-20261008T164432Z/qualification.json)
and [direct local replay](../results/nonparallel-soc-v112b-20261008T164432Z/raw-comparison.json)
pass all nine CPU/REMAG/native comparisons, eight-bank exports, forces/stress,
moments and effective convergence checks. Equally tighter SCF thresholds resolve
the original saved-DM discrepancy without changing comparison tolerances.
Final edge-to-central angles remain about 97.82 degrees. These fixed, unrelaxed
regressions do not establish a physical ground state or Fe3 speedup. Independent
memory/filesystem coverage remains unknown; explicit-copy counters alone do not
prove strict residency.

## Earlier strict-residency target (not a current delivery gate)

After one-time input staging, the steady-state SCF loop must use device-resident electronic state. The strict target is zero host-to-device and device-to-host transfers inside that loop, including scalar convergence/error readbacks, and zero file reads/writes. GPU memory reads and writes are, of course, still needed. CPU launch orchestration can remain; it must not require host copies of electronic state or perform numerical SCF work.

The target is a measured execution property, not the absence of an explicit `cudaMemcpy` call. Unified-memory migrations, library transfers, CPU serialization, file-backed scratch state and host-side array validation count as boundaries too. Moving a file to a RAM disk does not make the calculation GPU-resident.

Input parsing, pseudopotential loading and initial staging are explicitly outside the steady-state interval. At the end, state should remain on the GPU for another electronic calculation, geometry step or orientation. Export only requested final energies, moments, forces, densities, wavefunctions or checkpoint files. Regression runs must still export the artifacts needed to compare against upstream; exports should occur at explicit validation boundaries and remain included in end-to-end solution timings.

Removing error checks or scientific outputs to make transfer counters read zero is not acceptable. Errors and convergence must be computed/stored on device, with safe termination and explicit final reporting. If a library needs host involvement, record and resolve that boundary rather than silently exempting it.

## Latest accepted checkpoint

The latest completed [FePt16 SOC upstream comparison](../results/upstream-gpu-refresh-v143-20261009T101214Z/qualification.json)
measures the integrated v142 implementation at **58.2746×** median directional-pair
speedup over pristine native ELPA GPU SIESTA on one A100-SXM4-80GB. Three matched
x/z repetitions give **573.311 s upstream versus 9.838 s REMAG**. Scientific/export,
MAE, GPU-execution and timing gates pass at the recorded DM 1e-9 / H 1e-8 eV
stopping criteria. The earlier v117 18.4347× result remains historical; subsequent
incremental optimization ratios are not multiplied into either result. This
workload-specific result qualifies performance; strict residency remains unproven
and is not a delivery prerequisite. The larger FePt128 z comparison subsequently
measures a provisional **43.1289×** in one matched pair, with direct local
science/export checks passing for both contenders and their GPU comparison.
[Selected benchmark evidence](../benchmarks/evidence/README.md) records the timing details and limits.

Run [20261007T184850Z](../results/20261007T184850Z/qualification-summary.json)
is the latest broad scientific and CPU-comparison checkpoint for the promoted
compact-query implementation. All 99 material comparisons, six directional MAE
checks and 43 independent audit stages pass across 33 input/output cases and
three cold GPU repetitions. Its 321 geometries contain 4,599 SCF iterations;
every complete SCF region records zero explicit H2D/D2H bytes and copy calls.
All requested outputs are verified, with 984 raw files recovered from the same
completed attempt. The allocation is released.

GPU query expansion replaces 4,006,097,184 bytes of expanded query uploads with
185,989,080 bytes of compact metadata across the three repetitions, a 95.36%
reduction for that representation. Nonlocal H0 matrices remain resident, while
the four-byte setup status readback per geometry remains outside SCF.
[Fresh matched timings](../results/20261007T184850Z/PERFORMANCE.md) against
serial upstream CPU measure 6.11× for Fe16 NC, 7.32× for FePt SOC history
relaxation and 14.16× for Fe54. These are combined solution speedups, include
setup and requested output, and can include different SCF iteration counts;
they do not isolate the newest optimization. CPU setup/ionic consumers,
independent managed/mapped-memory observation and complete filesystem coverage
remain unfinished. Explicit-copy counters alone do not prove strict residency.

| Boundary | Accepted implementation and remaining work |
| --- | --- |
| Electronic SCF | Device graph owns iteration, convergence, error state, eigensolver, occupations, sparse/grid state and mixing. Explicitly accounted SCF copies are zero; complete independent memory/I/O coverage remains unknown and is not a delivery prerequisite. |
| Initial density/populations | Device density owner, normalization and initial population calculation pass the full corpus. Initial D stages once; subsequent supported geometries reuse the device history owner. |
| Final observables | Device forces, stress, density fields and population observables pass the full corpus; requested files are checked independently. |
| Geometry density history | Production handoff and conditional terminal D export pass full v91b history and requested-output audits. Eighteen history trajectories cover 150 geometries and 1,622 SCF iterations across collinear, NC and SOC, with requested DM enabled and disabled. |
| Hamiltonian/spectrum exports | Final eigenvalues remain on GPU until one requested analysis export per calculation; occupations have no host export. H/D/E downloads are gated by supported consumers. All 18 no-matrix controls pass with zero matrix downloads over 84 geometries and 1,028 SCF iterations. |
| Geometry/setup | Compact nonlocal queries expand on GPU; nonlocal H0 is assembled on GPU and leased to SCF, and its integrals remain resident through forces. CPU neighbor selection, orbital mesh values, kinetic/overlap setup and ionic updates remain. Setup staging/status reads occur outside SCF and count toward end-to-end runtime. |
| Scientific output | Requested energies, forces, moments, EIG, density and matrix files remain explicit host/export boundaries. Omitting unrequested matrix files passes matched upstream controls; requested files remain verified. |

The next staged core/neutral-field owner (v111) replaces the two initial field
uploads with direct device leases through SCF and forces. Host compilation and
local reader/CPU-export checks pass; GPU and material qualification are pending
behind the v107 mesh production, v108 field components and v110 full-field
shadow. It is isolated from the accepted runtime. Its explicit VNA output path
downloads once when `SaveNeutralAtomPotential` requests it, using the upstream
binary or NetCDF writer. This planned boundary is outside SCF and remains part
of end-to-end timing. The change adds no strict-residency or speedup claim.

The new [v103 timing and consumer review](../results/preflight-initial-pair-matrix-v113/profile-and-consumers.json)
measures overlap plus kinetic setup at a median 0.64% of end-to-end time for
FePt2 SOC and 1.34% for its SOC history trajectory, versus 7.55% for Fe54
collinear. These are inclusive routine timers, including capture/staging, not
isolated CPU arithmetic. Removing these producers advances residency; these
measurements do not establish a large SOC speedup from that change alone.

The overlap migration must also move upstream's atomic/user density correction
`DM(ind,:) / S(ind)` before initial populations, across all eight SOC banks.
Loaded or extrapolated D must bypass that correction. Initial populations and
H/S assembly currently stage S separately; SCF already borrows H/S assembly's
device overlap. Early requested NetCDF S/gradS output, overlap-only output and
optional `DM.Init.RandomStates` remain explicit consumers. The prepared v113
component passes 104 CPU composition checks and eight sensitivity controls;
456 GPU scenarios are prepared but neither CUDA compilation nor execution has
occurred. Production integration and device lifetime qualification remain open.
An additional composed probe keeps geometric selection, radial S/T values,
sparse assembly and the atomic correction on device. Its 134 CPU references
and 286 prepared GPU scenarios are retained in the
[composition preparation](../results/preflight-initial-pair-matrix-v113/composition-preparation.json).
It stages no CPU integral/cutoff/distance arrays. It remains uncompiled and
unexecuted on GPU and removes no accepted production boundary yet.
Its guarded launcher and independent raw rechecker are prepared, with 36 local
controls and 46 bound files. Launch requires completed, independently checked
nonparallel SOC qualification and release of that allocation; the real missing-
prerequisite control stops before Modal. Image compilation is outside the GPU
call, and component scenarios execute sequentially on one GPU.

The earlier [v91b timing review](../results/preflight-history-rank-v91b/remaining-boundary-profile.json)
identified `Setup_H0/nlefsm` at 46% of Fe2 relaxation time before the later H0
and compact-query migrations. It remains historical evidence, not the current
performance-priority profile.

The complete nonlocal H0 shadow in
[20261007T141909Z](../results/20261007T141909Z/qualification-summary.json)
passes all 33 material/output cases, two directional MAE checks and 42 independent
audit stages. Scalar and SOC H0 matrices match upstream exactly across all
107 geometries; the same GPU integrals remain resident through authoritative
final forces. Requested TSDE/NetCDF variants and output-omission controls pass.
This diagnostic still uses CPU H0 for SCF and adds matrix exports before SCF.
Its 1,535 electronic iterations have zero explicitly recorded copies. It does
not qualify production H0 timing or strict residency. The earlier
[twelve-case shadow](../results/20261007T130409Z/qualification-summary.json),
including its correctly rejected full-output coverage claim, remains retained.

The production replacement passes the
[twelve-case host-poisoning diagnostic](../results/20261007T134522Z/qualification-summary.json):
42 audit stages, 42 geometries, 633 electronic iterations and two directional
MAE checks pass with the unused CPU H0 matrices filled with NaNs. It assembles
nonlocal H0 on device, leases those matrices to the SCF owner and retains the
integrals through final forces. H0 matrix uploads and downloads are zero.
The four-byte setup status readback totals 168 bytes, all before SCF in that diagnostic.

[Full unpoisoned production](../results/20261007T145602Z/qualification-summary.json)
now passes 99 material comparisons, six directional MAE checks and all 43 audit
stages. The complete requested-output and omission corpus passes, including
TSDE/NetCDF. H0 matrix uploads/downloads remain zero; 321 setup status readbacks
total 1,284 bytes before SCF. All 4,620 electronic iterations record zero explicit
copies. Fresh serial CPU references and three cold GPU repeats on one A100-SXM4-80GB
measure 5.07× for Fe16 NC, 11.88× for Fe54 and 4.84× for SOC history relaxation.
Fe54 uses 50/64/60 GPU iterations versus 61 on CPU, so its ratio is a solution-time
measurement. The preceding timing allocation used A100 PCIe; differences between
these batches do not isolate H0's incremental effect. The allocation is released.
CPU geometry, kinetic/overlap setup, ionic work and independent memory/I/O observation
remain unfinished.

The poisoned Fe2 relaxation profile records 9.818 seconds in `nlefsm` and
2.145 seconds in device SCF cycles. Nonlocal setup remains a priority; its CPU
preparation and GPU execution must be timed separately before choosing the next
optimization. These diagnostic timings do not establish a speedup or regression
against a different allocation.

The [seven-case setup profile](../results/20261007T165703Z/SETUP_PROFILE.md)
now separates these costs with monotonic host intervals outside SCF. All seven
raw scientific comparisons and the Gamma SOC MAE comparison pass. In Fe2
history relaxation, geometry preparation/staging takes 6.213 s and GPU graph
launch/wait takes 0.767 s within a 16.099 s calculation. In SOC history
relaxation those regions take 7.397 s and 1.402 s within 26.485 s. Fe16 NC and
Fe54 also spend more nonlocal setup time preparing geometry than executing the
graph. Tiny Gamma SOC cases differ: GPU launch/wait is about 0.57 s versus
0.034 s preparing geometry. These intervals include host/API/wait costs and
do not measure individual GPU kernels.

The v102 migration targets repeated CPU construction and staging of integral
queries: retain compact neighbor displacements and projector identities, then
expand the repeated query representation on device. Upstream neighbor selection,
query ordering, scalar/SOC arithmetic and retained force consumers must remain
verified. CPU preparation includes other work, so these timings are not an
estimate of the speedup this change will achieve.

The [isolated compact-query expansion](../results/20261007T174532Z/component-verification.json)
now passes all 22 direct/captured GPU checks with nine source files verified
against the frozen inputs. It reproduces the existing pair/displacement
representation exactly, including replay with changed inputs. An initial
large replay failure is retained; fixing test-harness stream ordering resolves
it without changing the CUDA expansion kernel. Its production integration is
now qualified in v102 above. This component alone does not qualify material
accuracy, speedup or strict residency. CPU neighbor selection also remains
outside its scope.

This is a one-repeat profile with reused CPU references. The full-corpus force
and requested-output gates correctly reject its seven-case coverage; those
failures are retained. A separate
[profile-scope check](../results/20261007T165703Z/profile-material-verification.json)
verifies the individual scientific, force/output and H0 lease checks without
granting full-corpus qualification. The GPU is released. The accepted full
production/performance checkpoint remains `20261007T145602Z`.

The [isolated observer calibration](../results/20261007T160805Z/capability-recheck.json)
compiled and executed four sequential controls on one A100-SXM4-80GB, using
CUPTI 12.6.80 and 12.8.90 with and without kernel activity. All four native
arithmetic controls pass. Independent rechecking validates managed/pinned/device
allocation lifetimes and Runtime/Driver host-registration records. All 39 source
fingerprints and archived source files match the prepared calibration.

Complete observer qualification fails: Unified Memory configuration succeeds,
but enabling its activity returns `CUPTI_ERROR_NOT_READY` in all four variants.
The zero migration records are therefore unobserved, not zero traffic. Mapped-host
access observation and UM phase attribution remain unknown. The device reports
both pageable-memory access attributes as zero in these controls. Modal's Linux
sandbox denies `/proc/self/clear_refs` for both temporary and material-volume
mapping tests; page-write tracking remains unavailable. The cold-storage control
was not reached in this first calibration. Its allocation is released.

The [initialization/scope follow-up](../results/20261007T163623Z/capability-recheck.json)
and [traced counter-selection control](../results/20261007T164248Z/capability-recheck.json)
also fail migration coverage while passing all native arithmetic and independent
allocation/registration controls. The traced run verifies all 39 source files.
Both late-init CUPTI 12.8 controls fail to open `/dev/nvidia-uvm-tools` with
`ENXIO`, despite readable/writable device-node metadata. The
[retained syscall evidence](../results/20261007T164248Z/profiling-device-open-evidence.json)
binds each failure to its native control PID and raw trace hash. This identifies
an unavailable profiling interface in this allocation; it does not prove every
CUPTI initialization error has the same cause. Zero migration records remain
unknown traffic. Mapped-host kernel access remains unobserved.

The follow-ups independently attempted cold-storage controls on temporary and
material-volume files. Both produced zero storage/page-fault counter deltas and
failed observer qualification; advisory eviction cannot guarantee a cold read.
Soft-dirty tracking remains unavailable. All three calibration allocations are
released. These observers remain outside production math. Further identical UM
initialization trials are deferred; setup profiling can proceed independently.
No strict-residency or performance claim follows from these calibrations.

A separate [effective-address observer preparation](../results/preflight-memory-access-observer-v114/preparation.json)
now explores NVIDIA's NVBit instruction tracing. Its pinned vendor sample is
modified to retain active-lane masks and memory-space/operand/width metadata,
without forcing managed-memory placement. Twenty-five tiny GPU controls are
prepared across device/managed/mapped/registered storage and direct, captured,
manual and conditional launches. Forty-three local synthetic reader controls
pass, including missing and duplicate access rejection. CUDA compilation and
GPU execution have not occurred; the packaged driver limit also conflicts with
the current repository requirements and needs a bounded compatibility check.
This diagnostic uses its own managed/pinned communication buffers, supplies no
migration-byte or filesystem coverage, and is not a production residency
observer. Full instruction and allocation-lifetime coverage remain unknown.
The magnetic comparison and nonparallel Fe3 qualification retain GPU priority.
Its [guarded launch preparation](../results/preflight-memory-access-observer-v114/launch-preparation.json)
now passes another 28 local launch/recovery controls and binds 66 files. The
image compiles before GPU allocation; untraced/traced controls run sequentially
on one GPU, stopping immediately if the first compatibility pair fails. The
launcher rejects missing nonparallel magnetic qualification before contacting
Modal. The independent rechecker requires completed/released evidence and
reproduces raw decisions using archived code. No CUDA build or GPU execution
has occurred, and the completed GPU recheck remains unexercised.

Broader-material preparation now includes two pristine CPU references for a
16-atom FeN supercell derived from upstream's zincblende FeN test. Collinear
and noncollinear calculations converge in 123 and 126 iterations; independent
reparsing verifies eigenvalues, geometry, finite density fields and raw output
hashes. [Reference evidence](../results/preflight-nitride-corpus/reference-verification.json)
and [input provenance](../results/preflight-nitride-corpus/source-provenance.json)
record the exact structure and test pseudopotentials. These are numerical
regression preflights, with no GPU, speedup or materials-convergence claim.
The staged [nitride corpus adapter](../results/preflight-nitride-corpus-v100/status.json)
reproduces both preflight inputs and pseudopotentials byte-for-byte while leaving
all 39 existing generated cases unchanged. It is not promoted or GPU-qualified.
At that preparation checkpoint, boride, carbide, complex-material and
larger-spinor coverage remained pending. Subsequent direct upstream checks
qualify Fe3C collinear/noncollinear (v169), full-SOC Fe2B x/z and MAE (v173), and
the 128-atom, 2,304-state FePt full-SOC x/z pair (v167). The larger FePt z case
also has a verified single-pair upstream-GPU timing comparison (v172). Their
inputs, limits and evidence are described in the [benchmark guide](../benchmarks/README.md).
More complicated materials and the difficult Fe4N noncollinear case remain
unqualified.

The historical inventory and run narratives below retain their original scopes.
The acceptance-gate table is updated to the latest checkpoint stated above.

## Historical boundary inventory

The inventory reflects the qualified opt-in sparse-state path in run `20261006T034347Z`; legacy adapters retain their older boundaries. The byte-count table below retains the first Fe128 candidate in run `20261005T170415Z` as a historical reference point, not a fully resident baseline. Later sections record subsequent consumer migrations and their evidence.

| Boundary | Current reason | Required device consumer/owner |
| --- | --- | --- |
| H/S assembly | Qualified sparse owner supplies H directly to device assembly; immutable S/topology is staged on first use | Move all first-use staging before the complete device-controlled SCF region |
| Eigenvalue/occupation export | Optional deferred export now keeps E/Q on device through validated SCF paths and restores host arrays afterward | Device final observables/workflows; restrict eventual exports to requested data |
| FD result download | Host energy and convergence bookkeeping consumes a 32-byte Fermi/entropy/status result each iteration | Device bookkeeping, error propagation and convergence control |
| Scientific scalars in launch arguments | Several kernels still receive k coordinates, grid volume or normalization targets as host launch arguments; memcpy counters do not measure argument payloads | Stage immutable scientific configuration on device and pass handles/shape metadata in launches |
| Solver status download | Host checks cuSOLVER `info` after every call | Device status propagation and safe loop control; keep error detection |
| Sparse density and energy-density export | Qualified sparse owner retains D/E through normalization, scalar reductions, grid contraction and mixing; CPU forces still require final export | Resident final observables and repeated workflows |
| Density grid download | Validated combined resident path retains density grids through Hartree/PBE; final force/analysis consumers still use the original route | Keep subsequent sparse-state and final-observable consumers resident |
| Grid potential and sparse H updates | Qualified combined path retains potential and writes directly into device-owned H | Preserve ownership through device-controlled execution and final observables |
| SCF scalar summaries | Grid and sparse-state adapters export 224 and 144 bytes per iteration; mixer and normalization status also download | Device convergence, diagnostics, error propagation and continuation |
| Mutable density ownership | Qualified owner checks identities and stage transitions without routine host-content scans; optional debug snapshots remain | Move ownership transitions onto device for a complete captured SCF loop; extend lifetime through forces |
| SCF checkpoint and diagnostic I/O | Corpus end-of-cycle policy and deferred diagnostics remove selected SCF file operations in measured runs; mappings and other limits remain | Complete no-I/O qualification across supported runs, including mapped-file effects |
| Final force/stress and geometry consumers | Upstream CPU implementation | GPU forces, stress and reductions; then resident geometry/workflow updates |

Measured whole-process custom transfers for that candidate:

| Component | Host → device bytes | Device → host bytes |
| --- | ---: | ---: |
| Eigensolver bridge | 3,737,124,864 | 1,622,720 |
| Grid bridge | 2,300,955,912 | 2,076,622,848 |
| Density-matrix bridge | 41,573,376 | 2,998,206,464 |
| Total custom accounting | 6,079,654,152 | 5,076,452,032 |

These counts include setup/final work and exclude unmeasured library internals. They are not steady-state transfer counts. The 7.11× median speedup is useful evidence of the existing implementation, but does not satisfy the residency target. [Measured validation](DENSITY_VALIDATION.md).

## Implementation sequence

1. **Establish ownership and measurement.** Introduce one persistent electronic-state context with explicit geometry/sparsity epochs, spin layout, capacities and versioned device buffers. Give normalization, occupation, energy, mixing and convergence consumers a device API. Keep host exports explicit. Add phase markers for initialization, each SCF iteration, final observables and export; audit CPU/GPU copies, migrations, synchronization and file I/O by phase. Current whole-process counters cannot certify residency.
2. **Close the electronic-solver cycle.** Retain eigenvalues as well as vectors. Port upstream Fermi-level/occupation logic, density normalization, charge/energy/residual reductions and the applicable Pulay mixing path without changing numerical definitions. Preserve SOC/NC conventions and separate H-mixing from DM-mixing correctly. Remove D/E and occupation round trips only when every needed consumer has moved; otherwise retain and count the temporary boundary.
3. **Close density → potential → Hamiltonian.** Keep real-space density, Hartree/PBE XC and their derivative/spin intermediates on device. Assemble local, nonlocal and SOC Hamiltonian contributions into persistent sparse storage; construct k-space matrices there. Move invariant overlap/setup data once. Replace repeated host-content scans with validated lifetime tracking. Select sparse/grid layouts for the complete pipeline, including bounded memory growth.
4. **Close convergence and control.** Compute stopping/error state on device and implement a device-controlled or bounded execution scheme that safely avoids per-iteration scalar readbacks. Verify the numerical libraries' control requirements. CPU launch activity alone is permitted; host convergence decisions, hidden migrations and synchronous diagnostic writes are not part of the strict target.
5. **Keep final work and repeated workflows resident.** Port forces/stress and their energy-density consumers. Reuse device state across orientation sweeps and geometry steps, including structural invalidation when atoms move. Migrate geometry updates/setup where practical. Any remaining geometry-boundary downloads must be measured and disclosed separately from a resident fixed-geometry SCF claim.
6. **Qualify the baseline.** Run collinear, non-collinear, SOC/MAE, force and relaxation regression against pinned upstream, including representative sizes. Repeat timings with identical requested outputs and retain raw traces. Only then reconsider learned acceleration, using the resident conventional path as its baseline and fallback.

Steps overlap where dependencies allow. A port that leaves a consumer on CPU is partial progress, not a reason to skip required physics or declare the boundary removed. Library/solver experiments and k-point batching serve this architecture; they do not replace the residency work.

## Acceptance gates

Status below reflects accepted v103 evidence. Staged initialization experiments
at that checkpoint do not count as production qualification.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| Device ownership | All SCF arrays and history have explicit device lifetimes and invalidation; no host snapshots needed for routine validity | Device-controlled SCF, final observables and supported geometry-density history pass the current corpus; setup and ionic boundaries remain |
| Bulk-state residency | Phase-resolved trace shows zero electronic-array transfers during SCF; no disk scratch/checkpoint dependencies | All 4,599 SCF iterations across 321 geometries record zero explicit array or scalar copies; complete independent memory/I/O coverage remains open |
| Strict loop residency | Zero scalar readbacks/transfers and zero disk I/O inside SCF, including library and diagnostic activity | Unqualified: hidden migration, mapped-memory and complete filesystem observation are not established |
| Complete observable path | Device energy, moment, force and stress evaluation; explicit requested exports; repeated workflow boundaries accounted for | Current corpus passes device observables and requested-output checks; remaining CPU setup/ionic work and orientation reuse prevent a complete repeated-workflow claim |
| Numerical equivalence | Existing upstream gates pass for all target spin modes, MAE and relaxation; nonconverged/unsupported execution cannot qualify | 99 material comparisons and six directional MAE checks pass across 33 input/output variants and three GPU repeats; broader material/size coverage remains |
| Credible performance baseline | Repeated end-to-end and SCF measurements with documented hardware, output policy, transfers, allocations and memory | Matched end-to-end timings qualify the current implementation against fresh serial CPU references; strict-resident and optimized multicore CPU baselines remain pending |
| Learned work may start | All applicable residency/observable/scientific/performance gates above satisfied; unresolved boundaries explicit and agreed | Deferred |

A stage with zero bulk transfers but scalar readbacks may be reported as **bulk-state resident**. It does not satisfy the stricter zero-transfer loop target. Unknown counters and missing trace coverage never count as zero. No host fallback may silently turn an unsupported calculation into a claimed fully resident run.

## First I/O change

New benchmark inputs now default to `Write.DM false` and `Write.DM.End.Of.Cycle true`. This uses upstream's existing controls to skip iterative density-matrix checkpoints while preserving one end-of-SCF-cycle DM artifact, including each completed geometry cycle during relaxation. Both CPU reference and GPU candidate receive the same policy. Historical prepared inputs and evidence remain unchanged. `--io-policy checkpoint` restores the former checkpoint behavior for a controlled comparison.

This is a limited first step: stdout, final result files, timing/CML output, harness telemetry and all current GPU/CPU data boundaries still exist. Checkpoint availability during an interrupted SCF cycle changes intentionally; final validation/restart artifacts are retained. We do not label this policy “no I/O.”

`scripts/check_io_policy.py` validates the change on unmodified upstream for displaced forces, relaxation and both FePt SOC directions, comparing all existing scientific observables and MAE. Its separate output-policy experiment allows exactly the `Write.DM` boolean difference. Original input hashes and the ordinary comparator's provenance rejection are retained, and no performance speedup is qualified across differing output policies.

The [local validation](../results/io-policy-20261005/REPORT.md) passed all four pairs and the direct MAE check. Every compared observable was identical at its stored/output precision, with unchanged SCF counts (25, 96, 35 and 32). All twelve local tests passed. That policy-only experiment required no GPU allocation or native code changes. The subsequent geometry-lifetime validation below also uses the new policy on the GPU derivative.

Final DM files exist in both policies but are not byte-identical. With the default `SCF.MixAfterConvergence false`, the iterative checkpoint can be the preceding mixed state, whereas the end-of-cycle path writes the final output state. The policy therefore changes the restart artifact's timing intentionally. It does not alter the compared converged observables. File syscall counts have not yet been traced, so this validation does not certify the complete absence of other iterative I/O.

Modal capacity remains capped at one GPU instance. This direction does not authorize multiple GPUs, concurrent instances, learned-model work or changes to scientific tolerances.

## Geometry lifetime and phase accounting

The first ownership change makes stored grid basis/topology lifetimes explicit. The integration announces geometry changes in `dhscf_init` and resets in `dhscf_reset`; dimensions and addresses identify a staged layout within that lifetime. New geometries and unversioned callers still validate content. Normal execution avoids repeated host scans of the immutable basis. `REMAG_GRID_VERIFY_EPOCH=1` retains full byte hashing and rejects an unannounced mutation; `REMAG_GRID_EPOCH=0` provides a content-hashing control. Mutable density still uses the previous snapshot checks; this change does not establish ownership of all electronic state.

`remag-residency.json` now records successful explicit copies separately for initialization, every SCF iteration, post-SCF work and analysis/export. Boundary names include H/S, eigenvalues, eigenvectors, solver status, grid inputs/outputs, density topology, occupations, energy weights and D/E exports. The runner rejects inconsistent phase/bridge totals or SCF marker coverage. Whole-process, all-SCF and post-first-iteration traffic remain distinct. Library transfers, migrations and file I/O are unknown; this audit cannot yet certify strict residency.

The [debug validation batch](../results/20261005T190557Z/REPORT.md) passed six upstream pairs (Fe forces, nine-geometry relaxation, Gamma non-collinear Fe, both sampled-k FePt SOC directions and Fe54) plus the direct MAE check. Nineteen native scenarios and thirteen local tests passed. Full hashing verified every epoch hit: relaxation performed nine basis uploads and 220 verified reuses, with all 96 SCF intervals accounted for. All candidate transfer totals reconciled. The subsequent [normal-mode validation](LIFETIME_VALIDATION.md) passed 24 upstream pairs, six GPU controls and FePt16 MAE, including Fe128. Its median Fe128 time is 100.354 s, 1.596× faster than the hashing control; iteration counts and timing limits are disclosed. Both runs preserve source snapshots matching their implementation and harness hashes.

The opt-in `REMAG_NORMALIZE=1` path performs the charge trace and D/E scaling on device before their existing host export. Its [first validation](NORMALIZATION_VALIDATION.md) passes 16 upstream pairs, eight GPU controls and two MAE checks through Fe54, with normalization in every SCF iteration. It retains upstream tolerance checks, a counted four-byte error-status readback, and normalized D/E exports for remaining CPU consumers. The subsequent untraced batch also passes FePt16 and Fe128, with 20 upstream pairs, five controls and MAE. It demonstrates no incremental end-to-end normalization speed benefit. This is consumer migration, not removal of the D/E download boundary.

## Device control capability experiment

CUDA 12.6 supports conditional graph loops whose continuation condition is evaluated on device; their bodies permit a restricted set of node types on one device. This provides a possible control mechanism, but does not establish that our solver library calls are compatible. [NVIDIA CUDA 12.6 programming guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#conditional-graph-nodes).

The optional `--probe-control` experiment captures real/complex generalized cuSOLVER routines, inventories visible graph copy/host nodes, and attempts two iterations in a conditional graph. Input staging and final validation downloads occur outside that loop. The first attempt in `20261005T200824Z` encountered capture errors, followed by alternating warmup failures in the shared process. The [fresh-process probe](IO_VALIDATION.md) resolves that ambiguity: all twelve ordinary warmups succeed, while generalized divide-and-conquer and generalized Jacobi reject capture for every tested size/type. In run `20261005T203343Z`, six further probes separate Cholesky, triangular reduction, standard Jacobi and back-transformation; their warmups also succeed but capture errors occur at the standard Jacobi call. That batch did not validate a device loop. The production solver remains unchanged; later small-matrix capability results are detailed below.

The probe also checks full overlap orthogonality and changes the Hamiltonian between device iterations, to reject stale results. Its later generic-batched variant passes this check at size 64 for real and complex matrices, while larger sizes fail capture. These experiments are not SCF residency traces; the separate CUPTI audit below measures copies and kernels but leaves CPU-library work, migrations and mapped-host access unresolved.

## Filesystem measurement

The optional `--trace-io` diagnostic wraps each candidate in `strace` and records selected file syscalls, descriptor reads/writes and mappings across all its threads. Native phase records now carry contiguous wall-clock timestamps to attribute calls to SCF, initialization and final work. Calls touching a phase boundary are conservatively counted in both phases. Raw `syscalls.log` and parsed phase/operation/byte counts remain in the run evidence. Missing markers, truncated traces and unparsed records leave coverage incomplete; they never establish zero activity. Scientific comparisons remain available, but all traced timings are excluded from speedup qualification and CPU/RSS samples are left unknown rather than sampling the tracer as SIESTA.

The first complete Modal trace, Fe2 forces in `20261005T200824Z`, reports 26 stdout writes (7,022 bytes) across SCF, of which 25 writes (4,610 bytes) occur after the first iteration. It reports no steady-state reads and one steady-state mapping call. All 1,539 selected syscall records parsed with complete phase coverage. This is process-level evidence, not a whole-container disk trace: the tracer and harness observer are separate processes. It does not observe memory-mapped page faults or asynchronous kernel writeback, and deliberately excludes CUDA driver ioctl submission from filesystem accounting. Strict no-I/O certification therefore remains false.

The opt-in `--defer-scf-log` path suppresses the three routine stdout flush sites in the convergence reporter and flushes before/after each SCF cycle. Diagnostic formatting, energy bookkeeping and text are preserved. The harness selects a 1 MiB formatted-output buffer and buffered preconnected units using GNU Fortran's [buffer-size control](https://gcc.gnu.org/onlinedocs/gcc-13.4.0/gfortran/GFORTRAN_005fFORMATTED_005fBUFFER_005fSIZE.html) and [preconnected-unit control](https://gcc.gnu.org/onlinedocs/gcc-15.1.0/gfortran/GFORTRAN_005fUNBUFFERED_005fPRECONNECTED.html). `--log-ablation` supplies the original flush behavior as a same-host control. The [five-case validation](IO_VALIDATION.md) shows zero selected file operations during every SCF interval, with ten upstream comparisons, five direct controls and MAE passing. Buffering is finite and can spill, so other workloads still need measurement. Live SCF text is delayed until the cycle ends. Fatal-error handling is unchanged.

The harness now buffers GPU sampler output in memory and writes `gpu.csv` only after the SIESTA process exits. Telemetry remained populated throughout the deferred-log validation. The sampler still runs as an external observer; its activity is outside the SIESTA syscall trace.

## CUDA-library activity audit in development

The new optional `--trace-cuda` observer loads `libremag_activity.so` only into the requested candidate process. Normal runs do not load CUPTI. It records memory copies and concurrent kernels, including library-issued work, into bounded host memory; output is written only at bridge shutdown after device synchronization and activity-buffer delivery. Raw activity retains copy directions, byte sizes, kernel names, stream/context identifiers and correlation identifiers. Schema 2 additionally retains runtime/driver API records and thread-local external phase IDs. Events are attributed to their launching API region; timestamps are retained only as diagnostics. Asynchronous completion may occur later. Unmapped worker-thread activity and conflicting IDs invalidate attribution. This follows NVIDIA's [external correlation API](https://docs.nvidia.com/cupti/12.9.1/api/structCUpti__ActivityExternalCorrelation.html). Counters for dropped records, storage overflow and callback failures are mandatory evidence. Incomplete timestamps, missing markers or fewer observed bytes than the bridge counted invalidate coverage. The whole-process difference is labeled bytes beyond explicit copies; it does not identify their exact caller.

Local tests exercise hidden scalar copies, boundary overlaps, drops, timestamp gaps, missing metadata and explicit-counter reconciliation. GPU compilation and [six-case trace/scientific validation](CUDA_ACTIVITY_VALIDATION.md) have now passed after correcting shutdown order. Traced timings cannot qualify speedups. Unified-memory migrations, kernels touching mapped host memory and CPU library synchronization remain unknown; even complete memcpy/kernel coverage cannot certify strict residency.

The next isolated control probe additionally tries the pinned toolkit's [generic batched eigensolver](https://docs.nvidia.com/cuda/archive/12.6.2/cusolver/index.html#cusolverdnxsyevbatched) with batch size one after Cholesky reduction. Requested host/device workspace sizes are recorded. A changing Hamiltonian, full residual and overlap-orthogonality check guard against captured stale work. This remains a capability experiment and does not replace the production solver.

In diagnostic run `20261005T211820Z`, the known-array/scalar CUPTI check passes, but the first SIESTA traces fail coverage because final synchronization runs after CUDA begins unloading (`cudaDeviceSynchronize: 4`). These traces are retained as incomplete evidence. Finalization is now explicitly called from `siesta_end`, after scientific work and FDF shutdown but before process/library destruction. The updated Fortran seam compiles locally and 26 local tests pass. The [corrected six-case diagnostic](CUDA_ACTIVITY_VALIDATION.md), `20261005T212723Z`, now passes all coverage/scientific checks and MAE, exposing three extra scalar downloads per solve. Both apps completed sequentially and their GPU containers were released.

That run's isolated generic-batched probes succeed for real and complex size 64: two changing-Hamiltonian iterations, 170 captured nodes, no visible host/copy-to-host/copy-to-device/default-copy nodes, residuals below 1.2e-15 and overlap-orthogonality errors below 2.8e-15. Both use zero requested host workspace. Size 576 and 1152 warmups succeed but capture fails at the standard batched solve with cuSOLVER status 6, despite also requesting zero host workspace. All 18 older generalized/Jacobi probes still reject capture. A small successful graph does not establish a solver for the larger workloads or prove absence of all library-side activity.

## Device occupation component

`occupations.cu` implements the FD branch of upstream `fermid.F` as one device kernel. Bounds, charge bisection, the 150-iteration stopping/error decision, occupations and entropy remain on device. Parameters and scalar results are device pointers; the routine itself allocates no memory, copies no data, synchronizes no host thread and writes no files. Its host adapters perform explicitly counted staging and exports. SIESTA now has an opt-in integrated path described below.

The first GPU check in `20261005T211820Z` passes 25 scenarios. Nine fixtures use compiled, unmodified pinned upstream `fermid.F` with FD/default configuration stubs: spinless, collinear, NC/SOC weighted k points, up to 1,152 spinor bands, padded spectra, degenerate levels, the temperature floor and full bands. Each is checked with ordinary and captured execution. Seven additional captured checks retain device-side errors for invalid shape, nonfinite energies/temperature, negative charge, invalid weights, insufficient bands and nonconvergence. Maximum reference differences are 1.388e-17 in Fermi energy, 1.422e-13 in entropy and 1.777e-15 in occupations; charge error remains below upstream's 1e-10 stopping tolerance. These are component checks, not end-to-end scientific validation or a full-SCF transfer trace.

The [integrated validation](OCCUPATION_VALIDATION.md), `20261005T221744Z`, passes 16 upstream comparisons, eight GPU controls, both MAE checks and 16 CUDA traces through Fe54. All 306 candidate iterations compute device occupations; 5,349 density consumers borrow the retained E/Q bank, with zero SCF occupation/energy-weight uploads. All 28 source hashes verify. Native ownership checks and debug host-mutation guards pass. A Gamma non-collinear half-weight fixture expands the isolated FD suite to 27 scenarios.

The bank captures first-pass eigenvalues directly from the existing eigensolver and freezes them before sampled-k vector solves. Expected spectrum blocks must be present before solving occupations; geometry/layout and one-use consumer checks enforce ownership. In the original occupation mode, temporary E/Q exports and 32-byte FD result downloads remain for host consumers, so transfer volume is not uniformly lower. Fixed spin, single excitation, MP/cold smearing and other unsupported routes fail explicitly.

The subsequent [spectral residency validation](SPECTRAL_RESIDENCY_VALIDATION.md), `20261005T223820Z`, removes host band selection and defers E/Q export until after SCF. All 18 upstream comparisons, nine controls, both MAE checks and 36 traces pass. E/Q cross no host/device boundary in 365 candidate SCF iterations; 17 exports occur afterward. Selected SCF file operations are zero, with mappings and tracing limits disclosed. Larger untraced validation also passes 20 upstream comparisons, five controls and FePt16 MAE across 733 candidate SCF iterations; Fe128 median is 95.498 s (12.036× CPU). No uniform incremental gain over the GPU control is established. The full target still requires grid/H/S/D/E ownership, final observable migration, scalar control and the launch-argument configuration audit.

## Device matrix assembly in development

`REMAG_HS=1` / `--matrix-assembly` adds an experimental sparse-to-dense producer directly inside the existing cuSOLVER buffer owner. It supports real Gamma, complex sampled-k, non-collinear and eight-component SOC input layouts. Topology is grouped by destination orbital pair with a stable ordering; each device thread sums that pair's periodic images in upstream sparse order without floating-point atomics. Spinor order and Fourier-phase conventions are preserved. Only this CUDA translation unit disables FMA contraction, to preserve the explicit upstream multiply/add sequence.

Overlap, sparse topology, translations and k points have geometry lifetimes. Normal repeated iterations do not scan those host arrays. Debug mode checks byte snapshots and rejects undeclared mutation. Every diagonalization still uploads the host-owned sparse H, then each solve assembles into the solver's device H/S buffers. Host dense arrays remain allocated as opaque identifiers; their numerical assembly is skipped. A one-use selector checks block identity, shape, spin, k point and epoch; incomplete, duplicate or unexpected solves fail. Gamma expects one solve per spin block, and sampled-k expects the spectrum and vector passes. The CLI requires resident spectra so the second pass always requests the full needed band range. Unarmed post-SCF/band routines retain their existing behavior.

`remag-hs.json` separates sparse/geometry upload bytes, avoided dense upload bytes, debug host reads and assembly dispatch time. Dispatch time measures CPU submission, not device kernel duration. Transfer reconciliation includes the new component; enabling it requires one prepare/end per SCF iteration, nonzero matrix production, no matrix downloads and zero SCF dense solver uploads. `--matrix-ablation` retains the previous GPU path as a same-host control.

The patched Fortran module, Fermi seam, diagonal dispatcher, all six diagonal builders and post-SCF seam compile locally. The 26 local tests pass. Eight new native scenarios cover spinless/collinear Gamma/k, NC Gamma/k and SOC Gamma/k, periodic duplicate folding, all matrix components, padded sparse capacity, changed H, geometry invalidation, debug mutation rejection, missing/duplicate consumers, foreign host identities and wrong types/shapes. Independent assembly is compared by eigenvalues, generalized residual and overlap orthogonality; NaN host dense sentinels must remain unchanged. All eight scenarios now pass CUDA execution. The [matrix/Hartree diagnostic](MATRIX_HARTREE_VALIDATION.md) passes 27 upstream comparisons, 18 controls, six MAE checks, 54 traces, 38 native checks and all 34 archived source hashes. Across 351 candidate iterations there are 11,266 device matrix builds, zero dense H/S uploads and no H/S downloads. Traced timings do not qualify speedups.

This removes the dense staging boundary for the validated paths. Sparse H remains host-owned, D/E and grids still return to CPU consumers, and final observables and scalar control remain unfinished.

## Periodic Hartree component in development

The next grid dependency is `native/hartree.cu`, a device-pointer equivalent of upstream `poison.F` for the serial periodic FP64 grid. An experimental adapter is now connected behind `REMAG_HARTREE=1` / `--hartree`, and now passes the [matrix/Hartree diagnostic](MATRIX_HARTREE_VALIDATION.md), including forces, relaxation, spinor/SOC and all six candidate/control MAE pairs. Its plan owns reusable device FFT and reduction scratch; scientific geometry data, density, potential, energy, stress and error status are device pointers. `apply()` performs no explicit host/device copy, allocation, synchronization or file operation. Plan creation/destruction belong to geometry setup. Library activity still needs measurement before any zero-transfer claim.

The component preserves the upstream positive-exponent GPFA forward transform, reciprocal vectors including `2*pi`, positive Nyquist-index convention, strict `1e-15 < G² < G²max` filter, Rydberg `8*pi/G²` factor, inverse normalization, energy and all nine fixed-density stress components. A full complex transform keeps the original mesh convention for oblique cells. Fixed tree reductions replace the serial energy/stress sum and require numerical qualification. It uses caller-owned cuFFT workspace and a fixed stream; single-GPU graph capture and workspace APIs follow the pinned [CUDA 12.6 cuFFT documentation](https://docs.nvidia.com/cuda/archive/12.6.3/cufft/index.html).

`scripts/check_hartree.py` builds a small executable linked against pristine `libsiesta.a` and its original dependencies. The numerical Poisson, FFT, cutoff and lattice routines are unmodified, and their source/library hashes are retained. Six local CPU fixtures pass an independent NumPy FFT sanity check: constant density, Fourier modes crossing the cutoff, random orthogonal and oblique grids, a left-handed cell and a 40×36×32 grid. Odd/even dimensions and nonzero mean charge are covered. Constant charge contributes only the excluded zero mode.

The native adapter passes ordinary and captured execution against that oracle, including two executions on the same buffers after a device-only density change. It inventories graph nodes and rejects host or non-device-copy nodes. Seven additional captured fixtures require a device error for nonfinite density, overflow, nonfinite reciprocal geometry/cutoff, zero cutoff, negative volume and a geometry/plan shape mismatch. All 19 GPU scenarios pass; valid captured graphs contain 11 nodes and no forbidden host or non-device-copy nodes. Maximum potential/energy/stress errors are 2.34e-15 Ry, 4.45e-15 Ry and 1.22e-17 Ry/Bohr³. Captured execution is a component capability check, not a full SCF residency claim. Integrating this component still requires device grid reorderings, spin-sum/difference handling, reference-density subtraction, Hartree/neutral-atom energy reductions, PBE XC and the potential-matrix consumer.

The SIESTA adapter reuses its FFT plan across cell-only changes and refreshes reciprocal geometry per epoch; debug checks reject a changed cell inside that lifetime. The Fortran seam obtains the reciprocal lattice and cutoff from the original routines only when setup is needed. The alias-permitting density/potential contract is retained through explicit C pointers. Until surrounding consumers move, each call uploads density, downloads an 88-byte energy/stress/status record, checks the device error and then exports the potential. Invalid device data leaves host outputs untouched. These added transfers are temporary, counted boundaries, not a residency improvement by themselves. The runner requires Hartree transfer coverage in every SCF iteration, and `--hartree-ablation` adds the previous CPU-Hartree GPU control. Native checks additionally cover repeated changed density, cell invalidation/mutation, aliased input/output and error recovery. All patched Fortran seams compile, 26 local tests and the new native ownership checks pass, and the full diagnostic scientific/trace gates pass. Across its 351 candidate iterations, the temporary adapter adds 74,509,632 uploaded density bytes, the same number of downloaded potential bytes and 30,888 result/status bytes. Full grid residency remains unfinished.

The next XC port must preserve the actual pinned GridXC mesh functional: `cellxc.F90` uses `nn=3` (a seven-point derivative stencil on each cell axis), the reciprocal mesh Jacobian and the adjoint stencil for the potential. It also returns four energy/double-counting reductions and all nine stress components. **`ggaxc.F90` sets `old_scheme=.true.`**: its active NC/SOC path rotates density and gradients using theta/phi and transforms derivatives back with the legacy off-diagonal factors. The newer derivative-of-spin-transformation branch is present but inactive; porting it would change the numerical specification. A port of collinear pointwise PBE alone would leave those required consumers unfinished. The intended oracle is the unmodified full `cellXC` routine, with pointwise PBE checks only as a supplement.

`scripts/check_pbe.py` now links that pristine full-mesh routine and retains inputs, complete potential grids, four reductions and stress tensors in [local CPU evidence](../results/local-pbe-reference-20261005/reference.json) and its checksummed NPZ companion. All nine CPU fixtures pass: constant scalar/collinear/spinor densities, smooth scalar/collinear fields, a non-collinear texture, a 2×3×4 periodic spinor mesh, full polarization and sub-threshold density. The double-counting identity agrees within 3.1e-13 Ry; collinear finite-difference potential variations agree within 2.9e-9 Ry and stress variations within 1.2e-11 Ry/Bohr³. These are checks on the oracle, not a GPU implementation. A new NC variational convention is deliberately not imposed on the pinned legacy routine.

The next component is now implemented in `native/pbe_math.hpp` and `native/pbe.cu`, with GridXC attribution and redistribution terms retained in `native/GRIDXC_NOTICE`. The shared CPU/device algebra passes 900 pointwise comparisons against unmodified `ggaxc`, including zero gradients, low densities, full polarization, negative collinear-channel clipping and NC rotation. Combining that algebra with independent Python gradient/adjoint stencils reproduces all nine full-mesh oracle cases: maximum energy, potential and stress differences are 3.7e-13 Ry, 1.8e-13 Ry and 5.0e-16 Ry/Bohr³. [CPU algebra evidence](../results/local-pbe-math-20261005/check.json) includes source/oracle hashes. This does not validate CUDA execution.

The device plan uses reusable pointwise and fixed-tree reduction scratch, a gather adjoint without floating-point atomics, and device-owned configuration/result records. Duplicate neighbors on meshes smaller than the stencil remain separate contributions. All four energy/double-counting reductions and nine stress components are retained; input/output grids must not alias. Its four-kernel `apply()` contains no explicit copy, allocation, synchronization or file operation. The standalone test adapter passes 13 ordinary/captured mesh fixtures, including a vacuum interface, nearly unpolarized NC, a left-handed cell and a larger grid, plus eight captured invalid-data/configuration scenarios. Each valid check compares the second device-only density update with a fresh upstream result. **All 34 CUDA scenarios pass; the subsequent SIESTA connection below still requires full SCF validation.** Each captured graph has four kernel nodes and no forbidden nodes. Maximum full-grid energy/potential/stress errors are 2.19e-12 Ry, 6.67e-15 Ry and 2.84e-15 Ry/Bohr³. All 41 recorded source hashes verify in the retained component snapshot from completed batch `20261006T001018Z`. See [PBE component validation](PBE_VALIDATION.md) for scope and remaining consumer integration.

## Resident grid integration

`--resident-grid` / `REMAG_RESIDENT_GRID=1` composes the native grid contraction, periodic Hartree and built-in PBE components during declared SCF iterations. The density producer writes directly into a geometry-owned device buffer; the potential consumer borrows another device buffer. Integer indexing preserves original `reord` cluster ordering without a host reorder or a large mapping table. Spin sum/difference, reference/core-density arithmetic, neutral-atom potential, Hartree cross energies and XC reductions follow pinned `dhscf.F`. Geometry, integration volume and other scientific configuration are staged once per epoch. The composed device operation has no explicit transfers, allocations, host synchronization or file operations.

The temporary adapter exports a **224-byte energy/stress/status record per SCF iteration** for existing CPU bookkeeping. Sparse H seeds still upload, updated H still downloads, and sparse D/E, mixing, solver/normalization/occupation scalar results and final force work remain separate unfinished boundaries. Host density/potential allocations currently serve as opaque identity tokens; normal mode does not scan them. Debug mode snapshots their bytes and checks immutable geometry data. A one-use state machine rejects missing, duplicate, unexpected or stale consumers. Device failure leaves host results unchanged and requires a fresh density producer.

The current opt-in scope is serial periodic FP64 grids with 1/2/4 spin channels and one unit-weight built-in PBE functional. MPI, spin spirals, applied fields/dipole corrections, doping, geometry charge/potential additions, QMMM, TranSIESTA, ELSI export, density-only calls and in-SCF grid exports are rejected. Post-SCF forces, stress and requested density exports use the original path. `--grid-flow-ablation` runs the previous host-grid path as a direct control; it cannot be combined with a Hartree-only ablation.

`scripts/check_grid_potential.py` links pristine reordering, Poisson/GPFA and full GridXC routines with literal DHSCF arithmetic. Nine composite CPU fixtures pass, including partial core, NC/SOC, small grids and cluster sizes 1/2/3. This is a composite numerical oracle, not an unchanged full-DHSCF executable. **All 32 CUDA component scenarios now pass** in [run 20261006T010819Z](../results/20261006T010819Z/grid-potential-check.json), with all 51 source hashes verified. Eighteen ordinary/captured cases exercise two changing densities; ten cases reject invalid inputs or plan shapes. Captured graphs contain 19 permitted nodes and zero forbidden nodes. Four production-bridge cases exercise real density/potential contractions, unchanged opaque host buffers, geometry reuse, same-address invalidation and error recovery, rejecting 48 invalid ownership operations. Partial-core-off checks intentionally provide only a one-element NaN core buffer.

**Full grid-stage SCF regression now passes** in [run 20261006T014308Z](RESIDENT_GRID_VALIDATION.md): 18 upstream pairs, nine direct controls, four MAE checks, 36 complete traces and 57 verified source hashes. All 435 candidate iterations retain density/potential arrays, with one complete grid lifetime and a 224-byte result export per iteration. No immutable grid geometry is restaged after its first iteration. The grid bridge records 414,771,840 avoided density-download bytes and the same avoided potential-upload bytes. Sparse H/D/E traffic remains substantial.

The trace also exposes first-use setup: 24 extra H2D bytes and 39 selected CUDA compute-cache file operations per candidate process during its first SCF iteration. Later iterations have zero selected file operations, with mapping/page-fault limits still unresolved. One relaxation spectral export touches the SCF/post-SCF timestamp boundary and remains conservatively counted in both phases. Strict residency/no-I/O is not certified. Fe54 requires 120 candidate iterations versus 61 upstream and 48 control; traced timings cannot qualify performance. The completed larger untraced batch passes 20 upstream pairs, five controls and four FePt16 MAE comparisons through Fe128, with all 62 source hashes verified. Candidate medians are 4.33–18.65× faster than the single-thread CPU reference; iteration-count differences are disclosed in the [performance record](RESIDENT_GRID_VALIDATION.md).

The failed precursor `20261006T013412Z` remains retained. Its adapter rejected ordinary inputs because `QMMM_does_pcpot()` is true even with no MM charges. The corrected guard rejects actual nonzero `mm_charges%n`, matching upstream's immediate return for an empty list; QMMM physics remains excluded.

## Sparse SCF state after the grid boundary

The accepted corpus actually mixes the **Hamiltonian**, using stable Pulay, all spin components, weight 0.05 and history 6 (Fe) or 8 (FePt). `SCF.Mix.First` is true and `SCF.MixAfterConvergence` is false. These settings are recorded in the retained upstream FDF logs; the legacy `DM.NumberPulay` input name must not be mistaken for density-matrix mixing. The next owner therefore needs separate input/output H, old/new D and E, and mixer history with explicit geometry resets. The [sparse-state integration contract](SPARSE_STATE_INTEGRATION.md) records the upstream order and the consumers that must move together.

| CPU consumer in the current route | State needed on device |
| --- | --- |
| `setup_hamiltonian`: save H, initialize kinetic/nonlocal/SOC terms, add grid potential and offsite SOC | Separate H input/output, geometry-owned H seeds and SOC components; preserve the original before/after-grid addition order |
| `compute_max_diff`, `dm_charge` | Old/new D/H/E, overlap-weighted charge and maximum residual reductions |
| `compute_energies` | Spin-aware band/Harris traces, kinetic/nonlocal/SOC reductions, grid energy record, FD entropy and immutable ionic terms |
| `mixer` / `m_mixing` | Stable-Pulay residual and input-difference history, inverse checking, SVD fallback and mixed H |
| `scfconvergence_test` and loop bookkeeping | Device tolerances, residuals, energy history, minimum/maximum iteration policy, errors and deferred diagnostic records |

`scripts/check_mixing.py` provides an oracle linked to the pristine full `m_mixing` implementation. Nine retained CPU sequences cover 180 steps, 1/2/4/8 flattened spin components, history lengths 2/6/8/16, rollover, a same-shape geometry reset, separate initial/regular weights, small residuals and exact rank-deficient history. Independent array arithmetic agrees within 8.99e-15; the rank-deficient sequence takes the original SVD fallback 14 times. [Inputs and results](../results/local-mixing-reference-20261006/reference.json) include source/library hashes and a checksummed NPZ. Porting a generic DIIS formula would miss the current stable-Pulay history and fallback behavior.

`native/pulay_math.hpp` now implements device-capable coefficient algebra: pivoted LU, the original simple-inverse fallback and a symmetric pseudoinverse. The last uses Jacobi diagonalization with the original singular-value cutoff; it is a different numerical backend from upstream DGELSS. [CPU algebra validation](../results/local-mixing-math-20261006/check.json) reproduces all nine upstream sequences within 8.78e-15, passes 36 independently checked pseudoinverses (sizes 2–15 and scales 1e-25–1e25), and rejects eight invalid configurations without changing output coefficients. The largest scale-normalized pseudoinverse difference is 5.91e-11. Adversarial ill-conditioned histories, the successful simple-inverse fallback and full SCF trajectories still require broader qualification.

`native/pulay.cu` adds a reusable device plan for stable all-component Pulay, history 2–16 and fixed positive weights within an epoch. Eight kernels update history, reduce products into persistent Gram/projection buffers, solve coefficients, form a trial update and commit only a finite successful result. Invalid steps leave output unchanged and poison history until a new geometry epoch. Scientific configuration, history, coefficients and status remain device-owned; `apply()` performs no explicit transfers, allocations, host synchronization or I/O. There is no SIESTA adapter yet.

**All 24 CUDA scenarios now pass**, along with three direct reciprocal probes, in [retained component evidence from 20261006T013412Z](../results/resident-components-20261006T013412Z/pulay-check.json). All 57 source hashes and retained array hashes verify. Eighteen ordinary/captured sequences each reproduce the nine original trajectories (180 steps); the maximum mixed-vector difference is 1.083e-14, with matching iteration/history counts and 14 pseudoinverse fallbacks in the rank-one sequence. Two failure/reset scenarios and four invalid-configuration cases also pass. Captured graphs contain 11 kernel nodes and no forbidden nodes. Maximum plan scratch is 139,856 bytes in this fixture set.

Five preceding failed preflights are retained (`20261006T010819Z`, `20261006T011311Z`, `20261006T011755Z`, `20261006T012202Z`, `20261006T012713Z`). The local-array Gram/solve implementation produced a nonfinite reciprocal on the second history step, although direct global/local reciprocal probes passed and CUDA memcheck reported zero invalid accesses. Separating collection and solving through persistent device storage resolves all current sequences; the underlying code-generation/local-storage cause has not been independently established. No tolerance was changed. Full SCF trajectories, adversarial ill-conditioned histories, GR/Broyden variants, alternate spin subspaces and mixer switching/restart policies remain unqualified.

`scripts/check_scf_scalars.py` prepares the next consumers by linking unmodified `compute_energies`, `dm_charge` and `compute_max_diff` against explicit sparse state. Twelve CPU scenarios cover 1/2/4/8 components and capacities 19/128/513, including nonzero unused capacity. This distinguishes full-capacity energy/residual loops from topology-selected charge sums. The NC energy trace doubles its off-diagonal components; SOC uses all eight components, while its charge reduction hermitifies the two off-diagonal pairs. [Reference inputs and scalar results](../results/local-scf-scalars-reference-20261006/reference.json) are retained with hashes. Kinetic/nonlocal/SOC contributions enter as supplied energy seeds here: their construction, device scalar reductions and full SCF control remain unimplemented by this check.

`native/scf_scalars.cu` now implements the isolated device consumer, including H−Hold for the next Pulay step. Five kernels validate configuration, reduce full-capacity traces/residuals and topology-selected charge, finalize the 13 scalar outputs, and commit the residual only on success. Inputs, energy seeds, topology and results are device-owned; `apply()` has no explicit copies, allocations, synchronization or I/O. The supported energy formula assumes zero external-field, DFT+U, dispersion, meta, MM and bulk-bias corrections. The checker stages two complete changing states, captures an eight-kernel graph including device selection/packing, and exports only at the end.

**All 39 CUDA scenarios pass** in [run 20261006T011755Z](../results/20261006T011755Z/scf-scalars-check.json), with all 56 source hashes verified. Twenty-four ordinary/captured cases cover the 12 two-step upstream oracle sequences; 14 cases reject invalid shapes, seeds, matrix values, active overlap, topology or overflow and check unchanged outputs/recovery. The final case proves NaN unused overlap capacity is not consumed. Maximum scalar error is 1.07e-14, residual arrays agree exactly, every captured graph has eight kernel nodes and no forbidden nodes, and maximum plan scratch is 32,984 bytes for the tested dimensions. This is component qualification only. Sparse Hamiltonian seed/energy construction, state ownership, SCF integration and device loop control remain separate work. The same run stops afterward at the still-failing isolated Pulay preflight.

`scripts/check_hamiltonian_seeds.py` retains the next construction contract: unchanged blocks extracted from pristine `setup_hamiltonian.F` initialize kinetic/nonlocal/onsite-SOC terms, compute their energies and add offsite SOC after a supplied grid contribution. [Fifteen CPU fixtures](../results/local-hamiltonian-seeds-20261006/reference.json) cover scalar, collinear, NC, onsite SOC and offsite SOC at three sparse capacities. Independent matrices agree exactly and energies within 8.66e-15 Ry. Extracted-block, wrapper, library and array hashes are retained. This checks the literal arithmetic with a supplied grid contribution, not a full `setup_hamiltonian` call or a CUDA implementation.

`native/hamiltonian_seeds.cu` implements that component with separate preparation and post-grid commits. Configuration, sparse inputs, three energy reductions and stage/error state are device-owned. The second operation rejects missing/duplicate preparation and changes to the selected SOC convention. Both operations leave H unchanged on their own validation failure; neither contains explicit copies, allocation, synchronization or I/O.

**All 52 GPU scenarios pass** in [retained component evidence from 20261006T020310Z](../results/sparse-components-20261006T020310Z/hamiltonian-seeds-check.json), with all 62 source hashes and oracle/array/wrapper hashes verified. Thirty ordinary/captured checks reproduce 15 two-step upstream sequences; matrices agree exactly and maximum energy error is 8.89e-15 Ry. Twelve invalid-input/recovery scenarios, five stage/missing-input scenarios and five unused-SOC-input cases pass. Normal captured execution has 12 kernel nodes and zero forbidden nodes, including input selection, supplied grid accumulation and output packing. Maximum plan scratch is 32,904 bytes for these fixtures. In that validated image, the plan is linked only into its standalone checker. No production SCF transfer is qualified as removed until [sparse-state ownership](SPARSE_STATE_INTEGRATION.md) connects it to the other consumers.

## Sparse-owner adapter awaiting validation

The optional `REMAG_RESIDENT_SCF` path now connects device-owned H/Hold/D/Dold/E/Eold to the eigensolver, normalized-density producer, resident grid, Hamiltonian/SOC plan, scalar reductions and stable Pulay. Its lifecycle and unsupported-setting guards are in [SPARSE_STATE_INTEGRATION.md](SPARSE_STATE_INTEGRATION.md). Twenty affected Fortran units compile, all 26 local tests pass, and ten coupled upstream CPU sequences are retained. All 17 production-owner GPU checks now pass, with maximum scalar error 1.25e-14 and 1,157 rejected invalid calls. The first full integration fails before SCF because of an overly strict inactive-restart guard; a subsequent pre-SCF failure exposed upstream’s optional host Eold allocation, and the diagnostic with both fixes is running. [Evidence and remaining scope](SPARSE_STATE_INTEGRATION.md). The completed grid-performance batch uses the preceding immutable implementation.

The new explicit boundaries are `sparse_state_setup` outside SCF, `sparse_state_scalars` (144 bytes per SCF step), `sparse_state_mixer_status` (four bytes per mixing call) and `sparse_state_export` after SCF. The intended bulk-array gate forbids spectral and H/D/E copies within SCF and checks producer/consumer counts against actual iterations. Existing grid/FD/normalization/solver scalar transfers, first-step setup, host control, topology processing and final forces remain unfinished. Qualification will report observed bytes rather than infer residency from the wiring.

## Upstream convergence semantics before device control

`scripts/check_scf_convergence.py` now calls the full unchanged `scfconvergence_test` against the pristine library. All 34 CPU sequences pass: every nonempty combination of the five conventional criteria (DM, H, EDM, Harris and free energy), exact strict-threshold neighbors, the outer minimum-iteration guard, and history reset. Inputs, decision flags, source/library/wrapper hashes are retained in `results/local-scf-convergence-20261006`.

A consequential upstream detail is now explicit: `m_convergence.add_value` **latches** Harris/free-energy convergence after the first qualifying difference and leaves it true until reset, even if a later difference is larger. Matrix criteria instead compare the current residual with strict `<`. The minimum-iteration check can therefore defer exit while the energy flags stay latched. A device port must preserve these decisions; substituting a fresh energy-difference test each step would change the algorithm. This is CPU oracle evidence only. Device convergence, loop termination, error propagation, logging after loop completion and removal of library scalar boundaries are still pending.

`native/scf_convergence.cu` now implements that conventional decision logic as a standalone device component, not yet linked into production SCF. One kernel consumes the device scalar record and a staged configuration; it preserves energy latches, applies strict matrix comparisons, and emits convergence, mixing, continuation and iteration-exhaustion flags. Device errors remain sticky until an explicit nonzero epoch change. Calling beyond the configured maximum is an error. Diagnostic callers may continue evaluating criteria after convergence; an eventual loop owner must honor the returned continuation flag.

The new checker stages all inputs before execution, uses a three-kernel graph (input selection, decision, output packing), and exports only final records. All **92 CUDA scenarios pass**, covering ordinary/captured versions of the 34 pristine sequences plus twelve error/recovery conditions. Every decision matches the upstream oracle exactly; captured graphs have three kernel nodes and no forbidden nodes. Plan scratch is 40 bytes. [GPU evidence](../results/20261006T030311Z/scf-convergence-check.json), all 71 archived source hashes, and oracle/array/wrapper hashes verify. This component does not resolve cuSOLVER's separate graph/copy limitations or remove current scalar downloads.

The pinned-runtime solver probes remain the authority for library residency. NVIDIA documents device and host workspace parameters for the batched eigensolver, but that interface alone does not establish a transfer-free captured loop. No library upgrade is assumed to fix the measured larger-matrix capture failures. [Pinned-generation cuSOLVER API](https://docs.nvidia.com/cuda/archive/12.6.2/cusolver/index.html#cusolverdnxsyevbatched).

## Alternative eigensolver feasibility, CPU only

The larger-matrix graph failure warrants testing a conventional blocked eigensolver. `scripts/check_block_jacobi_math.py` reduces the generalized problem with Cholesky and triangular solves, then uses disjoint 64×64 Hermitian block eigensolves and accumulated unitary rotations. The motivation is the retained CUDA probe's successful size-64 graph, not an assumption that the same library works at larger sizes. Block-Jacobi eigensolvers are established methods; parallel ordering and convergence require care. [Scott, Heath and Ward, 1986](https://www.sciencedirect.com/science/article/pii/002437958690176X), [Yamamoto, Okša and Vajteršic, 2021](https://www.sciencedirect.com/science/article/pii/S0024379521001312).

The first CPU ordering stalled at dimension 128; its source, inputs and failure are retained in `results/local-block-jacobi-failed-order-20261006`. Keeping each pair in ascending global block order resolves that observed failure. Padding is decoupled above the physical spectrum, and the stopping norm uses the unpadded matrix so padding cannot loosen convergence. Eight small synthetic generalized problems now pass at dimensions 64/128/257, including dense nonidentity overlaps with condition up to 1000 and changing real/complex Hamiltonians. The accepted CPU prototype evidence is `results/local-block-jacobi-unpadded-norm-20261006`.

Four further real/complex problems at dimensions 576/1152 with diagonal overlap of condition 1000 converge in 7–9 sweeps; maximum generalized residual is 3.02e-15, overlap-orthogonality error 6.55e-13 and eigenvalue error 4.60e-14 against SciPy. Inputs and outputs are retained in `results/local-block-jacobi-large-20261006`. These larger dimensions need no padding, so their stopping norm is unchanged. This is a numerical feasibility result only: the small eigensolves still run through SciPy on CPU. GPU batching, capture, device convergence/sorting, ill-conditioned and clustered spectra, real SIESTA matrices, end-to-end scientific agreement and performance all remain untested. The production eigensolver is unchanged.


`native/check_small_eigensolver.cu` and `scripts/check_small_eigensolver.py` prepare the next isolated GPU probe, enabled by `--probe-small-solver`. It tests real/complex 64×64 batches of 1/2/4/18 matrices, three changing device-only inputs, ordinary calls and conditional captured loops. Independent worker processes retain failures. Library-default, zero-initialized and 0xff-filled device workspaces are separate experiments, with initialization visible as graph memset nodes. Eight CPU fixture sets are staged in `results/local-small-eigensolver-fixtures-20261006`; the intended 48 GPU scenarios compile successfully but have not run. This code is not linked into the production solver and is not executed by the active sparse-state diagnostic. No batching, capture or performance result is claimed yet.


The successful Fe2 force pair from `20261006T031913Z` now provides a first measured sparse-state milestone: unchanged 25-iteration convergence, passing scientific/trace gates, and removal of 22,094,400 H2D plus 33,141,600 D2H bytes from the four H/D/E SCF round trips. Full qualification remains pending: subsequent relaxation attempts exposed a legacy cache lifetime and upstream first-step mixing policy during geometry reuse. Both fixes preserve the original algorithm; the active diagnostic tests them. See the [integration record](SPARSE_STATE_INTEGRATION.md) for retained failures, exact scope and remaining scalar/observable boundaries.


The known-copy check in `20261006T033337Z` exposes a limitation in the earlier schema-1 observer: device timestamps can precede host markers enough to assign an explicit final array export to SCF. Correct whole-process totals do not repair those phase counts. Existing raw traces are preserved, but their timestamp-only phase residency claims require requalification. Schema 2 never falls back to timestamp assignment; its parser has 22 passing tests, including deliberately shifted clocks, asynchronous completion, missing links and conflicting runtime/driver tags. GPU calibration and full-SCF correlation coverage are pending in the current sequential diagnostic.


Twelve additional CPU feasibility cases at dimensions 128/129 pass with dense overlap condition 1000, real/complex arithmetic, exact repeated eigenvalues, tightly clustered bands and a cluster across zero. Padding is exercised at 129. They converge in 4–6 sweeps, with maximum generalized residual 1.67e-16 and eigenvalue error 2.03e-14; [retained source/arrays/results](../results/local-block-jacobi-clustered-20261006/check.json). These tests strengthen numerical feasibility only; the small solves remain SciPy CPU operations and no GPU residency or performance result follows.


The [completed sparse-state diagnostic](SPARSE_STATE_VALIDATION.md), `20261006T034347Z`, now passes all 18 upstream pairs, nine controls, four MAE checks and 36 correlation-based CUDA/filesystem traces. Across 349 candidate iterations, the targeted H/D/E array transfers are absent, steady H2D is zero, and selected steady file operations are zero. There are still 347,128 steady D2H bytes, first-use setup/cache traffic, mappings and CPU final observables. All 73 source hashes verify. This qualifies the scoped electronic-array milestone, not strict residency. The next single-GPU probe adds graph replay calibration and the 48 small-eigensolver scenarios; that newer calibration is not part of the accepted nine-case build.


The small-solver workspace experiments also target a documented library issue: NVIDIA reports that CUDA 12.8 fixes incorrect complex `cusolverDnXsyevBatched` results for batch sizes of at least two when the workspace was not zeroed. The active 12.6.3 probe deliberately compares default, zeroed and 0xff-filled workspaces before choosing a library path. [NVIDIA release note](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cusolver-release-12-8). No upgrade or workaround is presumed validated.


The small-solver/graph probe **20261006T040015Z** now passes all 48 GPU scenarios, the ordinary and graph-replay known-copy calibrations, and the Fe2 force pair/control with complete traces. Eight real/complex fixture sets exercise batches 1/2/4/18, three changing inputs, repeated eigenspaces, ordinary/captured execution and all three workspace fills. Maximum residual is 2.59e-16, orthogonality error 2.18e-14 and eigenvalue error 1.00e-14. Requested host workspace is zero throughout; device workspace reaches 16,535,520 bytes. Captured bodies have at most 162 nodes and no forbidden nodes. All 73 source hashes and 56 input/output array hashes verify. [Probe evidence](../results/20261006T040015Z/small-eigensolver-check/check.json), [graph attribution calibration](../results/20261006T040015Z/activity-graph-check.json).

The documented complex workspace defect does not reproduce in these size-64 fixtures, including 0xff-filled workspaces. This establishes only the tested building block, not a large generalized eigensolver or strict SCF residency. The production solver remains unchanged. A blocked generalized solver must next preserve Cholesky reduction, ascending block-pair order, physical stopping normalization, sorting/back-transformation and device error/convergence handling at representative sizes. Nested conditional graph nodes are supported by the pinned [CUDA 12.6.3 graph API](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-driver-api/structCUDA__CONDITIONAL__NODE__PARAMS.html), but their use with this solver still requires execution evidence.

The generalized blocked CUDA prototype subsequently gained an opt-in production adapter and passed its numerical SCF regression, as recorded in [BLOCK_JACOBI_EXPERIMENT.md](BLOCK_JACOBI_EXPERIMENT.md). It remains slower than the standard GPU solver and retains a temporary host status boundary.

Run `20261006T065114Z` now qualifies zero H2D/D2H in all 36 isolated blocked-solver and 75 device-control graph regions under calibrated memcpy/memset collection on driver 580.95.05. All 119 worker/calibration traces reproduce under the exact archived parser, with raw hashes retained. This component evidence excludes complete kernel activity, managed/mapped memory, CPU work and filesystem I/O. Capturable H/S and density interfaces also pass their native numerical checks and nine-case SIESTA regression. Production still records 358,384 steady scalar/library download bytes and three steady relaxation filesystem calls; full device SCF and final-observable residency remain unfinished. See the [current integration contract](DEVICE_SCF_INTEGRATION.md).

The synthetic composed electronic cycle now passes [whole-region transfer and selected-filesystem qualification](../results/20261006T080125Z/REPORT.md). All 32 CUDA and 32 separate syscall traces cover the first accepted iteration through terminal replay and show zero H2D/D2H, selected file operations and mappings. Each suite validates 96 accepted iterations plus eight expected failures against independent CPU/upstream numerical oracles. Bounded per-step diagnostics are exported afterward. This executor is still isolated from production SIESTA; its handoff and material regressions, final GPU observables, complete conditional kernel activity, managed/mapped memory and filesystem page-fault/writeback coverage remain open.

## Conditional energy-density export qualified (v84)

The current candidate removes the terminal energy-density matrix download
when GPU integral forces are enabled and neither TSDE nor NetCDF EDM is
requested. `remag_cycle.F90.in` resolves the consumers before SCF and passes
a reason mask to the native owner. The owner validates legacy-force mode,
keeps E on device for forces, and records any requested E copy in schema-3
terminal telemetry. NetCDF's reason is present only when `CDF.Save` and
`Write.DM.End.Of.Cycle` are enabled in a NetCDF-4 build; per-iteration output
remains excluded by the existing device-SCF guard.

Run [20261007T044323Z](../results/20261007T044323Z/qualification-summary.json)
qualifies this route across 18 cases and three GPU repetitions: 54 material
comparisons and six repeated directional MAE checks pass independent raw-output
reanalysis, including final XV geometry and input hashes. All 226 source and
1,749 component hashes verify. Nine TSDE/NetCDF/both variants cover collinear,
non-collinear and SOC matrices; both-format files agree exactly with each other
within each calculation. Reason masks 0/2/4/6 reconcile with actual copy counts.
The no-TSDE/NetCDF cases omit 50,672,544 bytes of E download across three repeats
(16,890,848 bytes per base corpus). GPU force consumers retain E throughout.

All 1,755 accepted electronic iterations across 78 geometry cycles record zero
explicit H2D/D2H bytes and calls. This is untraced copy accounting, not complete
memory/I/O observation. The single A100-SXM4-80GB allocation (driver 610.57.04)
is stopped with zero tasks and no containers. [Fresh matched performance](../results/20261007T044323Z/PERFORMANCE.md)
is qualified for the combined implementation; no incremental timing benefit
is attributed to E-export omission without a same-executable control.

The preceding v83 batch stopped on a
validator assumption: upstream `CDF.Save` writes requested final density fields
inside `remag.nc`, rather than separate RHO/DRHO files. Retained raw NetCDF
outputs pass the corrected reader and unchanged numerical tolerances; that
diagnosis does not qualify v83 or its untested no-export path. The corrected
reader passes 83 local checks and is part of the qualified v84 build above.

Density history, initialization, CPU ionic terms and complete independent
memory/I/O observation still prevent a full residency claim. Learned SCF
remains deferred.

The next geometry-boundary migration has a [density-history preflight](DENSITY_HISTORY_RESIDENCY.md).
Its linked pristine reference checks cover destructive sparse remapping,
supercell coordinate translation, bounded history eviction and default SVD
coefficients. A multi-entry shared-sparsity diagnostic exposes an upstream
supercell aliasing issue; it is retained as a limitation, not counted as a
passing GPU check. No GPU density-history owner is implemented yet.

## Initial device state and requested populations qualified (v90)

The later [v90 qualification](../results/20261007T081045Z/qualification-summary.json)
extends the combined route to GPU initialization and initial spin/Mulliken
populations. All 54 material comparisons and six MAE checks pass independent
audits; all 78 initial-population outputs match upstream. The initial owner
stages D once and lends the same allocation to SCF. The 1,730 accepted SCF
iterations have zero explicit copies. Source/component checks verify 238/1,755
hashes, and the standalone owner probe covers 16 cases and 264 rejections.
The single allocation is released. [Fresh repeated performance](../results/20261007T081045Z/PERFORMANCE.md)
is qualified for the combined implementation, without an incremental claim.

The GPU density-history component, including sparse capacity growth, is now
qualified separately. Its production handoff and conditional D download are
staged locally as v91 but have not run on GPU. Small geometry/status/coefficient
boundaries remain explicit outside SCF; H/spectral exports, CPU ionic/setup
consumers and complete independent memory/I/O observation remain unfinished.
