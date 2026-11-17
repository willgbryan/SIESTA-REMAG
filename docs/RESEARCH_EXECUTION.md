# Controlled research execution

An operator can enable `run_prepared_query` for a research agent session. The
action takes a prepared bundle, a new attempt directory and optional spectra.
It uses the existing `scripts/run_modal.py` account-wide GPU guard and shared
launch lock, `scripts/research_modal.py` single-GPU function, cumulative budget
ledger, native experiment runner and compact collection format. It introduces
no second solver, reference calculation, warm pool or automatic retry.

The action is absent from the default session/tool catalog. Enabling it requires
an explicit `--execution-config` supplied by the operator. A model cannot choose
the image, runtime, budget ledger or authorization through tool arguments.
Trusted policy commands remain local programs, not a security sandbox. A JSON
file does not itself establish human consent: operators must supply only an
actually authorized grant. The closed FePt convergence campaign is not such a
grant; this feature does not reopen it or authorize its remaining funds.

## Saved discovery observations

After `execute_relaxation` has collected, stopped and reviewed an attempt,
`import_relaxation_result({"attempt": "runs/candidate-1"})` creates
`runs/candidate-1/research-observation/report.json`. This is an explicit
`REMAG-DISCOVERY-OBSERVATION/1`, not a fixed-geometry experiment. Its `structure`
is the unchanged exact source; `evaluated_geometry` contains the accepted
numeric relaxation. The native `dft-relax` stage is exposed as `scalar` with
its original name retained in `native_stage`. The raw collection is untouched.

Import replays the source, input, runtime, native acceptance, magnetic-branch
and geometry checks, then copies collected stage bytes into the research view.
Subsequent reads verify both the original evidence and the view. Captured
producer hashes must match their original launch policy; unrelated later
research-code changes do not make saved observations unreadable. A fresh
execution still requires exact current producer hashes. This is provenance
checking and current-validator replay, not cryptographic attestation of a
remote machine. Interrupted imports are retained rather than overwritten.

The direct tool host can inspect this report, record E/F predictions and
outcomes, score them, and record evidence-bound revisions. Relaxation metrics
use stage `relaxation`: `Js_T` (T), `volume_a3` (angstrom^3),
`max_force_ev_ang` (eV/angstrom), and `max_stress_gpa` (GPa).
A completed low-moment screen is a measured negative screen; its unrun SOC
and anisotropy outcomes are inconclusive, never zero. Native/branch failures
remain failed outcomes. These checks do not establish phase stability or
synthesis feasibility. Prescribed-strain response experiments remain distinct
from these relaxed observations, including in catalog comparison groups.

New controlled discovery runs request native k-point diagnostics and retain
HSX/KP/EIG, density, pseudopotential and ion XML bytes. The physics-profile
`export_electronic_spectrum` tool requires the bound HSX/KP/EIG files for an
accepted stage; its output feeds the existing state inspection and DOS tools.
Existing matrix/orbital-moment exports can use the copied native stage files
when their required evidence is present. No missing diagnostic is silently
recalculated. Native-mesh correspondence uses the accepted final cell for
relaxation outputs while retaining the input's declared integer grid.

The local regression path uses synthetic Ni/Co native transport and an analytic
native-format spectral fixture. Live non-FePt measurements, model-selected
revisions and exploratory latency remain separate acceptance work.

## Operator configuration

`REMAG-RESEARCH-EXECUTION/1` requires these fields. Values below are illustrative
placeholders, not an executable authorization:

```json
{
  "format": "REMAG-RESEARCH-EXECUTION/1",
  "authorization_id": "new-research-allocation",
  "authorization": "Owner's explicit scope and spending authorization",
  "budget_ledger": "authorized-budget.json",
  "budget_stage": "authorized-stage",
  "runtime": "pinned-remag-runtime.json",
  "runtime_sha256": "SHA256_OF_THAT_MANIFEST",
  "image": "im-EXISTING_IMAGE",
  "previous_app_id": "ap-PREVIOUS_OWNED_APP",
  "app_timeout_seconds": 600,
  "hourly_reservation_usd": 6,
  "max_queries": 4,
  "max_reserved_cents": 400
}
```

An optional `volume` field selects a separate result volume by name. It is
operator configuration, bound to both the immutable authorization and execution
policy; the model cannot change it through tool arguments. Omitting it preserves
the existing `siesta-remag-results` default. Put compilation scratch on ephemeral
disk and publish only required build artifacts to a result volume.

The ledger and stage must already exist with sufficient authorized capacity;
this controller cannot create, extend, refund or reallocate them. Each query
reserves its full app ceiling through the existing launcher. Reservations remain
distinct from invoices. The authorization ID cannot be reused with edited
configuration, and its attempt count survives new agent sessions. A failed or
uncertain attempt also consumes one of its allowed attempts, even if no budget
reservation was recorded. This conservative attempt limit is not an invoice.

The runtime manifest supplies the existing REMAG executable path/hash,
environment and thread count. The controller snapshots the query and runtime,
records the expected Python source identity, and later checks the returned
source, input, executable, environment, threads and collected artifacts. No
input bundle or completed result is overwritten.

```sh
python -m siesta_remag.research agent --workspace /path/to/workspace \
  --output session-001 --objective 'Test the recorded structural hypothesis' \
  --policy-command policy-command.json --max-seconds 1800 \
  --execution-config execution-config.json
```

The same option works with the JSON-lines `tools` server. For a single explicit
invocation, use `execute-query --workspace ... --execution-config ... --bundle
prepared --output attempt-001`. No configuration is enabled automatically.

## General-material relaxation

`execute-relaxation` connects an exact GPAC source and an accepted CPU Orb
discovery bundle to the **same** allocation owner, authorization history,
launcher, ledger and authoritative shutdown checks as `execute-query`. The
configuration must explicitly contain `"operation": "relaxation"`; an existing
query grant cannot be repurposed. Omitting `operation` retains query behavior.
This command is operator-only and is absent from the direct policy tool catalog.

Prepare with `research.relaxation.prepare_relaxation(export, pseudos, weights,
output, initial_moments=[...])`. Optional signed moments are in μB in exact
export site order; the original and Orb-prepared structures must retain that
branch. Without explicit moments the existing element defaults are used and
labeled as such. Preparation is bounded CPU work using existing weights, not
GPU submission. The currently supported discovery condition is unconstrained
positions and a fully variable cell at zero external pressure; input constraints
are rejected. This is not a substrate or pressure-constrained relaxation mode.

```sh
python -m siesta_remag.research preflight-relaxation \
  --runtime workspace/runtime.json --structure workspace/prepared/structure.json \
  --bundle workspace/prepared/bundle

python -m siesta_remag.research execute-relaxation --workspace workspace \
  --execution-config relaxation-grant.json --structure prepared/structure.json \
  --bundle prepared/bundle --output attempts/candidate-001
```

Offline preflight verifies exact-source geometry/site order, prepared hashes,
per-species fully relativistic PBE PSML files, magnetic starts, mesh and actual
build capability metadata before account calls or reservations. The worker
rechecks preparation/runtime bindings before native work. The existing discovery
runner performs native relaxation and, when its moment screen passes, sequential
z/x/y SOC evaluations without changing the density-producing integer mesh.

Unlike older compact discovery collections, this path collects the native
DM/HSX/PSML bytes needed for independent local review. Review reconstructs each
FDF and manifest from the bound preparation, repeats native completion,
force/stress, geometry and magnetic-branch checks, verifies collected hashes,
and reproduces the directional screen. Returned feedback keeps the exact source
separate from relaxed numerical observations. It does not rewrite GPAC geometry.

Both `screened` and `screened_low_moment` are completed observations and yield
controller status `measured`; a negative moment screen is not a solver failure.
Native/branch failures, changed artifacts and uncertain shutdown leave the
shared owner held, preventing follow-on work. No retry, rebuild or extension is
performed. The result retains native status and both energy definitions, with
stability and synthesis qualification false. Source-bound observation import,
notebook scoring and IDE discovery-cycle views have local integration coverage;
general electronic analysis and a completed live feedback loop remain open.

The [local integration record](../tests/evidence/research/general-material-execution-2026-11-17.json)
uses synthetic Ni/Co native results through the actual remote-function code.
It is engineering evidence, not measured material or GPU acceptance.

An agent call may request just the spectra relevant to its question:

```json
{"op":"run_prepared_query","args":{"bundle":"prepared","output":"attempt-001","spectra":[{"stage":"scalar","spin":0},{"stage":"scalar","spin":1},{"stage":"soc-x","spin":0}]}}
```

Omitting `spectra` requests no downstream spectral reconstruction. The list can
select only stages already requested in the prepared query; it cannot add an
SOC direction. Scalar channels 0 and 1 are distinct, and SOC uses its one spinor
channel (argument 0). Legacy campaigns retain their existing `diagnostic_axes`
behavior. A missing or rejected requested spectrum makes the new query unusable
even when its directional energies were collected successfully.

## Offline capability preflight

Device-SCF execution now requires metadata from the **actual linked build**.
The adapter and graph planner share a 4,096-k-point ceiling (previously 128).
The 2,048 spatial-orbital / 4,096 spinor-dimension bounds remain unchanged.
Sampling storage and spectral jobs are dynamic; this ceiling does not establish
memory fit, native acceptance or usable-result latency for every supported shape.

In a trusted CPU builder, after linking and publishing the candidate, generate
its runtime manifest without invoking the solver:

```sh
python -m siesta_remag.research runtime-manifest \
  --executable /results/new-build/siesta \
  --library /results/new-build/libremag_gpu.so \
  --environment runtime-environment.json --threads 1 \
  --output new-runtime.json
```

The command reads the library's pure `remag_device_scf_capabilities` symbol and
binds the returned limits to executable and library SHA-256 hashes. The runtime
environment must select that library directory first in `LD_LIBRARY_PATH`.
Manifest paths preserve public mount aliases such as `/results`; they must not
be replaced with a provider-internal symlink target. Hashing and probing still
check the actual bytes through those aliases, both in the builder and worker.
The output must be new; old manifests and failed-attempt evidence stay intact.
Source-only metadata and component test libraries are planning evidence, not a
substitute for this fully linked candidate manifest.

Before authorizing GPU submission, run this entirely local check on every bundle:

```sh
python -m siesta_remag.research preflight-query --runtime new-runtime.json \
  --bundle prepared-baseline-k6 --bundle prepared-baseline-k12
```

It verifies the frozen input inventory, request/manifest/native mesh agreement,
SOC runtime requirements and build-specific mesh limit. It uses the full mesh
product, never an assumed scalar symmetry reduction: 6³ requires capacity 216;
12³ requires 1,728. Missing capability metadata, stale executable bindings and
unsupported meshes fail closed. Legacy host-controlled manifests remain valid.

Execution preflight also holds the exact executable/library pair involved in
the unresolved acceptance-r6 native SIGSEGV. Offline symbolization locates the
failure inside `cudaGraphInstantiate` during `ScfCyclePlan::Impl::build`.
Capacity checks still pass for that build, so capacity alone cannot release it.
The hold applies before account access, image creation or reservation, and is
rechecked by the worker before loading the native library. Historical manifests
remain readable. This hold prevents repeat spending; it is not a native crash
repair, and changing graph settings is not evidence that the crash is fixed.

The first non-FePt live iteration also holds the exact r8 executable/library
pair after a Ni native SIGSEGV. Saved instrumentation reaches
`instantiate_begin` with 27,501 graph nodes; stack addresses map to CUDA driver
and runtime libraries. This localizes the failing phase but does not establish
the cause or a repair. Astra authoring, prospective predictions and real CPU Orb
preparation passed; native measurement, scoring and revision did not complete.
The owned app is stopped and no next case is queued. See the
[preserved failure record](../tests/evidence/research/general-material-first-native-failure-2026-11-17.json).

Preflight now also checks the generated FDF against the runner's export and
density-history policies before account access, intent or reservation. CG and
Broyden both select native `idyn=0`. Their accounting is now supported after
100 option combinations matched compiled fragments from the pinned native
reader/history routine and REMAG export adapter. Other dynamics and unknown
analysis consumers remain rejected. This proves policy arithmetic, not native
relaxation health, and does not release the failed runtime or alter frozen
inputs. The original attempt retains both failures: native return code -11 and
the subsequent unsupported-MD audit exception that obscured it in the summary.

For new attempts, a postprocessing exception after the native process finishes
produces an explicitly invalid `result.json` containing the actual exit status
and elapsed native time, plus `postprocessing-error.json`. The discovery driver
retains that failed stage and stops before SOC. Raw artifacts stay intact;
input/setup errors still raise, and existing attempt directories are never
rewritten. Exit code zero with a failed review is also unusable.

The same check is mandatory in the controlled executor before account access,
durable intent, image preparation or reservation, and in the low-level Modal
entry before any Image/App/Volume API. Successful execution retains
`preflight.json`. The remote worker also checks actual executable/library bytes
and compiled metadata before native work. These checks use operator-trusted
build metadata; they do not independently attest remote files before allocation.
The commands themselves make no cloud calls or budget reservations.

## Lifecycle and failure behavior

Before each launch, the controller checks the previous owned app's authoritative
lifecycle. Subsequent queries use the last controller-owned app rather than the
initial configuration ID. The existing launcher then rechecks account-wide GPU
availability. One synchronous query owns submission through collection, shutdown
and review. Local durable intent exists before launch.

The receipt must match the ledger's app/run ownership before the controller may
stop it. It stops only that app and confirms its authoritative `stopped` state.
The [Modal app CLI](https://modal.com/docs/reference/cli/app) supplies lifecycle
inspection and termination. Failed observations never imply release; bounded
repeated observations refer to the same app. Local process loss never licenses
another allocation. Missing receipts, uncertain shutdown, failed native health
or incomplete review leave a held owner record and prevent follow-on work.

The controller returns paths to the verified query report and original compact
archive manifest. These go directly to the existing inspection and notebook
outcome tools. `measured` means the requested native stages/diagnostics passed
their implementation checks, not that numerical convergence, predictive error
or a scientific mechanism has been established. The agent stops its session on
a failed execution result. Evidence can be inspected or recovered through the
existing collection tools; recovery must never resubmit the native calculation.

A session must have enough time remaining for the configured app ceiling plus
480 seconds of bounded client/cleanup allowance before it dispatches execution.
The default five-minute inspection session therefore does not implicitly start
a cloud job. A local timeout terminates the client process group and still
attempts owned-app cleanup. Parent termination gives an executing tool worker a
120-second cleanup grace. A hard process kill can prevent cleanup; its unresolved
owner record and remote finite deadline remain, and the next query stays held.

## Timing and validation

Timing includes request-to-submission preparation, submission-to-collection,
submission-through-review and full request-through-review. `usable_result_seconds`
is populated only for a successful reviewed result. Remote timing separates the
native experiment pipeline (including its preparation, reviews and checkpoints)
from requested spectral postprocessing. Function entry/finish timestamps are
recorded. Queue, allocation startup, packaging and download-only durations remain
unknown where the provider evidence does not separate them. Cached inspection
does not become a new-calculation latency measurement.

[Native crash diagnostics](NATIVE_CRASH_DIAGNOSTICS.md) preserve flushed graph
setup boundaries and bounded Linux process mappings through the same collection
format. Exit status is saved before scientific output parsing. This improves
failure diagnosis; it does not clear the unresolved r6 execution hold or authorize
a new build or calculation.

Tests execute the actual experiment/review/collection/ledger paths with a
synthetic native backend and cloud transport. They cover prediction → controlled
execution → outcome → evaluation, scalar/SOC diagnostic selection, legacy
diagnostics, failed stages, interruption, missing ownership, transient shutdown
observations, immutable authorization limits and no retries/refunds. They do not
allocate a GPU, call a model service, demonstrate scientific policy quality or
establish the 1–5 minute exploratory-query target. Live authorized integration
and the equivalent-budget research comparison remain empirical acceptance work.

### Review a stopped failure for one changed-runtime diagnostic

A stopped `held_native_review` remains failed. A relaxation `held_review` can
also be reviewed when its verified archive contains a nonzero native exit or
timeout matching the retained executable and environment; a Python review
error alone is insufficient. After the owner authorizes a
new diagnostic, the operator may call `review_failed_runtime_for_diagnostic`
with a new one-query execution grant, the exact prepared bundle, saved diagnosis
paths and a review reason. Relaxation additionally requires the exact authored
structure and binds its provenance separately from the prepared bundle.
It checks the retained execution/owner/archive,
requires changed executable or native-library bytes, and freshly confirms the
owned app is stopped. Environment/path relabels cannot release the hold.

The append-only review binds the prior failure, evidence hashes, next grant and
exact request hash, operation and authored-source binding. The executor rechecks those bindings and the authoritative
lifecycle before its existing account-wide launch guards. This performs no
launch or reservation by itself and never marks the old result measured. An
unknown allocation, a different bundle/grant, altered evidence or an unchanged
runtime still blocks. The review API is intentionally absent from policy and
browser catalogs. Operator-supplied text is not itself owner authorization, and
this diagnostic does not establish a causal repair or scientific acceptance.

### Recover a saved review error

`recover_saved_review(attempt, reason=...)` is an operator-only, local operation
for an already stopped relaxation `held_review`. It replays native, input,
source, runtime and branch checks, then writes an append-only
`review-recovery.json`. The original execution, collection, owner and ledger
are unchanged. Import and later explicitly granted execution independently
replay the recovery and require a healthy measured result. Recovery never
launches compute. Usable-result latency includes the elapsed time through
recovery, including diagnosis; the earlier review-error return is not usable
latency.

Input reproduction verifies original artifact hashes exactly. The sole
serialization allowance is eight binary64 epsilons in fractional coordinates
recomputed by ASE's Cartesian-to-fractional solve across BLAS implementations.
Both manifests must reproduce their own hashes and rendered coordinates.
Species, cell, options, all other input bytes and metadata remain exact, and
native convergence/force/stress/branch thresholds are unchanged. The reviewer
records the measured reconstruction difference instead of claiming byte-equal
regeneration when it is not.

## Isolated graph-setup diagnostic

`native/probe_scf_setup.cpp` constructs a synthetic `ScfCyclePlan` against an
existing REMAG library and stops after graph instantiation/upload. It never
calls `launch`, loads material inputs or runs a pristine solver. Constructor
warmup does execute private GPU component work. The diagnostic preserves raw
process status, CUDA setup checkpoints, runtime/driver versions and mapped
libraries; successful setup is neither an SCF result nor proof of native repair.

The finite operator harness `scripts/scf_setup_probe_modal.py`, launched through
`scripts/run_modal.py --module-file`, uses one GPU, zero retries and a 420-second
allocation limit. Its fixed five shapes vary 30/76 orbitals, 132/948 k points
and a 4/40 grid side. Each shape runs once in a fresh process with a 60-second
limit. A timeout ends the batch. The operator must retain the material hold,
verify its prior allocation stopped under the shared ownership lock, bind the
existing library/headers, and supply the reviewed 53-cent reservation. The
harness cannot release a material-runtime hold. Reservations are not invoices.

The [first setup probe](../tests/evidence/research/general-material-setup-probe-2026-11-17.json)
constructed all five graphs with the exact failed r8 library on an A100 using
driver 610.57.04 (CUDA driver API 13030). The Ni-sized control had 27,501 nodes,
matching the failed run's count. The actual Ni failure used driver 620.70.06
(API 13050), and material data and prior initialization also differ. These
controls show those synthetic shapes can construct on 610; they do not identify
the cause on 620 or clear the native hold. No native material calculation was
repeated and the diagnostic app was authoritatively stopped.

### Original-input initialization with a mandatory cycle barrier

`scripts/native_setup_probe_modal.py` compiles a diagnostic-only preload library
from `probe_cycle_barrier.cpp` and `probe_graph_interpose.cpp`. It preserves the
retained executable/library bytes and original input/pseudopotential. The CUDA
interposer dumps each graph before forwarding the identical graph and flags
exactly once. The cycle interposer writes a receipt and exits with code 90 before
the first `remag_cycle_execute` body. It never returns a fake successful result.
These sources must never be linked into a production library.

The operator harness checks a real Linux preload control and the executable's
imported cycle symbol before native initialization; immediate loader bindings
record which library resolves the actual entry point. One native process has a
60-second limit within the 180-second, one-GPU allocation. There are no retries.
The separate 23-cent reservation remains charged; invoices remain unknown.
Shared owner/account guards apply, and the held material owner stays unchanged.
DOT files and loader records are collected as hash-verified `.log` artifacts.

The [first original-input diagnostic](../tests/evidence/research/general-material-native-initialization-2026-11-17.json)
reached the barrier at geometry 0 on driver 610.57.04. Both instantiations returned
success; the actual SCF graph contains 27,501 nodes and 836 distinct conditional
handles. No SCF cycle ran. This narrows the earlier driver-620 failure but does
not prove its cause: instrumentation and allocation differ. The app is stopped,
the incident hold remains active, and no full native or scientific acceptance
is inferred from initialization alone.

### Explicit native runtime admission

The candidate library now exports a pure `remag_device_scf_runtime_policy`
metadata symbol, bound into the runtime manifest alongside the executable and
library hashes. Its initial profile is CUDA runtime 12060, driver API 13030 and
NVIDIA driver 610.57.04, based on the retained setup and original-input checks.
This is a restricted deployment candidate, not proof of a causal driver fix or
successful full Ni calculation. Driver 620 remains unvalidated.

Before invoking the native solver, the research worker checks the actual CUDA
versions and the single visible GPU's NVIDIA driver version. Unknown, missing or
different observations stop the attempt without a solver call or fallback.
Within the native library, the SCF graph helper independently rejects failed
version queries or unsupported CUDA runtime/driver API pairs before calling
`cudaGraphInstantiate`. The pure metadata query itself is safe in CPU builders.

Removing or widening the manifest policy fails against the linked library's
metadata. Collection review recomputes admission from the retained observation;
a builder's metadata-only proof cannot replace the worker's deployment proof.
Historical runtime manifests and failed receipts remain readable and unchanged,
including their incident holds. Full native health and scientific review are
separate requirements, even on an admitted profile.
