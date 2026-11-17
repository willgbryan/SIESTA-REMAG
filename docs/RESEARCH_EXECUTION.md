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

A stopped `held_native_review` remains failed. After the owner authorizes a
new diagnostic, the operator may call `review_failed_runtime_for_diagnostic`
with a new one-query execution grant, the exact prepared bundle, saved diagnosis
paths and a review reason. It checks the retained execution/owner/archive,
requires changed executable or native-library bytes, and freshly confirms the
owned app is stopped. Environment/path relabels cannot release the hold.

The append-only review binds the prior failure, evidence hashes, next grant and
exact request hash. The executor rechecks those bindings and the authoritative
lifecycle before its existing account-wide launch guards. This performs no
launch or reservation by itself and never marks the old result measured. An
unknown allocation, a different bundle/grant, altered evidence or an unchanged
runtime still blocks. The review API is intentionally absent from policy and
browser catalogs. Operator-supplied text is not itself owner authorization, and
this diagnostic does not establish a causal repair or scientific acceptance.
