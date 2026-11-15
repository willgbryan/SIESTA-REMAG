# Scalar versus physics-guided research

The paired-study runner implements the comparison structure in spec §12. Both
arms use the same trusted policy command, GPAC authoring tools, objective,
session bounds and declared common inputs. The scalar arm sees standard
energies, magnetic moments, forces, stress and anisotropy. The physics arm also
has electronic spectra, projections, matrices and orbital-moment tools.

The registration records both complete initial file inventories, common input
hashes, policy artifacts, Python tool implementations, held-out source hashes
and evaluation definitions. It requires disjoint arm workspaces without prior
session, notebook or execution history. Common input files must match. Extra
initial electronic evidence in the physics arm is visible in its inventory;
the operator must account for its acquisition cost when designing a real
equal-compute study.

This is protocol isolation, not an OS sandbox. Policy commands are trusted local
programs. Their external data access, hidden model context, prior knowledge and
service use are not audited by this runner. A registration or green test suite
does not prove blinding or empirical equal compute.

## Register before running either arm

A `REMAG-PAIRED-STUDY/1` definition contains:

```json
{
  "format": "REMAG-PAIRED-STUDY/1",
  "objective": "Predict the registered structural responses and test competing explanations",
  "policy_command": ["/path/to/python", "/path/to/workspace/policy.py"],
  "policy_artifacts": ["policy.py", "policy-settings.json"],
  "common_inputs": ["initial-structures.json", "initial-scalar-observations.json"],
  "limits": {"max_steps": 64, "max_seconds": 1800, "allow_cpu_relaxation": false},
  "arms": {
    "scalar": {"workspace": "scalar-arm", "execution_config": null},
    "physics": {"workspace": "physics-arm", "execution_config": null}
  },
  "evaluation": {
    "heldout_sources": ["EXACT_HELDOUT_SOURCE_SHA256"],
    "observables": {
      "anisotropy.total_energy_ev": {"unit": "eV/cell", "resolution": 0.00005}
    }
  }
}
```

Paths and the source hash above are placeholders. The policy bridge and initial
inputs must exist. Policy artifacts live outside both mutable arm workspaces.
The same bridge receives `information_profile` and `withheld_sources` in its
context, allowing its research choices to depend on available observations.
It still returns the ordinary one-tool-or-finish decision protocol.

For an electronic or anisotropy **response** observable, add a `target` to its
registered `unit` and `resolution`. This is the full frozen definition described
in [response evaluation](RESEARCH_EVALUATION.md). Both arms must use that same
definition and unit; they cannot choose different windows or quantities after
seeing results. The source hashes select the held-out cohort before either arm
runs. Resolutions are declared evaluation choices, not calibrated error bounds.

```sh
python -m siesta_remag.research study-register --workspace /path/to/workspace \
  --definition study-definition.json --output paired-study
python -m siesta_remag.research study-run-arm --workspace /path/to/workspace \
  --study paired-study --arm scalar
```

The second command runs only the scalar arm. A separate explicit invocation
with `--arm physics` runs the other arm. Either order is allowed and recorded.
There is no automatic queue, retry, replacement of an arm or concurrent launch.
Study intent, decisions, responses and terminal receipts remain on disk.
Unresolved process ownership prevents a following invocation. Initial inputs,
policy artifacts and tool implementations are checked again at the boundaries.

Each arm uses its own notebook. All its experiment/prediction events are hashed
when the session finishes. Evaluation rejects changed events or predictions
added afterward. Native calculations and report inspection for known held-out
source hashes are rejected through the tool protocol; exact structural authoring
remains available for forming predictions. Registered initial inventories
cannot include known held-out research calculation reports. Bare spectral
artifacts do not carry a universal source identity, so proper initial-data
provisioning and external access control remain the operator's responsibility.

The scalar tool catalog omits electronic inspection/export operations and
enforces that restriction at dispatch. It cannot request spectral postprocessing
or ingest electronic outcomes while active. It may prerecord electronic
predictions for the separate evaluator. Both profiles have identical structural
operations and the same optional CPU relaxation permission. A standalone
`agent` or `tools` command also accepts `--information-profile scalar` and repeated
`--withheld-source HASH` arguments; physics is the existing default.

## Optional native execution

With null execution configurations, registration and execution enable no native
calculation. They can exercise policy decisions against authorized saved data.

To include native work, each arm must already have its own explicitly authorized
[execution configuration](RESEARCH_EXECUTION.md) and existing ledger/stage.
Registration requires equal query/reservation limits, per-query deadline/rate,
runtime and image, with enough remaining stage capacity for the declared arm
ceiling. It neither creates nor extends a ledger, reserves money, reallocates
funds, nor turns a JSON declaration into owner consent. The closed FePt campaign
and its untouched final-stage funds remain unavailable.

The existing execution controller still enforces one GPU and account-wide
availability for every query. Before a following arm, a prior native-enabled
arm must have terminal evidence, healthy reviewed ownership and an authoritative
STOPPED observation for its last owned app. Unknown ownership, a failed health
review or an observation failure holds the study for explicit diagnosis. No
failure automatically spends another reservation.

## Grade once, after both policies finish

The evaluator can use a shared saved-evidence manifest outside the arm
workspaces. A scalar example is:

```json
{
  "format": "REMAG-STUDY-OUTCOMES/1",
  "cases": {
    "EXACT_HELDOUT_SOURCE_SHA256": {
      "anisotropy.total_energy_ev": {
        "report": "heldout/query/report.json",
        "collection": "heldout/archive-manifest.json",
        "stage": "anisotropy",
        "observable": "total_energy_ev"
      }
    }
  }
}
```

For a response prediction, the observation instead contains `before` and
`after` endpoints with reports, optional collection manifests and spectral
metadata where required. The evaluator uses the existing verified outcome
readers; it accepts no caller-supplied measured number. Source, native health,
units, artifacts and frozen definitions must agree. Both arms receive the same
declared held-out evidence after they have finished.

```sh
python -m siesta_remag.research study-grade --workspace /path/to/workspace \
  --study paired-study --outcomes heldout-outcomes.json
python -m siesta_remag.research study-evaluate --workspace /path/to/workspace \
  --study paired-study --output paired-scores.json
```

All grading values are checked before notebook writes. Durable grading intent
prevents replay after interruption; partial writes stay available for diagnosis.
An existing held-out outcome is not replaced. Outcomes appended outside this
grading step, late predictions and multiple experiments for the same held-out
source are rejected instead of retrospectively selecting the best result.
An evaluation before grading reports pending cases. Missing predictions,
missing sources, failed calculations and inconclusive outcomes remain visible.

## What the report establishes

The report audits session/request/decision/response hashes and registered limits.
Per-observable scorecards include errors, resolved signs and stated-interval
coverage. A paired MAE difference uses only common measured source/observable
pairs and names that subset; the full missing/failure inventories remain beside
it. No aggregate mixes unlike units or target definitions.

Execution accounting distinguishes attempted queries, confirmed submissions,
completed/failed results, unresolved attempts, recorded reservations, native
pipeline time, policy/tool wall time and full session time. Cached inspection
is not counted as a newly completed native experiment. Unknown reservation or
timing data remain unknown. Native pipeline time is not GPU-hours, and
reservations are not invoices. Actual GPU-hours, model-service compute and
initial-data acquisition cost require separate measurements before an empirical
equal-compute claim can be made.

The report does not infer independent candidate validation, stability,
saturation magnetization, validated candidates per GPU-hour, cross-material
transfer or a verified mechanism. Those require their own evidence. Synthetic
tests exercise both profiles, controlled execution handoff and shared grading;
they do not demonstrate that a scientific policy outperforms scalar search.
