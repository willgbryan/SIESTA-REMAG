# Candidate and compute scorecard

The paired evaluator can report the complete persisted authoring cohort, saved
candidate validation and imported allocation usage. This implements the
reporting parts of spec §12; it does not launch a solver, query a cloud account,
authorize spending, or establish that physics-guided research outperforms a
scalar policy.

## Register candidate criteria before the sessions

Add `discovery` to a paired definition's `evaluation` object. Existing definitions
without it retain prediction-only evaluation. For example:

```json
{
  "assessment_policy": {
    "format": "GPAC-PROPOSAL-POLICY/1",
    "cutoff_angstrom": "3",
    "minimum_distance_angstrom": "1",
    "max_sites": 64,
    "max_orbits": 64
  },
  "excluded_findings": ["short_distance"],
  "controls": [{
    "varied": "kmesh",
    "tolerance_ev": 0.00005,
    "successive_changes": 2,
    "transverse_tolerance_ev": 0.00001
  }]
}
```

This example registers one mesh control, not full physical qualification. The
existing convergence assessor supports mesh, grid, SCF, basis, radial extent and
smearing controls. A study must declare the controls appropriate to its claims;
this interface never changes the closed campaign's thresholds or failed results.
Assessment values cross GPAC's existing exact boundary. The cohort's geometry
policy cannot request reference-cell strain without a common reference cell.

Every successful `author_structure` action in the frozen session is included,
including explicit imports and repeated sources. The evaluator verifies the
candidate's original artifact inventory against the recorded tool reply. It
groups exact source hashes and retains all authoring step indices. Unpersisted
low-level worker results are not silently treated as candidates. An exact source
count is not a count of crystallographically unique or previously unknown
materials; that field remains unassessed.

GPAC reruns its small advisory geometry assessment for each candidate with the
registered policy. `excluded_findings` defines the study's declared geometric
screen, not a universal feasibility verdict. All rows remain in the report,
including those with findings, failed native results or missing validations.

## Supply saved validation evidence

After both arms finish, supply a `REMAG-STUDY-CANDIDATE-EVIDENCE/1` object with
`arms.scalar` and `arms.physics` maps keyed by exact authored source hash. A case
contains `report`, optional `collection`, and `controls`. Each control maps its
registered variable to an ordered list of report endpoints:

```json
{
  "report": "validation/fine/report.json",
  "controls": {
    "kmesh": [
      {"report": "validation/coarse/report.json"},
      {"report": "validation/middle/report.json"},
      {"report": "validation/fine/report.json"}
    ]
  }
}
```

Paths are relative to the study workspace. An endpoint may additionally name
the original compact `archive-manifest.json`. Source/geometry, available native
artifact hashes, acceptance and magnetic branches are checked. Each declared
ladder must end at the scored report. The existing assessor reproduces both E/F
changes, unchanged configuration and transverse checks; cross-case moment
branches are also checked. Missing or failed controls cannot validate a row.

The scored report must record a native start after the arm finished to count as
post-session validation. A legacy report without a start timestamp retains its
properties and finite control checks, with unknown chronology and no new
validation credit. This is an auditable recorded chronology, not proof
against copied data or hidden outside knowledge. Provider authentication,
blinding and independent scientific judgment are not inferred from timestamps
or hashes.

Rows include signed E/F anisotropy per cell, per atom and per volume, together
with the native total spin moment and its volume-normalized polarization.
Energy density avoids rewarding a larger simulation cell simply for containing
more atoms. Rankings are grouped by recorded numerical settings, pseudo hashes,
executable and runtime; unlike configurations are not silently ranked together.
Neither the directional energy difference nor the spin moment is
automatically fitted K1, a saturation value or a magnetic ground-state result.
Thermodynamic stability, structural robustness and crystallographic uniqueness
remain explicit unassessed fields unless supported by separate scientific work.
The numerical protocol is finite and does not establish physical accuracy.

## Import allocation measurements

`--usage` accepts `REMAG-STUDY-GPU-USAGE/1`, with a `records` list. Each record
names `arm` (`scalar`, `physics` or `shared`), `phase` (`initial_data`, `search`,
`validation` or `grading`), `source`, its `source_sha256`, and zero-based `row`.
The source is an explicitly normalized JSON measurement export:

```json
{
  "format": "REMAG-ALLOCATION-MEASUREMENTS/1",
  "provider": "Provider/export identifier",
  "allocations": [{
    "allocation_id": "provider-allocation-id",
    "app_id": "provider-app-id",
    "gpu_type": "actual-device-type",
    "gpu_count": 1,
    "started_at": "2026-11-15T00:00:00Z",
    "stopped_at": "2026-11-15T01:00:00Z",
    "report_sha256s": ["ACTUAL_SAVED_REPORT_SHA256"]
  }]
}
```

Use measured terminal allocation intervals, not submission timestamps, timeout
ceilings, native wall times, hourly-rate estimates or budget reservations.
Preserve the original provider export when normalizing it. The importer checks
the supplied bytes, identities, timezone-aware positive intervals, integer
device counts, duplicate rows and overlapping segments for the same app/device.
Covered intervals must contain the bound report's recorded start and measured
duration. Legacy reports without timing keep complete accounting unknown.
It does not authenticate the provider or fetch missing measurements.

GPU-hours are duration × device count, retained separately by GPU type. Shared
acquisition/validation/grading time is charged half to each arm; policy search
time cannot be shared. The comparison checks coverage of known session apps,
native reports, initial calculation reports, candidate validation reports and
graded held-out outcomes. Missing coverage or unresolved submissions leaves the
complete usage total unknown while preserving the imported subtotal. Outside
activity not represented in the registered evidence remains unaudited.

The candidate rate uses all exact sources that passed the declared validation
controls, divided by the positive covered allocation total. It is not a rate
of unique stable compounds. The prediction rate is scalar MAE minus physics
MAE on the shared measured prediction cohort, divided by physics-arm GPU-hours;
negative improvements remain negative. Missing predictions and failed outcomes
remain in the full scorecard. Different hardware types remain visible, and
neither rate alone establishes equal compute or scientific advantage. Actual
invoices stay separate and unknown; imported usage never refunds reservations.

```sh
python -m siesta_remag.research study-evaluate \
  --workspace /path/to/study-workspace --study paired-study \
  --discovery-evidence candidate-evidence.json --usage usage.json \
  --output scores-with-discovery.json
```

Omit either optional file to preserve unknown evidence. The output path must be
new and outside the owned study/arm directories. Evaluation does not alter the
registration, sessions, original reports, budget ledger or frozen GPAC evidence.
