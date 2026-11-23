# From an exact edit to a prepared experiment

The research environment can now save a GPAC operation as a candidate, expose
it in the workspace and prepare a REMAG query from that candidate. Exact
structural operations, validation and correspondence remain in GPAC. REMAG
owns the artifact handoff and calculation preparation; no second structural
implementation or compiler service is introduced.

## Agent and Python workflow

Use the existing JSON-lines tool command:

```sh
python -m siesta_remag.research tools --workspace /absolute/path/to/research
```

Read the source and its editable degrees of freedom:

```json
{"op":"inspect_structure","args":{"structure":"baseline/structure.json"}}
```

The result includes the verified `GPAC-PHYSICAL/1` export, exact source program
and hash, parameter-space hash, current exact values, and numerical periodic
coordination. An optional `candidate` path also verifies saved authoring lineage.

Save a volume-preserving tetragonal intervention, using the actual source hash
returned by inspection:

```json
{"op":"author_structure","args":{"structure":"baseline/structure.json","output":"candidates/shape-001","request":{"op":"physical-tetragonal","scale":"101/100","expected_sha256":"ACTUAL_PARENT_SOURCE_SHA256"}}}
```

`structure` supplies the exact parent program; do not also provide `program`
inside the request. Supported parent edits are GPAC's `apply`, `physical-strain`,
`physical-isotropic`, `physical-tetragonal`, `physical-displace` and
`physical-coordination-scale`. Their existing exact fields and constraints apply.
`apply` also requires the current `expected_space_sha256`. Local symmetry
lowering remains an explicit `explicit-p1` request. No float is coerced into an
exact edit. Stale hashes, invalid constraints and inexact numbers are rejected
before creating a candidate directory.

For a new root, omit `structure` and use `inspect` with an exact program,
`construct` with a motif recipe, `assemble` with its document, or `physical-export`
with an exact program and `length_unit: "angstrom"`. Failed assembly assertions
are not saved as accepted candidates. Canonical compilation, approximate
recognition and finite completion are separate explicit GPAC operations; they
are not silently run by this saving tool.

The result returns `structure_path` and `candidate_path`. Pass both to preparation:

```json
{"op":"prepare_query","args":{"structure":"candidates/shape-001/structure.json","candidate":"candidates/shape-001/candidate.json","pseudos":"pseudos","output":"queries/shape-001","query":{"kind":"scalar"},"mesh":[6,6,6]}}
```

This mesh is an example input, not a calibrated exploratory default. Preparation
copies verified authoring evidence into the new bundle, checks that it belongs
to the selected source, and binds every copied file in the experiment inventory.
The bundle remains self-contained if the original candidate moves. Native
execution verifies those prepared bytes without needing a GPAC installation.
Preparation performs no DFT calculation and reserves no compute.

The equivalent CLI preparation supports `--candidate`:

```sh
python -m siesta_remag.research prepare candidates/shape-001/structure.json \
  --candidate candidates/shape-001/candidate.json \
  --pseudos pseudos --output queries/shape-001 --query scalar-query.json
```

To include a candidate in the local viewer:

```json
{"op":"register_structure","args":{"structure":"candidates/shape-001/structure.json","candidate":"candidates/shape-001/candidate.json","identifier":"shape-001","label":"Shape candidate"}}
```

Registration appends to `workspace.json` under the existing local lock. IDs are
unique; existing sources, calculations and entries are never replaced. A new
catalog is created only when explicitly registering the first structure. Both
scalar and physics research profiles have the same authoring tools.

## Editable workspace

Open an explicitly registered workspace with:

```sh
python -m siesta_remag.research serve --workspace /absolute/path/to/research --editable
```

The Structure panel's **New candidate** control offers exact parameter edits,
uniform/tetragonal scale and axial strain. Values are integers or fractions.
**Save candidate** creates a distinct source, retains its parent, appends it to
the catalog and selects it. It starts with no inherited calculation results.
The control is unavailable while viewing relaxed geometry: numerical relaxed
coordinates are not silently promoted to an exact program.

The server remains loopback-only. Writes require the explicitly enabled mode,
a local Host, same-origin JSON and a bounded static edit of a registered source.
The browser endpoint can prepare registered experiments but cannot run calculations, execute shell commands,
invoke a canonical search, or choose arbitrary output paths. Without
`--editable`, the existing viewer is read-only. Exported HTML snapshots always
remain read-only, including when saved from a workspace used for editing.

## Retained evidence

Each new directory contains `request.json`, the original worker `response.json`,
`program.xcl`, `structure.json`, optional `parent.json`, and `candidate.json`
with hashes of the other files. GPAC's operation-specific correspondence stays
in the worker result. Verification checks the exact program/export and parent
bindings as well as artifact hashes. Output directories are exclusive; an
interrupted or completed attempt is not overwritten or automatically retried.

These records establish authored provenance, not physical feasibility, equilibrium,
accuracy or a measured response. Root motif construction is supported, while
interactive motif editing, general response controls and experimental discovery
validation remain separate items in the spec-completion record.

See [workspace experiments](WORKSPACE_RESEARCH.md) for saved-settings preparation,
parameter response views and coordination-motif selection in this same interface.

## Exact-source editor

With editable mode enabled, `/ide` also presents a GPAC source editor beside the
crystal on wide screens and below it on narrow screens. Validation uses GPAC's
existing exact physical export; it neither saves a candidate nor invokes a solver.
The canvas continues to show the selected saved source while a draft is edited.
Saving creates a distinct independent source and selects it in the workspace.
No calculation is inherited and no atom correspondence is invented for arbitrary
text edits. Use the parameter controls when an operation-specific parent mapping
is needed. Drafts persist per selection within the page until it is reloaded.

`/api/source/preview` and `/api/source/save` accept the selected source ID and hash
with bounded exact program text (and a label when saving). Both require editable
mode, same-origin JSON, the current source hash and static GPAC validation.
Neither accepts arbitrary filesystem paths. Unsupported source, stale state or
failed validation leaves the catalog unchanged. The selected saved source and
previous evidence remain immutable.
