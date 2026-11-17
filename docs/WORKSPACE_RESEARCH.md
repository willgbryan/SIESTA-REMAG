# Prepare and inspect experiments in the workspace

The local workspace connects an exact selected structure to REMAG input
preparation and saved parameter responses. It does not execute or queue a
calculation. The existing execution APIs retain their separate authorization,
resource and budget controls.

## Register saved settings and evidence

Begin with a registered exact structure as described in
[the authoring handoff](AUTHORING_HANDOFF.md). Register an existing immutable
prepared experiment to make its numerical settings available in the interface:

```sh
python -m siesta_remag.research workspace-register --workspace /path/to/workspace \
  --kind profile --path baseline/query --identifier baseline-settings \
  --label 'Saved numerical settings'
python -m siesta_remag.research serve --workspace /path/to/workspace --editable
```

`--path` must resolve inside the workspace. A profile is a verified prepared
bundle, including its pseudopotentials, explicit mesh, basis definition and full
numerical settings. It is not a calibrated exploratory default. No profile is
created or inferred automatically. The visible summary shows the mesh, basis,
mesh cutoff, smearing and radial energy shift; the API state retains every setting
and pseudo hash. Profiles must supply the species required by the selected source.

In **Structure → Prepare experiment**, select saved settings, scalar or z/x/y
anisotropy, and either the displayed structure or a parameter response. For a
response choose axial strain, uniform strain, volume-preserving tetragonal scale
or an exact metric parameter/rate. Enter 3–9 ordered rational samples containing
zero. These controls call the existing [response preparation](STRUCTURAL_RESPONSE.md)
API; no geometry is reconstructed from the rendering. Other local displacement
and coordination response designs remain available through Python/agent APIs.

The server checks the selected exact-source hash and profile manifest hash before
preparation. New bundles get new directories and registry entries. Candidate
lineage is retained for a selected-structure query; response children record their
exact selected parent and atom mapping. SOC operator exports are not silently
inherited from the settings profile. Source, profile and previous evidence are
not overwritten. An interrupted/failed preparation can leave diagnostic files;
it is not automatically retried or presented as a registered success.

Read-only servers and HTML snapshots have no preparation controls. Editable
writes require the existing loopback Host and same-origin JSON protections.

## Inspect parameter responses

Use `analyze-response` or `analyze_response_series` for saved measurements. Register
the original exact series before its resulting analysis:

```sh
python -m siesta_remag.research workspace-register --workspace /path/to/workspace \
  --kind series --path c-strain --identifier c-strain --label 'c strain'
python -m siesta_remag.research workspace-register --workspace /path/to/workspace \
  --kind response --path analysis/response.json --identifier c-response \
  --label 'Saved c response'
```

**Response → Parameter response** shows analyses whose exact parent matches the
selected structure. Select an observable to inspect its original values, deltas
from zero, sample status, consecutive finite differences and supplied error
bounds. Missing/failed samples and changed magnetic branches remain visible;
there is no interpolation across gaps. Units and observable definitions are
retained. Truncation and convergence errors remain unknown.

Viewing verifies the registered report file hash/self-digest and its correspondence
to the verified exact series. It displays a **saved analysis snapshot**: it does
not recollect native files, rerun the analysis, or refresh its scientific
qualification. Original evidence paths and source/runtime bindings remain in the
saved result. Re-analyze saved evidence explicitly to create a new analysis.

`workspace.json` has an optional `research` list. Each item has distinct `id`,
`label`, `kind` (`profile`, `query`, `series`, `response`), confined `path` and
`sha256`. Existing catalogs remain valid. Registrations append under the same
catalog lock; at most 128 research items are admitted. `register_research` and
`prepare_workspace_research` provide the equivalent Python operations.

## Coordination selection

The [saved band and reciprocal-region workspace](RECIPROCAL_WORKSPACE.md) links
this same atom/orbit/motif selection to band points, spin/occupation details and
regional signed DOS, with original k weights and explicit native provenance.

The site/orbit selector also offers the center's periodic coordination motif at
the displayed 3.4 Å cutoff. Its unique unit-cell sites feed the existing DOS,
state-weight, orbital-moment and coupling views. Multiple images of one site do
not multiply electronic weights. The canvas shows one unit cell, including sites
on its boundary faces, and only connections whose endpoints lie inside that
cell. Periodic coordination counts and distances remain available in the
inspector; neighboring-cell atoms and the expanded coordination-shell overlay
are not drawn. The camera fits the unit cell as it rotates. This is a geometric
selection, not a chemical-bond or motif-stability claim.

Engineering tests cover real loopback requests, preparation and saved-response
bindings, and the shipped JavaScript against DOM/canvas stubs. Stub execution is
not visual rendering verification. The Project14 text and colors are retained.

## Viewport and editing

The workspace fits the browser viewport. At desktop widths the source editor,
crystal and inspector share one row. Below 1150 CSS pixels, **Source**, **Crystal**
and **Inspect** select one panel; switching panels preserves unsaved drafts.
Read-only and relaxed-observation views omit the source editor. Long source lines
scroll inside the editor, and long inspection content scrolls inside its
panel. The structure selector and navigation stay outside those scrolling areas.

The 17 November browser check rendered the actual local IDE at 320×568, 390×844,
844×390, 1024×768, 1150×500, 1280×720 and 1440×900. Recorded page dimensions match
the viewport at every measured check. Source/crystal/inspector switching, draft
retention, saved electronic data, the bottom of the electronic inspector and the
activity deep link were exercised. See the
[bounded browser record](../tests/evidence/research/ide-viewport-2026-11-17.json).
This verifies the sampled interface behavior, not every scientific workflow or
arbitrary viewport size.

## Research records

The IDE presents Structure, Response and Electronic inspection panels. The activity
log and its polling client have been removed from the interface. Existing
`#activity` links open the normal structure workspace.

The read-only `/api/activity` endpoint and `research-activity.json` records remain
available to development tools. The manifest format is unchanged:

```json
{
  "format": "REMAG-RESEARCH-ACTIVITY/1",
  "title": "Astra · acceptance",
  "status": "Awaiting native outcomes",
  "summary": "Predictions sealed; no outcome scores yet.",
  "next": "Collect and review the registered measurement.",
  "updated_at": "2026-11-16T12:00:00Z",
  "stages": [{"label": "Predictions", "status": "Sealed", "detail": "Four predictions."}],
  "policies": [{"label": "Physics", "status": "Finished; outcomes pending", "calls": "policy/calls"}]
}
```

The controller updates this status manifest atomically as work advances. These
are controller-reported milestones, not independent cloud-allocation observations.
Register each desired policy explicitly: the endpoint reads its existing
`calls/*/request.json` and matching `response.json`, including new calls on later
requests. A missing response is pending; a response is accepted only when its
request hash matches. Successful tool dispatch is not scientific acceptance.

Only tool names, supplied public rationales, timestamps, dispatch status and public
notebook arguments are returned. Model contexts, ambient agent histories,
arbitrary tool arguments and raw tool results are not served. Paths (including
symlinks) must remain inside the workspace. Files are bounded to 256 KiB, with at
most 8 policies and 256 calls per policy. The endpoint has the same loopback Host
checks as the workspace and no execution/write action. Invalid or incomplete
records produce an unavailable response; they are not reported as success.

The activity view was rendered in the in-app browser at narrow panel width and
its recorded-hypothesis disclosure was checked. This is not blanket visual
acceptance of every scientific inspection panel.
