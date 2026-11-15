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
not multiply electronic weights. The geometry still displays periodic neighbors;
this is a geometric selection, not a chemical-bond or motif-stability claim.

Engineering tests cover real loopback requests, preparation and saved-response
bindings, and the shipped JavaScript against DOM/canvas stubs. Stub execution is
not visual rendering verification. The Project14 text, colors and compact
accordion layout are retained; browser visual acceptance remains pending.
