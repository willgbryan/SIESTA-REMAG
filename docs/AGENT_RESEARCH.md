# Agent research tools

The experimental `REMAG-RESEARCH-TOOLS/1` interface connects an external research
agent to existing GPAC authoring, saved REMAG observations and durable research
memory. It is an internal repository-pinned interface. It does not require an
API key, introduce a model service or change GPAC's public model/service status.

```sh
python -m siesta_remag.research tools --workspace /path/to/research
```

Send one JSON object per line; receive one `{"ok":true,"result":...}` or
`{"ok":false,"error":...}` response. Start with `{"op":"catalog"}`.
Paths are relative to the selected workspace and cannot escape it through
absolute paths, parent traversal or symlinks. Arrays remain in hashed artifacts.
The same operations are available through `ResearchTools(workspace).call(...)`.

For example, inspect a selected atom's near-Fermi d states:

```json
{"op":"inspect_electronic_structure","args":{"spectrum":"zero/electronic-z/electronic.json","selection":{"atom_indices":[1],"l":[2],"energy_window_ev":[-0.5,0.5],"limit":20}}}
```

State replies include reciprocal fractional k points/weights, eigenvalues,
reconstructed occupations, signed Mulliken weight and spinor Pauli expectations
when present. The query reports omitted states. Failed native EIG comparisons
and changed NPZ artifacts are rejected. `projected_dos` supports the same
selection, a Gaussian width and a bounded energy grid. Selecting k points does
not renormalize their weights. A collinear export represents one channel;
spin-degenerate exports receive the factor of two.

`compare_electronic_states` checks orbital/spin metadata and k sampling, then
compares projected DOS on a common energy grid relative to each spectrum's
Fermi level. It reports the Fermi shift separately. Matching orbital labels
does not establish identical radial functions, band identity through crossings,
SOC attribution, a unique partition or a verified mechanism.

`calibrate_exploratory_response` accepts two lists of saved report paths,
`exploratory` and `reference`, plus `baseline_source_sha256` and a declared
`resolution_ev`. Each tier must contain the same baseline and at least two
matched structural interventions, with one numerical configuration per tier.
It verifies the native stages and artifacts, reports E/F response discrepancies,
response signs and pair rankings, and exposes every changed setting between
tiers. Near-zero responses and ties remain unresolved. These observed errors
do not become uncertainty bounds or a universal default, and the reference
tier does not acquire a physical-accuracy certificate from its name. This is
saved-data analysis; no convergence campaign or calculation is started.

The research loop uses existing implementations, rather than separate copies:

1. Use `structural_operation` with a GPAC worker request to inspect, author,
   modify or describe exact structural degrees of freedom.
2. `record_hypothesis` with explicit predictions and competing explanations.
3. `propose_experiment` with the source hash, action and discriminating rationale.
4. `record_prediction` before observing the outcome. For energy ingestion use
   an observable such as `anisotropy.total_energy_ev` or `scalar.free_energy_ev`,
   in `eV/cell`.
5. Inspect saved evidence, or `prepare_query` using a verified structure export,
   explicit query and numerical settings. Preparation neither executes a solver
   nor reserves compute. Paid execution continues through the existing guarded
   `scripts/run_modal.py` path and the owner's compute authorization.
6. `record_calculation_outcome` verifies saved artifact hashes, native acceptance
   and matching experiment source before ingesting a stage energy or E/F
   anisotropy. `evaluate_hypothesis` reports prediction error without certifying
   a mechanism. Failed experiments remain failed.
7. `record_analysis` binds interpretation to an evidence-file hash. Record a new
   hypothesis for a revision; retrieve prior findings using
   `retrieve_research_history`. SQLite transactions preserve prediction/outcome
   ordering across processes.

The adapter is functional plumbing for the agent's scientific choices. It does
not implement an autonomous selection policy, queue or uncertainty calibration.
Electronic mechanisms, end-to-end latency and the equivalent-budget comparison
remain distinct empirical evaluations in the v3 completion checklist.
