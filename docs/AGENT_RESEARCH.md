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

`export_electronic_matrices` reconstructs selected-k H/S, eigenvectors, density
and explicit Löwdin orbital-coupling blocks from accepted, hash-bound native
evidence. `compare_electronic_matrices` requires actual same-radial-basis and
geometry checks before associating endpoint states, including degenerate
subspace overlap diagnostics. Total self-consistent Hamiltonian changes are
kept distinct from isolated SOC. See [matrix conventions and runnable examples](ELECTRONIC_MECHANISMS.md).

`export_orbital_moments` exposes native-density orbital/atom moments, their
Cartesian frame and explicit onsite approximation. It verifies original
DM/HSX/log hashes and checks every orbital, atom and total against the final
native printed table. Large density arrays and contraction kernels stay in a
hashed artifact. [Conventions and examples](ORBITAL_MOMENTS.md) distinguish this
native density from reconstructed selected-k occupations and this local moment
from bulk orbital magnetization. The tool performs saved-data analysis only.

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
   nor reserves compute. When explicitly enabled by an operator configuration,
   `run_prepared_query` executes through the existing guarded `scripts/run_modal.py`
   path, collects, stops the owned app and reviews the result. See
   [controlled execution](RESEARCH_EXECUTION.md) for authorization, limits and timing.
6. `record_calculation_outcome` verifies saved artifact hashes, native acceptance
   and matching experiment source before ingesting a stage energy or E/F
   anisotropy. `evaluate_hypothesis` reports prediction error without certifying
   a mechanism. Failed experiments remain failed.
7. `record_analysis` binds interpretation to an evidence-file hash. Record a new
   hypothesis for a revision; retrieve prior findings using
   `retrieve_research_history`. SQLite transactions preserve prediction/outcome
   ordering across processes.

The tools provide the operations used by the agent's scientific choices. The
opt-in [model policy](MODEL_RESEARCH_POLICY.md) now supplies the service-backed
decision adapter; it does not add a queue or uncertainty calibration.
Electronic mechanisms, end-to-end latency and the equivalent-budget comparison
remain distinct empirical evaluations in the v3 completion checklist.

Several observables can be predicted for one experiment before its first
outcome. `record_response_prediction` freezes an anisotropy or electronic
response definition; `record_response_outcome` derives it from verified saved
endpoints. `evaluate_predictions` retains failures and pending cases while
reporting per-observable error, sign and stated-interval coverage. See
[response definitions and scoring](RESEARCH_EVALUATION.md) for conventions,
examples and the distinction between local ordering and verified blinding.

For a controlled comparison, [paired research studies](PAIRED_RESEARCH_STUDY.md)
register scalar/physics information profiles with the same authoring tools,
common inputs and session/native ceilings. They freeze held-out definitions,
keep separate histories and grade shared evidence after both policies finish.
The profiles enforce tool-level visibility; trusted policy commands remain
outside any OS sandbox and the runner does not certify blinding.
# Relaxation feedback

`prepare_relaxation` accepts an exact structure export, local pseudopotentials
and local Orb weights. It binds the exact source before running the existing
bounded **CPU** Orb preparation. This operation performs CPU work; it does not
submit native or GPU work. Its returned bundle uses the existing discovery
runner and its execution guards. There is no second relaxation implementation.

`inspect_relaxation` accepts `structure`, `bundle` and a collected discovery
`report`. It verifies the prepared inputs, report binding and accepted artifacts,
then reads the accepted `STRUCT_OUT`. It reports site/orbit displacement,
deformation, neighbor changes and fixed-setting authored symmetry residuals
through GPAC. Saved older bundles are labeled as retrospective geometry matches;
new bundles verify the pre-preparation source binding. Failed native relaxation
is rejected, while an accepted relaxation can be inspected even if a later SOC
stage failed. Changed atom counts, species/order or lattice presentation are not
silently matched. Numerical observations remain separate from exact sources.

## Compact saved collections

`inspect_calculation` and `record_calculation_outcome` accept an optional
`collection` path to the original `archive-manifest.json`. This verifies the
bound report, every collected artifact and every locally present artifact named
by the calculation. Accepted stages require their collected native output,
effective FDF log, input, result and review. Missing remote binary artifacts
remain explicitly listed in `verification.omitted_artifacts`; their bytes are
not claimed to have been reverified. Without a collection manifest, all report
artifacts must exist locally. This supports the existing compact collection
format without recreating files or rerunning calculations.

## General workspace

The same local viewer and snapshot command accept a `workspace.json` catalog at
the workspace root. Paths stay inside that root. Each entry names an exact
physical export and optional calculations and spectra:

```json
{
  "format": "REMAG-RESEARCH-WORKSPACE/1",
  "title": "Crystal research",
  "structures": [{
    "id": "candidate", "label": "Candidate A", "structure": "candidate.json",
    "calculations": [{
      "report": "collected/candidate/report.json",
      "collection": "collected/archive-manifest.json",
      "spectra": {"soc-z": "collected/candidate/electronic-z/electronic.json"}
    }]
  }]
}
```

`collection` is optional for a complete local result. The viewer verifies the
exact source and Hamiltonian bindings, supports atom/orbit selections, signed
projected DOS, bounded near-Fermi k/state inspection, spin expectations and
occupations. A per-structure `relaxation: {"bundle": "...", "report": "..."}`
adds authored/relaxed geometry comparison and feedback. The state preview is
limited to the nearest 100 states within ±3 eV, with omitted counts visible;
the agent inspection tool supports other bounded selections. Labels are text,
never executable markup. The original FePt-family workspace remains available
when no catalog is present. Viewing never submits calculations.

A calculation can also register `matrices: {"soc-z": "inspection/z/matrices.json"}`.
The viewer checks the report, stage and HSX bindings and shows the strongest
Löwdin atom/l coupling blocks touching the selected atom or orbit, with a
selected-k picker. It displays sampling coverage and the basis-dependent
coupling convention. Matrix-only stages do not inherit another direction's DOS.

## Bounded policy sessions

`agent` runs a trusted operator-supplied policy bridge over these tools. The
bridge can connect an external Astra session or another research policy. The
package includes an opt-in [OpenAI Responses bridge](MODEL_RESEARCH_POLICY.md)
with explicit model selection and environment-only credentials. A bridge
reads one JSON context from stdin and returns one decision on stdout:

```json
{"tool":"inspect_electronic_structure","args":{"spectrum":"run/electronic-z/electronic.json","selection":{"atom_indices":[1],"limit":10}},"rationale":"Test whether the selected atom dominates the near-Fermi states"}
```

It receives the objective, available tools, prior decisions/results and remaining
limits on every invocation. It ends with `{"finish":"Evidence-based conclusion"}`.
The operator stores its command as a JSON argument array, for example
`["python", "/absolute/path/to/policy_bridge.py"]`, and runs:

```sh
python -m siesta_remag.research agent --workspace /path/to/research \
  --output sessions/question-001 --objective "Distinguish the competing mechanisms" \
  --policy-command /path/to/policy-command.json --max-steps 24 --max-seconds 300
```

Each decision and action is saved before the tool runs. Responses, process
streams, phase timings and the terminal reason remain in the new session
folder. Step/time/context limits, tool failures and policy failures stop the
session; there are no automatic retries or replay of interrupted actions.
Notebook events may have committed before a timeout, so inspect the retained
history before starting a separate session. CPU Orb preparation is unavailable
unless `--allow-cpu-relaxation` is explicit. The default allowlist excludes native/GPU
submission; preparing a query leaves execution to the existing separately
controlled runtime. The trusted policy command itself is not a filesystem or
network sandbox, and must not perform unrecorded calculations or use hidden
holdout information. A completed policy session certifies neither its scientific
conclusion nor a model's research quality. No live model or new DFT run is needed
to test the policy/tool plumbing.

An explicit operator `--execution-config` enables the single additional
`run_prepared_query` action. The session checks that sufficient deadline remains
for the bounded query and cleanup, then preserves the same intent/result flow.
A failed native execution ends the session. The configuration fixes the existing
budget stage and runtime; the policy cannot supply or enlarge them. See
[execution configuration, ownership and latency](RESEARCH_EXECUTION.md).


The physics profile also exposes `export_soc_matrices` for the separately
produced [native nonlocal SOC operator](NATIVE_SOC.md). `prepare_query` accepts
an explicit `soc_matrix: true` option for SOC stages with retained host H0.
These tools neither grant compute nor fabricate operators from saved total H.
The scalar comparison profile rejects this diagnostic.
