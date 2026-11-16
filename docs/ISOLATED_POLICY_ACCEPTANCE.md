# Isolated policy acceptance

The optional `siesta_remag.research.isolated_policy` broker fixes the r5 startup
and shared-agent exposure paths. The operator imports
`siesta_remag.research.toolhost.ToolHost`; the policy never imports Python.
The direct-chat research tools remain available as before. This broker is for
the controlled scalar-versus-physics acceptance comparison.

## Boundary

The model receives only a fixed brief, registered parent/scalar packet, its own
receipted history and an allowlisted research catalog. The Responses request
offers `select_research_action` and `finish_research` only, with no conversation
or previous-response identifier and `store=false`. A local broker validates
the returned action before calling ToolHost. There is no desktop agent harness,
agent listing, messaging, shell, plugin loading or native execution in this
model interface. Policies cannot choose the model transport or extensions.

Preflight accepts exactly `parent.json` and `packet.json` for the scalar arm.
The physics arm additionally accepts the registered SOC-z/SOC-x `electronic.json`
and `spectrum.npz` pair. It rejects prior notebooks, extra files, symlinks,
non-scalar fields and withheld-source observations. The operator must bind the
electronic exports to the saved baseline using the retained native/export audit;
spectrum-file integrity alone is not a native provenance certificate.

The narrow catalog supports scalar inspection, physics-only electronic
comparison, exact authoring from the registered parent, hypotheses, experiments,
E/F response predictions and post-score analyses. Readable paths and candidate
output paths are checked independently of the tool name. Shared workspace files
and ambient desktop environment values are never assembled into model context.
The operator and Python tool implementations remain trusted code; this is not
an OS sandbox against a malicious operator.

Each arm permits ten pre-outcome decisions and two post-score revisions. A
600-second phase deadline is checked before each request; requests also retain
the adapter timeout. These checks do not promise usable completion within that
time. An error, timeout, forbidden action or early finish holds the context
without retry. At ten actions, the broker requires one hypothesis, two exact
registered source experiments and four E/F predictions, then freezes a notebook
copy and receipt bindings. No further model call is allowed until the operator
releases four typed, arithmetically verified scores. Each arm receives only its
own scores and appends two evidence-bound revisions. Native outcome ingestion
and evaluation happen in separate operator notebook copies.

## Local setup proof

From an environment with REMAG and its pinned GPAC dependency installed:

```sh
python -m siesta_remag.research.isolated_policy preflight \
  --workspace /absolute/path/to/clean/scalar \
  --information-profile scalar \
  --withheld-source <minus-source-sha256> \
  --withheld-source <plus-source-sha256> \
  --config /absolute/path/to/model.json
```

Preflight makes no model, solver or allocation request. `create` records a new
private context only; `step` is the separately authorized live model operation.
Credentials are supplied outside chat through `OPENAI_API_KEY`. A syntactically
valid configuration and credential presence do not prove service/model access.
The optional model adapter records requested model, returned model, tokens,
timeouts and uncertain attempts. Equal limits do not prove equal actual compute.

The local regression suite runs both twelve-action histories through the
production request builder, response decoder, ToolHost and GPAC using synthetic
model replies. It checks sibling prediction IDs and environment canaries never
enter the other request, rejects agent/shell/compute calls and path traversal,
and verifies preflight, deadlines, immutable seals and score release. Fresh
process preflights also pass on hash-identical copies of the actual r5 seed
inputs. These are software tests, not live Astra acceptance or forecasts.

Two local fake-server probes of desktop CLI 0.165.0-alpha.1.2 still advertised
collaboration tools despite disabled feature flags, including with a clean child
environment. That route is rejected for clean acceptance. No live model or
shared-agent tool was invoked by those probes. The new broker does not use it.

## Smallest remaining measurement plan

Preserve the closed r5 attempt, its five-cent reservation, the closed convergence
campaign and all original failures. No new execution is authorized by this guide.

1. Reuse the passed saved-build CPU preflight and verify exact bundle/runtime
   hashes locally. Check fresh-source census and freeze the common protocol,
   model configuration and two clean broker registrations before any paid call.
   Bind both seals to a newly reviewed operator handoff; never restart the old
   r5 driver or contexts.
2. With authorized, funded Astra extra-high access, use ten decisions per arm
   to inspect the permitted data, author the fixed ±1/320 axial strains and seal
   four E/F response predictions each. Both seals must pass before GPU spending.
3. Reuse the existing build for the two unchanged 12³ bundles, one GPU at a time.
   Each gets one 600-second app / 480-second native / 180-second stage attempt
   and a 75-cent reservation. Verify prior authoritative STOPPED status and
   account availability before each launch using the existing guarded launcher.
   Verify the saved executable/library inside each funded app before solving.
   No rebuild, separate paid preflight, retry or extension is included.
4. Collect and review native/branch/identity health, score the same two outcomes
   against both sealed notebooks, release arm-local scores, execute two revisions
   per arm, verify shutdown and close with pass/fail/pending scope.

The remaining GPU reservations total **$1.50**, separate from actual invoices.
They leave no reserved room for new model-service charges. Local access has no
configured API key, and the desktop subscription has not been connected to this
decision-only adapter. At the checked standard Astra price of $50 per million
output tokens, 24 requests with a 2,048-token output ceiling allow **$2.4576 in
output charges alone**, plus input. This is a configured ceiling component, not
an invoice or a minimum cost. An all-in $1.50 completion is therefore not ready
to authorize with that paid configuration. Existing funded/credited model access
would have to cover it, or a separately reviewed budget/configuration is needed.
See [official model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

The two outcomes finish the clean tool-workflow exercise only. They do not erase
the completed response-calibration failure or establish scientific advantage,
full empirical spec completion, or physical qualification.
