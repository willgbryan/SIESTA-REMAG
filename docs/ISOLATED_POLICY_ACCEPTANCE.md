# Isolated policy acceptance

The `siesta_remag.research.isolated_policy` broker fixes the r5 startup
and shared-agent exposure paths. The selected transport is the local
`codex_policy.CodexPolicy` client using existing Codex-managed ChatGPT access. The operator imports
`siesta_remag.research.toolhost.ToolHost`; the policy never imports Python.
The direct-chat research tools remain available as before. This broker is for
the controlled scalar-versus-physics acceptance comparison.

The [authorized live attempt](../tests/evidence/research/finite-tool-acceptance-r6-2026-11-16.json)
completed both ten-action policy phases and sealed eight predictions. Its first
GPU bundle segfaulted during scalar startup; the health gate prevented the second
bundle and scoring/revisions. The app is stopped and this attempt is closed:
$0.75 reserved, $0.75 unreserved, actual invoices unknown. The plan below records
the approved scope, not a new execution authorization. No rebuild or retry ran.

## Boundary

The model receives a fixed brief, registered parent/scalar packet, its own
receipted history and an allowlisted research catalog. A local broker validates
one structured JSON action before calling ToolHost. Each arm uses its own
app-server process, private state directory and ephemeral decision threads;
its context is reconstructed from that arm's receipts. It never resumes or forks
this chat, another policy, or the shared daemon.

The documented `agents.enabled=false` control removes collaboration tools in
this installed build. The earlier `features.multi_agent=false` flags alone did
not do so. Shell, Code Mode host, apps, plugins, memories, environment access,
project instructions and configured MCP servers are disabled. Four native
execution/input tool names remain advertised by this build; execution has no
Code Mode host, and the client holds and terminates the context on any native
tool attempt or server tool/approval request. The local forced agent-list,
agent-spawn and correctly encoded custom shell calls all stopped before a
research action. No policy-selected transport or extension is accepted.

Codex owns authentication: the client checks `account/read` for `chatgpt`, never
extracts credentials and never falls back to an API key. The unmodified built-in
provider is used. Its connection internals cannot be overridden through a custom
provider definition; reported transport errors terminate the owned process.
There are no automatic policy/context or native-calculation restarts. The
separate Responses adapter remains an optional interface, outside this plan.

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
A `REMAG-CODEX-POLICY/1` configuration pins the installed binary SHA-256,
`gpt-6-astra`, `xhigh`, the finite request timeout, disabled MCP server names and
the common research brief. The CLI selects this backend from the config format.
Existing ChatGPT login is required; no separate API account is required. Native
provider prompt inspection and managed-login initialization have passed without
inference. Live model availability is still subject to existing account quota.
Stop on unavailable access or limits; do not purchase credits, switch models or
restart the context.

Codex receipts retain requested model, reasoning effort, reported token usage,
context/decision hashes and process events. The broker enforces ten actions
before scoring and two afterward. No API-style output-token ceiling is claimed
for Codex. Equal action/time bounds do not establish equal actual model compute.

The local regression suite runs both twelve-action histories through the
production request builder, response decoder, ToolHost and GPAC using synthetic
model replies. It checks sibling prediction IDs and environment canaries never
enter the other request, rejects agent/shell/compute calls and path traversal,
and verifies preflight, deadlines, immutable seals and score release. Fresh
process preflights also pass on hash-identical copies of the actual r5 seed
inputs. These are software tests, not live Astra acceptance or forecasts.

The original two failed local probes are retained. A later test used the actual
installed Codex app-server with a loopback fake model: both twelve-decision
histories completed, sibling prediction IDs and environment/file canaries never
crossed contexts, collaboration tools were absent, and all three forced native
tool attempts held their contexts. Normal-provider prompt rendering confirms
the same collaboration exclusion without a fake-provider override. These local
proofs make no claim about live Astra prediction quality.

## Smallest remaining measurement plan

Preserve the closed r5 attempt, its five-cent reservation, the closed convergence
campaign and all original failures. No new execution is authorized by this guide.

1. Reuse the passed saved-build CPU preflight and verify exact bundle/runtime
   hashes locally. Check fresh-source census and freeze the common protocol,
   model configuration and two clean broker registrations before any paid call.
   Bind both seals to a newly reviewed operator handoff; never restart the old
   r5 driver or contexts.
2. Using existing Codex access with Astra extra-high, use ten decisions per arm
   to inspect the permitted data, author the fixed ±1/320 axial strains and seal
   four E/F response predictions each. Both seals must pass before GPU spending.
3. Reuse the existing build for the two unchanged 12³ bundles, one GPU at a time.
   Each gets one 600-second app / 480-second native / 180-second SOC-stage attempt
   and a 75-cent reservation. Verify prior authoritative STOPPED status and
   account availability before each launch using the existing guarded launcher.
   Verify the saved executable/library inside each funded app before solving.
   No rebuild, separate paid preflight, retry or extension is included.
4. Collect and review native/branch/identity health, score the same two outcomes
   against both sealed notebooks, release arm-local scores, execute two revisions
   per arm, verify shutdown and close with pass/fail/pending scope.

The remaining GPU reservations total **$1.50**, separate from actual invoices.
Policy reasoning uses the existing Codex access selected by the owner; there is
no separate API purchase, new subscription or credit top-up in this plan. The
prior API-account funding prerequisite is superseded. Reuse the passed CPU
preflight; all remaining staging, scoring and review are local. No new CPU app
reservation is needed. If existing Codex quota is unavailable, hold before GPU
spending. The old five-cent reservation and all other ledgers remain unchanged.

See the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
for `agents.enabled`, and [app-server authentication](https://learn.chatgpt.com/docs/app-server#authentication)
for Codex-managed ChatGPT access. Local evidence is retained in
[the Codex isolation record](../tests/evidence/research/codex-policy-isolation-2026-11-16.json).

The two outcomes finish the clean tool-workflow exercise only. They do not erase
the completed response-calibration failure or establish scientific advantage,
full empirical spec completion, or physical qualification.
