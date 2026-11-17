# Model-driven research decisions

The OpenAI Responses adapter connects an explicitly selected model to the
existing research runner. It receives the objective, an operator research brief,
the permitted tool catalog and prior decisions/observations. It chooses one
action or a conclusion. The runner records and executes that action, then
returns the observation on the next model call. Notebook IDs and evidence are
carried through this loop; the model does not execute tools directly.

This implements the model-service connection for spec §§4.4, 8–9. Engineering
tests exercise the Responses protocol and real local notebook tools with
injected replies. They do not demonstrate live model reasoning or scientific
advantage. A model account/key and an authorized model identifier are required
for an actual service run. There is no default model, account discovery, hidden
credential lookup, model substitution or automatic retry.

## Operator setup

Phase-specific tool bindings can be added with `bind_tool_arguments`. It copies
the catalog and advertises `fixed_arguments`, such as the exact experiment ID,
observable/unit pair, or score-evidence path. Both model transports enforce those
same values before local tool execution. A revision must not be rejected for an
unadvertised choice between two otherwise valid evidence paths. This change
does not rewrite failed phases or count operator-reviewed recovery as unattended
success. Binding tests exercise isolated catalogs without model calls.

Create a policy configuration outside the research workspace when registering
a paired study, and include it in `policy_artifacts`. For example:

```json
{
  "format": "REMAG-MODEL-POLICY/1",
  "model": "YOUR_AUTHORIZED_MODEL_ID",
  "max_calls": 8,
  "max_output_tokens": 2048,
  "max_reserved_output_tokens": 16384,
  "request_timeout_seconds": 45,
  "research_brief": "Describe the available exact sources, artifact paths and question here. Distinguish saved observations from withheld outcomes."
}
```

Replace the model placeholder with an identifier available to your account.
Optional `reasoning_effort` is sent exactly as configured; unsupported settings
cause a retained failure, never a fallback. Supply `OPENAI_API_KEY` in the
policy process environment using your normal secret-management mechanism.
Do not put the key in this file, the research brief or the command arguments.

Create `policy-command.json` containing the actual Python interpreter and
absolute configuration path:

```json
["/absolute/path/to/python", "-m", "siesta_remag.research.model_policy", "--config", "/absolute/path/to/model-policy.json"]
```

Run the existing agent command with a new output directory:

```sh
python -m siesta_remag.research agent \
  --workspace /absolute/path/to/research \
  --output session-001 \
  --policy-command /absolute/path/to/policy-command.json \
  --objective "Compare the recorded competing explanations using the supplied evidence" \
  --max-steps 8 --max-seconds 300
```

The policy module requires the runner's step-directory environment binding; it
is not a standalone chat client. The key enables only the model API request.
Native calculations still require the separate explicit
[execution configuration](RESEARCH_EXECUTION.md). Without it, the model cannot
select `run_prepared_query`. CPU relaxation is likewise separately enabled.
The closed FePt campaign and its ledger are not a grant for new model or GPU work.

The brief should identify usable workspace-relative paths and relevant exact
source hashes/programs. There is no general filesystem browser or shell tool.
Only the listed tool operations are exposed. Existing structural-worker request
and prepared-query conventions still apply; a model cannot make an unsupported
operation available by naming it.

## Request and failure behavior

The adapter uses the fixed HTTPS Responses endpoint with `store:false`, two
strict function definitions (`select_research_action`, `finish_research`),
`tool_choice:required` and `parallel_tool_calls:false`. It sends no hosted
execution tools and no shared conversation/previous-response identifier.
Each request includes the full bounded local history. The data sent to the
service consists of that history, the catalog, the objective and research brief.
`store:false` is not a claim about all provider retention policies.

The outer function schema limits the choice to available tools. Its
`arguments_json` string is decoded locally with duplicate/nonfinite JSON
rejection and checked against the selected tool's required/optional fields.
The actual tool then performs its domain validation. Incomplete responses,
refusals, multiple calls, unknown tools, malformed arguments, transport failures
and exhausted ceilings stop the loop. Redirects are refused, credentials are
never written into artifacts, and transport error bodies are not logged.
The request/response bounds are 1 MiB/8 MiB; the runner enforces the overall
wall-clock deadline in addition to the transport timeout.

The initial registration pins the config, implementation, instructions, catalog,
objective, profile and withheld sources. Each step retains its request, raw
successful HTTP response, decoded decision, reported usage and receipt. An
intent is written before the HTTP request. A killed process can therefore leave
an unresolved attempt; it must not be replayed. A finished session cannot be
resumed and existing step outputs cannot be overwritten. Manual follow-up work
requires a separate reviewed session, not a reset of recorded reservations.

## Accounting and paired evaluation

`max_calls` and `max_reserved_output_tokens` are per-session ceilings. Each
attempt reserves its full `max_output_tokens` before sending. Small responses
and failed calls do not refund that reservation. These are output-token limits,
not a bound on input tokens, total tokens, dollars or invoices. Usage is retained
separately as reported by the service; unavailable usage remains unknown.

`report.json.model_policy` audits all attempt receipts, request/response hashes,
actual runner observations and decisions. It reports requested model, reserved
output tokens, reported input/output/total tokens, unresolved calls and unknown
billing. The receipt also preserves the returned model identifier, which can
name a service-resolved snapshot. Injected test transports are explicitly
labeled `injected-test`; the CLI offers no injected-transport option.

[Paired studies](PAIRED_RESEARCH_STUDY.md) use the same policy command and hashed
config in separate scalar/physics sessions. Each arm's model usage is included
in its audited cost inventory. Matching ceilings does not establish equal
actual usage, GPU consumption, invoices, blinding or scientific benefit.

The protocol follows the official
[function-calling guide](https://developers.openai.com/api/docs/guides/function-calling)
and [Responses create reference](https://developers.openai.com/api/reference/resources/responses/methods/create),
checked on 15 November 2026.
