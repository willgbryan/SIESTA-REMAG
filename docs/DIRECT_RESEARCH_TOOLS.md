# Research tools for an existing chat

The chat supplies scientific decisions and calls local tools. No second model
process, API key or model-service subscription is needed by this interface.
Model and reasoning-effort selection belong to the chat client; the tool layer
does not silently select or change them. The owner's current preference is
Astra with extra-high reasoning.

`tool-catalog` discovers the available operations. `tool-call` performs one
operation, with a rationale and a new receipt directory. Both reuse the same
`ResearchTools` implementations as the optional standalone policy runner.
This is a command-line tool interface usable through the chat's shell tool;
it is not an installed MCP server or a new native tool namespace.

## Call from a source checkout

Use the existing Python environment with REMAG research dependencies and GPAC.
For adjacent source checkouts, from the REMAG repository root:

```sh
PYTHONPATH=src:../GPAC/src ../.venv/bin/python -m siesta_remag.research \
  tool-catalog --workspace /absolute/research-workspace

PYTHONPATH=src:../GPAC/src ../.venv/bin/python -m siesta_remag.research \
  tool-catalog --workspace /absolute/research-workspace --name author_structure

PYTHONPATH=src:../GPAC/src ../.venv/bin/python -m siesta_remag.research \
  tool-call --workspace /absolute/research-workspace \
  --request request.json --receipt calls/unique-call-name
```

For installed packages, omit `PYTHONPATH` and use that environment's `python`.
`--request -` accepts JSON on stdin. A request contains exactly:

```json
{
  "tool": "inspect_structure",
  "args": {"structure": "structures/parent.json"},
  "rationale": "Inspect the exact parent and its editable parameters before an intervention."
}
```

Paths supplied to built-in tools resolve inside the workspace. The catalog
provides descriptions, required/optional fields, an object input schema and
effects. Value types, units and scientific invariants are validated by each
existing handler; a catalog schema is not a substitute for that validation.
Check both the transport `ok` field and the scientific status inside `result`.

## Tool families

| Purpose | Operations |
| --- | --- |
| Exact structures | Inspect, author, assess, register; GPAC structural operations |
| Query design | Prepare individual queries and exact response families |
| Saved results | Verify calculations; analyze responses and exploratory calibration |
| Electronic diagnostics | Projected DOS, selected states, matrices, SOC, orbital moments, band paths |
| Research memory | Hypotheses, experiments, predictions, outcomes, evaluation, revision and retrieval |

The direct host omits native execution and CPU relaxation and rejects calls to
them. Preparing input does not launch it. The separate execution adapter retains
its existing explicit runtime, account and budget guards for separately
authorized work. `--information-profile scalar` retains the existing scalar
observation restrictions; `--withheld-source HASH` retains held-out source gates.
These are trusted-command protocol controls, not filesystem isolation from a
chat that also has shell access.

## Receipts and continuation

Every call records `request.json` before dispatch, then `response.json` with
the request hash, result or error, and elapsed tool time. Request records include
the tool declaration and directly used adapter/handler source hashes. These
are not a complete Python environment attestation; scientific handlers verify
their own saved sources, inputs, runtimes and artifacts.

Receipt directories must be new. Reusing one fails before dispatch, preventing
accidental duplication across processes. If a request exists without a response,
execution was interrupted and side effects may exist. Inspect the notebook and
artifacts before deciding any next action; the host never retries. Failed
responses explicitly do not claim retry safety. Large arrays remain artifacts;
requests are limited to 1 MiB and replies to 8 MiB.

The notebook remains the research history. A typical chat sequence is:
inspect parent → record hypothesis → author intervention → propose experiment →
record prediction → ingest saved outcome → evaluate → append revision → retrieve.
Use returned IDs and source hashes, and preserve failed hypotheses. A replay of
known data exercises operation; it cannot count as an unseen prediction study.

## Add a tool without changing the host

Implement a reviewed module exporting `register_tools(host)`:

```python
from siesta_remag.research.toolhost import ToolDefinition

def count_samples(context, arguments):
    values = arguments["values"]
    if not isinstance(values, list):
        raise ValueError("Expected a list of saved samples")
    return {"count": len(values)}

def register_tools(host):
    host.register(ToolDefinition(
        name="count_samples", description="Count supplied saved samples",
        required=("values",), optional=(), handler=count_samples,
        effect="inspect", profiles=("physics",)))
```

Add `--extension your_package.tools` to catalog and call commands. Modules are
explicit process configuration, never imported from request JSON. A duplicate
or reserved name is rejected. Extensions receive the workspace-bound
`ResearchTools` context and can reuse `context.path()` and `context.call()`.
They must validate scientific arguments before effects, preserve provenance,
and have tests for failures as well as successful operation.

Extensions are trusted Python with the process's privileges. Their declared
effect and profile are metadata and dispatch filters, not a Python sandbox.
Do not install unreviewed extensions or use one to bypass compute authorization.

## Verified direct-chat example

The 15 November example uses the saved three-point FePt archive. The chat authors
the −1/400 axial-strain source, freezes the parent's 2.469067 meV/cell anisotropy
as a no-change replay prediction, and ingests 2.439251 meV/cell from the matching
saved report. The resulting −0.029816 meV/cell error is retained with an
evidence-linked revision. All five notebook event kinds survive fresh processes.
No separate model-service call or solver is used. See the
[receipt summary](../tests/evidence/research/spec-direct-tools-2026-11-15.json).

The response-calibration tool also accepts explicit `{report, collection}`
endpoints, so the chat can use compact saved archives without recreating remote
native binaries. It reuses the same report verifier as calculation inspection
and preserves omitted-byte disclosures. The saved identity control is recorded
in [archive-input evidence](../tests/evidence/research/spec-calibration-archives-2026-11-15.json);
that control does not validate a faster numerical setting.
