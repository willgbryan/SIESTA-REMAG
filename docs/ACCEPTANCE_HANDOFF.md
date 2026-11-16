# Acceptance handoff for the delivered v3 software

The initial software implementation is delivered. Full spec acceptance still
requires measured operation, and scientific advantage is an outcome to test.
The closed FePt campaign, original qualification failures and budget ledger
remain unchanged. No new compute is queued by this document.

## Primary workflow: this chat uses the research tools

The owner selected Astra with extra-high reasoning and directed that the research
capabilities be an extensible toolset usable in the current chat. The
[direct tool host](DIRECT_RESEARCH_TOOLS.md) implements that interface. The chat
chooses actions and invokes `tool-catalog` / `tool-call` through its shell tool;
no separate model-service account or API credential is required. Client model
selection is outside the tool host; this does not claim to change the chat's
model setting. Native execution and CPU relaxation remain disabled in this host.

The chat completed a saved-data replay with ten direct calls: exact parent
inspection, hypothesis, GPAC intervention, experiment, verified baseline,
immutable prediction, saved outcome, evaluation, revision and history retrieval.
The no-change prediction of 2.469067 meV/cell was compared with 2.439251 meV/cell
for the authored −1/400 axial-strain source. All five notebook event kinds and
individual request/result receipts persisted. This tests live chat/tool
operation; the previously exposed data cannot establish unseen predictive
quality, scientific superiority or a calibrated exploratory default. See
[direct-tool evidence](../tests/evidence/research/spec-direct-tools-2026-11-15.json).

New trusted Python modules can register tools without editing the host, while
reusing existing workspace paths, source validation and notebook operations.
The catalog describes their contracts and effects. Receipts prevent a reused
call directory from executing twice and retain failures without automatic retry.
The extension interface is trusted code, not an OS sandbox.

The former separate model-policy session plan and its unset-model preflight are
preserved as dated evidence. No such sessions were run. The standalone model
adapter remains optional for deployments that need a separate service; its key
requirement is not a blocker for this chat's selected architecture. The original
scalar and physics input staging remains available, with the demonstrated
physics arm now containing its direct-call notebook and receipts.

## Acceptance that requires separate compute scope

The full scalar-versus-physics comparison needs a genuinely new held-out cohort,
matching declared budgets, frozen predictions/targets and measured allocation
costs. Existing exposed reports cannot be relabeled unseen. That run must have
its own exact inputs and finite compute authorization; the historical ledger
and untouched final-stage reservation capacity do not authorize it.

Production SOC acceptance needs a full CUDA-linked build and a measured export
from the target runtime. Exploratory product acceptance measures submission
through usable result and observable-specific accuracy against saved evidence.
The 1–5 minute target remains unproven. Neither item justifies another open-ended
convergence campaign or a new pristine reference calculation.

Optional visual acceptance needs a permitted browser route. The prior browser
URL-policy denial must not be bypassed; DOM/canvas doubles establish protocol
behavior only.

A completed replay, green CI or finished implementation cannot guarantee the
scientific advantage required by §17. Report software delivery, live workflow
acceptance and scientific outcomes separately.
