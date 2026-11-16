# Acceptance handoff for the delivered v3 software

The initial software implementation is delivered. Full spec acceptance still
requires measured operation, and scientific advantage is an outcome to test.
The closed FePt campaign, original qualification failures and budget ledger
remain unchanged. No new compute is queued by this document.

## Next runnable scope: a live agent using existing results

Use the existing model adapter and two separate `agent` sessions, with the same
explicitly selected model and limits for both profiles. Provision two clean
workspaces from the existing three-point, 32-cubed DZP FePt strain example:
axial strains −1/400, 0 and +1/400. Their exact source, native report and artifact
hashes are recorded in the candidate-metrics evidence. Provide matching exact
structures and standard properties to both arms; electronic observations go
only to the physics profile through its brief and permitted tool responses. Use
separate notebooks and invoke each session separately. The initial briefs differ
only in that electronic evidence. Raw archives support report verification;
this is trusted-command protocol isolation, not an OS sandbox.

This replay uses no invented held-out hash and no `study-register` call. The
paired-study runner requires a real held-out cohort and remains reserved for the
separate comparison described below. Exposed-data replay cannot fill that role.

Bound this first service acceptance run to **eight model calls per arm**, at
most **2,048 output tokens per call**, **16,384 reserved output tokens per arm**,
a **45-second request timeout** and **300-second session limit**. Native
execution configurations remain null and CPU relaxation disabled. There are
**zero GPU launches or reservations**. Failed calls and sessions are retained;
no retry, fallback model or automatic follow-on run is included.

The task is to formulate competing explanations, author one exact structural
intervention, record predictions before inspecting the relevant saved outcome,
then evaluate and revise the hypothesis. The acceptance check is whether live
model decisions complete that tool/notebook workflow with correct sources,
units and retained outcomes. Missing or failed steps are reported as such.
This is a replay of exposed evidence, so it cannot establish unseen prediction
quality or scientific superiority. Output-token reservations do not cap input
tokens or monetary billing; reported service usage must be retained separately.

**Missing configuration:** the owner's authorized model identifier and service
account/key. Supply a key through the process environment or normal secret
manager, never in chat or a committed file. No credential is searched for or
service request sent by this handoff. The local input preflight copies and
verifies saved reports, exact parent and intervention sources, profile-specific
evidence and initial file inventories. After the model/account is supplied,
configure the model and freeze the actual policy bytes
before the first call. The existing agent records its request, configuration
and complete decision history; staging inputs does not create a session.
[Input-preparation evidence](../tests/evidence/research/spec-live-acceptance-inputs-2026-11-15.json)
records 405 unchanged source files, matching common briefs, native report and
diagnostic validation, and the unset-model rejection before any request.

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
