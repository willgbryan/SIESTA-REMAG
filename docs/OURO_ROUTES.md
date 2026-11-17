# Ouro temporary-file adapter

Install `.[ouro]` for the pinned `ouro-py==2.24.0` transport and JSON-schema
preflight. The adapter implements local file preparation and operator APIs for
one submission, one status observation and collection. It has **not executed a
live route**. A selected route, configured access, provider allocation admission
and route-specific scientific result interpretation are still required.

The [Ouro guide](https://ouro.foundation/guides/route-input-output-assets#file-inputs-that-dont-need-to-be-assets)
describes temporary uploads: bytes still leave the machine, but no persistent
input asset is created. Scratch uploads expire after seven days. Outputs can
still be assets. Input-priced (`cost_accounting: variable`) routes and request
fields expecting asset-ID strings do not support this mechanism.

## Local preparation

First retrieve a complete route through the SDK's `routes.retrieve`, then save
its `model_dump(mode='json')` to a local route snapshot. This is a read, not a
route execution. The agent tool `prepare_ouro_request` takes workspace-relative
paths:

```json
{"op":"prepare_ouro_request","args":{
  "route":"ouro/route.json",
  "inputs":{"structure":"candidates/Ni.cif"},
  "body":{"fmax":0.05,"include_report":false},
  "output":"ouro/prepared-Ni"
}}
```

The field names/settings above are illustrative, not a verified available route.
Preparation copies the original bytes into a new directory and records hashes,
schema, parameters, private output configuration and a USD metadata ceiling. It
does not upload, reserve funds, infer a material's stability or submit compute.
Every input must be a declared file input. Local preparation accepts at most 16
files totaling 64 MiB. Unknown/inherited pricing, incompatible extensions,
asset-ID fields, external schema references, missing parameters and enabled
post/comment outputs fail before any write/upload. Schemas requiring metadata
absent from temporary inputs also fail. This is conservative validation of the
documented metadata shape; it cannot certify an upstream service implementation.

`price_ceiling_cents` rounds up USD per-call costs, or per-second rates times an
explicit `max_billable_seconds`. A billing ceiling is not a provider execution
deadline, a reservation or an invoice. Missing USD pricing is not treated as
zero. The initial adapter supports resolved free and pay-per-use pricing only.
The SDK has no atomic quote/hash condition on execution: a fresh preflight does
not install a server-side spending cap or prevent a concurrent price change.
An operator requiring a hard cap must establish that guarantee with the provider
before supplying admission.

The tool is unavailable to scalar-only and withheld-source policies: arbitrary
external files have no verified binding to their allowed scalar packet. Input
hashes identify bytes; they do not assert GPAC source/relaxed-geometry identity.

## Operator execution boundary

`research.ouro.submit_request(prepared, attempt, client,
expected_sha256=..., maximum_cents=..., admit=...)` requires:

1. An operator-reviewed request digest and sufficient per-action ceiling.
2. The pinned SDK with an access-token client (`can_refresh == False`). The
   normal API-key SDK client can automatically repeat a POST after an auth
   failure, so it is rejected. Token acquisition/renewal is outside this call;
   keep credentials outside source, requests and evidence.
3. A provider-specific `admit(digest, cents)` callback that enforces reviewed
   route/resource/deadline authorization, the shared account lock, append-only
   budget ledger and durable allocation ownership. It returns a JSON receipt.
   It must retain ownership after this function returns or fails. No default
   admission controller is supplied: Ouro action state alone cannot establish
   one-GPU use or authoritative release of a provider's allocation. The existing
   Modal guard cannot see arbitrary external provider jobs.

The adapter refetches the route and requires its contract to match before calling
admission or uploading. It retains a second local copy of input bytes in the
attempt, uploads via `files.upload_input`, saves each returned upload ID, then
submits once with `wait=False`. It never creates persistent input assets and
requests private outputs. Policy tools have no Ouro submission entry point.

An attempt directory is an exclusive submission marker. An exception after the
submission intent leaves `submission_uncertain`; do not delete it or resubmit.
A returned action is saved before interpretation, so a malformed response can be
reconciled from evidence. HTTP timeouts and missing local handles do not prove
remote termination. Admission/upload failures retain their partial evidence.
There are no automatic retries, queues or provider shutdown claims.

`observe_action(attempt, client)` performs one read of the existing action.
`collect_action(attempt, client, fresh_output_directory)` retains a terminal
action's raw response/usage metadata and downloads named file outputs to fixed
local names with hashes. Failed/cancelled/timed-out actions remain distinct.
Missing required outputs on a successful action or unexpected outputs leave an
incomplete manifest. Non-file output metadata stays in the raw action; the
adapter does not claim to have downloaded dataset contents. Partial collections
are preserved for diagnosis. Original inputs, action IDs and collected bytes
survive expiry of scratch uploads.

Service success is not native acceptance, stability, synthesizability or GPU
release. Collections keep `scientific_acceptance: not_evaluated` and
`allocation_release_verified: false`. Route-specific validators and authoritative
provider release evidence must precede promotion to the REMAG measured notebook
or the next compute launch. Old convergence evidence and ledgers are unchanged.

## Local evidence

`tests/test_research_ouro.py` uses the actual pinned SDK with intercepted HTTP,
including temporary upload reservation/PUT, async submission, one-read polling
and output download. Regression controls cover changed price/schema/input bytes,
denied admission, unsupported inputs, 401/timeout without POST retry, partial
uploads, missing outputs, action identity and information-profile restrictions.
The fixture responses are protocol evidence, not live scientific measurements.
