# Ouro temporary-file adapter

Install `.[ouro]` for the pinned `ouro-py==2.24.0` transport and JSON-schema
preflight. The adapter implements local file preparation and operator APIs for
one submission, one status observation and collection. A live saved-Ni hull call has now succeeded in Ghost; its
[checkpoint](../tests/evidence/research/ouro-ghost-Ni-hull-2026-11-17.json) records
input/result bindings and the original upload-only failure. The generic adapter
does not turn a successful action into a scientifically qualified observation.
Each route still requires explicit admission and result interpretation.

## Saved hull inspection

`inspect_ouro_hull` makes a collected crystal-db/ehull screen available to the
research agent without a network call. It verifies the request and input bytes,
submission/action identities, organization, complete collection and every listed
artifact. It validates the response against the captured route schema and checks
the energy surface, input model, composition, finite eV/atom values and counts.

```json
{"tool":"inspect_ouro_hull","args":{
  "attempt":"ouro/attempt-Ni",
  "collection":"ouro/collected-Ni",
  "structure":"prepared/Ni/structure.json",
  "geometry":{"kind":"local_orb","bundle":"prepared/Ni/bundle"}
},"rationale":"Inspect the saved hull screen for this locally relaxed geometry."}
```

Geometry is an explicit choice: `{"kind":"authored"}`, `local_orb` with a
verified preparation `bundle`, or `native_relaxed` with `bundle` and accepted
native `report`. The tool verifies the exact source against the original input
and the submitted CIF against the selected geometry. Species/order must match;
cell metrics and periodic fractional sites use an absolute `1e-8` tolerance.
Cartesian rotation and periodic wrapping in CIF serialization are permitted.
Multiple CIF structures, partial occupancies and provider-side relaxation are
rejected. A matching source hash alone cannot transfer a hull screen between
the authored, locally relaxed and native-relaxed cells.

The result contains the provider's classification, energy above hull,
decomposition, reference counts and effective request settings. It keeps
mechanical/dynamical stability and synthesizability separate and unestablished.
Provider weight and database-snapshot hashes are unknown; matching model names
do not certify identical weights. The PBE option is explicitly labeled as
mixing an Orb input energy with PBE reference energies. Neither a successful
action nor a hull value proves physical stability or provider GPU release.
The metadata quote remains separate from unknown billing.

Both scalar and physics profiles can inspect the same source-bound scalar hull
screen. Withheld sources and escaping nested paths remain rejected. The earlier
restriction on arbitrary external-file **preparation** still applies. Raw
actions, upload handles and signed URLs are not returned to the agent.

The saved live Ni call reproduces `0.000004 eV/atom` for its **local Orb** cell;
the authored and DFT-relaxed Ni geometry selections must reject that same CIF.
This is an evidence-ingestion capability, not a new hull calculation or a
calibrated thermodynamic acceptance gate.

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
An explicit zero fixed rate in USD or BTC is zero without an exchange-rate
lookup; absent rates or contradictory nonzero metadata do not become free.
Optional service headers are left to Ouro, never supplied by this adapter.
Required headers and attempts to pass header names as query/path parameters
are rejected. Schema-reference checks also apply to omitted headers.
Zero BTC-only routes use `currency=btc` on execution; a zero quote does not make
USD an accepted payment currency.

An operator can pass `organization_id` and `team_id` to preparation. These UUIDs
are part of the request digest. Submission and observation require a client
pinned to that organization, including its `X-Ouro-Org` header and selected team.
The route call receives the same organization and output team. Collected assets
with another organization/team are retained as a failed review and never
downloaded. This binds Ouro execution/output context, not an independent
service's database tenancy.
To persist an operator's workspace choice, set `research-ouro-context.json` at
the workspace root to `{"organization_id":"<UUID>","team_id":"<UUID>"}`.
The policy preparation tool uses this context by default and rejects conflicting
arguments before writing a request. The generic library remains usable by other
workspaces and accounts.
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
Temporary upload IDs are opaque strings, not asset UUIDs. Each raw upload
receipt is saved before validation in a private attempt directory. Keep those
handles out of public evidence; publish their hashes instead.

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

The working integration uses local CPU Orb relaxation, REMAG electronic/magnetic
evaluation, and a separate Ouro hull screen. The saved Ni preparation already
used `orb-v3-conservative-inf-mpa`: 4.0877 seconds including the local worker,
10 optimizer steps, maximum atom force 0.0147412 eV/Å and stress 0.0819901 GPa.
This is a measured preparation, not end-to-end research latency or stability.

On 17 November the existing account successfully retrieved the live
[hull route](https://ouro.foundation/routes/mmoderwell/calculate-energy-above-the-convex-hull),
ID `75fe7f4b-22f5-4883-aa1d-378fd42da47c`. Its contract accepts a temporary CIF
object with URL/filename, and currently quotes an explicit zero fixed BTC rate.
That initial snapshot preceded all uploads and route execution. The request uses the saved local
Orb geometry, Orb MPA surface, `allow_unrelaxed=false`, health checks enabled,
and no user-material/generated-P1/phase-diagram outputs. The saved force and
cell-force check passes the route's 0.05 eV/Å screen. CIF conversion preserves
the cell metric and periodic fractional sites; Cartesian cell orientation is
not a CIF invariant. Exact GPAC source and observed geometry hashes are retained.

The Orb screen refers to the Orb-relaxed geometry, not the distinct DFT-relaxed
geometry used for the magnetic result. The route's PBE option mixes an Orb-scored
candidate with published DFT references, so it is not selected as a consistent
DFT hull. The same named Orb model family is not proof of identical provider
weights or settings. Live outputs still need model, geometry, reference coverage
and scientific validation before use in a stability conclusion.

The route documentation says novel inputs are added to its crystal database;
the retrieved contract has no opt-out. Temporary input storage and private
output settings do not prevent that service-side database write. Submission is
authorized by the owner, who permits future Ouro uploads into **Ghost**. The
verified Ghost organization is `0194e5db-d49e-7d25-ac3d-fde542f594cb`, with default
team `0194e5db-d4c6-7bc2-9a6e-14e628284eb8`. Provider hardware and allocation
release are not reported by the route. A terminal service action must not be
reported as verified provider GPU release.
The local preflight and its two retained earlier failures are recorded in
`tests/evidence/research/local-relaxation-ouro-hull-preflight-2026-11-17.json`.
No old Desktop campaign or closed FePt campaign is resumed.

The first Ghost temporary upload completed, then the local adapter rejected its
opaque handle as a UUID before making any route call. That failure is retained;
the handle check is now corrected and covered by a real-SDK/mock-HTTP regression.
A subsequent explicit invocation was held by the account guard before upload
because an unrelated REMAG GPU app was active. No hull action or stability result
has been obtained, and no follow-on job is queued. Current receipts and counters
are in `tests/evidence/research/ouro-ghost-context-2026-11-17.json`.

`tests/test_research_ouro.py` uses the actual pinned SDK with intercepted HTTP,
including temporary upload reservation/PUT, async submission, one-read polling
and output download. Regression controls cover changed price/schema/input bytes,
denied admission, unsupported inputs, 401/timeout without POST retry, partial
uploads, missing outputs, action identity and information-profile restrictions.
The fixture responses are protocol evidence, not live scientific measurements.
