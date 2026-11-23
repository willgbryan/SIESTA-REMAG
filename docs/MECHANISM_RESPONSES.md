# Electronic mechanism responses

The structural-response and prediction tools accept three additional observables:
onsite orbital moments, Löwdin orbital-coupling norms and occupied band sums.
They reuse saved, report-bound exports. They do not run a native solver.

These tools implement the response plumbing in specification §§5.2–5.3. A
response can distinguish candidate explanations; observing it does not establish
a causal mechanism or a unique partition of anisotropy energy.

## Define the observable

Use the same definition in `analyze_response_series` or a prerecorded prediction:

```json
{
  "Fe-Pt-d-coupling": {
    "kind": "coupling", "stage": "soc-z", "spin": 0,
    "groups": [{"site": 0, "l": 2}, {"site": 1, "l": 2}],
    "metric": "frobenius", "k_indices": [0, 5, 13]
  },
  "Fe-orbital-z": {
    "kind": "orbital_moment", "stage": "soc-z",
    "selection": {"sites": [0], "l": [2]}, "component": 2
  },
  "selected-band-sum": {
    "kind": "band_sum", "stage": "soc-z", "spin": 0,
    "reference": "own_fermi", "k_indices": [0, 5, 13]
  }
}
```

Sites refer to the exact family parent and follow its atom permutation. Coupling
groups contain all radial functions and spin components for each site/l pair.
Choose `frobenius` or `largest_singular_value` for the block of
`S^-1/2 H S^-1/2`. This is a basis-dependent hybridization proxy in eV. Geometry
and overlap may change across samples, but radial functions must agree within
`1e-10`. It is not an isolated SOC matrix or a unique bond-energy decomposition.

Band sums use Fermi–Dirac occupations from each saved Hamiltonian. `absolute`
uses its eigenvalue origin; `own_fermi` subtracts that endpoint's Fermi level.
The latter is invariant under a common shift of H and the Fermi level. Neither
is a self-consistent total energy, free energy or force-theorem MAE.

Both matrix observables preserve the original selected native weights, including
their incomplete coverage. At most eight native indices are accepted. No weight
renormalization or inferred missing-k contribution occurs. Per-k values and
weights are retained in the evidence. Full native KP/input evidence must prove
the same declared mesh and ordered grid labels across endpoints, including
printed-coordinate precision. This is sampling correspondence, not band or
eigenvector tracking between deformed structures.

Orbital moments use the native integrated density and the pinned onsite
same-atom/l/zeta approximation described in [ORBITAL_MOMENTS.md](ORBITAL_MOMENTS.md).
`component` is 0, 1, 2 or `norm`; omitted site/l selection includes all orbitals.
The vector is summed before taking its norm. Units are muB. Common orbital
inventory and conventions are checked; this export does not contain radial
arrays, and its evidence explicitly leaves radial identity unverified.

## Supply saved evidence

Export matrices and moments with the existing
[matrix](ELECTRONIC_MECHANISMS.md) and [moment](ORBITAL_MOMENTS.md) tools. Each
series endpoint's optional `diagnostics` object maps an observable name to its
`matrices.json` or `orbital-moments.json` path. Missing entries remain pending.
The [series analyzer](STRUCTURAL_RESPONSE.md) retains source, runtime, acceptance,
branch and parent-site checks. Branch changes retain values but withhold smooth
finite differences. Unspecified measurement and truncation errors remain unknown.

H/S-derived values are recomputed from the bound array artifact, rather than
copied from JSON summaries. Moment exports reproduce their saved contraction
kernels. Native report/source/executable/stage and HSX/KP/EIG or HSX/DM/stdout
bindings are required; matrix radial files are also bound to the report.

To prerecord an endpoint change, use `record_response_prediction` with:

```json
{
  "kind": "diagnostic_response",
  "baseline_source_sha256": "<exact baseline source SHA-256>",
  "diagnostic": {"kind": "orbital_moment", "stage": "soc-z",
    "selection": {"sites": [0]}, "component": 2}
}
```

Each `before`/`after` outcome endpoint supplies `report`, `diagnostic`, and
optionally the original compact `collection` manifest. Pair outcomes require
unchanged ordered atoms; structural families additionally transport parent
selections through their explicit mapping. An incompatible branch returns a
failed outcome with no scoreable value. Prediction targets are immutable and
must precede outcomes. Existing error/sign/interval scorecards apply unchanged.
Scalar-profile agents cannot inspect these diagnostics; reserved held-out
sources remain unavailable to the research policy.

Tests use analytic synthetic H/S/density data to check responses, unequal-step
derivatives, atom reordering, energy-origin shifts, partial native weights,
artifact/convention rejection and actual local-worker scoring. Synthetic scores
do not establish calibrated uncertainty, blinding or predictive performance.
