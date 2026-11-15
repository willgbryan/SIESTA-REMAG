# Recorded response predictions

The notebook supports several distinct observables for one structural
experiment. Every prediction must be recorded before the first outcome for
that experiment. A prediction or outcome for the same observable cannot be
replaced. Existing single-observable notebooks retain their original events
and evaluation result shape; no database rewrite is required.

`record_response_prediction` freezes a response definition as well as a value.
The intervention source is the experiment's exact source hash; the definition
names its baseline. For example:

```json
{"op":"record_response_prediction","args":{"experiment_id":"EXPERIMENT_ID","observable":"Pt-d-window-response","target":{"kind":"spectral_response","baseline_source_sha256":"BASELINE_SOURCE_SHA256","stage":"soc-z","spin":0,"metric":"projected_occupation","selection":{"atom_indices":[1],"l":[2],"energy_window_ev":[-0.5,0.5]}},"value":0.02,"uncertainty":0.01,"interval_probability":0.8}}
```

Values in that example are illustrative, not a physical prediction. The tool
derives units from the metric. Supported targets are:

| Kind | Metric | Observed value | Unit |
| --- | --- | --- | --- |
| `anisotropy_response` | `total_energy_ev` or `free_energy_ev` | Change in self-consistent x-minus-z anisotropy between structures | eV/cell |
| `spectral_response` | `projected_weight` | Change in signed projected spectral weight in the specified window | states/cell |
| `spectral_response` | `projected_occupation` | Change in reconstructed occupied Mulliken weight in the window | electrons/cell |
| `spectral_response` | `projected_energy_moment` | Change in the signed first energy moment of projected weight | eV/cell |

An anisotropy target has only `kind`, `metric` and `baseline_source_sha256`.
A spectral target also requires `stage`, `spin` and `selection`. Scalar channels
0 and 1 are separate; SOC uses the single spinor channel, argument 0. Selection
supports atom, orbital, angular-momentum and k-point indices plus a window;
there is no top-N state truncation. The default window is explicitly stored as
[-3,3] eV. Window edges are included and energies are relative to each endpoint's
own Fermi level. Features crossing a window boundary can change the response.

For selected states, the weight is the original k weight times signed Mulliken
projection. Selected k weights are not renormalized. Spectral weight and its
energy moment include the twofold factor only for a spin-degenerate export.
Projected occupation multiplies the exported occupations, which already carry
that factor. Occupations are the documented Fermi–Dirac reconstruction, not an
export of native occupations. The energy moment is neither total energy nor
an MAE decomposition. These features do not identify a mechanism by themselves.

`record_response_outcome` takes the experiment ID and observable plus `before`
and `after` endpoints. Each endpoint names a `report`, optionally its original
`collection` manifest, and for spectral targets a `spectrum` metadata path:

```json
{"op":"record_response_outcome","args":{"experiment_id":"EXPERIMENT_ID","observable":"Pt-d-window-response","before":{"report":"baseline/query/report.json","collection":"baseline/archive-manifest.json","spectrum":"baseline/query/electronic-soc-z-spin0/electronic.json"},"after":{"report":"intervention/query/report.json","collection":"intervention/archive-manifest.json","spectrum":"intervention/query/electronic-soc-z-spin0/electronic.json"}}}
```

It checks source hashes, original report artifacts, native acceptance and
branches, numerical settings/runtime, ordered atom species, spectrum HSX/KP/EIG
hashes, spin channel, orbital metadata, occupation convention and common k
sampling. It computes the frozen quantity directly from those arrays; the
caller cannot supply an observed value or change the selection at ingestion.
Failed native endpoints become failed outcomes with unknown values. Invalid or
tampered evidence is rejected. Both endpoint hashes and the definition are
retained in the immutable outcome. Compact-archive omissions remain explicit.

The selection refers to literal ordered indices. Matching orbital labels and
numerical settings does not prove radial-basis identity, map reordered atoms,
track crossing bands or isolate SOC. Such stronger interpretations require the
separate matrix/correspondence tools and appropriate evidence.

## Prediction scores

`evaluate_hypothesis` accepts an optional `observable`. Without one, an experiment
with several quantities returns each evaluation instead of silently selecting
the last outcome. `evaluate_predictions` takes up to 200 distinct experiment IDs
and an explicit nonnegative resolution for each observable:

```json
{"op":"evaluate_predictions","args":{"experiment_ids":["EXPERIMENT_ID"],"resolutions":{"Pt-d-window-response":0.005}}}
```

The result includes per-quantity MAE, RMSE, bias, resolved sign accuracy and
interval coverage. Units and response definitions must match within a group;
unlike quantities are never pooled into one error. Failed, inconclusive and
pending outcomes remain in the denominator inventory. Unpredicted outcomes and
experiments without predictions are listed. Unique source counts are reported,
but repeated measurements are not assumed independent.

Sign means the recorded quantity's sign relative to zero. Observations within
resolution stay unresolved; a prediction within resolution abstains. Sign
accuracy counts an abstention on a resolved observation as unsuccessful.
Resolutions are evaluation inputs, not calibrated physical error bounds.

`uncertainty` is a symmetric interval radius in the prediction's unit. Optional
`interval_probability` states its nominal coverage. Scores report observed
coverage, sample count, mean width and deviation from the stated nominal level
separately for each level. A radius without a nominal probability remains an
error tolerance; it is not silently interpreted as a statistical interval.
These finite coverage frequencies do not establish general calibration.

The same scores can be saved with:

```sh
python -m siesta_remag.research prediction-scores --workspace /path/to/workspace \
  --experiments experiment-ids.json --resolutions resolutions.json \
  --output new-prediction-scores.json
```

Local notebook ordering does not prove that an agent had never seen saved data.
The scorecard therefore does not certify blinding, independent holdout,
material transfer, numerical qualification or scientific advantage. An
equivalent-budget scalar-versus-physics-guided study must separately control
information access, computational cost and selection. This increment provides
electronic-response outcomes and evaluation machinery; it reports no such
empirical win and launches no calculation.
