# Controlled structural responses

`research.series` connects exact GPAC perturbations to independently runnable
REMAG queries and saved-result analysis (§§5.3, 7C–D, 8–9 of the v3 spec).
Preparation creates no calculations, reservations, retries or execution queue.
Each prepared point remains subject to the existing explicit single-query
execution controls. Analysis consumes existing evidence only.

## Declare a path

```json
{
  "format": "REMAG-RESPONSE-DESIGN/1",
  "parameter": "epsilon_c",
  "values": ["-1/50", "-1/100", "-1/200", "0", "1/200", "1/100", "1/50"],
  "control": {"kind": "axial_strain", "axis": 2}
}
```

Use 3–9 strictly increasing exact rational samples in `[-1/2, 1/2]`, including
zero. Inexact floating-point inputs are rejected by GPAC's exact admission
boundary. One parent, pseudopotential set, explicit k mesh, basis, query and
numerical settings apply to the entire family. Distinct exact sources that
collapse to the same numerical solver geometry are rejected before writing.

| Control | Meaning of dimensionless `q` | Additional fields |
| --- | --- | --- |
| `axial_strain` | Scale selected authored cell vector by `1+q` | `axis`: 0, 1 or 2 |
| `isotropic_strain` | Scale every cell vector by `1+q` | None |
| `fixed_volume_tetragonal` | GPAC's volume-preserving basal scale `1+q`; c scales inversely by its square | None |
| `fractional_displacement` | Add `q × direction_fractional` to selected sites | `sites`: records with `site_index`, exact three-component `direction_fractional` |
| `coordination_strain` | Scale selected periodic neighbor vectors about a center by `1+q` | `center_site`, `neighbors`: GPAC site/image records |
| `metric_path` | Add `q × rate` to declared free Gram-matrix parameters | `rates`: map such as `{"metric.g11":"2","metric.g33":"-1"}` |

Metric rates use the parent's angstrom-squared Gram convention. Local site
operations use explicit P1 symmetry lowering and record that choice. Every
point is generated directly from the common parent, never cumulatively from
the previous point. Exact parent-to-child atom permutations are verified and
persisted; orbit correspondence is not substituted for atom correspondence.

```sh
python -m siesta_remag.research prepare-response parent.json \
  --design design.json --query query.json --pseudos pseudos \
  --mesh 16 16 16 --output response-family
```

The Python API is `prepare_series(...)`. The agent tool
`prepare_response_series` returns the family path and individual bundle paths.
`series.json` binds the original parent, exact design, candidates, prepared
queries and every prepared file. Outputs must be new directories. A preparation
failure retains its partial files and failure record; it does not resume or
overwrite an existing family.

## Analyze saved measurements

Evidence maps planned IDs to `report` paths and optionally an original compact
`collection` archive-manifest path. Optional `spectra` maps named spectral
observables to saved checked electronic metadata. CLI evidence paths are
resolved from the current working directory; agent paths are confined to its
workspace. A report can predate the family declaration, provided its exact
source, geometry and scientific configuration match. This is retrospective
analysis, not a prerecorded prediction or held-out validation claim.

```json
{
  "K_E": {"kind":"anisotropy", "stages":["soc-x","soc-z"], "quantity":"total_energy_ev"},
  "K_F": {"kind":"anisotropy", "stages":["soc-x","soc-z"], "quantity":"free_energy_ev"},
  "moment": {"kind":"moment", "stage":"soc-z", "site":null, "component":"norm"},
  "force_z": {"kind":"force", "stage":"scalar", "site":0, "component":2},
  "stress_zz": {"kind":"stress", "stage":"scalar", "component":[2,2]},
  "Fe_d_weight": {"kind":"spectral", "stage":"soc-z", "spin":0,
    "metric":"projected_weight", "selection":{"atom_indices":[0],"l":[2],"energy_window_ev":[-1,1]}}
}
```

`energy` selects one stage and `total_energy_ev` or `free_energy_ev`.
Anisotropy is explicitly the first selected SOC stage minus the second, in
eV/cell. Force components use eV/angstrom, stress uses eV/angstrom³ and moments
use muB. Moment `site:null` selects the cell total; site indices always refer
to the original parent. A moment component may be 0, 1, 2 or `norm`.

Spectral metrics are signed Mulliken window weight, occupation or energy moment,
with the same selection and units as [response evaluation](RESEARCH_EVALUATION.md).
Negative projected contributions are retained. Parent atom selections follow
the exact permutation. Literal orbital-index selections require unchanged atom
ordering. Both profiles can prepare families and analyze standard observables;
the scalar profile cannot request SOC operator exports or spectral responses.
Reserved held-out sources remain unavailable to the research policy.

```sh
python -m siesta_remag.research analyze-response response-family \
  --evidence evidence.json --observables observables.json --output response-analysis
```

Python: `analyze_series(...)`; agent: `analyze_response_series`.
Source, scientific and acceptance settings, pseudopotentials, executable and
runtime environment must agree. Newly generated reports explicitly record
thread count; historical reports may leave it unknown. Source-code inventories
are retained for each report; equality of historical runner revisions is not
inferred. Compact collections verify all collected and locally present bytes,
while explicitly naming omitted remote artifacts.

## Responses and limits

Every original sample remains in `response.json`: measured, failed or pending.
A branch change preserves the measured value and withholds the smooth-branch
derivative. The existing projected-site and net-moment tests compare each point
against zero using parent atom correspondence. Near-zero net reference moment
is explicitly unresolved by this relative-net branch test. It does not certify
antiferromagnetic branch continuity.

Every consecutive three-point window gets its own local quadratic first and
second derivative, including unequal spacing. All three points must have
accepted native evidence and compatible branches; missing points are never
skipped to make a wider artificial stencil. Differences from the baseline are
also reported. Optional absolute `error_bounds` are keyed by observable and
point ID. Only three supplied bounds yield a conservative propagated bound.
Unknown observable error, finite-difference truncation and numerical convergence
remain unknown. Three samples do not prove a derivative is physically accurate.

Vectors remain in each authored right-handed Cholesky solver frame; there is
no hidden rotation or changed-cell correspondence fit. Electronic comparisons
require matching parent orbital inventories and actual ordered fractional
k points/weights within `1e-12`. Native KP decimal rounding can differ across
strained cells even on the same intended mesh; such historical spectra are
currently rejected rather than silently relabeled. Precision-aware native
sampling correspondence remains a follow-up. Radial-basis identity, band
tracking and unique SOC or MAE attribution are not inferred by window integrals.

Engineering tests cover all six paths, reordered atoms, analytic unequal-step
responses, explicit errors, missing/failed/changed-branch samples, native hash
and runtime mismatches, signed spectra, CLI and actual local agent processes.
Their calculations use synthetic runners. A separate saved FePt example uses
the existing 6³ DZP results at −0.5%, 0 and +0.5% c strain. Four unmeasured points
in the seven-point design remain pending. That example verifies the integration;
it establishes neither convergence nor a new scientific prediction.
