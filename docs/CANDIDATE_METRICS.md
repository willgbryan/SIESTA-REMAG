# Stability, structural robustness and candidate groups

Paired studies can preregister three additional discovery criteria. The evaluator
recomputes them from exact sources and saved reports, retaining every missing or
failed case. These policies extend `evaluation.discovery`; they create no
calculation or allocation. Existing definitions continue to work unchanged.

```json
{
  "candidate_equivalence": "periodic-origin-proper-signed-axes/1",
  "stability": {
    "stage": "scalar",
    "reference_sources": ["EXACT_REFERENCE_PHASE_SOURCE_SHA256"],
    "maximum_energy_above_reference_hull_ev_atom": 0.01,
    "maximum_force_ev_ang": 0.04,
    "maximum_stress_ev_ang3": 0.001
  },
  "robustness": {
    "designs": [{
      "format": "REMAG-RESPONSE-DESIGN/1",
      "parameter": "axial",
      "values": ["-1/100", "0", "1/100"],
      "control": {"kind": "axial_strain", "axis": 2}
    }],
    "maximum_anisotropy_change_MJ_m3": 0.1,
    "maximum_spin_polarization_change_tesla": 0.1
  }
}
```

The thresholds are examples of explicitly chosen screening rules, not defaults
or scientific acceptance standards. Every other required discovery field
(geometric assessment and numerical controls) remains required.

## Candidate groups

GPAC owns the exact `physical-candidate-key` operation. Its declared policy
identifies periodic origin shifts, site order and the 24 proper signed-axis
permutations. The returned representative uses the exact species-decorated
primitive metric/sites and contains no floating matching tolerance. Source
hashes and GPAC pins remain attached. Different keys establish different groups
under this policy; general lattice shears, supercell reduction, magnetic states
and novelty are unassessed. Improper transformations are not included.

The scorecard reports all groups, geometrically plausible groups and groups with
at least one source that passed every declared screen. Each group retains its
member source hashes and each source's independent status. The global
`crystallographically_distinct_count` remains unknown because this bounded
policy is not a general crystal canonicalizer.

## Finite reference-phase stability screen

Add `reference_phases` to the candidate evidence manifest, mapping each declared
source hash to a `report` and optional compact `collection`. All declared phase
reports are checked and included in the usage coverage, including missing or
failed validation cases. No undeclared reference phase may be added after the
search.

Accepted fixed-geometry candidate and reference reports must have compatible
settings, executable, runtime, thread count and mesh. Shared species must use
identical pseudopotentials and explicit basis shells. The selected stage is
`scalar` or `soc-z`; source identity and native artifacts are verified. The
chemical reference set may contain different compositions, up to eight species
and 64 declared phases.

For E and F separately, the evaluator minimizes reference energy per atom over
nonnegative atomic phase fractions at the candidate composition. It rechecks
composition conservation and retains the decomposition fractions. This is a
linear program solved with [SciPy's `linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).
The signed candidate-minus-reference energy and the nonnegative energy above
the hull including the candidate are both reported. A reference set that cannot
span the composition yields no stability verdict. Missing or failed phases
cannot yield a passing screen.

This follows the finite compositional-hull comparison described in the
[Materials Project methodology](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds).
The result is limited to the supplied structures and native energy model. It
does not establish that the phase set is exhaustive or each phase is relaxed.
Native electronic free energy is not a complete finite-temperature Gibbs free
energy. Phonons, competing magnetic orders, entropy, numerical uncertainty and
physical accuracy remain separate evidence. Maximum candidate residual force
and absolute stress component are reported and checked against the declared
thresholds; passing stationarity alone is not a stability result.

## Robustness on declared structural paths

Each candidate evidence row may add `robustness`, keyed by the registered design
parameter. A value has `series` (the existing prepared response-series directory)
and `reports` (a map from point ID to report/optional collection):

```json
{
  "axial": {
    "series": "candidate-variation",
    "reports": {
      "point-000": {"report": "variation-minus/report.json"},
      "point-001": {"report": "variation-zero/report.json"},
      "point-002": {"report": "variation-plus/report.json"}
    }
  }
}
```

The parent source, exact design, every prepared sample and its atom mapping are
reproduced through the existing series verifier. Source/artifact/configuration
and runtime checks precede property comparison. Native and cross-geometry
magnetic branches must pass. Each row reports signed E/F anisotropy density and
spin polarization changes from the selected candidate, including the zero
sample. Both E and F must satisfy the declared amplitude threshold. Missing,
failed or branch-changing points leave the path inconclusive; no interpolation
fills a gap. Complete paths that exceed a threshold fail the declared screen.

These are finite sampled robustness measurements. They imply no unsampled
response bound, disorder tolerance, magnetic transition temperature or calibrated
uncertainty. All supplied variation reports contribute to the known compute
coverage; their costs cannot be omitted from efficiency totals.
