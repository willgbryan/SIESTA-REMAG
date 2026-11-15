# Structural proposal assessment

The research tool `assess_structure` delegates geometry and exact-source checks
to GPAC's `physical-assess` worker operation. Both scalar and physics profiles
receive the same capability. It launches no calculation and does not prevent
an unusual candidate from being prepared for DFT.

Example tool arguments, with illustrative thresholds chosen by the researcher:

```json
{
  "structure": "candidate/structure.json",
  "reference": "parent/structure.json",
  "output": "candidate-assessment",
  "policy": {
    "format": "GPAC-PROPOSAL-POLICY/1",
    "cutoff_angstrom": "3",
    "minimum_distance_angstrom": "1",
    "max_sites": 32,
    "max_orbits": 16,
    "max_stretch_deviation": "1/20"
  }
}
```

Optional GPAC policy fields add per-species coordination bounds and same-species
isolation, and a declared family label/composition ratio/authored-Hall list.
The policy is explicit; no chemical thresholds or material family are inferred.
Distance/stretch numbers use rational strings or integers, not decimal tokens.

The result returns `assessment`, a provenance `record`, `evidence_path` and
`record_path`. Findings identify periodic short contacts, coordination concerns,
exact geometric translation redundancy, complexity preferences, declared family
mismatches and reference-cell strain. They are review items, not chemical
feasibility or magnetic-stability verdicts. Neighbor counts are geometric, and
reference strain compares literal authored cell presentations without aligning
lattice bases or sites. A geometric supercell may be intentional for magnetism.

The output directory must be new. It retains verified source/reference exports,
canonical policy, exact worker request, assessment and file-hash inventory.
`siesta_remag.research.proposals.verify_assessment(directory)` checks the inventory
and reproduces the assessment from those exact saved inputs. Invalid input is
rejected before creating the directory. Existing results are never overwritten.
The returned evidence path can be cited by `record_analysis` in research memory.

This delivers original-spec §4.5 and `validate_structural_proposal` behavior
through the existing tool contract. A model can inspect the findings, revise a
candidate, or explain why a flagged feature is intentional before preparing a
query. The choice remains a research decision. No solver run, model API call,
physical qualification or new spending follows from assessment.
