# GPAC × REMAG v3 implementation completion

The owner directed completion of the original **Physics-Guided Autonomous
Crystal Discovery, Development Specification v3.0** on 15 November 2026.
This is the active development objective. The FePt convergence campaign is
closed and is not an implementation prerequisite or an active work queue.
Existing SIESTA implementation comparisons remain evidence; no new pristine
reference is requested. Empirical outcomes must not be inferred from software
tests or marked complete merely because the interface exists.

| Spec scope | Delivered foundation | Remaining acceptance |
| --- | --- | --- |
| §§4.1–4.2, 4.5–4.6 physical authoring | Exact facade, motifs, parameter editing, volume/shape/axial controls, geometric neighbors | Targeted local interventions, coupled constraints and informative proposal assessment exposed to the agent |
| §§4.3, 6 reverse observations | Numeric geometry domain, explicit site IDs, displacement/deformation comparison; existing REMAG relaxation | Integrated relaxation feedback, symmetry/motif/coordination changes and explicit correspondence policies |
| §§4.4, 8–9 research tools | Python/worker tools, immutable hypothesis/prediction/outcome notebook | A reusable, bounded agent workflow spanning observation, intervention, execution, analysis and revision |
| §§5.1–5.2 electronic diagnostics | H/S, checked spectra, Mulliken projections, spinor expectations, native moments/forces/stress | Selectable electronic-state inspection and comparisons; documented SOC/orbital mechanism diagnostics |
| §§5.3–5.4 response/anisotropy | Finite differences, structural response fits, three SOC directions and branch checks | Question-specific calculations, angular sweeps/fits, broader structural/electronic response orchestration |
| §5.5 and exploratory target | Existing GPU execution paths and saved timing evidence | Full query latency accounting and observable-specific calibration; 1–5 minutes remains a product target |
| §§6, 7 FePt integration | Reproducible fixed-geometry dataset, source/input/runtime hashes, independent response controls | Connect new tools to saved data and complete mechanism-oriented experiment capabilities |
| §§7E–F, 12 evaluation | Prerecorded response predictions, held-out tests against simple fits | Electronic-response predictions, uncertainty assessment and equivalent-budget agent-versus-scalar evaluation |
| §10 workspace | Project14-style FePt structure/response/projected-DOS viewer | General structures and linked structural/electronic selections, observations and research actions |
| §11 / Phase 5 expansion | General GPAC authoring and REMAG discovery screening | Demonstrated workflow for Fe₂B/selected structures, theory-guided candidates and transfer evaluation |

Implementation order: (1) observable-specific queries and reusable electronic
inspection; (2) integrated agent workflow; (3) local authoring and relaxation
feedback; (4) linked workspace; (5) comparative evaluation and expansion.
Each delivery includes existing-package checks and a runnable example. Reuse
saved native results for local analysis. New compute is not silently authorized
by this checklist. Optional force-theorem, automatic-differentiation and
approximate-to-exact methods are deferred unless needed for a concrete query.

## Delivered development increments

- 15 November: scalar-only, selected-direction and angular queries reuse native
  execution/branch checks. Angular E/F fits retain one-plane scope. Generic
  physical exports can be prepared through the CLI. Historical bundles retain
  their four stages; no saved campaign bundle is changed.
- 15 November: [local agent tools](AGENT_RESEARCH.md) provide bounded electronic
  selections, signed projected DOS, matched spectral comparisons, existing GPAC
  worker access and persistent hypothesis/prediction/outcome operations. Actual
  report ingestion binds the recorded experiment source and artifact hashes.
  This delivers tool access, not an autonomous experiment-selection policy.
- 15 November: a saved-report calibration tool compares matched intervention
  responses, signs and pair rankings separately for E/F. It rejects unmatched
  sources and inconsistent tiers and retains unresolved small signals. Actual
  calibration coverage and new-calculation latency remain empirical work.

Scientific status remains in `FEPT_QUALIFICATION_PLAN.md`; it is separate from
this implementation checklist. No claim of verified mechanisms, transferable
predictions, physical accuracy or the latency target follows from code delivery.
