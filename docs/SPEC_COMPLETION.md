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
| §§4.1–4.2, 4.5–4.6 physical authoring | Exact facade, motifs, parameters, volume/shape/axial and local coordination/displacement controls, full lineage | Broader coupled constraints and informative proposal assessment |
| §§4.3, 6 reverse observations | Integrated relaxation preparation/feedback, site/orbit motion, neighbor changes, authored-operation residuals | Broader motif classification and changed-cell/atom-count correspondence; maximal symmetry remains unassessed |
| §§4.4, 8–9 research tools | Python/worker tools, immutable notebook, bounded policy-driven inspection/authoring/prediction/outcome/revision sessions | Connect controlled new-calculation execution and demonstrate an actual research policy |
| §§5.1–5.2 electronic diagnostics | H/S, checked spectra, selected-k eigenvectors/reconstructed density, native density and onsite orbital moments, Löwdin coupling blocks, same-basis state/subspace correspondence, selectable Mulliken/spin inspection, native moments/forces/stress | Isolated SOC operators; changed-basis correspondence and empirical mechanism tests |
| §§5.3–5.4 response/anisotropy | Finite differences, response fits, scalar/selected-direction/angular queries, one-plane K1/K2 and branch checks | Broader structural/electronic response orchestration |
| §5.5 and exploratory target | Existing GPU execution paths and saved timing evidence | Full query latency accounting and observable-specific calibration; 1–5 minutes remains a product target |
| §§6, 7 FePt integration | Reproducible fixed-geometry dataset, source/input/runtime hashes, independent response controls | Connect new tools to saved data and complete mechanism-oriented experiment capabilities |
| §§7E–F, 12 evaluation | Prerecorded response predictions, held-out tests against simple fits | Electronic-response predictions, uncertainty assessment and equivalent-budget agent-versus-scalar evaluation |
| §10 workspace | General exact-source catalog, atom/orbit/k/state selection, signed DOS/spin, authored/relaxed comparison; original FePt response viewer | Motif selection, editable research actions and richer linked band/response views |
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
- 15 November: exact local displacements and selected periodic coordination
  scaling are available through GPAC, with explicit symmetry lowering, rational
  parameters and full parent/child site correspondence.
- 15 November: CPU relaxation preparation and saved native feedback reuse the
  existing discovery pipeline. The agent sees source-linked atom/orbit motion,
  geometric neighbor changes and authored symmetry residuals. Maximal symmetry
  recognition and changed-cell/atom-count correspondence remain explicit limits.
- 15 November: the general workspace verifies registered source/calculation and
  Hamiltonian/spectrum bindings, links atom/orbit selection to sampled states,
  and compares authored/relaxed geometries. Compact original archives are usable
  with explicit uncollected-artifact scope. Saved FePt spectra and a separate
  uncalculated Fe₂B authored example exercise the general catalog; this is not
  a Fe₂B phase or transfer demonstration.
- 15 November: a saved-report calibration tool compares matched intervention
  responses, signs and pair rankings separately for E/F. It rejects unmatched
  sources and inconsistent tiers and retains unresolved small signals. Actual
  calibration coverage and new-calculation latency remain empirical work.

- 15 November: bounded external-policy sessions select tools from observations,
  persist intent/results, and stop on limits or failure. A synthetic full
  hypothesis→prediction→outcome→revision loop passes. Native execution remains
  separately controlled; no live-model research quality is inferred.

- 15 November: [selected-k matrix diagnostics](ELECTRONIC_MECHANISMS.md) export
  eigenvectors, reconstructed density and explicit orbital-coupling blocks.
  Same-basis endpoint association handles energy-order changes and degenerate
  subspaces. Analytic tests and saved FePt z/x postprocessing exercise the agent
  interface; no native solver runs. Self-consistent Hamiltonian differences
  remain distinct from isolated SOC, and partial band sums from total MAE.

- 15 November: [native-density orbital moments](ORBITAL_MOMENTS.md) expose
  orbital/atom vectors, native zero-cell density components and explicit
  contraction kernels through Python, CLI, agent tools and linked workspace
  selection. Analytic angular checks and saved FePt z/x/y tables reproduce the
  pinned onsite approximation. Isolated SOC and bulk orbital magnetization
  remain outside this delivered capability.

Scientific status remains in `FEPT_QUALIFICATION_PLAN.md`; it is separate from
this implementation checklist. No claim of verified mechanisms, transferable
predictions, physical accuracy or the latency target follows from code delivery.
