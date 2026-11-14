# FePt response controls

The original c-axis series changes volume and tetragonal shape together. It
cannot identify their separate derivatives. The new exact GPAC controls supply
independent perturbations: isotropic factors `(s,s,s)` and determinant-preserving
factors `(s,s,s^-2)`. The response coordinates are `v=log(V/V0)` and
`q=log((c/a)/(c/a)0)`. A c-only edit has `v=q`; that rank deficiency is now rejected
by the model fitter.

The [prospective plan](../tests/evidence/research/fept-response-plan-r1-2026-11-13.json)
specifies four independent training controls around the separately measured
reference, a fixed additive quadratic, and constant/shape-only/volume-only
baselines using identical training observations. New ±0.25% c-axis structures
are withheld until numerical predictions have been committed. Old exposed
±0.5% results are not relabeled as blind data. These are initially fixed-setting
16³ experiments; they do not substitute for numerical convergence.

The shape-dominance hypothesis is an inference to test, not a demonstrated
orbital mechanism. Primary work identifies significant Pt spin-orbit
contributions to FePt anisotropy ([Kota and Sakuma](https://arxiv.org/abs/1307.5961));
independent full-potential calculations also emphasize separate numerical and
SOC controls ([Khan et al.](https://arxiv.org/abs/1604.00176)). Their different
electronic-structure approximations and reference geometries are not numerical
calibration targets for this calculation.

A [retrospective descriptor check](../tests/evidence/research/fept-descriptor-sensitivity-2026-11-13.json)
uses the original three training spectra. The Fe d-DOS at the Fermi level
increases with c strain with Gaussian widths 0.08 and 0.12 eV, but decreases
with width 0.20 eV. This qualitative reversal rules out treating that simple
correlation as a robust explanation. Signed Mulliken projections and native
spectral checks are retained; neither establishes SOC transition matrix elements
or a unique electronic mechanism.

Acceptance of the structural response hypothesis requires prospective outcomes.
An advantage over baselines, if observed, remains scoped to the tested numerical
settings until response convergence is demonstrated.
