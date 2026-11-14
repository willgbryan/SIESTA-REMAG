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

## Independent controls at 16³

The five pristine measurements completed with native scalar and SOC checks.
The [fitted response](../tests/evidence/research/fept-response-model-r1-2026-11-14.json)
retains all source and report hashes. The original negative-shape-derivative
hypothesis fails for both E and F: shape dominates in magnitude, but its slope
is positive. The registered additive model remains unchanged.

| Control | E anisotropy (meV/cell) | F anisotropy (meV/cell) |
| --- | ---: | ---: |
| Reference | 2.861 | 2.716 |
| Isotropic scale 399/400 | 2.913 | 2.757 |
| Isotropic scale 401/400 | 2.787 | 2.653 |
| Volume-preserving in-plane scale 399/400 | 2.968 | 2.805 |
| Volume-preserving in-plane scale 401/400 | 2.647 | 2.567 |

At the reference, the E derivatives are −8.404 meV per unit log(volume) and
+21.418 meV per unit log(c/a). The corresponding F derivatives are −6.937 and
+15.877 meV. These are fixed-setting local fits with no inferred uncertainty;
five observations interpolate five coefficients and do not estimate an error bar.
The [mixed-control predictions](../tests/evidence/research/fept-response-predictions-r1-2026-11-14.json)
were pushed in `c388dc0` before either target calculation. E predictions for
−0.25%/+0.25% c strain are 2.821208/2.886315 meV/cell; F predictions are
2.689032/2.733753 meV/cell. All three same-training baselines were sealed too.

## Blind mixed outcomes at 16³

Both [prospective targets](../tests/evidence/research/fept-response-evaluation-r1-2026-11-14.json)
completed after the prediction commit. Measured E anisotropies are
2.821/2.894 meV/cell for −0.25%/+0.25% c strain; F values are 2.690/2.742.
The unchanged additive model predicts both response directions correctly and
has lower mean absolute error than each registered baseline for both observables.

| Same-training model | E MAE (meV/cell) | F MAE (meV/cell) |
| --- | ---: | ---: |
| Additive volume + shape | 0.003946 | 0.004607 |
| Constant | 0.036500 | 0.026000 |
| Shape only | 0.017052 | 0.013692 |
| Volume only | 0.057486 | 0.043328 |

The positive target is underpredicted by 0.007685/0.008247 meV for E/F.
The negative E error is only 0.000208 meV, below the native energy print
increment of 0.001 meV. Rounding does not provide an SCF or sampling error bar.
These are two nearby structures at fixed settings, not a significance test or
a generalization result. The original negative shape-slope hypothesis remains
falsified at 16³ even though the separately registered response fit performs well.

The [control audit](../tests/evidence/research/fept-response-controls-audit-2026-11-14.json)
finds matching numerical settings, consistent reference magnetic moments under
the existing branch criterion, and zero x/y splits at native printed precision
for all six new local controls. This does not establish a global magnetic ground state.

The already [registered 32³ transfer](../tests/evidence/research/fept-response-transfer-plan-2026-11-14.json)
will retain every coefficient and use the same reference-only E/F offset for
the model and all baselines. It also changes from pristine to REMAG execution;
the independent 16³ reference replication supports that bridge, but the target
comparison is not a strictly isolated k-mesh response-convergence experiment.

## Post-training orbital diagnostic

The [independent descriptor analysis](../tests/evidence/research/fept-independent-descriptors-2026-11-14.json)
uses only the five training SOC-z spectra and the previously used Gaussian
widths 0.08, 0.12 and 0.20 eV. The analysis plan was committed before these new
spectra were reconstructed; the training energies were already known. This is
a post-training diagnostic, separate from the blind predictions.

All native spectrum comparisons pass; maximum eigenvalue discrepancies range
from 0.0849 to 0.1266 meV. Fe and Pt d-DOS derivatives are positive under volume
expansion and negative under increasing c/a at fixed volume at all three widths.
At 0.12 eV, the volume derivatives are +3.060/+0.641 and the shape derivatives
−4.232/−0.992 states/eV per respective log coordinate for Fe/Pt. Magnitudes
remain width-sensitive: the Pt shape derivative changes from −2.580 to −0.205
between the narrowest and broadest widths.

These signs are opposite to the fitted anisotropy derivatives at this mesh.
That association does not identify SOC transition matrix elements or establish
an orbital mechanism. The earlier 6³-to-16³ descriptor sampling sensitivity,
unqualified basis and absent response-convergence result remain limitations.
