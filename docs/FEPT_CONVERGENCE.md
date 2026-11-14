# FePt convergence continuation

The [first continuation batch](../tests/evidence/research/fept-convergence-r1-evaluation-2026-11-14.json)
and [28³/32³ continuation](../tests/evidence/research/fept-convergence-r2-evaluation-2026-11-14.json)
completed all six fixed-geometry experiments. The preregistered criterion is
0.05 meV/cell for both E[x]−E[z] and F[x]−F[z], with two successive k refinements
and an x/y split below 0.01 meV/cell. It has not yet passed.

| Mesh | E anisotropy (meV/cell) | F anisotropy (meV/cell) |
| --- | ---: | ---: |
| 16³ | 2.860936 | 2.715598 |
| 20³ | 2.365251 | 2.382505 |
| 24³ | 2.397985 | 2.459264 |
| 28³ | 2.497739 | 2.476167 |
| 32³ | 2.469067 | 2.480508 |

The 20→24 total-energy change passes, but the free-energy change is 0.076759
meV/cell and fails. The threshold is unchanged. A conditional
[28³/32³ plan](../tests/evidence/research/fept-convergence-plan-r2-2026-11-14.json)
was committed before submission and has now completed. The 24→28 E change is
0.099753 meV/cell and fails. The final 28→32 changes are −0.028671 meV for E
and +0.004341 meV for F; both pass individually. Two successive passing steps
are still absent. The original threshold and acceptance rule are unchanged.
The 28³/32³ magnetic branch and transverse checks pass, and their recovered
native binary artifacts are hash-bound separately from the collected text archive.

At 16³, tightening DM/H tolerances tenfold changes E/F anisotropy by
−0.0000156/−0.0000407 meV/cell. Increasing the grid cutoff from 300 to 500 Ry
changes them by −0.038369/−0.040433 meV/cell. Both controls pass the declared
single-change threshold; this is a finite sensitivity check, not an asymptotic
grid error bound. All four continuation cases remain consistent with the 16³
reference magnetic branch under the existing moment checks.

A separate [pristine SIESTA 16³ replication](../tests/evidence/research/fept-pristine-k16-replication-2026-11-14.json)
uses an identical scalar input hash and agrees with REMAG within native printed
energy precision. Its HSX reconstruction passes the native eigenvalue comparison.
These checks support implementation agreement, not basis or physical accuracy.

Basis size, radial extent, electronic smearing, pseudopotential/functional
accuracy and magnetic ground state remain unqualified. The fixed lattice is not
an equilibrium relaxation, and the 300 K setting is an electronic occupation
temperature. The directional energy difference is not a fitted K1 constant.

The [response study](FEPT_RESPONSE_CONTROLS.md) uses separately preregistered
volume/shape controls and new blind mixed strains. Its initial 16³ predictions
remain explicitly scoped to those numerical settings.

## Additional local controls

A [separate CPU plan](../tests/evidence/research/fept-local-controls-plan-2026-11-14.json)
tests TZP versus DZP, radial confinement shift 0.01 versus 0.02 Ry, and electronic
occupation temperature 150 versus 300 K. Each varies only its named control
against the pristine 16³ reference, using the same 0.05 meV/cell E/F and
0.01 meV/cell transverse thresholds. These are single-change sensitivity tests,
not basis or zero-temperature extrapolations. They run sequentially on the
existing local CPU and add no GPU reservation. The sealed strain models and
their evaluations remain unchanged.

The [completed TZP control](../tests/evidence/research/fept-basis-control-2026-11-14.json)
gives E/F anisotropies of 2.735/2.625 meV/cell, changes of
−0.126/−0.091 meV from DZP. Both exceed the declared threshold. All four native
stages and the reference magnetic-branch comparison pass; x/y splits are zero
at printed precision. Basis sensitivity therefore remains unresolved, at a scale
larger than the 16³ mixed-target prediction errors.
