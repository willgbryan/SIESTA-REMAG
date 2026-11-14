# FePt pilot — 13 November 2026

The exact GPAC → fixed-geometry DFT → electronic-diagnostic → research-memory
path works on seven FePt strain samples. The physical response is **not yet
converged**. Increasing the reference k mesh from 6³ to 8³ changes MAE from
1.311 to 3.296 meV/cell, larger than the observed strain effect. A mechanism
claim or materials ranking would be premature.

These measurements use pristine SIESTA at the pinned upstream commit, on CPU.
They are reference controls, not REMAG GPU validation. Inputs use fully
relativistic PBE Fe/Pt PSF files from that source, DZP with a 0.02 Ry energy
shift, 300 Ry mesh cutoff, 300 K smearing, and fixed fractional sites. Each
sample has one scalar seed and independent [001], [100], [010] SOC runs.

| c strain | E[100] − E[001], meV/cell |
|---:|---:|
| −2% | 2.424 |
| −1% | 2.080 |
| −0.5% | 1.710 |
| 0 | 1.311 |
| +0.5% | 0.919 |
| +1% | 0.692 |
| +2% | 0.752 |

The agent selected a local quadratic response using only −2%, zero and +2%
and recorded predictions before either ±0.5% outcome existed. Predictions were
1.5373 and 1.1193 meV/cell. The agent-selected fit, a straight-line fit and
piecewise interpolation all have held-out mean absolute error 0.1865 meV/cell.
Their RMSE values are 0.1870, 0.2600 and 0.1978 meV/cell respectively. Two
points at unconverged settings do not demonstrate superiority to scalar
optimization. No electronic mechanism is certified.

The [machine-readable evaluation](../tests/evidence/research/fept-pristine-2026-11-13.json)
records input and report hashes, the k-mesh check, diagnostics, limitations and
preserved failures. The [notebook events](../tests/evidence/research/fept-notebook-2026-11-13.json)
retain timestamped hypotheses, alternatives, predictions and outcomes. Large
raw/native arrays remain in the calculation workspace rather than Git.

Native runs caught two defects that toy controls missed: integer-looking
origin coordinates were misread by libfdf, and spinor HSX sparse spatial
indices were transposed incorrectly. Coordinates now include decimal points.
Spinor assembly follows pinned `diag3kp.F90`; reconstructed spectra agree with
native EIG within the KP/EIG decimal precision. The viewer withholds unchecked
or failed spectra. The earlier failed outputs remain separate.

The next decision is reference numerical qualification: compare REMAG with
these pristine controls, extend k sampling, then assess mesh, basis, smearing
and SCF sensitivity before revisiting the strain mechanism. No extra GPU
allocation may overlap an existing one.

## REMAG replication and reference sampling

Two sequential A100 campaigns completed after the shared GPU became free.
At the shared 6³ and 8³ settings, REMAG differs from pristine SIESTA by
−0.230 and +0.427 µeV/cell in MAE, within the precision of the native text
energy output. All scalar and directional SCF stages passed.

| REMAG k mesh | MAE, meV/cell |
|---:|---:|
| 6³ | 1.310770 |
| 8³ | 3.296427 |
| 12³ | 2.405574 |
| 16³ | 2.860936 |

The last change is 0.455362 meV/cell. **K-point convergence remains open.**
The [reference evaluation](../tests/evidence/research/fept-remag-reference-2026-11-13.json)
binds the measurements and twelve checked SOC spectra. The 8³ spectra were
initially held because rounded KP weights sum to 0.99999744. Their exact
printed-precision error bound is 0.00000256. They were recovered locally from
hash-matched saved HSX files using a per-token decimal rounding bound, without
rerunning DFT or erasing the original held diagnostics.

The original locked predictions were also evaluated against independent REMAG
calculations at ±0.5%, yielding 1.709773 and 0.919618 meV/cell. This is a
replication of the existing preregistration, not a fresh blind test after seeing
native outcomes. All three predictors again tie in MAE, at 0.186077 meV/cell;
the quadratic, linear and interpolation RMSE values are 0.186575, 0.259568 and
0.197287. See the [GPU prediction evaluation](../tests/evidence/research/fept-remag-predictions-2026-11-13.json)
and [final notebook](../tests/evidence/research/fept-notebook-final-2026-11-13.json).

The two runs reserved $20 and $6.67 within the owner's $100 authorization,
using one GPU at a time. Recorded function allocation times were 1054.24 and
182.01 seconds. These reservations and times are not a billing invoice.
No claim of mechanism verification, physical convergence or superiority to
scalar optimization follows from this pilot.
