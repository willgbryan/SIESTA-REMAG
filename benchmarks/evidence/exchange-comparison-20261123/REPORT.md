# Focused FePt exchange comparison — 23 November 2026

REMAG's GPU real-space Green response completed the matched FePt calculation in
**29.157 s including first-call interpolation**, versus **150.961 s** for TB2J's
full-basis calculation: **5.18× faster**. All 12 unique interactions agree within
**4.50e−14 eV** after applying the declared pair-counting conversion. Eight TB2J
workers took 158.038 s and agreed within 9.50e−14 eV. This is one small,
scientifically matched screening workload, not a general scaling claim.

## Matched calculation

Both implementations consume the saved parent FePt native generator from
`discovery-response-20261123T175305Z`; no new SCF was run for this comparison.
The same finite-range H/S, onsite exchange fields, nonorthogonal PAO basis,
cell-positive Bloch gauge, 18 electrons, chemical potential −3.966119060854283 eV,
electronic temperature 0.025851999786436097 eV (the native 300 K value), 9³
Gamma-containing mesh and 2.8 Å pair cutoff are used. SOC is excluded from this
isotropic exchange calculation. Fe is a local-moment site; Pt channels are
reported as induced electronic responses, not independent spins.

The mesh is odd because TB2J rounds requested even dimensions upward. Its
continued-fraction contour contains 128 positive and 128 negative poles plus
the zero-weight infinity point; REMAG evaluates the equivalent positive-pole
symmetry. TB2J receives 48 orbitals per spin; REMAG retains the full 96-spinor
basis. HamiltonIO performs the existing CPU eigenproblem and Fourier conversion;
no independent CPU exchange solver was implemented.

The input adapter reconstructs the upper triangle from the native lower
triangle using opposite translations, preserving SIESTA's eigensolver convention.
The transported Hamiltonians agree within 5.33e−15 eV and overlaps within
4.47e−16 at the checked k point. TB2J's ordered-pair J is multiplied by two for
REMAG's `E=-sum_unique_pairs e_i.J_ij.e_j` convention. There are 24 directed,
12 unique interactions. The nearest Fe–Fe value is about 20.2098 meV at 9³.
Its difference from the previously saved 8³ result confirms that exchange is
still mesh-sensitive; implementation agreement is not physical convergence.

## Wall time and accuracy

| Calculation | Setup / interpolation, s | Exchange, s | Total, s | Maximum error vs REMAG, eV |
|---|---:|---:|---:|---:|
| REMAG, first call | 5.408635 | 23.748478 | 29.157113 | — |
| REMAG, warm exchange on the same sampled state | excluded | 2.779102 | 2.779102 | — |
| TB2J, full basis, 1 worker | 10.299878 | 140.661514 | 150.961392 | 4.50e−14 |
| TB2J, full basis, 8 workers | 6.234271 | 151.803551 | 158.037822 | 9.50e−14 |
| TB2J, default band window, 1 worker | 14.484358 | 122.084775 | 136.569133 | 3.116e−3 |

The headline uses the full first-call REMAG cost and the faster full-basis TB2J
setting. The warm GPU timing is retained separately and is not a matched warm
CPU speedup. Both implementations include their eigenvalue preparation in these
stage totals. Common source loading, Python imports, cloud provisioning, image
building and evidence packaging are excluded. These are single observations,
with no error bars or broad timing campaign.

TB2J normally discards bands above EF+5.1 eV in this route. The full-basis run
subclasses only that reduction hook to retain the supplied basis; TB2J's exchange
algorithm is unchanged. Its default window retained 13 up / 12 down bands and
changed interactions by up to 3.116 meV, so that faster timing does not meet the
matched numerical condition. The [frozen harness](harness/single/compare_exchange.py)
shows the exact conversion and single override.

The main comparison ran sequentially on one A100-SXM4-80GB, driver 610.57.04,
CUDA 12.6.3, with eight allocated CPU cores and one BLAS thread. TB2J used one
worker. The eight-worker follow-up used an eight-core CPU allocation with the
same image and one BLAS thread per worker; it requested no GPU. CPU model was
not captured and remains unknown. No TB2J GPU path or OpenMX timing is claimed.

TB2J source: [4d7a65bd7a2b582f7ede53e9fec88efe8d6895ed](https://github.com/mailhexu/TB2J/tree/4d7a65bd7a2b582f7ede53e9fec88efe8d6895ed),
version 0.9.19; HamiltonIO 0.3.8, NumPy 2.5.3, SciPy 1.18.1 and CuPy 14.2.0.
The source lists an unpublished `pypao` dependency; required imports were
installed explicitly and TB2J installed without dependency resolution. The
reused image's SciPy is outside TB2J's declared `<1.17` bound. The reported
accuracy and performance were measured in this retained environment; this is
not a claim of upstream-supported packaging. No numerical source was patched.

## OpenMX and retained evidence

OpenMX `jx` requires an OpenMX collinear SCF `scfout`, as specified in its
[exchange procedure](https://www.openmx-square.org/openmx_man4.0/s45_3_procedure.html).
The saved SIESTA generator cannot be passed directly. A new OpenMX SCF would
change the basis/pseudopotentials, while a custom converter would add substantial
work outside this focused comparison. It was therefore not run.

[Summary](summary.json), [matched pair values](single/matched_full_basis.json),
[eight-worker values](parallel/matched_full_basis_8_workers.json), full compact
outputs and raw calculation logs are retained here. Raw source archives remain
on Modal volume `siesta-remag-magnetic-results` under the run IDs in the summary.
Two image-only dependency failures allocated no GPU. The initial generator
transport attempt failed its matrix check and is retained under
`input-check-failure`; it is excluded from the performance conclusion. All
allocations were sequential. A diagnostic for driver 620 was prepared but the
comparison hosts ran driver 610, so this work establishes no driver-620 fix.
