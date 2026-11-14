# Physics-guided research workspace

The experimental `siesta_remag.research` package connects GPAC's exact structural
authoring to fixed-geometry REMAG measurements. It reuses the existing discovery
input generator, execution harness, density rotation, SCF review and magnetic
branch checks. The existing Orb/relaxation screening workflow remains available.

Install REMAG with `.[research]` and install the GPAC checkout in the same Python
environment. GPAC is required for authoring and the local viewer, not for remote
execution of prepared numerical bundles. Versions and frozen GPAC semantics are
unchanged. This is an internal repository-pinned research interface.

```sh
python -m siesta_remag.research fept --pseudos /path/to/pseudos \
  --output runs/fept --mesh 6 6 6
python -m siesta_remag.research run runs/fept/zero \
  --runtime runtime.json --output runs/measurements/zero
python -m siesta_remag.research serve --workspace runs --port 8765
```

Use `python -m siesta_remag.research snapshot --workspace runs --output fept.html`
to save a self-contained interactive view with exact sources and saved observations.
The snapshot needs no server or network and refuses to overwrite an existing file.

The viewer binds only `127.0.0.1`. Its white, charcoal, cool gray and `#FF0860`
palette follows Project 14. Drag or use arrow keys to rotate; select an atom,
toggle its coordination shell and choose exact strain samples. Source words and
calculation details are collapsed. The viewer never launches paid calculations.
Unmeasured quantities stay empty. Comparisons require identical numerical settings.

## Structure and execution

`prepare_experiment` consumes a verified `GPAC-PHYSICAL/1` export, preserving the
source word, pins, site IDs, numerical geometry and conversion policy. It prepares
a scalar calculation and records every input hash. FePt uses a=b=2.73 Å and
c=3.79 Å, then varies c by −2, −1, −0.5, 0, +0.5, +1 and +2 percent at fixed
fractional coordinates. The ±0.5% points are designated as held-out samples in
the preparation manifest; blinding must be maintained by the research driver.

Pseudopotentials are caller-supplied PBE Dirac PSML or explicitly enabled PBE
relativistic PSF. The latter supports pristine SIESTA's existing FePt reference
files, renamed `Fe.psf` and `Pt.psf`. PSF header validation is a compatibility
check, not independent pseudopotential accuracy qualification. Preserve origins
and hashes when preparing a dataset.

Each experiment performs one scalar seed and independent full SOC [001], [100]
and [010] calculations from rotated copies of that seed. No ionic relaxation
occurs. Completion, native SCF tolerances, spin layout, unchanged geometry,
final moment direction, branch consistency and common directional integration
grids are checked. A failure stops that experiment and preserves its evidence.
Completed calls can be reused only with matching inputs, code, runtime and
artifact hashes. Interrupted runs require an explicitly new attempt.

MAE is `E[100] - E[001]`, in eV per computational cell, with positive values
favoring [001]. Free-energy differences are retained separately. K1/K2 and
numerical uncertainties remain unknown; an x/z pair is not a K1 fit.
The defaults are preliminary screening settings, not a physical convergence
qualification. Basis, mesh, k sampling, smearing, SCF thresholds and magnetic
branches need explicit sensitivity studies and pristine-SIESTA comparison.

## Electronic diagnostics

`research.electronic` reconstructs Bloch H and S from the existing HSX v2 export
and solves `H c = E S c`. It supports scalar, collinear and spinor layouts with
the pinned upstream `diag3kp.F90` spatial/spin index convention. Lattice-gauge
phases use reciprocal fractional k points; energy units are eV. Hermiticity and
S-orthonormality are checked before projection.

Orbital Mulliken weights are `Re(c* · (S c))`, not squared coefficients. Negative
weights are retained. Atom, orbital index, n, l, zeta, spin convention and basis
metadata accompany the array artifact. The angular m label is deliberately not
inferred from metadata that does not encode it. These partitions are basis
dependent, not unique physical observables. Spinor Pauli expectation values
use the same overlap metric.

```sh
python -m siesta_remag.research electronic-run runs/measurements/zero/soc-z \
  --output runs/measurements/zero/electronic-z
```

`electronic-run` reads native `.KP` and `.EIG` and records a spectral comparison.
The viewer withholds untested spectra and residuals above 1 meV; this diagnostic
threshold is not an MAE accuracy criterion. The plot uses 0.12 eV Gaussian
broadening and retains signed Mulliken contributions. The lower-level
`electronic --sampling sampling.json` command accepts custom sampling:
`sampling.json` supplies `kpoints` (fractional reciprocal triples) and `weights`
(nonnegative, sum one). Spectra and orbital weights go to `spectrum.npz`, with
hashes and conventions in `electronic.json`. The Python API accepts matched
native eigenvalues for an independent residual check. Without that comparison,
reconstructed spectra remain explicitly unvalidated. Occupations are documented
Fermi–Dirac reconstructions from HSX chemical potential/temperature, not a native
occupation export. No SOC perturbation decomposition is inferred from H alone.

The three-point `finite_difference` helper supports unequal parameter spacing
and propagates supplied absolute error bounds. Unknown numerical and truncation
errors remain unknown. A derivative is a local response, not a stability claim.

## Research memory

`research.memory.Notebook` stores immutable hypotheses, competing explanations,
experiment actions, prerecorded predictions, evidence-bound outcomes and
evaluations in SQLite. Transactions prevent an outcome/prediction race. Failed
and inconclusive outcomes are retained. Predictions cannot be revised after an
outcome is known. Retrieval is local text search; no model service or API key is
assumed. `evaluate_holdout` compares supplied agent predictions with linear and
interpolation baselines on disjoint bracketed samples. Matching a prediction
does not automatically verify a mechanism.

## One-GPU campaigns

`scripts/research_modal.py` consumes a prepared campaign, an explicit runtime
and a cached native image. Always launch through `scripts/run_modal.py`, which
checks the pinned pristine checkout and refuses any active/unknown GPU allocation.
The function has one A100-80GB, one container, zero retries and a bounded timeout.
It records an allocation cost reservation; actual billing remains unknown until
reconciled with provider usage. No solver build occurs on the paid GPU.

```sh
REMAG_RESEARCH_CAMPAIGN=/absolute/path/to/campaign \
REMAG_RESEARCH_RUNTIME=/absolute/path/to/runtime.json \
REMAG_RESEARCH_IMAGE=im-EXISTING_VALIDATED_IMAGE \
python scripts/run_modal.py --module-file scripts/research_modal.py \
  --receipt runs/submission.json --output runs/collected
```

`execution.json` in the campaign contains ordered `experiments`,
`timeout_seconds` (120..7200), `hourly_reservation_usd`, and `budget_usd`.
The deadline reservation must fit the declared budget, at most $100. That
configuration records a bound; the operator still needs authorization to spend.
The campaign holds on any failed experiment. Use the existing discovery
collector to recover a submitted call; do not submit again to recover output.

## Remaining scientific work

The [first FePt pilot](FEPT_PILOT.md) records seven native reference samples,
pre-observation predictions, baseline evaluation and a failed k-convergence check.

This implementation establishes a working structural/physical bridge, fixed
interventions, electronic analysis, research memory and local inspection.
It does not establish production MAE accuracy, a verified FePt mechanism,
an autonomous model policy outperforming scalar optimization, or transfer to
novel materials. Those require new calculations and recorded evaluations.

SIESTA's [official analysis documentation](https://docs.siesta-project.org/projects/siesta/en/stable/reference/siesta.html)
describes native HSX and electronic exports; the executable/source pin remains
`upstream.lock.json`, not the moving online documentation.
