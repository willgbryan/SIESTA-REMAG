# Frozen-potential MAE qualification — 25 October 2026

The experimental evaluator passes the executed implementation checks, but its
directional energies are not accurate enough to replace full SCF at these
settings. Six of seven planned calculations completed; the budget reserve
prevented starting the FeCo eigenvalues-only control. All six completed SCFs
and all 18 remote comparisons passed. Local raw-spectrum, occupation, spin,
science and export checks also pass after fetching 54 output files from the
exact completed attempt `1369ed8d5a31421199b89b09d79a144c`.

The stopping reason is **budget limitation**, not physical convergence. The
separate 5 µeV diagnostic target for frozen/full-SCF differences is missed.
The default build remains unchanged: the probe requires explicit experimental
build and runtime options and is not a qualified screening capability.

## Numerical scope

Both systems use FP64, full offsite SOC, PBE, TZDP with non-perturbative
polarization, global SplitNorm 0.25, zero k offsets and 300 K electronic
smearing. SCF uses eight solver lanes, DM tolerance 1e-9, Hamiltonian tolerance
1e-8 eV and a 300-iteration cap. Geometry and pseudopotentials are fixed and
identified in the archived inputs and metadata. No new CPU timing run was
performed; FePt retains comparisons with the prior pristine-SIESTA outputs.

| System | Orbitals / spinor bands | Mesh / actual k count | Grid cutoff | Confinement |
| --- | --- | --- | --- | --- |
| FePt, two atoms | 48 / 96 | 4³ / 64 | 400 Ry requested, 443.309 Ry actual | 0.01 Ry |
| B2 FeCo, two atoms, a = 2.8562 Å | 56 / 112 | 16³ / 4096 | 2000 Ry requested, 2168.227 Ry actual | 0.00125 Ry |

Every executed frozen direction has the corresponding number of GPU
eigensystem jobs. The evaluator uses one reusable dense solver workspace,
retaining spectra and optional band spin expectations across k points. Explicit
plan storage is 17,020,708 bytes for FePt with spins, 16,283,428 bytes for
FePt eigenvalues-only, and 30,329,508 bytes for FeCo with spins. These figures
exclude borrowed SCF storage, occupation scratch and library-internal memory.
Production still admits 128 k points; the staged 32768 ceiling is not validated
capacity. Only these tested meshes are covered.

## Directional energies and method bias

All entries below are µeV per formula unit at **300 K electronic smearing**.
Full SCF columns contain total energy E and free energy F. Frozen columns
contain band energy and band free energy from a common converged z/[001]
generator Hamiltonian, rotating exchange while keeping lattice SOC fixed.
They are different approximations, not interchangeable observables.

| Difference | Full SCF E / F | Frozen band E / F | Frozen − SCF E / F |
| --- | ---: | ---: | ---: |
| FePt x − z | −1357.583369 / −541.785341 | −1357.944825 / −579.582156 | −0.361456 / −37.796815 |
| FeCo [110] − [001] | 7.445336 / 8.560232 | 103.604430 / 105.006986 | 96.159093 / 96.446754 |
| FeCo [111] − [001] | 20.738187 / 22.220042 | 42.037428 / 43.677484 | 21.299241 / 21.457443 |

FePt's total-energy agreement alone would hide the free-energy discrepancy.
For FeCo the frozen method reverses the relative [110]/[111] ordering, although
[001] remains lowest among the three sampled directions. Passing electron
counts and moment checks therefore does not establish accurate anisotropy.
Raw band, band-free and fixed-electron-number canonical-free differences are
retained separately. None is a zero-temperature MAE claim.

Full-SCF spin magnitudes are 3.378620/3.382100 µB per formula unit for FePt z/x,
and 4.592510/4.592502980/4.592498075 for FeCo [001]/[110]/[111]. The corresponding
frozen-potential magnitudes are 3.378622440/3.381366104 and
4.592512971/4.592582590/4.592536217. Full vectors and direction checks are in
`local-qualification.json`; all executed directions pass the 1° check. These
are spin moments, not total saturation magnetization including orbital terms.

## What passed, and what remains untested

An additional local analysis interpolates the existing FeCo points with
`E = constant + K1*(mx²my² + my²mz² + mz²mx²) + K2*mx²my²mz²`, using normalized
final spin directions in the cubic crystal frame. The coefficients below are
**provisional interpolation parameters**, in µeV/f.u. at 300 K electronic
smearing; they are not experimentally corroborated or converged constants.

| Estimator | K1 | K2 |
| --- | ---: | ---: |
| Full-SCF total energy | 29.781346 | 291.898943 |
| Full-SCF free energy | 34.240929 | 291.772756 |
| Frozen band energy | 414.417719 | −2594.748922 |
| Frozen band free energy | 420.027945 | −2600.959438 |

Three directions determine the constant and two coefficients exactly: the
zero fit residual does not test omitted angular terms. For the ideal
[001]/[110]/[111] directions, `K2 = 9*E001 − 36*E110 + 27*E111`; bounded
per-point energy errors can therefore be amplified by up to 72 in K2. No
error bar is assigned because a physical per-point error bound is unknown.
Generic [210]/[211] holdouts and a [100]/[001] cubic-equivalence control remain
unrun. This analysis used existing outputs only and did not change the pending
GPU batch. Inputs, source hashes, residuals, estimator labels and explicitly
unmeasured model predictions are retained in [angular-analysis.json](angular-analysis.json).

Identity spectra agree with the accepted SCF generator within 2.65e-12 eV for
FePt and 1.24e-10 eV for FeCo. Independent long-double sums reproduce the raw
directional reductions. Electron count retains its 1e-10 gate. FePt's
eigenvalues-only/vector control agrees within 1.07e-8 eV per eigenvalue and
8.53e-10 eV in directional band/free differences. In the original
`eigenvalues-only-comparisons.json`, error values are in eV even where keys
retain the source field's `_ry` suffix; the local report states this explicitly.
FeCo's eigenvalues-only control was **not run**.

These checks cover identity, arithmetic, occupations, spin and unchanged
full-SCF outputs. An independent construction of the rotated frozen Hamiltonian
is still needed before attributing all observed bias to the approximation.
Reference-direction sensitivity and a common scalar-relativistic reference
are the next scientific diagnostics. Keep the energy/free-energy targets
unchanged before further throughput optimization.

One measured rotated solve takes 0.158 s for FePt with spins, 0.091 s for its
eigenvalues-only control, and 11.73–11.78 s for FeCo with spins. These timings
include occupations, explicit downloads and raw CSV output but exclude the
initial SCF reference. Independent full-SCF anchors take 15.31 s for FePt x
and 105.18/106.04 s for FeCo [110]/[111]. They are single observations of
different work with unacceptable method bias: **no scientifically equivalent
speedup or upstream-GPU speedup is established**.

## Corrections and retained failures

- The initial image build failed because its embedded CMake cache selected an
  architecture too old for existing FP64 CUDA kernels. The staged build now
  explicitly targets sm_90. No GPU function ran in that failed app.
- Two runtime attempts failed the spin-normalization check. The diagnostic
  preserved C and S at the first failure. Independently accumulated overlap
  triangles differ by up to 3.99e-9; using both triangles gave a 2.04e-7 norm
  error. The spin consumer now uses the same lower Hermitian triangle as
  cuSOLVER. The saved-data norm error becomes 7.69e-13 without renormalization
  or tolerance relaxation.
- The next attempt completed the probe but was rejected because transfer
  reconciliation omitted its 541,352 recorded download bytes. Accounting now
  includes these bytes and still rejects mismatches. That attempt remains
  marked failed; separate post-hoc checks retain what was verified.

The four calculation attempts and image-build failure have separate evidence
directories. Raw outputs, failure matrices, CSV spectra, implementation archives
and launcher logs remain under their original `results/` run directories and
the Modal results volume. Compact copies omit large spectra/telemetry from
individual result records and retain the original paths and SHA256 hashes.
The local suite passes **149 tests**, with two existing NumPy deprecation
warnings. Focused commits include `97becc5f`, `72758ba0`, `3c1e12ad` and
`2359d7d4`.

## Allocation and remaining scientific limits

All five apps are STOPPED with no account containers. The conservative sum of
app lifetimes is **987.091659 s (16.451528 min)**, including CPU image builds,
within the approved 1200 s allowance. Actual billed GPU seconds are unknown.
The last app stopped at 21:18:58.966 UTC, before the unchanged 21:19:20.615 UTC
deadline. See `allocation-release.json` and `budget-report.json`.

Basis/confinement, grid and k/smearing convergence, geometry, independent
pseudopotentials/semicore, angular and orbital treatment remain open. This
batch establishes neither physical convergence, complete angular anisotropy,
general dense-k capacity nor cross-package agreement. Energy-only full SCF
remains the usable reference path while the frozen-potential discrepancy is
investigated.
