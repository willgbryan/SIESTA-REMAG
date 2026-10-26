# Scalar-preserving SOC-free MAE validation — 26 October 2026

**The corrected implementation passes its sampled runtime checks; the frozen
approximation still fails the unchanged 5 µeV/f.u. total/free-energy target.**
All nine planned calculations completed. All 26 remote comparisons and the
local reanalysis pass. No screening capability is promoted.

Run `mae-soc-free-scalar-20261026T210526Z`, completed attempt
`677395ac67de45dea1d3e7b75dd8adc9`. Native overlays include scalar preservation
(`e4f3a0be`) and input-staging ordering (`d08b6c50`). The original failed run
`mae-soc-free-reference-20261026T182811Z` remains unchanged. The executable
SHA256 is `507748c71585b3f7cf986aed84b4a6223dc0f62e8a509afb2196ac41f34324f1`;
the rebuilt library is
`3d0b5097156a6adff8a93d5c228acd8ccf9044cbf5e8ed40c6f7d7eb1d904871`.
Only six declared native overlays enter the verified base image; unrelated
working-tree native changes are excluded. Upstream stays pinned and unmodified.

## What the correction establishes

The SOC-free reference keeps half the spin trace of the unsplit offsite
nonlocal operator; frozen targets restore only its spin-traceless remainder.
FePt retains a 3.401840 µB/f.u. SOC-free spin moment, replacing the rejected
zero-moment result. This is a fixed-projector decomposition, not equivalence
to regenerated scalar-relativistic pseudopotentials or an upstream Hemstreet
projector split.

FeCo SOC-free SCFs initialized along [001] and [110] agree in total/free energy
within 5.3151e-9/5.2960e-9 eV and in spin magnitude within 3.5880e-7 µB.
Their restored directional energies differ by at most **6.7168e-12 eV**;
the free-energy spread is at most **6.6264e-12 eV**. The earlier SOC-containing
reference gave a 211.520211 µeV/f.u. [110]−[001] free-energy spread.
This removes the observed reference-direction inconsistency at these settings,
without establishing agreement with full SCF.

39 selected-k matrix captures pass 345 local checks. Maximum rotation error is
5.55e-15 Ry; maximum tested spectral error is 3.22e-10 eV. The restored operator
passes the spin-trace checks. FeCo eigenvalues-only and spinful probes agree
within 4.80e-8 eV per eigenvalue and 8.58e-10 eV in directional band/free energy.
These controls include independently repeated reference SCFs; final directions,
actual k counts/jobs, electron counts, convergence and requested exports pass.
Fresh ordinary full-SOC anchors retain historical REMAG regressions and the
FePt pristine-SIESTA checks. No new CPU SCF timing run was performed.

## Directional energies and moments

All values below are **µeV per formula unit at 300 K electronic smearing**.
Full-SCF columns contain total/free-energy differences. Frozen columns contain
band/band-free differences from the scalar-preserving SOC-free reference.
FeCo uses the [001] reference; the independently converged [110] reference
agrees to the precision described above.

| System / direction difference | Full SCF E | Full SCF F | Frozen band E | Frozen band F | Bias E | Bias F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FePt x−z | −1357.583638 | −541.785568 | −1441.387813 | −680.211548 | −83.804175 | −138.425980 |
| FeCo [110]−[001] | +7.444049 | +8.558903 | −1.901792 | −0.765918 | −9.345840 | −9.324820 |
| FeCo [111]−[001] | +20.732441 | +22.214283 | −2.541188 | −1.014880 | −23.273629 | −23.229163 |

Every tested SOC-free-reference direction fails the 5 µeV E/F bias target.
FeCo's sampled ordering is reversed relative to full SCF. FePt's ordinary
SOC-containing frozen control still has E/F biases −0.360813/−37.796196 µeV;
the new reference therefore worsens its approximation bias. Reference
consistency, implementation equivalence and approximation accuracy are
separate outcomes. The full-SCF anchors themselves are not physically
converged MAE references.

| System / direction | Full-SCF spin magnitude (µB/f.u.) | SOC-restored frozen spin magnitude (µB/f.u.) |
| --- | ---: | ---: |
| FePt z | 3.378620 | 3.380299 |
| FePt x | 3.382100 | 3.382102 |
| FeCo [001] | 4.592510 | 4.593142 |
| FeCo [110] | 4.592503 | 4.593134 |
| FeCo [111] | 4.592498 | 4.593132 |

The SOC-free FeCo reference magnitude is 4.596350 µB/f.u. Both reference
directions and all spinful restored targets pass the signed one-degree
direction gate. Eigenvalues-only targets have no moment measurement.

## Exact scope and timings

FePt: two atoms, 48 spatial / 96 spinor functions, 4³ = 64 actual k points,
400 Ry requested grid (443.309 Ry actual), 0.01 Ry confinement.
FeCo: two-atom B2, lattice parameter 2.8562 Å, 56 spatial / 112 spinor functions,
16³ = 4096 actual k points, 2000 Ry requested grid (2168.227 Ry actual),
0.00125 Ry confinement. Both retain TZDP, non-perturbative polarization,
global SplitNorm 0.25, PBE, fixed geometry and full offsite pseudopotentials,
zero offsets, FP64, upstream radial setup, eight SCF lanes, DM 1e-9,
H 1e-8 eV and a 300-SCF cap. Frozen evaluation uses one streamed dense lane;
its extra SOC-free identity solves are explicitly counted. Production admission
remains 128; staged 32768 is only an admission ceiling.

| Calculation | Iterations | Process seconds |
| --- | ---: | ---: |
| FePt z, SOC-free reference + probes | 45 | 16.508014 |
| FePt z, ordinary reference + probe | 51 | 16.291827 |
| FePt x, full-SOC anchor | 77 | 16.505078 |
| FeCo [001], full-SOC anchor | 33 | 104.031949 |
| FeCo [110], full-SOC anchor | 39 | 111.026410 |
| FeCo [111], full-SOC anchor | 40 | 112.537529 |
| FeCo [001], SOC-free reference + probes | 38 | 153.552053 |
| FeCo [110], SOC-free reference + probes | 37 | 153.842269 |
| FeCo [001], repeated reference + eigenvalues-only probes | 42 | 140.087853 |

FeCo's individual spinful frozen points take 11.47–11.88 s; eigenvalues-only
points take 6.94–7.00 s, including diagnostic exports. Explicit frozen-plan
workspace is 30,329,508 versus 18,516,644 bytes. These are single observations
with different SCF iteration counts and different requested probe outputs,
not paired end-to-end performance measurements or a useful MAE speedup.
All individual values remain in the native reports and `calculations.csv`.

## Release, evidence and next work

The single H100 app `ap-xrKTXHTfEM0cctD4nZyvoX` is STOPPED with zero account
containers. Conservative app lifetime, including image building and evidence
packaging, is **1164.137622 seconds (19.402294 minutes)**. It stopped at
21:24:13.111 UTC before the fixed 21:24:43.931 UTC deadline. Function work
before packaging took 976.189552 seconds; actual billed GPU seconds are
unknown. The 1200-second allowance is closed, with no follow-on allocation.
The stopping reason is completion of the prepared diagnostic, not convergence
or budget exhaustion.

All 81 requested density/matrix outputs were fetched with the exact completed
attempt ID. Local qualification passes all nine calculations and retains the
method-bias failure. Raw spectra, matrix captures, inputs, logs, binaries and
the executed implementation archive remain in local `results/` and the Modal
volume. `raw-evidence-manifest.json` identifies the retained raw files. This
compact copy includes the reports, settings, ledger, release and launcher
sources; it does not replace the raw evidence.

Keep frozen-potential evaluation experimental. The next acceleration priority
is combining energy-only finalization with the existing full-SCF geometry-reuse
sweep, preserving independent atomic guesses and exact SCF acceptance. The
unqualified retained-density warm start remains deferred. This targets the
measured setup cost while retaining self-consistency; it requires its own
runtime and timing qualification. Do not build interpolation or reduced-SOC
production screening on the present inaccurate frozen estimates.

No zero-temperature MAE, total saturation magnetization, complete angular
anisotropy, physically converged screening, general dense-k capacity or
upstream-GPU speedup is established. Basis/confinement, grid, k/smearing,
geometry, independent pseudopotential/semicore, angular/torque and orbital
qualification remain open.
