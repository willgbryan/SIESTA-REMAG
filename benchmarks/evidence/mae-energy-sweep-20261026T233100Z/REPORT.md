# Energy-only full-SCF geometry reuse — 26 October 2026

All 14 processes and 21 independently initialized SCF directions passed the
remote checks, including 36 historical comparisons and four fresh matched
comparisons. Local output reanalysis also passes. All 221 supplementary files were
downloaded (including 165 RHO/DRHO/DM/HSX/XV files), with the completed attempt
identity unchanged. This batch measures
geometry reuse within REMAG; it does not measure upstream-GPU speedup or
establish physical MAE convergence.

## Matched whole-process timings

| Material / repetition | Separate processes (s) | Cached sweep (s) | Ratio |
| --- | ---: | ---: | ---: |
| FePt z/x, 1 | 29.920128 | 17.034139 | 1.756480 |
| FePt z/x, 2 | 30.123973 | 17.130329 | 1.758517 |
| FePt z/x, 3 | 29.938199 | 17.640327 | 1.697145 |
| FeCo [001]/[110]/[111], 1 | 334.284995 | 227.840221 | 1.467190 |

FePt's median paired ratio is **1.75648×**, with a 1.69715–1.75852× observed
range. Median separate/cached times are 29.938199/17.130329 s. Contender order
alternates between repetitions. FeCo has only one sample; repeatability remains
unmeasured. Both contenders use the same executable, one H100 allocation,
thread/solver settings, requested electronic exports and 100-ms CPU/200-ms GPU
observers. Times include SIESTA setup, SCF and exports, but exclude image build
and external validation. Individual sweep-direction wall times remain unknown.

SCF iteration counts differ slightly between executions; all converge under
the same criteria. These are measured workflow gains, not an isolated estimate
of geometry-setup savings. Each direction starts from an independent atomic
guess. Retained density, a frozen potential and reduced precision are not used.

Existing device-SCF timing records provide this narrower breakdown. The
residual includes all work outside the recorded SCF interval, including setup
and requested outputs; it is not a direct setup-only measurement.

| Sample | Recorded SCF, separate/cached (s) | Remaining process time, separate/cached (s) |
| --- | ---: | ---: |
| FePt repetition 1 | 3.001308 / 2.980825 | 26.918821 / 14.053313 |
| FeCo repetition 1 | 165.108915 / 151.079177 | 169.176080 / 76.761045 |

FePt remains dominated by work outside SCF. SCF accounts for about 66% of the
cached FeCo process, so its next optimization needs to address that cost too.

The 19.547419-s FePt z/x/z return-to-start check is excluded from performance
ratios. Returning to z changes total/free energy by 1.06866e-10/6.68479e-11 eV;
the printed spin norm is unchanged. All signed final moments pass the
one-degree direction gate.

## Scientific agreement and scope

Maximum fresh separate-versus-cached directional E/F discrepancy is
**0.000326 µeV/f.u. for FePt** and **0.001687 µeV/f.u. for FeCo**, below the
unchanged 5 µeV/f.u. gate. Energy, free energy, density, spin populations,
eigenvalues, DM/HSX exports, convergence and actual k-count/eigensystem-job
checks pass remotely and locally. FePt also retains the saved pristine-SIESTA comparisons;
FeCo retains saved REMAG regressions. No new CPU timing calculation was run.

Representative cached directional differences at **300 K electronic smearing**
(FePt repetition 1; FeCo's single repetition), in µeV per formula unit:

| Difference | Total energy | Free energy |
| --- | ---: | ---: |
| FePt x−z | −1357.583239 | −541.785156 |
| FeCo [110]−[001] | +7.447334 | +8.562209 |
| FeCo [111]−[001] | +20.733039 | +22.214886 |

Spin magnitudes are FePt z/x **3.378620/3.382100 µB/f.u.** and FeCo
[001]/[110]/[111] **4.592510/4.592502980/4.592498075 µB/f.u.** Full vectors,
energies, iterations, actual eigensystem jobs and recorded SCF seconds for all
21 directions are in `calculations.csv`.

FePt uses two atoms, 48 spatial/96 spinor functions, 4³ = 64 actual k points,
400 Ry requested grid and 0.01 Ry confinement. FeCo uses two-atom B2,
2.8562 Å lattice parameter, 56 spatial/112 spinor functions, 16³ = 4096 actual
k points, 2000 Ry grid and 0.00125 Ry confinement. Both retain TZDP,
non-perturbative polarization, global SplitNorm 0.25, PBE, fixed geometry and
pseudopotentials, full offsite SOC, zero offsets, FP64, upstream radial setup,
eight solver lanes, DM 1e-9/H 1e-8 eV and a 300-SCF cap.

Retained pair, projector, H0, initial-mesh and compact-orbital records show
accepted electronic retirement and zero new geometry setup/allocation in later
directions. Force-completion flags stay false; no forces/stress or derivative
generation are fabricated. These records do not imply that every setup or SCF
allocation is eliminated. Production admission remains 128 k points; the
staged 32768 ceiling is not validated general capacity.

This qualifies selected implementation equivalence. Basis/confinement, grid,
k/smearing, geometry, pseudopotential/semicore, angular and orbital convergence
remain open. These are finite-smearing directional energies and spin moments,
not zero-temperature MAE or orbital-inclusive saturation magnetization. The
frozen-potential approximation remains experimental after its earlier failed
accuracy checks.

## Evidence and allocation

Run `mae-energy-sweep-20261026T233100Z`, completed attempt
`166fb639d1e8420296ff10da0f37cd0c`, app `ap-FXn1BFmFiJTIgqAvbYU4Rv`.
Candidate commit `cd201c34` includes the reviewed geometry lifecycle,
per-direction publication/collection and final-snapshot reset corrections.
The disposable Fortran source admits cached energy-only orientations; source
provenance is retained in `metadata.json` and `implementation.tar.gz`.

Executable SHA256:
`31ede34f14857ad4d495a0c4102639ccd43ea9440c2554bcb8a0c013ea5a881b`.
Native library SHA256:
`2bd581ea0a59a9c62294240ff34b4899305f42467bb397b88eb696f3a7f6c40a`.
Upstream remains pinned and unmodified; unrelated working-tree changes were
not copied wholesale into the image.

An initial CPU image build failed because `nonlocal_force_state.hpp` omitted
`<cstdint>`. Commit `cd201c34` fixes the direct dependency; 12 host lifecycle
and header tests pass. The failure, original recipe and log are preserved.
It started no GPU calculation. The corrected launch used the same fixed
deadline without extension.

The batch stopped because all prepared calculations completed. Both apps are
authoritatively STOPPED and the final check found zero account containers.
Their combined app lifetimes, including CPU image construction, total
**928.618437 s (15.48 min)**, a conservative bound below the approved 1200 s.
The final app stopped before the fixed deadline. Exact billed GPU seconds are
unknown. The allowance is closed; no further calculation is authorized here.
