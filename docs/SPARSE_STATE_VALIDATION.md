# Resident sparse SCF validation

Run **20261006T034347Z** passes 18 upstream comparisons, nine resident-grid controls, four FePt MAE comparisons and all 36 CUDA/filesystem coverage checks. The supported conventional route now keeps sparse H/Hold/D/Dold/E/Eold, Pulay history, spectra/occupations and real-space density/potential on the GPU during SCF. This is an electronic-array milestone; scalar downloads, initial setup and CPU final observables remain.

[Scientific report](../results/20261006T034347Z/REPORT.md), [machine-readable ownership/transfer summary](../results/20261006T034347Z/sparse-state-summary.json), [source verification](../results/20261006T034347Z/source-verification.json). All 73 archived hashes verify, as do the retained owner/convergence arrays and owner wrappers. The single Modal A100 was released before the next probe launched.

## Scope and scientific agreement

The batch requested one A100 40 GB; the retained hardware inventory identifies the supplied device as an A100-SXM4-80GB with NVIDIA driver 580.95.05. It uses serial SIESTA and one BLAS thread. Every candidate enables resident spectra, density normalization, device matrix assembly, resident Hartree/PBE grid, sparse ownership, epoch verification and deferred logs. The control differs by keeping sparse-state consumers on CPU. Source, pseudopotentials, FDF files, raw output and traces are retained. Instrumented times do not qualify speedups.

The nine cases cover displaced forces, a nine-geometry relaxation, non-collinear Gamma and sampled-k Fe16, Gamma and sampled-k FePt SOC along x/z, and Fe54. All 349 candidate SCF iterations have exactly one density publication, seed/grid/H completion, scalar reduction and H/density borrow. Seventeen electronic cycles have seventeen setup/export pairs; 332 nonconverged steps mix. All eight reused relaxation geometries preserve upstream's disabled first mix while still updating Pulay history. The [integration record](SPARSE_STATE_INTEGRATION.md) explains the corrected adapter defects and retained failures.

Maximum upstream comparison errors across candidates and controls are 1.852e-8 eV/atom for energy, 1.0e-6 eV for Fermi level, 1.08e-6 eV for eigenvalues, 1.0e-6 eV/Å for forces, 1.0e-6 eV/Å³ for stress, 1.0e-5 μB for total moment, 1.0e-6 μB for atomic moments, zero at stored precision for atom charges, 1.193e-7 e/Bohr³ for density, and 1.6e-8 Bohr for relaxed coordinates. All original gates pass. Both variants reproduce the reference E(x)−E(z) at stored precision: −0.021465 eV at Gamma and −0.007489 eV with sampled k points. These small regression inputs are not production MAE convergence studies.

## Measured boundaries

CUDA schema 2 uses external correlation to associate each event with its launching API's phase. The known-copy calibration passes, every candidate/control trace has complete correlation coverage, and no event uses timestamp fallback. The candidate traces contain 365 events whose timestamp interval disagrees with their launch tag. GPU execution can outlast a host region; these counts describe launch attribution. Unified-memory migrations and kernel access to mapped host memory remain unmeasured.

Steady state means iterations after the first in each electronic cycle. File operations exclude mapping syscalls; the last column shows selected file operations / mappings. Missing evidence is never interpreted as zero.

| Case | Candidate / control iterations | Cycles | Steady H2D bytes | Steady D2H bytes | Steady file operations / mappings |
| --- | ---: | ---: | ---: | ---: | ---: |
| fe2_forces | 25 / 25 | 1 | 0 | 36,668 | 0 / 6 |
| fe2_relax | 96 / 96 | 9 | 0 | 132,900 | 0 / 13 |
| fe2_noncollinear_gamma | 10 / 10 | 1 | 0 | 3,848 | 0 / 0 |
| fept2_soc_gamma_z | 36 / 36 | 1 | 0 | 14,976 | 0 / 0 |
| fept2_soc_gamma_x | 38 / 38 | 1 | 0 | 15,832 | 0 / 0 |
| fept2_soc_z | 35 / 35 | 1 | 0 | 50,588 | 0 / 14 |
| fept2_soc_x | 32 / 32 | 1 | 0 | 46,124 | 0 / 11 |
| fe16_noncollinear | 46 / 45 | 1 | 0 | 32,756 | 0 / 5 |
| fe54_collinear | 31 / 31 | 1 | 0 | 13,436 | 0 / 3 |

All four targeted sparse-state boundaries are absent inside candidate SCF: H upload for matrix assembly, sparse D upload for the grid, grid H download and density/energy-density output downloads. Their control counterparts total **1,204,988,416 H2D bytes and 1,748,334,272 D2H bytes**. Fe16 has 46 candidate versus 45 control iterations, so these are measured control volumes, not an equal-iteration timing claim.

All candidate SCF D2H copies are 4, 8, 32, 144 or 224 bytes. The remaining **347,128 steady D2H bytes** come from explicit scalar/status boundaries plus library activity. The owner still exports 144 bytes per step and four bytes per mix; grid and FD adapters export 224 and 32 bytes per step. Normalization and eigensolver status downloads remain. Immutable S/topology/geometry staging and CUDA cache reads still occur on first use. Explicit final matrix exports feed CPU force/stress and analysis work after each cycle.

The coupled production-owner preflight passes 37 GPU scenarios and rejects 3,071 invalid ownership operations. The separate device-convergence plan passes 92 scenarios, but production SCF still uses host convergence. Other numerical preflights pass unchanged.

## Larger untraced qualification

Completed run [20261006T042922Z](../results/20261006T042922Z/REPORT.md) passes all 12 upstream comparisons, three direct sparse-state controls and four same-instance FePt16 MAE checks. All 77 archived source hashes verify. The earlier interrupted run [20261006T040628Z](../results/20261006T040628Z/PARTIAL_SCOPE.md) provides complete Fe54 and Fe16 groups; its missing work remains explicitly unqualified. Together these five complete case groups provide 20 upstream comparisons and five direct controls. The [machine-readable performance summary](../results/20261006T042922Z/sparse-performance-summary.json) retains individual result hashes, iteration counts, owner counters and explicit transfer totals. Measurements from different allocations are not pooled as repetitions of one case.

| Case | Run suffix | Candidate median (range), s | Serial CPU / median | GPU control / median | CPU / candidate / control SCF iterations |
| --- | --- | ---: | ---: | ---: | --- |
| Fe54 collinear | 040628Z | 15.077 (15.001–15.136) | 8.743× | 1.035× | 33 / 33,33,34 / 33 |
| Fe16 non-collinear | 040628Z | 16.026 (15.941–16.032) | 8.372× | 1.105× | 45 / 46,46,46 / 45 |
| FePt16 SOC z | 042922Z | 22.358 (22.357–22.370) | 5.351× | 1.214× | 47 / 47,47,47 / 48 |
| FePt16 SOC x | 042922Z | 21.434 (21.251–21.546) | 5.070× | 1.166× | 42 / 39,40,41 / 39 |
| Fe128 collinear | 042922Z | 55.773 (55.248–62.419) | 24.798× | 1.321× | 98 / 75,105,79 / 98 |

Each candidate has three cold process repetitions and one CPU reference/control on the same allocation. Timings include initialization, SCF, final forces and requested output, but exclude cloud provisioning/build/artifact transfer. These compare against serial upstream with one BLAS thread, not an optimized parallel CPU implementation. Iteration differences materially affect end-to-end ratios, especially Fe128. Its candidate mean SCF iteration times are 0.253–0.256 s versus 0.387 s for the control; unchanged scientific thresholds still pass, but the ratio is not an equal-trajectory experiment.

Across these 20 comparisons, maximum errors are 1.2501e-7 eV/atom in energy, 1.0001e-6 eV in Fermi energy, 2.9751e-5 eV in eigenvalues, zero at output precision in force, 2.0001e-6 eV/Å³ in stress, 4.0001e-5 μB in total moment, 4.0001e-6 μB in atomic moments, 1.0001e-6 e in atomic charges and 5.1857e-6 e/Bohr³ in density. The four completed FePt16 candidate/control MAE errors are at most 1.000004e-6 eV against the unchanged 5e-6 eV gate; all final-axis checks pass. Reference E(x)−E(z) is 0.056827 eV at stored precision. These remain regression settings, not production magnetic convergence studies.

This batch validates the larger owner/scientific path and repeated performance. It is deliberately untraced and therefore does not extend the independent phase-resolved zero-array-transfer claim beyond the nine-case diagnostic above. Scalars and CPU final observables remain required migrations. Its GPU allocation was released before the next solver diagnostic started.

## Next qualification

Ordinary graph replay calibration and 48 small batched-eigensolver scenarios now pass in a subsequent isolated run. The generalized blocked solver and conditional-loop trace attribution remain experimental; see [the retained failures and next gates](BLOCK_JACOBI_EXPERIMENT.md). Remove the remaining library/scalar control boundaries without altering convergence or mixing semantics. Fully resident final observables and geometry workflows remain required before learned SCF acceleration begins.
