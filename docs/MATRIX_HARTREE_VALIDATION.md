# Device matrix assembly and periodic Hartree

[Run 20261005T235359Z](../results/20261005T235359Z/REPORT.md) passes all 27 upstream comparisons, 18 direct GPU controls, six MAE comparisons and 54 CUDA/filesystem coverage checks. All 38 native checks, 27 FD scenarios, 19 isolated Hartree scenarios and 26 local tests pass. The implementation archive verifies all 34 recorded hashes. The single GPU was released before the subsequent untraced run launched. This diagnostic uses tracing and debug ownership checks; its timings do not qualify speedups.

`--matrix-assembly` builds dense H/S directly in the solver's device buffers from sparse upstream matrices. It removes all dense H/S uploads inside the 351 candidate SCF iterations, across real/complex, collinear, NC and eight-component SOC paths. There are 11,266 device matrix builds, 351 balanced prepare/end lifetimes and 17 geometry uploads, including nine relaxation geometries. Sparse H still uploads once per diagonalization; overlap/topology/k geometry have geometry lifetimes. Host dense buffers remain allocated as opaque identifiers.

`--hartree` computes the periodic potential, energy and all nine stress components with a reusable device FFT/reduction plan. Its temporary adapter still uploads density and exports potential plus an 88-byte energy/stress/status record. This migrates a necessary consumer but does not itself remove the density-grid boundary. Device errors are checked before changing host outputs. CPU grid handling, PBE, mixing, final observables and scalar control remain required work.

## Scientific and ownership checks

Eight H/S native scenarios preserve periodic duplicate ordering, complex phases, Gamma/k spinor conventions and padded layouts. Changed H reuses geometry; a new epoch refreshes changed geometry. Debug mutation, missing/duplicate consumption, wrong shape/type and foreign buffer identities are rejected. NaN sentinels prove host dense buffers are not used or overwritten. Maximum eigenvalue, generalized residual and overlap-orthogonality errors against independent NumPy/SciPy assembly are 1.12e-15, 4.33e-15 and 4.50e-15 respectively.

Hartree uses the unmodified pristine SIESTA Poisson/GPFA/cutoff routines as its oracle. Six fixtures cover constant density, Fourier modes crossing the cutoff, random orthogonal/oblique grids, a left-handed cell, mixed odd/even dimensions and a 40×36×32 grid. Ordinary and captured execution both pass after two changing-density iterations on the same buffers. Every valid captured graph has 11 nodes and no forbidden host or non-device-copy nodes. Seven captured invalid-data/configuration cases retain the expected device errors. Maximum potential, energy and stress errors are 2.34e-15 Ry, 4.45e-15 Ry and 1.22e-17 Ry/Bohr³. Native bridge checks additionally pass geometry mutation/invalidation, aliased density/potential, unchanged host outputs on failure and subsequent error recovery. Capture is a component capability check, not full SCF residency certification.

| Case | SCF iterations CPU / candidate / CPU-matrix control / CPU-Hartree control | Device matrix builds | Avoided dense H/S upload bytes | Sparse H + geometry upload bytes |
| --- | --- | ---: | ---: | ---: |
| Fe2 forces | 25 / 25 / 25 / 25 | 1,400 | 14,515,200 | 12,047,000 |
| Fe2 relaxation | 96 / 96 / 96 / 96 | 5,376 | 55,738,368 | 51,419,448 |
| Fe2 NC Gamma | 10 / 10 / 10 / 10 | 10 | 414,720 | 112,784 |
| FePt2 SOC Gamma z | 36 / 36 / 36 / 36 | 36 | 1,492,992 | 755,600 |
| FePt2 SOC Gamma x | 38 / 38 / 38 / 38 | 38 | 1,575,936 | 797,072 |
| FePt2 SOC z | 35 / 35 / 35 / 35 | 1,890 | 78,382,080 | 61,981,344 |
| FePt2 SOC x | 32 / 32 / 32 / 32 | 1,728 | 71,663,616 | 56,753,184 |
| Fe16 NC | 45 / 45 / 45 / 45 | 720 | 1,911,029,760 | 326,445,344 |
| Fe54 collinear | 33 / 34 / 33 / 31 | 68 | 256,981,248 | 129,698,744 |

The avoided-byte column measures the dense uploads that the previous bridge would perform for these same calls. It is not a runtime comparison with a control taking a different number of iterations. Across candidates, 2,391,793,920 dense bytes are replaced by 640,010,520 sparse/geometry bytes. No H/S downloads occur. Debug mode deliberately scans host snapshots; normal-mode performance must be measured separately.

Maximum errors across the 27 upstream comparisons are 1.852e-8 eV/atom in energy, 1.0e-6 eV in Fermi energy, 1.0e-6 eV/Å in forces, 1.0e-6 eV/Å³ in stress, 1.0e-5 μB in total moment, 1.0e-6 μB in atomic moment, zero charge error at stored precision, 1.060e-6 eV in eigenvalues, 1.193e-7 e/Bohr³ in density and 6.1e-9 Bohr in relaxed coordinates. Gates and convergence limits are unchanged. Each candidate/control SOC x/z pair is checked directly for MAE and final moment direction: all six MAE errors are zero at stored precision. E(x)−E(z) is −0.007489 eV for sampled k and −0.021465 eV at Gamma; these small regression inputs are not production convergence studies.

## Residency evidence and next work

For equal-iteration Fe16 NC, steady-state explicit uploads decrease from 2,261,766,144 bytes in the CPU-matrix control to 704,292,864 bytes with device assembly, about 69%. Downloads remain 1,015,390,288 bytes, and copy calls decrease from 2,596 to 1,232. This is a measured boundary reduction, not a traced speedup claim. Fe54 takes differing numbers of iterations, so its totals do not isolate the feature.

The Hartree adapter adds 74,509,632 density-upload bytes, the same number of potential-download bytes and 30,888 result/status-download bytes during SCF across the nine candidates. Its arrays must be joined to device grid producers and consumers before those temporary copies can disappear. The Fe16 NC CPU-Hartree control has 687,869,952 steady-state upload bytes and 998,963,504 download bytes; this explicitly shows the adapter's added traffic.

Explicit accounting reconciles in every variant. CUPTI SCF uploads match those counters, and additional downloads remain exactly 16 bytes per solver call. Retained spectral E/Q arrays still cross no SCF host/device boundary. FD, normalization and solver status/result readbacks remain; launch-argument scientific values also need device ownership.

All 27 candidates/controls have zero selected SCF filesystem operations. Mapping counts range from 7 to 20. Mapped-file faults, asynchronous writeback, mapped-host accesses and unmeasured CPU library work still prevent strict residency/no-I/O certification. Final host exports and observable computation remain unfinished.

## Larger untraced validation

[Run 20261006T001018Z](../results/20261006T001018Z/REPORT.md) passes all 25 upstream comparisons, ten direct controls and five FePt16 MAE comparisons. All 41 archived source hashes verify. The isolated device PBE component also passes all 34 checks before SCF, with [component evidence retained separately](PBE_VALIDATION.md). The app completed and released its sole GPU before resident-grid validation began.

Each case has three independent cold candidate processes, one serial CPU reference and one control for each feature on the same host. Times include initialization, SCF, final forces and output, excluding cloud provisioning/builds/artifact transfer. Controls are single measurements; ratios against the candidate median are descriptive, not confidence intervals.

| Case | CPU seconds | Median GPU seconds (range) | CPU / median GPU | CPU-matrix control / median GPU | CPU-Hartree control / median GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fe54 collinear | 303.898 | 32.090 (29.629–34.495) | 9.47× | 0.96× | 0.93× |
| Fe16 NC | 162.411 | 23.852 (23.452–23.951) | 6.81× | 1.25× | 0.99× |
| FePt16 SOC z | 138.155 | 35.833 (35.823–37.026) | 3.86× | 1.20× | 1.06× |
| FePt16 SOC x | 121.977 | 33.875 (33.193–34.393) | 3.60× | 1.10× | 0.95× |
| Fe128 collinear | 1259.268 | 103.407 (103.273–103.483) | 12.18× | 1.01× | 1.04× |

SCF iterations (CPU / three candidates / CPU-matrix control / CPU-Hartree control) are 61 / 56,63,48 / 51 / 49 for Fe54; 43 / 43,43,43 / 43 / 43 for Fe16 NC; 43 / 45,42,42 / 42 / 43 for FePt16 z; 36 / 36,36,36 / 36 / 36 for FePt16 x; and 73 / 67,67,68 / 68 / 67 for Fe128. Matrix assembly benefits the equal-iteration spinor cases, but neither feature has a uniform incremental end-to-end benefit. Hartree is a necessary consumer migration; its temporary array transfers remain unfinished work.

Across all 25 upstream comparisons, maximum errors are 1.25e-7 eV/atom in energy, 1.0e-6 eV in Fermi energy, zero force error at stored precision, 1.0e-6 eV/Å³ in stress, 7.0e-5 μB in total moment, 8.0e-6 μB in atomic moment, 1.0e-6 e in atomic charge, 2.23e-5 eV in eigenvalues and 3.994e-6 e/Bohr³ in density. All unchanged gates pass. The CPU FePt16 MAE is 0.056830 eV; errors for all five candidate/control x/z pairs are at most 2.001e-6 eV, with final moment directions passing. These inputs remain regression fixtures, not production magnetic convergence studies.

This untraced batch does not certify strict residency or no-I/O behavior. The next diagnostic joins grid producers and consumers to remove the temporary density/potential copies. Capacity remains one GPU instance; learned SCF remains deferred.
