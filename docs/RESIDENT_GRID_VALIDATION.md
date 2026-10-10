# Resident density/potential grid validation

Run `20261006T014308Z` qualifies the opt-in combined density → periodic Hartree/PBE → sparse-potential path for the tested serial corpus. All **18 upstream comparisons, nine direct controls, four MAE checks and 36 CUDA/filesystem traces pass**. All 57 retained implementation/harness hashes verify. This is a grid-stage milestone; sparse H/D/E, scalar control and final forces remain outside full device ownership.

[Machine-readable summary](../results/20261006T014308Z/grid-flow-summary.json), [full report](../results/20261006T014308Z/REPORT.md), [source verification](../results/20261006T014308Z/source-verification.json). Raw outputs, arrays, CUDA activity and syscall logs are retained.

## Scientific and ownership evidence

| Case | SCF CPU / resident-grid / control | Grid geometry setups |
| --- | ---: | ---: |
| fe2_forces | 25 / 25 / 25 | 1 |
| fe2_relax | 96 / 96 / 96 | 9 |
| fe2_noncollinear_gamma | 10 / 10 / 10 | 1 |
| fept2_soc_gamma_z | 36 / 36 / 36 | 1 |
| fept2_soc_gamma_x | 38 / 38 / 38 | 1 |
| fept2_soc_z | 35 / 35 / 35 | 1 |
| fept2_soc_x | 32 / 32 / 32 | 1 |
| fe16_noncollinear | 43 / 43 / 43 | 1 |
| fe54_collinear | 61 / 120 / 48 | 1 |

The relaxation case covers nine geometry cycles. Each of the 435 candidate SCF iterations has exactly one density producer, potential solve, potential consumer and completed grid lifetime. All 17 geometry setups occur once per SCF cycle; no immutable grid geometry is restaged after its first iteration. Debug byte checks were enabled.

Across all upstream pairs, maximum errors are 1.86e-8 eV/atom in total energy, 1.0e-6 eV in Fermi energy, 1.6e-6 eV in eigenvalues, 1.0e-6 eV/Å in forces, 1.0e-6 eV/Å³ in stress, 1.0e-5 μB in total moment, 3.0e-6 μB in atomic moments, zero stored atomic-charge difference, 2.39e-7 e/Bohr³ in density components and 7.0e-9 Bohr in relaxed positions. All existing thresholds remain unchanged.

Both candidate and control reproduce the Gamma MAE of −0.007489 eV and sampled-k MAE of −0.021465 eV at stored precision, with passing moment-direction checks. These small regression inputs do not establish production magnetic convergence.

The preflight also passes 38 native checks, 27 occupation checks, 19 Hartree checks, 34 PBE checks, 32 combined grid checks, 39 scalar-consumer checks and 24 device Pulay scenarios plus three reciprocal probes. Scalar/Pulay plans are still standalone components.

## Transfers and I/O

No SCF density-grid download, potential-grid upload or standalone Hartree density/potential/result boundary occurs in the resident path. The grid bridge records **414,771,840 avoided density-download bytes and the same number of avoided potential-upload bytes** across the candidate iterations. Its remaining array output exactly equals the uploaded sparse H matrix size. The combined grid exports exactly 224 bytes of energy/stress/status each iteration: 97,440 bytes in total.

Important remaining explicit SCF boundaries are:

| Boundary | Host → device bytes | Device → host bytes |
| --- | ---: | ---: |
| sparse_hamiltonian_input | 911,240,096 | 0 |
| grid_sparse_input | 911,240,096 | 0 |
| grid_output | 0 | 911,240,096 |
| dm_outputs | 0 | 1,763,331,840 |
| grid_flow_geometry | 7,188,832 | 0 |
| grid_flow_result | 0 | 97,440 |
| occupation_result | 0 | 13,920 |
| normalization_status | 0 | 1,740 |
| solver_status | 0 | 45,624 |

CUDA activity additionally records 24 H2D bytes per candidate process during its first SCF iteration. None of those extra uploads recur in steady state. Ordinary library D2H activity remains 16 bytes per solver call. In relaxation, one 4,032-byte spectral export touches the SCF/post-SCF timestamp boundary and is conservatively counted in both phases. The retained summary does not subtract this overlap or claim exact phase attribution for it.

Each candidate makes 39 selected filesystem calls to `/root/.nv/ComputeCache` during the first SCF iteration: 351 calls and 637,001 read bytes across nine processes. There are **zero selected file operations after the first SCF iteration of each geometry**, but 38 mapping calls remain across candidates. First-use plan/JIT preparation should move before the loop. Page faults, asynchronous writeback, unified-memory migration and mapped-host accesses remain outside the observer coverage; strict no-I/O and strict residency are not certified.

## Performance and next integration

The nine-case diagnostic used CUPTI, syscall tracing and debug ownership verification, so those timings do not qualify speedups. Fe54 needed 120 resident-grid iterations versus 61 upstream and 48 in the direct control; that trajectory difference is retained without changing tolerances.

The larger **untraced batch `20261006T020310Z` is complete**. All 20 upstream pairs, five grid-disabled controls and four FePt16 MAE comparisons pass, and all 62 source hashes verify. Each case has three independent cold candidate processes, one CPU reference and one control on the same single Modal A100. Times include initialization, SCF, final observables and requested outputs, and exclude cloud provisioning/builds/artifact transfer. [Full report](../results/20261006T020310Z/REPORT.md), [machine-readable summary](../results/20261006T020310Z/grid-performance-summary.json), [source verification](../results/20261006T020310Z/source-verification.json).

| Case | CPU seconds | GPU median (range) seconds | CPU / GPU median | Grid-disabled control seconds | SCF CPU / GPU repeats / control |
| --- | ---: | ---: | ---: | ---: | --- |
| Fe54 collinear | 376.995 | 22.765 (21.958–23.285) | 16.56× | 47.086 | 61 / 54, 49, 59 / 79 |
| Fe16 noncollinear | 186.804 | 22.289 (21.851–22.721) | 8.38× | 26.525 | 43 / 43, 43, 43 / 43 |
| FePt16 SOC z | 163.218 | 34.245 (34.044–34.862) | 4.77× | 40.091 | 43 / 42, 42, 43 / 42 |
| FePt16 SOC x | 140.323 | 32.429 (32.418–32.436) | 4.33× | 37.614 | 36 / 36, 36, 36 / 36 |
| Fe128 collinear | 1556.599 | 83.470 (82.725–83.726) | 18.65× | 123.085 | 73 / 68, 68, 67 / 69 |

Control/median-candidate ratios are 2.07×, 1.19×, 1.17×, 1.16× and 1.47× respectively. These are measured end-to-end ratios; differing iteration counts, particularly Fe54, prevent attributing the entire gain to faster per-iteration computation. The CPU reference is serial with one BLAS thread, not a tuned parallel CPU comparison.

The largest upstream-pair errors are 1.26e-7 eV/atom in energy, 1.01e-6 eV in Fermi level, zero stored force difference, 1.01e-6 eV/Å³ in stress, 7.01e-5 μB in total moment, 7.01e-6 μB in atomic moments, 1.0e-6 in atomic net charge, 2.163e-5 eV in eigenvalues and 3.756e-6 e/Bohr³ in density. Every existing gate passes. FePt16 reference MAE is 0.056830 eV; candidate repeats differ by 0, 1 and 2 micro-eV, and the control agrees at stored precision. No scientific tolerance was loosened.

This batch also passes all 52 standalone Hamiltonian seed/SOC GPU checks. That component is not integrated in the measured performance image. The batch provides no new CUDA/filesystem trace coverage and does not qualify strict residency.

The next production change is [persistent sparse H/D/E ownership](SPARSE_STATE_INTEGRATION.md), connecting device Hamiltonian seeds, energy/residual/charge consumers and stable Pulay. Final force/stress consumers and scalar loop termination remain subsequent requirements. Learned SCF work stays deferred.

## Failed precursor retained

Run `20261006T013412Z` passed all isolated components but rejected ordinary SIESTA inputs before SCF because the adapter treated `QMMM_does_pcpot()` as evidence of active MM charges. Upstream enables that option with an empty charge list and immediately returns from the potential routine when the count is zero. The corrected guard rejects actual nonzero `mm_charges%n`; it does not enable QMMM physics. The failed run and its source hashes remain in `results/20261006T013412Z`.
