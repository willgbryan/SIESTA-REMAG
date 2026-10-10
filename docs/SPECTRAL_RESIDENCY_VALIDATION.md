# Eigenvalues and occupations retained through SCF

Run [20261005T223820Z](../results/20261005T223820Z/REPORT.md) passes all 18 upstream comparisons, nine direct GPU controls, both Gamma/sampled-k MAE checks and all 36 CUDA/filesystem trace checks. All 29 native checks, 27 isolated FD scenarios and 26 local tests pass. The source archive matches all 28 recorded hashes. Its single GPU was released before the subsequent larger run launched. Traced timings do not qualify speedups.

With `--resident-spectra --occupations --dm`, eigenvalue and occupation arrays stay on device through all 365 candidate SCF iterations. All 5,725 density consumers borrow E/Q from the bank. There are zero explicit SCF eigenvalue/occupation downloads and zero density-builder E/Q uploads. Seventeen explicit exports restore the latest host E/Q arrays after SCF: one per fixed geometry and nine across relaxation. This closes the spectral-array boundary for these supported paths; it does not establish full SCF residency.

## Mechanism and ownership

The solver already computes a full spectrum. Sampled-k density builders now retain all requested eigenvectors and let the existing device density kernel apply the unchanged occupation cutoff. This removes the host Q scan without adding a band-count readback. Gamma builders already used all requested vectors. The first-pass spectrum stays frozen during the second, vector-only solve, matching upstream's energy-density weighting convention.

Host E/Q buffers are opaque during SCF. The solver suppresses their downloads only inside an explicitly armed resident bank; unarmed analysis/wavefunction calls retain their original outputs. Each expected spectrum block must be produced and consumed before the bank closes. Successive SCF iterations can replace unpublished state only for the same owner, layout and geometry. Changing ownership or geometry without export fails explicitly.

The integration exports E/Q immediately after entering the post-SCF phase, before existing host restart/analysis work. The export checks size, pointer identity, completed consumers and geometry, and rejects duplicate exports. Debug mode detects host mutations before consumption, subsequent iterations and export. Normal mode does not scan those host arrays. Native checks keep sentinel host arrays unchanged through two changing SCF iterations, verify the final exported spectrum/occupations, and exercise nonzero occupations below the density cutoff.

These post-SCF exports preserve the current host-array contract and validation outputs. They are still a temporary boundary while final observables and subsequent workflows migrate to device ownership.

## Transfer evidence

The byte columns count downloads suppressed within each candidate, compared with the previous bridge's per-solve/per-FD export behavior for the same number of calls. They are not total pipeline savings against a control with a different iteration count. Explicit export bytes occur after SCF.

| Case | SCF CPU / candidate / control | Suppressed eigenvalue bytes | Suppressed occupation bytes | Post-SCF E/Q export bytes |
| --- | --- | ---: | ---: | ---: |
| Fe2 forces | 25 / 25 / 25 | 201,600 | 100,800 | 8,064 |
| Fe2 relaxation | 96 / 96 / 96 | 774,144 | 387,072 | 72,576 |
| Fe2 NC Gamma | 10 / 10 / 10 | 2,880 | 2,880 | 576 |
| FePt2 SOC Gamma z | 36 / 36 / 36 | 10,368 | 10,368 | 576 |
| FePt2 SOC Gamma x | 38 / 38 / 38 | 10,944 | 10,944 | 576 |
| FePt2 SOC z | 35 / 35 / 35 | 544,320 | 272,160 | 15,552 |
| FePt2 SOC x | 32 / 32 / 32 | 497,664 | 248,832 | 15,552 |
| Fe16 NC | 43 / 43 / 43 | 1,585,152 | 792,576 | 36,864 |
| Fe54 collinear | 61 / 50 / 70 | 388,800 | 388,800 | 15,552 |

Total suppressed spectral downloads are 6,230,304 bytes; post-SCF exports total 165,888 bytes. For equal-iteration force and Fe16 NC pairs, steady-state explicit copy counts decrease by 1,368 and 714, respectively, and downloaded bytes decrease by 290,304 and 2,322,432. Fe54's differing iteration counts prevent assigning its total-traffic or runtime difference solely to this feature. Retaining more vectors can also change work and memory costs, so transfer reduction is not a speedup claim.

CUPTI upload totals match explicit counters, with complete coverage in all candidates and controls. Each candidate still downloads 16 additional library bytes per eigensolver call beyond the explicit counters. The FD bridge still downloads its 32-byte result/status every iteration; solver and normalization status readbacks also remain.

No selected file operations occur within SCF in any candidate or control. Mapping calls remain: candidate all-SCF counts range from 8 to 24. The trace covers SIESTA and its child threads, not whole-container observer activity. Mapped-file page faults and asynchronous writeback remain unmeasured, so strict no-I/O certification stays false.

## Scientific agreement and remaining work

Maximum errors across the upstream comparisons are 1.852e-8 eV/atom in energy, 1.0e-6 eV in Fermi energy, 1.0e-6 eV/Å in forces, 1.0e-6 eV/Å³ in stress, 2.0e-5 μB in total moment, 1.94e-6 eV in eigenvalues, 1.193e-7 e/Bohr³ in density and 1.2e-8 Bohr in relaxed coordinates. Both MAE errors are zero at stored precision. All existing scientific tolerances are unchanged.

Bulk traffic elsewhere remains large: the Fe16 NC candidate still transfers 2,143,282,176 bytes to the GPU and 953,556,072 bytes back after its first SCF iteration. Dense H/S input, grid potential/density, D/E, CPU mixing, final forces and scalar control remain unfinished. Scientific values supplied in launch arguments also need device configuration ownership; memcpy traces do not measure those argument payloads. The [full residency target](GPU_RESIDENCY.md) remains active, and learned SCF remains deferred.

## Larger untraced validation

Run [20261005T225818Z](../results/20261005T225818Z/REPORT.md) completes with all 20 upstream comparisons, five direct spectral controls and the main FePt16 MAE comparison passing. All 29 native checks, 27 isolated FD scenarios and 28 archived source hashes pass. There is no CUDA or syscall tracing in this batch, so the timings below are eligible end-to-end measurements. The single GPU was released before the next diagnostic launched.

Three cold candidate processes run sequentially for each case against one serial CPU reference and one GPU control that exports E/Q every iteration. Hardware and inputs are unchanged. All 15 resident candidate processes retain E/Q through 733 SCF iterations and export once afterward (15 exports total). Explicit transfer accounting reconciles; SCF E/Q downloads and density-builder E/Q uploads are zero. Normal-mode host validation bytes are zero. This batch does not add independent evidence about library scalar copies, mapped memory or filesystem activity.

| Case | CPU seconds | Candidate seconds, three runs | Candidate median seconds | CPU / median | GPU control seconds | SCF iterations CPU / candidates / control |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Fe54 collinear | 127.402 | 20.706, 21.829, 20.698 | 20.706 | 6.153× | 22.043 | 33 / 33,35,33 / 35 |
| Fe16 NC | 124.267 | 24.311, 23.649, 23.961 | 23.961 | 5.186× | 23.785 | 45 / 45,45,45 / 45 |
| FePt16 SOC z | 113.078 | 33.853, 34.508, 34.618 | 34.508 | 3.277× | 35.426 | 47 / 47,47,49 / 49 |
| FePt16 SOC x | 101.781 | 31.069, 30.769, 31.070 | 31.069 | 3.276× | 30.573 | 42 / 39,39,39 / 39 |
| Fe128 collinear | 1149.427 | 95.498, 90.691, 100.345 | 95.498 | 12.036× | 101.753 | 98 / 80,71,86 / 86 |

These CPU comparisons measure the cumulative GPU implementation. They do not isolate this feature's contribution. Control/median ratios are 1.065, 0.993, 1.027, 0.984 and 1.066 respectively. Differing iteration counts confound several apparent improvements; even equal-iteration Fe16 NC and FePt16 x are slightly slower than their controls. There is no uniform incremental end-to-end gain from spectral residency.

Maximum upstream-comparison errors are 1.251e-7 eV/atom in energy, 2.0e-6 eV in Fermi energy, zero forces at stored precision, 2.0e-6 eV/Å³ in stress, 4.0e-5 μB in total moment, 4.0e-6 μB in atomic moment, 1.0e-6 in charge, 3.052e-5 eV in eigenvalues and 4.530e-6 e/Bohr³ in density. All gates are unchanged. Main FePt16 MAE is 0.056827 eV and agrees within 3.638e-12 eV; this run's driver did not separately gate paired repeat/control MAEs.

The harness allows 3,600 seconds for cases with at least 100 atoms and 7,200 seconds for the sequential batch. Other cases retain 900 seconds; the 200-iteration limit, convergence thresholds and scientific gates are unchanged. Capacity remains one GPU instance.
