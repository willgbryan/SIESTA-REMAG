# Geometry lifetime validation

The grid bridge now uses explicit geometry lifetimes instead of hashing unchanged basis/topology arrays on every call. This is a validated ownership and performance improvement within the partial GPU implementation. It does not remove the remaining electronic-state transfers or establish a fully resident SCF loop.

The pinned reference, corpus, scientific gates, FP64 arithmetic and requested outputs are unchanged. Each process uses the default end-of-cycle DM checkpoint policy. One Modal A100 40 GB runs all work sequentially; CPU references use one BLAS thread. Wall times include setup, SCF, final forces and exports, and exclude cloud setup and artifact transfer.

## Scientific evidence

The full-content verification run [20261005T190557Z](../results/20261005T190557Z/REPORT.md) passed six CPU/GPU pairs and the FePt2 MAE check. It covers displaced forces, nine-geometry relaxation, Gamma non-collinear Fe, sampled-k SOC in both directions and Fe54. Debug hashing checked every epoch reuse. Relaxation staged nine basis layouts and verified 220 reuses; all 96 SCF markers reconciled with printed iterations and transfer counters. Nineteen native scenarios passed, including same-address mutation with proper invalidation and rejection when invalidation is missing.

The normal-mode run [20261005T191250Z](../results/20261005T191250Z/REPORT.md) passed all 24 upstream pairs, six direct old/new GPU comparisons and the FePt16 MAE check. Each case has three independent cold candidate processes and one content-hashing GPU control; each CPU reference and control was measured once. All transfers reconcile and all SCF marker counts agree. Its 19 native scenarios also passed.

Maximum errors across the 24 upstream comparisons:

| Observable | Maximum absolute error | Gate |
| --- | ---: | ---: |
| Energy (eV/atom) | 5.00e-7 | 1e-5 |
| Fermi energy (eV) | 1.00e-6 | 1e-5 |
| Eigenvalues (eV) | 2.919e-5 | 1e-4 |
| Forces (eV/Å) | 1.00e-6 | 1e-4 |
| Stress (eV/Å³) | 2.00e-6 | 1e-5 |
| Total moment (μB) | 5.00e-5 | 1e-4 |
| Atomic moment (μB) | 5.00e-6 | 1e-4 |
| Atomic charge (e) | 1.00e-6 | 1e-4 |
| Full density components (e/Bohr³) | 5.663e-6 | 1e-5 |
| Relaxed coordinates (Bohr) | 1.30e-8 | 1e-3 |

FePt16's direct E(x)−E(z) error is 1.000004e-6 eV against the unchanged 5e-6 eV gate. Final magnetic directions pass the one-degree checks. These are numerical-regression settings, not a production MAE convergence study.

## End-to-end timings

| Case | CPU seconds | Median GPU seconds | GPU range | CPU / median GPU | Hashing control seconds | Control / median GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fe2 relaxation | 28.489 | 16.407 | 16.322–16.527 | 1.736× | 17.315 | 1.055× |
| Fe54 collinear | 141.282 | 22.629 | 22.405–23.014 | 6.243× | 29.983 | 1.325× |
| Fe16 non-collinear | 132.467 | 26.268 | 26.073–26.490 | 5.043× | 29.442 | 1.121× |
| FePt16 SOC z | 121.134 | 38.418 | 37.781–39.121 | 3.153× | 41.397 | 1.078× |
| FePt16 SOC x | 110.784 | 34.812 | 33.813–34.996 | 3.182× | 37.453 | 1.076× |
| Fe128 collinear | 1403.406 | 100.354 | 91.527–107.566 | 13.985× | 160.134 | 1.596× |

These ratios compare converged, scientifically equivalent solutions, not identical SCF trajectories. Fe128 uses 98 CPU iterations, 83/66/75 candidate iterations and 89 control iterations. Its candidate mean SCF iteration times are 0.864/0.846/0.858 s versus 1.362 s for the control, supporting an iteration-cost improvement independently of iteration-count variation. FP64 atomic summation can change trajectories. Fe54 uses 33 CPU, 34/33/33 candidate and 33 control iterations; Fe16 non-collinear uses 45 throughout. The raw report preserves all other counts. The JSON ablation files compare the control to the first candidate; the table above compares it to the candidate median.

For the first Fe128 candidate, basis validation scans 212,229,416 host bytes once and avoids 36,079,000,720 bytes of repeated scanning across 170 epoch hits. Its grid bridge takes 12.571 s across 171 calls, versus 60.772 s across 183 calls with hashing. The cache still stages one basis layout in each process. Avoided host scans are not host/device transfer savings.

The new first-candidate exclusive profile is led by XC (36.813 s in `gridxc@cellXC`), final force contraction (20.195 s in `dfscf`), density-grid contraction (9.502 s), mixing (6.429 s) and density-matrix construction (6.339 s). XC and force migration remain substantial residency and performance work.

## Residency evidence and limits

The first Fe128 candidate still transfers 5,371,936,768 bytes host→device and 4,681,888,400 bytes device→host over iterations after the first, with 1,476 explicit copies. Across all 83 SCF iterations, D/E exports account for 2,827,853,824 downloaded bytes and grid outputs for 1,909,600,256. H/S uploads account for 3,524,788,224 bytes. These are successful explicit bridge copies only; library internals, migrations and file I/O were not traced in these runs.

Geometry lifetimes apply to immutable grid data. Mutable density still requires the existing host snapshot checks. Normalization, occupations, mixing, potentials, final forces and other consumers require further migration. The new opt-in device-normalization, syscall-trace and conditional-graph experiments postdate these measurements and are not validated by them.

Both evidence directories contain an `implementation.tar.gz` snapshot matching every recorded native, integration-patch and harness SHA-256 (16 files verified against the performance metadata). Preserve that snapshot to reproduce this milestone as development continues. The run completed and its Modal GPU container was confirmed released before the next batch started.
