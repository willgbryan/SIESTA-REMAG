# Device charge normalization and filesystem audit

Run [20261005T200824Z](../results/20261005T200824Z/REPORT.md) validates the opt-in device charge-normalization path against pinned upstream SIESTA. All 16 upstream comparisons, eight direct GPU normalization controls and two MAE checks passed. The batch covers displaced Fe forces, nine-geometry relaxation, Gamma non-collinear Fe, both Gamma and sampled-k FePt SOC directions, and Fe54. The same single Modal A100 executes all cases sequentially; no additional GPU instance is used.

The 27 native numerical/lifetime checks and 14 local tests pass. Native checks cover independent charge/scaling results for all six density-builder layouts, bad-charge rejection before host export, one-use request handling, and overlap changes with and without geometry invalidation. The debug epoch verifier is enabled throughout the DFT batch. All explicit transfer totals and SCF phase counts reconcile.

| Case | CPU / GPU SCF iterations | Device normalizations | Overlap uploads | Steady-state file writes | Steady-state written bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fe2 forces | 25 / 25 | 25 | 1 | 25 | 4,610 |
| Fe2 relaxation | 96 / 96 | 96 | 9 | 96 | 17,314 |
| Fe2 NC Gamma | 10 / 10 | 10 | 1 | 10 | 2,396 |
| FePt2 SOC Gamma z | 36 / 36 | 36 | 1 | 36 | 6,480 |
| FePt2 SOC Gamma x | 38 / 38 | 38 | 1 | 38 | 6,820 |
| FePt2 SOC sampled-k z | 35 / 35 | 35 | 1 | 35 | 6,310 |
| FePt2 SOC sampled-k x | 32 / 32 | 32 | 1 | 32 | 5,800 |
| Fe54 collinear | 61 / 64 | 64 | 1 | 65 | 15,364 |

Steady state means iterations after the first in each geometry. All 336 candidate SCF iterations normalize on device. The initial-state normalization remains upstream work. Subsequent SCF charge traces and scale factors remain on the GPU, followed by scaling of every D/E component. Normalized D/E still export for the remaining CPU consumers. A four-byte error status is read back each time: 1,344 bytes across these candidate loops. The overlap uploads once per geometry. This is consumer migration, not elimination of the D/E download boundary or strict scalar residency.

Across the 16 upstream pairs, the largest energy error is 1.852e-8 eV/atom, eigenvalue error 2.180e-6 eV, whole-cell moment error 2.0e-5 μB, force error 1.0e-6 eV/Å and density-component error 1.192e-7 e/Bohr³. Both direct MAE errors are zero at the stored output precision; magnetic directions and relaxed geometry pass their existing gates. Tolerances are unchanged. The subsequent [deferred-log batch](IO_VALIDATION.md) also validates normalized Fe16 non-collinear. FePt16, Fe128 and untraced performance are covered by the subsequent completed batch below.

## File-I/O findings

Every candidate and control trace has complete parsing and phase coverage. No selected steady-state read calls are observed. Stdout writes remain in every loop; Fe54 also flushes an FDF log during steady state. Several first iterations flush FDF startup/option records. One steady-state mapping call remains per candidate, separate from the file-operation counts. Anonymous/device mappings, memory-mapped page faults and asynchronous kernel writeback must not be conflated with measured read/write calls.

The trace covers SIESTA and its threads. The tracer and harness sampler are external observers, so this is not whole-container I/O accounting. All speedup fields are deliberately null: syscall tracing affects timing. No performance gain is claimed for normalization from this batch.

These findings motivated the subsequent deferred-diagnostic experiment, whose changes to stdout flush timing, Fortran buffering and sampler output passed their own [traced regression](IO_VALIDATION.md). Finite buffering alone cannot establish a no-I/O guarantee.

## Solver graph experiment

The separate six-case graph probe failed to demonstrate capture with the existing generalized divide-and-conquer solver. Three cases returned cuSOLVER status 7 during capture; alternating later cases returned status 6 during warmup in the shared process. Those later failures may reflect contaminated library state and are not independent compatibility measurements. The scientific regressions ran in separate fresh processes and all passed.

The subsequent [fresh-process probes](IO_VALIDATION.md) explicitly check warmup status and also try generalized Jacobi and a decomposed Cholesky/Jacobi route. They reproduce capture failures after successful warmups. Neither successful capture nor graph-node inspection alone could certify full-library or SCF residency.

The run's `implementation.tar.gz` matches all 20 recorded implementation, harness, checker and driver hashes. Raw syscalls, transfer intervals, numerical checks, stdout/stderr, profiles and ledgers are retained. The preceding CUDA graph-probe compilation error is retained under `results/build-failures/`; its corrected source built successfully before this run. The GPU container was confirmed released before the following experiment began.

## Larger workloads and untraced performance

Run [20261005T203343Z](../results/20261005T203343Z/REPORT.md) passes 20 upstream comparisons, five direct GPU normalization controls and FePt16 MAE. Each case has three cold GPU candidates, one CPU reference and one GPU control with normalization left on the CPU. All run sequentially on the same A100 host, with one BLAS thread, deferred SCF logging and no syscall/CUPTI tracing. Every candidate SCF iteration performs device normalization: 753 across the 15 candidates. All 27 native checks pass and the source archive matches all 20 recorded hashes.

| Case | CPU seconds | GPU median (range), seconds | CPU / median GPU | GPU host-normalization control, seconds | Control / median GPU | SCF iterations: CPU / candidates / control |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fe54 collinear | 127.328 | 21.507 (21.298–21.610) | 5.920× | 21.303 | 0.991× | 33 / 35,33,35 / 33 |
| Fe16 non-collinear | 125.045 | 24.634 (24.329–24.714) | 5.076× | 24.436 | 0.992× | 45 / 45,45,45 / 45 |
| FePt16 SOC z | 112.149 | 35.047 (34.200–35.364) | 3.200× | 35.422 | 1.011× | 47 / 49,47,49 / 49 |
| FePt16 SOC x | 102.167 | 31.205 (30.869–31.875) | 3.274× | 31.267 | 1.002× | 42 / 39,39,41 / 40 |
| Fe128 collinear | 1151.876 | 98.783 (95.164–104.272) | 11.661× | 92.633 | 0.938× | 98 / 91,77,83 / 73 |

Normalization is a validated consumer migration, with no demonstrated end-to-end speed benefit in this batch. Small-case differences are approximately flat. Fe128's candidate median is 6.6% slower than its single GPU control and takes more iterations; its mean SCF iteration times are 0.770–0.792 s versus 0.800 s for the control. These observations do not isolate a normalization speedup. The overall CPU/GPU ratios compare against a serial reference and must not be attributed to normalization alone or compared directly across different hosts/runs.

The maximum upstream-pair errors are 1.250e-7 eV/atom in energy, 2.986e-5 eV in eigenvalues, 5.0e-5 μB in total moment and 5.305e-6 e/Bohr³ in density. FePt16 MAE differs by 1.000004e-6 eV. Every existing gate passes without tolerance changes.

The first Fe128 candidate still transfers 5,896,028,160 bytes to the device and 5,138,658,360 bytes to the host after the first SCF iteration, across 1,710 explicit copies. All-SCF D/E exports account for 3,100,418,048 bytes, grid output for 2,093,658,112 bytes, H/S upload for 3,864,526,848 bytes, and normalization error readbacks for 364 bytes. Library transfers remain unknown in this batch. Its largest exclusive components are XC (36.127 s), forces (19.495 s), density update (17.093 s), Hamiltonian construction (14.828 s), and mixing (5.510 s).

All 18 isolated generalized/decomposed solver warmups succeed, but graph capture fails in every case. The decomposed failures occur at standard Jacobi. This run predates the changing-Hamiltonian and generic-batched solver probes. The app completed and its container was confirmed released before the queued CUDA-activity diagnostic started.
