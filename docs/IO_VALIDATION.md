# Deferred SCF diagnostics

Run [20261005T202238Z](../results/20261005T202238Z/REPORT.md) measures zero selected file operations during SCF for all five deferred-log candidates. All ten CPU/GPU comparisons, five direct GPU controls and the FePt2 MAE check pass. Forces, nine-geometry relaxation, both sampled-k SOC directions and Fe16 non-collinear calculations are covered. All 27 native checks pass; the local suite has 14 passing tests.

The change defers three routine convergence-reporter flushes until the SCF cycle ends, flushes pending startup text before the loop, and uses GNU Fortran's 1 MiB formatted-output buffer. All diagnostic text and numerical bookkeeping remain. The harness also buffers GPU samples in memory until SIESTA exits; GPU telemetry is present in every result. The default without `--defer-scf-log` retains the earlier flush behavior.

| Case | SCF iterations | All-SCF writes: control → deferred | Steady-state written bytes: control → deferred |
| --- | ---: | ---: | ---: |
| Fe2 forces | 25 | 26 → 0 | 4,610 → 0 |
| Fe2 relaxation | 96 | 105 → 0 | 17,314 → 0 |
| FePt2 SOC z | 35 | 37 → 0 | 6,310 → 0 |
| FePt2 SOC x | 32 | 34 → 0 | 5,800 → 0 |
| Fe16 non-collinear | 45 | 47 → 0 | 12,125 → 0 |

Candidate and control iteration counts match in every pair. All ten traces parsed completely with no phase-attribution errors. The controls write stdout and, in the SOC/non-collinear cases, flush an FDF log. The deferred candidates have no selected file reads, writes, opens, closes or metadata operations inside any SCF interval. Each still has one mapping call after the first iteration; mappings are a separate counter. Initial/final exports remain present, and regression still compares the full density grid, forces, moments, energies, eigenvalues, stress and geometry.

This is measured behavior for these runs. Finite buffers can spill on other workloads; the setting alone is not a no-I/O guarantee. Traces cover SIESTA and its child threads, not the external tracer or sampler. Memory-mapped page faults and asynchronous kernel writeback remain unmeasured. Electronic arrays and scalar status still cross the CPU/GPU boundary. Full residency is therefore still pending. No speedup is qualified from these traced timings.

The exact implementation, patch, harness, numerical checker and driver are preserved in `implementation.tar.gz`; all 20 recorded hashes were verified. The GPU container was released before the next performance batch.

## Isolated solver-control results

The revised solver probe ran each matrix in its own process on the same GPU, using real/complex matrices of sizes 64, 576 and 1152. All twelve ordinary warmups succeeded with device status zero. Both generalized divide-and-conquer and generalized Jacobi APIs then returned cuSOLVER status 7 during capture in every case. No graph or device loop was produced; node and numerical-loop measurements remain null. This result applies to the pinned CUDA 12.6 stack and tested APIs.

The subsequent experiment in `20261005T203343Z` explicitly separates Cholesky factorization, triangular reduction, standard Jacobi eigensolution and eigenvector back-transformation, retaining status on device between those operations. All six warmups succeed, but capture errors arise at the standard Jacobi eigensolve. No graph loop or numerical replay has been validated. The latest probe source additionally changes the Hamiltonian between iterations and checks full overlap orthogonality if capture succeeds; that extension awaits execution. The accepted production eigensolver remains unchanged.

## Later runtime read in combined tracing

The H/S and density interface batches `20261006T062841Z` and `20261006T065114Z` each record three steady relaxation filesystem calls: open, a one-byte read, and close of `/proc/sys/vm/overcommit_memory`. Both have complete syscall coverage; the access remains counted. It occurs at geometry 1, iteration 3 in the first batch. These newer runs do not establish zero steady filesystem I/O.

Read-only inspection of the subsequent grid-validation container finds that literal path in its loaded glibc binary. Glibc 2.35's [heap-shrinking implementation](https://raw.githubusercontent.com/bminor/glibc/glibc-2.35/malloc/arena.c) invokes a platform check; the [Linux helper](https://sourceware.org/pipermail/glibc-cvs/2022q1/077265.html) opens this sysctl and reads a character. This makes allocator activity a plausible explanation, but neither the binary match nor the current syscall trace identifies which application/profiler allocation triggered it. The trace has no caller stack. Preallocating bounded observation storage is a possible follow-up, not a validated fix; do not relabel or exclude these calls to produce a zero count.

The blocked solver/control probes have since passed their numerical and conditional memcpy/memset gates; see [their current evidence](BLOCK_JACOBI_EXPERIMENT.md). Those component traces do not measure filesystem I/O or qualify the complete production SCF loop.
