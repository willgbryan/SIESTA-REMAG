# Local Metal eigensolver: accurate prototype, performance gate failed

Measured on the Apple M3 Max Mac on 9 November 2026 local time (10 November
UTC). The isolated solver runs complex FP32 cyclic Jacobi on Metal, followed
by CPU FP64 correction against the original operators. Production is unchanged.
**It is slower than native CPU FP64 for every measured configuration.**

Source: `2492745a` (`experiments/metal-eigensolver/`), with exact source and
library hashes in [summary.json](summary.json). CPU baseline: Accelerate's new
LP64 LAPACK interface, direct `ZHEGVD`, or cached-overlap `ZHEGST` / `ZHEEVD`
and triangular back-transformation, with persistent per-worker workspaces.
The native comparator passed independent saved-matrix checks before timing.

## Matched results

Five alternating CPU/hybrid timing pairs after one untimed warmup per path.
The table uses four CPU workers for both paths. Values are median wall times
for each complete batch, not kernel-only timings or full DFT calculations.

| Input batch | CPU FP64, fresh | Hybrid, fresh | CPU FP64, cached overlap | Hybrid, cached overlap |
| --- | ---: | ---: | ---: | ---: |
| FePt: 12 captures, 96 states | 7.05 ms | 40.46 ms | 6.90 ms | 38.47 ms |
| FeCo: 27 captures, 112 states | 23.73 ms | 107.77 ms | 23.12 ms | 100.85 ms |
| FePt: same captures repeated 8×, batch 96 | 52.64 ms | 227.45 ms | 51.32 ms | 225.06 ms |
| FeCo: same captures repeated 8×, batch 216 | 176.46 ms | 819.03 ms | 175.67 ms | 874.18 ms |

The fresh hybrid path takes 4.32–5.74 times the native CPU time in this table
(ratios of medians). These are slowdowns. Individual paired ratios and every
one-worker result are retained. Replicated inputs measure batching behavior;
they are not new physical k points or validated larger scientific capacity.

The first Python/SciPy CPU comparator showed little benefit from four job
workers. Replacing that comparator with direct native Accelerate task dispatch
materially improved CPU throughput. Both earlier reports are retained; the
native result above is the appropriate performance comparison. The hybrid's
FP64 preparation/correction still uses Python/NumPy/SciPy, so this result
characterizes the current implementation, not an intrinsic limit on Metal or
all mixed-precision algorithms.

## Accuracy and scope

- All **39 distinct source captures** pass after Metal seeding and FP64
  correction; they are selected k points and reference/rotation combinations
  from earlier FePt 64-k and FeCo 4096-k calculations.
- All **16 configurations / 80 matched timing pairs** pass the component
  checks. Each method solved 7,020 eigensystems across timed repetitions;
  these mostly repeat the same captures. No timed hybrid job used fallback.
- Maximum eigenvalue error: **1.9323e-10 eV**, against the existing 1e-7 eV
  component threshold. Maximum generalized relative residual: **3.366e-13**;
  maximum S-orthogonality error: **1.547e-12** (both gates 1e-10).
- At most **two FP64 corrections** were needed. Fixed-source-chemical-potential
  density-operator error was <=9.62e-13; this is a diagnostic, not full-mesh
  electron-count or SCF validation.
- **11 focused tests passed**, covering complex/repeated spectra, the saved
  matrices, cached/native CPU equivalence, invalid inputs/lifetime, and
  explicit fallback after forced seed/correction exhaustion.

There is no full-SCF, directional total/free-energy, moment-direction, force,
stress, physical MAE, or upstream-GPU performance qualification here. The
earlier frozen-potential bias failures remain failures. No new cloud allocation
or production calculation was performed.

## Where time goes

For the 96-matrix FePt batch, four workers, fresh factors, median stages are
33.58 ms preparation, 34.62 ms Metal seed wall time (30.21 ms GPU execution),
and 163.01 ms FP64 correction/back-transformation/acceptance. For FeCo's
216-matrix fresh batch they are 104.28, 142.31 (126.06 GPU), and 569.78 ms.
Medians of individual stages need not sum to the median total.

FP64 correction is the largest cost in these batches. For the small original
batches, even the GPU seed alone costs more than the entire native four-worker
FP64 solve. Moving correction to native code may remove interpreter/scheduling
cost, but does not remove its cubic matrix work. A faster Metal seed algorithm
is also needed; the current cyclic Jacobi method repeatedly accesses device
matrix storage and synchronizes within each threadgroup. No bandwidth or
occupancy measurements were collected, so neither is asserted as the measured
hardware bottleneck.

The next optimization candidates are a native batched correction path and a
tiled/block-Jacobi Metal seed. Measure those individually before extending the
backend. Software FP64 on Metal remains a separate, unmeasured arithmetic
experiment; it is not a demonstrated fix for this overhead. Production stays
on CUDA/FP64, and the MAE goal remains paused.

## Timing and evidence details

The timed hybrid call includes CPU standardization, FP32 packing, shared-buffer
copies, command submission/completion, CPU FP64 correction, original-H/S
acceptance checks, and any fallback. Baseline packing/copies and native task
dispatch are also timed. Both paths receive the same inputs and overlap-cache
opportunity. Cache-factor construction, shader compilation and persistent plan
allocation are separate setup measurements. Inputs are already in RAM; input
I/O and independent verification are excluded from both timed paths.

`VECLIB_MAXIMUM_THREADS=1` is set before process startup; independent jobs use
one or four CPU workers. The GPU runs one batched command buffer per solve.
Its elapsed time uses Apple's completed-command-buffer
[GPU timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime),
separately from wall time. Fast math is disabled. No IEEE FP64 emulation runs
on the GPU. Source H/S and saved reference outputs remain untouched.

Raw reports are gzip-compressed JSON, preserving every repetition, check,
correction history, setup time and source hash:

- [Initial matched check](benchmark-01.json.gz): 3 repetitions, original batches,
  Python/SciPy comparator.
- [Expanded Python comparator](benchmark-02.json.gz): 5 repetitions, original
  and repeated batches, one/four workers, fresh/cached factors.
- [Final native CPU comparator](benchmark-03-native.json.gz): the reported
  5-repetition benchmark.
- [Development outcomes](development-attempts.json): initial device-access
  failure, pre-normalization seed failures, fixes and passing checks.

The initial FP32 seed had column norm drift above the test's 1e-5 threshold.
Column normalization fixed it; the test threshold was unchanged. Raw source
captures and the CPU precision replay are linked through hashes in the reports.
