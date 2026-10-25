# Selected benchmark evidence

These existing records are copied verbatim from completed experiments. The
[manifest](manifest.json) records their original local paths and SHA-256 hashes.
No new calculations were performed to populate this directory.

## Energy-only full SCF, 25 October 2026

[Qualification](mae-energy-only-c-20261025T200237Z/qualification.json) covers
FePt x/z against pristine CPU SIESTA and all five FePt/FeCo directions against
the ordinary REMAG path. Electronic science, requested density/matrix exports,
final directions and total/free directional energies pass. The largest
ordinary-REMAG directional discrepancy is 0.006401 micro-eV/f.u.; the CPU
FePt total-energy difference agrees within 0.584 micro-eV/f.u., limited by
its six-decimal energy output. These are numerical comparisons at fixed inputs.

The [per-calculation record](mae-energy-only-c-20261025T200237Z/energy-only-analysis.json)
shows FeCo orbital storage falling from 6,468,547,416 to 137,894,360 bytes:
6,330,653,056 derivative bytes are no longer allocated. This is one owner's
allocation change, not a measured whole-device peak-memory reduction. Timings
are individual observations with different output scope and varying SCF
counts; no repeatable runtime gain or upstream-GPU speedup is established.

The [final regression](mae-default-check-20261025T201839Z/summary.json) checks
the default force/stress path, rechecks energy-only execution after the shared
export-initialization fix, and confirms relaxation rejection before SCF.
All 131 local tests and ten local science/export comparisons pass. Five serial
apps, including two failed harness/validator attempts, total 31.85 minutes of
app lifetime, conservatively bounding GPU use below the authorized hour.
All allocations are released; billed GPU time is unknown. Original failed
summaries and the separate validator diagnosis remain alongside the passes.

FePt used 4³ k sampling/400 Ry/0.01 Ry confinement; FeCo used 16³/2000 Ry
nominal/0.00125 Ry. Both used full SOC, TZDP, 300 K and default FP64. FeCo's
4096-k execution used staged capacity changes and does not expand the
production 128-k guard. Physical convergence and broader capacity remain open.

## MAE profiling, 25 October 2026

The [FeCo profile](mae-profile-20261025-20261025T193930Z/profile-analysis.json)
retains three full-SCF direction runs at 16³ k sampling, 300 K smearing,
TZDP/0.00125 Ry confinement and nominal 2000 Ry (actual 80³ grid).
Assembly/eigensolving takes about 70–72% of the profiled electronic cycle;
geometry/H0 setup takes about 25–26 seconds per direction. Final force work
takes about 1.3–1.4 seconds but retains 6,330,653,056 bytes of orbital
derivatives. This motivates removing derivative storage and avoiding repeated
SCF/setup work. Profiling perturbs timing; these are individual observations,
not paired speedups or physically converged anisotropy. Raw result hashes,
source identities and allocation release are retained with the record.

## H100 native-architecture qualification, 12 October 2026

[Hopper qualification](hopper-qualification-v187-20261012T214738Z/qualification.json) records 72 passing
solver checks and ten passing material calculations across eight inputs,
including nonparallel SOC, FePt128 and the stronger DZP FePt x/z pair. The
SM80/SM90 native library ran on one H100 with PTX JIT disabled and direct
retained pristine-CPU science/export/convergence comparisons. Both directional
checks pass. The GPU allocation is released.

This is numerical portability coverage. Timings are individual observations;
there is no new upstream-GPU speedup, A100/H100 comparison or physical
convergence claim. See [Hopper support](../../docs/HOPPER.md) for the tested
configuration and limits. Raw outputs remain in the local archive.

## Expanded magnetic campaign, 10 October 2026

The [campaign summary](easy-magnetic-campaign-v181-20261010T060613Z/summary.json)
and [per-family results](easy-magnetic-campaign-v181-20261010T060613Z/workflow-results.json)
record four qualified comparisons across three workflows. Each finalist has
three fresh upstream GPU repetitions and three REMAG repetitions; the two Fe54
families share the same three REMAG runs.

| Workload / upstream GPU family | Upstream median | REMAG median | Speedup |
| --- | ---: | ---: | ---: |
| Fe54 collinear / ELSI ELPA | 104.237 s | 7.426 s | 14.037× |
| Fe54 collinear / native ELPA | 129.122 s | 7.426 s | 17.388× |
| Fe3C collinear / native ELPA | 144.065 s | 2.839 s | 50.736× |
| Fe3C noncollinear / native ELPA | 360.351 s | 3.255 s | 110.694× |

All contenders ran sequentially on one A100-SXM4-80GB, driver 580.95.05, with
16 reserved CPU cores. REMAG uses the v178 numerical binary, one thread and
default tile modes. All upstream finalists use two-stage ELPA. Fe54 ELSI and
Fe3C noncollinear use four MPI ranks, one thread per rank and block size 32;
the two native collinear finalists use one rank, four threads and block size 64.
These settings were screened, without a claim of exhaustive tuning.

All 21 timed runs pass [direct pristine-CPU checks](easy-magnetic-campaign-v181-20261010T060613Z/upstream-comparison.json)
of science, requested exports and convergence. Independent diagnostics verify
upstream GPU execution. Exact inputs, source/binary/pseudopotential identities
and runtime settings are retained in the [metadata](easy-magnetic-campaign-v181-20261010T060613Z/metadata.json).
The [49-calculation ledger](easy-magnetic-campaign-v181-20261010T060613Z/ledger.json)
includes all screening failures, diagnostics and timed repetitions.

Coverage is incomplete, and the full-suite aggregate fields are **null**:

- Fe54 collinear and Fe3C noncollinear pass every required solver-family gate.
  Fe54's primary ratio uses the faster passing ELSI finalist. Its separate
  eight-rank/block-64 ELSI screen reached the 200-SCF cap; that failure does not
  invalidate the historical passing ELSI comparison below.
- Fe3C collinear qualifies against native ELPA only. Both valid-layout ELSI
  screens (16 ranks, block size 8) converged in 65 iterations but failed the
  required eigenvalue-output check. Their timings do not qualify for speedup.
- Fe2B full-SOC x/z completed screening and GPU diagnostics but no timed
  repetitions. FePt16 full-SOC x/z, Fe16 noncollinear and Fe5PB2 full-SOC x/z
  were unstarted when the compute budget ended. This campaign adds no repeated
  full-SOC or directional MAE result.

SCF counts differ despite matching input thresholds and equivalent converged
outputs: Fe3C noncollinear uses 100 upstream iterations versus 71/65/69 for
REMAG; Fe54 ELSI uses 79 versus 107/127/114. These are end-to-end solution
speedups, not fixed-iteration kernel ratios. The Fe3C fixtures use
nonrelativistic pseudopotentials and do not establish SOC performance.

## Earlier comparisons

| Record | What it supports |
| --- | --- |
| [FePt16 v143](upstream-gpu-refresh-v143-20261009T101214Z/qualification.json) | Three complete x/z pairs: 573.311 → 9.838 s medians, 58.275× versus native upstream GPU; measured REMAG v142 |
| [Fe16 NC v127b](spin-layout-performance-v127b-20261008T224249Z/summary.json) | Three fresh repetitions: 177.814 → 5.980 s medians, 29.733× versus native upstream GPU |
| [Fe54 collinear v127](spin-layout-performance-v127-20261008T220559Z-interrupted-c7b6eddcebea4576b95627dfced200ef/summary.json) | Three completed repetitions: 79.627 → 9.512 s medians, 8.372× versus the selected upstream ELSI GPU route |
| [FePt128 v172](large-upstream-comparison-v172-20261009T172335Z/summary.json) | Scientifically checked 10,060.055 → 233.256 s single pair; 43.129× remains provisional, not repeated performance qualification |
| [FP64 matrix rotation v178](fp64-matrix-qualification-v178-20261009T215709Z/production-integration.json) | Three large FePt128 pairs, 203.275 → 190.832 s medians: 1.065× incremental REMAG gain; smaller-case gain not established and option remains off by default |

FePt used A100-SXM4-80GB; the v127 Fe tests used A100 80GB PCIe. Each individual
comparison used one matched allocation. Source versions, inputs, baseline
settings and the exact trial times remain attached to each record. These rows
must not be combined into a common-version suite average, and incremental
REMAG factors must not be multiplied into upstream-GPU ratios.

The Fe54 calculation group completed before a later Fe16 calculation in that
app was preempted. Its [recovery record](spin-layout-performance-v127-20261008T220559Z-interrupted-c7b6eddcebea4576b95627dfced200ef/recovery.json)
preserves that distinction. Other failed experiments remain in the full local
archive; the table is a selection of published results, not a complete attempt
ledger. The expanded campaign above retains its full compact ledger and
must be interpreted within its separate hardware, build and coverage scope.

All timings include process startup, initialization, SCF, final observables
and requested output exports. Build/provisioning, diagnostic traces and artifact
transfer are excluded. SCF counts may differ despite equivalent converged
outputs. CPU upstream validates science; CPU timings are not substituted for
upstream-GPU baselines.

These SZP/100 Ry magnetic regressions establish implementation behavior at their
recorded settings. They do not establish production convergence, universal
speedups, performance on other hardware or physically converged MAE predictions.

## Reproduction boundary

Included: compact qualification/timing records, available measured metadata,
exact FDF inputs and checksums of the copied files. Metadata records source,
binary and pseudopotential identities. Some paths inside the records name the
original remote `/results` or `/opt` filesystem and are retained as provenance.

Not included: raw density/matrix binaries, stdout for every trial, historical
implementation archives, pseudopotentials, account-bound Modal images, and the
complete development archive. Those are retained locally under the original
run IDs. The compact records let readers inspect timing claims, but they do not
alone let readers independently repeat all scientific comparisons. Historical
documents' `results/` links refer to that separate local archive.

Current source and build scripts are in this repository. The default Modal
harness uses a CPU oracle; repeating the upstream-GPU table also requires the
recorded GPU-enabled upstream build/configuration. See [running instructions](../../docs/RUNNING.md).
