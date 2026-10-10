# Selected benchmark evidence

These existing records are copied verbatim from completed experiments. The
[manifest](manifest.json) records their original local paths and SHA-256 hashes.
No new calculations were performed to populate this directory.

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
ledger. The newer easy-convergence suite is still being evaluated and is not
included in this snapshot.

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
