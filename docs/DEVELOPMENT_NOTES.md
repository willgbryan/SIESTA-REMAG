# Historical development notes

This preserves the pre-release README and its chronological experiments. Some
sections describe superseded implementations. Links into `results/` refer to
the retained local raw-evidence archive, which is excluded from Git. See the
[current README](../README.md) and [versioned benchmark records](../benchmarks/evidence/README.md)
for the source-release overview and selected public data.

# SIESTA-REMAG

GPU-first SIESTA development for magnetic materials, using pinned upstream SIESTA as the numerical oracle. The full build specification is in [docs/SPEC.md](../docs/SPEC.md).

This repository implements an experimental single-GPU derivative. Its device-controlled SCF and supported final observables pass upstream regression. The current priority is measured speedup through GPU-first execution and purpose-built data structures, using upstream SIESTA directly for numerical validation. The derivative applies an integration patch to disposable copies of pinned upstream SIESTA.

**The v142 FePt16 full off-site SOC baseline measures 58.27× faster
than upstream native ELPA GPU SIESTA** on the same A100-SXM4-80GB allocation.
Three fresh repetitions of the complete x/z directional
pair give median times of **573.31 s upstream and 9.84 s REMAG**, including
initialization, SCF, final observables and requested exports. The
[completed v143 comparison](../results/upstream-gpu-refresh-v143-20261009T101214Z/qualification.json)
passes all 17 calculations, six direct scientific/export comparisons and 11 MAE
checks. Separate execution diagnostics confirm upstream GPU use; traced and
profiled timings are excluded. Upstream uses the previously tuned ELPA
two-stage/block-64/one-thread setting, without a claim of optimal settings on
this new allocation. All contenders use identical current DM 1e-9 / H 1e-8 eV
stopping criteria and unchanged scientific comparison tolerances. Iteration
counts differ, so this measures time to equivalent converged solutions.
The earlier [18.43× result](../results/upstream-gpu-refresh-v117-20261008T180328Z/qualification.json)
and [6.03× result](../results/upstream-gpu-performance-v109-20261008T033242Z/performance-recheck.json)
remain separate historical measurements at their recorded stopping criteria.
These results apply to this
regression workload; other materials and sizes require separate upstream GPU
measurements. The collinear comparison below is independent. The pinned SIESTA ELSI interface rejects NC/SOC, so
native ELPA is the applicable comparator.

The subsequently integrated **shared-spin grid-potential contraction** measures
another **1.054× incremental gain** over that REMAG baseline: three x/z pair
medians are **10.050 → 9.538 s**, with all three pairs faster. It shares orbital
lookup and basis loads across spin components and uses at most 2 KiB of block
shared memory without adding persistent storage or transfers. Existing grid,
SOC, collinear/noncollinear, relaxation and orientation-sweep checks pass.
Enable it with `--grid-potential-spins` on the GPU grid path, or
`REMAG_GRID_POTENTIAL_SPINS=1`. The earlier kernel remains the default/fallback.
[Measured source and results](../results/grid-potential-spins-v144-20261009T105702Z/production-integration.json)
retain the independent comparison; this factor is not multiplied into the
58.27× upstream result above.

**Fe54 collinear measures 8.37× faster than upstream GPU ELSI/ELPA**, the faster
qualified upstream route in its settings screen. On one A100-80GB PCIe,
three fresh median timings are **79.627 s ELSI → 9.512 s REMAG**; native ELPA
takes 110.915 s. ELSI uses two-stage/block-64, eight MPI ranks and one thread
per rank on the same GPU; native ELPA uses two-stage/block-64 with four threads.
Both upstream routes have actual GPU-execution evidence. All six direct
scientific/export comparisons pass, including saved DM coefficients within
the unchanged 1e-7 gate. [Case results](../results/spin-layout-performance-v127-20261008T220559Z-interrupted-c7b6eddcebea4576b95627dfced200ef/summary.json)
are complete even though Modal later preempted the subsequent Fe16 calculation;
[recovery details](../results/spin-layout-performance-v127-20261008T220559Z-interrupted-c7b6eddcebea4576b95627dfced200ef/recovery.json)
preserve that distinction.

**Fe16 noncollinear measures 29.73× faster than upstream native ELPA GPU SIESTA**
on one A100-80GB PCIe: three fresh median timings are **177.814 s upstream →
5.980 s REMAG**. The selected upstream setting is one-stage/block-64 with four
threads, after screening three settings; a separate diagnostic confirms GPU
execution. All 11 calculations and three direct science/export comparisons
pass at matched DM 1e-9 / H 1e-8 eV stopping criteria. Maximum direct saved-DM
error is 5.08e-8 against the unchanged 1e-7 gate. This independent sampled-k NC
regression has no directional MAE pair. [Completed comparison](../results/spin-layout-performance-v127b-20261008T224249Z/summary.json)
uses one uninterrupted allocation; its result was recovered after the local
launcher lost connectivity during retrieval, without rerunning calculations.

Three nonparallel Fe3 SOC regressions also pass CPU/native-GPU/REMAG comparison,
including forces, stress, moments and saved matrices. [Returned raw outputs
reproduce all nine comparisons](../results/nonparallel-soc-v112b-20261008T164432Z/raw-comparison.json).
These fixed, unrelaxed Gamma and sampled-k inputs qualify numerical coverage;
their traced native timings do not establish a Fe3 speedup.

Sourced **16-atom cementite (Fe3C)** now passes direct pristine-CPU comparison
in collinear and uniformly tilted noncollinear layouts with the current v163
implementation. Energies, forces, stress, moments, densities and requested raw
matrices pass; maximum saved-DM error is **1.586e-8** against 1e-7.
[Carbide qualification](../results/carbide-gpu-v169-20261009T170851Z/capability-qualification.json)
retains the published structure, exact inputs and measurements. These
nonrelativistic test-pseudopotential regressions do not qualify full SOC or a
carbide speedup against upstream GPU SIESTA.

Sourced **12-atom Fe2B** also passes full off-site SOC x/z comparisons using
fully relativistic PBE PSML pseudopotentials. Forces, stress, moments, densities
and requested matrices pass; the directly compared MAE is **-1.622 meV** in both
implementations at printed precision. [Boride qualification](../results/boride-gpu-v173-20261009T202117Z/capability-qualification.json)
records the coarse regression settings and scope. No boride upstream-GPU timing
comparison or production materials-accuracy claim is made.

Sourced **32-atom Fe5PB2** passes full off-site SOC x/z comparisons with fully
relativistic PBE Fe/P/B pseudopotentials and the v175 implementation at default
tile settings. Forces, stress, densities and saved matrices pass; final moment
directions agree and direct MAE is **−48 micro-eV** in both implementations at
printed precision. [Ternary-magnet qualification](../results/candidate-magnet-gpu-v179-20261009T222951Z/capability-qualification.json)
retains the published structure, exact settings, raw outputs and individual
timings. These are coarse implementation regressions; no Fe5PB2 upstream-GPU
speedup or production materials-accuracy claim is made.

The latest broad regression is [20261007T184850Z](../results/20261007T184850Z/qualification-summary.json): 33 input/output variants, three GPU repeats, 105 scientific checks and zero explicit copies across 4,599 SCF iterations and 321 geometries. Production GPU nonlocal H0, retained-integral forces, density history, initial populations, final-spectrum retention and conditional H/D/E exports pass. CPU geometry/setup and ionic consumers remain. Scientific tolerances are in [docs/VALIDATION.md](../docs/VALIDATION.md), and current work is in [docs/PROGRESS.md](../docs/PROGRESS.md).

Eigenbasis reuse, clustered grid contraction and parallel sparse charge reduction have reduced SCF cost. Initialization and the remaining eigensolver/density work now dominate the measured runtime; focused nonparallel SOC coverage has passed. Electronic state retention and reduced transfers remain architectural priorities; exhaustive zero-transfer/zero-I/O certification is not a delivery prerequisite. [Residency evidence and limits](../docs/GPU_RESIDENCY.md) remain available. Learned SCF acceleration stays deferred while the conventional performance baseline is established.

The [full nonlocal Hamiltonian shadow](../results/20261007T141909Z/qualification-summary.json) passes 33 material/output cases and both MAE comparisons: GPU scalar/SOC matrices match upstream exactly across 107 geometries. The [production host-poisoning control](../results/20261007T134522Z/qualification-summary.json) separately verifies that SCF and forces can use device H0 with unused host matrices filled with NaNs. The qualified production run above uses that matrix evidence and fresh CPU references. These preceding diagnostics do not themselves qualify speedups.

The integrated solver workspace pool separately measured **3.66× end-to-end gain
against the preceding REMAG implementation** on one A100-80GB PCIe: three fresh
FePt16 x/z paired repeats have medians of 89.33 s and 24.44 s. All 17 calculations
and six MAE checks pass. [Timing results](../results/scf-solver-lanes-v116-20261008T173132Z/summary.json)
and [magnetic regression details](../docs/PROGRESS.md#scf-performance-work) cover
nonparallel full SOC, real collinear matrices, sampled noncollinear matrices and
relaxation. This is a separate comparison from the upstream-GPU result above;
the two speedup factors must not be multiplied. Use `--solver-lanes 8` with the
existing `--device-scf --vector-cache` route, or set
`REMAG_SCF_SOLVER_LANES=8` for a standalone candidate. The pool adds about 94 MiB
of persistent workspace in FePt16 and leaves the common occupations and density
accumulation order unchanged. The default remains one workspace.

The opt-in `--solver-warm-start` setting reuses each k-point/spin eigenbasis on
the GPU, with reorthogonalization and the same FP64 solver convergence criterion.
With `--device-scf --vector-cache --solver-lanes 8`, it measures a further
**1.57× gain over the preceding REMAG path**: median FePt16 x/z pair time falls
from **25.73 s to 16.42 s** on one A100-SXM4-80GB. All 14 performance calculations
and direct MAE checks pass. [Results](../results/scf-seeded-eigensolver-v119-20261008T190153Z/summary.json)
and [integration record](../results/scf-seeded-regression-v119b-20261008T191317Z/production-integration.json)
cover nonparallel full SOC, collinear and noncollinear Fe, and nine relaxation
geometries. Fe16 requires matched tighter SCF stopping criteria (DM 1e-9,
H 1e-8 eV) for saved-DM agreement; its original loose-input failure is preserved.
The default remains off; standalone use is `REMAG_SCF_WARM_START=1` with retained
vectors. This incremental result is separate from the 18.43× upstream comparison
above; the ratios must not be multiplied. Additional persistent storage is about
10.1 MiB for FePt16.

The grid-density kernel now computes an eight-point mesh cluster together,
sharing sparse lookups and density reads while preserving each point's reduction
order. It measures **1.10× end-to-end gain over warm-start REMAG** on one
A100-80GB PCIe: paired medians **18.36 → 16.68 s**. Three fresh pairs, direct
scientific/export/MAE checks and four magnetic/relaxation regressions pass.
[Measurement](../results/grid-cluster-density-v120-20261008T193307Z/summary.json) and
[integration](../results/grid-cluster-regression-v120-20261008T193841Z/production-integration.json)
record the evidence. The layout is selected automatically for eight-point
clusters; other cluster sizes keep the existing kernel. It adds no persistent
storage. These independently measured incremental gains do not establish a new
combined speedup against upstream GPU SIESTA.

Sparse charge reduction now distributes the stored entries of a row across a
CUDA block. It measures **1.072× end-to-end gain over the clustered-density
baseline**, with matched FePt16 x/z medians of **16.99 → 15.86 s** on one
A100-SXM4-80GB. The diagnostic scalar stage falls from 7.78 to 0.58 ms per
iteration, adding 4,576 bytes of workspace. All 39 existing scalar checks,
three direct scientific/export/MAE pairs, and four magnetic/relaxation cases
pass. [Measurement](../results/scf-row-charge-v121-20261008T195053Z/summary.json)
and [integration](../results/scf-row-charge-regression-v121-20261008T195606Z/production-integration.json)
retain source identity, scope and unchanged tolerances. No new upstream-GPU
comparison or multiplied aggregate speedup is claimed.

The opt-in `--gpu-radial-tables` setting batches initialization's radial
Fourier transforms on the GPU. It measures **1.142× end-to-end gain over the
preceding REMAG baseline** on one A100-SXM4-80GB: median FePt16 x/z time falls
from **15.52 to 13.59 s**. Three paired science/export/MAE comparisons and all
four magnetic/relaxation regressions pass.
[Results](../results/radial-batch-v123-20261008T201716Z/summary.json) and
[integration](../results/radial-batch-v123-20261008T201716Z/production-integration.json)
record the measured source and unchanged gates. Use it with `--integral-force`;
standalone use is `REMAG_RADIAL_BATCH=1` on the same production force path.
The default is off. This initial version downloaded an 80.3 MiB transform bank;
the GPU compression update below reduces that setup boundary. Serial table setup
is supported; other materials, larger angular momenta and spin layouts need their
own performance measurements. This is an incremental REMAG comparison, separate
from the upstream GPU result above.

GPU phase adjustment and first-match table compression now reduce the FePt setup
download to **2.48 MiB (96.9% less)**. Against the preceding radial-batching path,
three paired medians improve modestly from **13.53 to 13.33 s (1.016×)** on one
A100-80GB PCIe; all three pairs are faster. This comparison uses fresh upstream
CPU references and matched **DM 1e-9 / H 1e-8 eV** stopping criteria. The original
DM 1e-8 / H 1e-7 eV comparison exceeded the saved-DM gate and remains unqualified.
The scientific gates are unchanged; all direct science/export/MAE checks and four
magnetic/relaxation regressions pass at the documented settings. CPU spline and
angular construction remain, and explicit SCF copies remain zero.
[Measurement and integration](../results/radial-compact-v124b-20261008T204611Z/production-integration.json)
include the original failure and tighter-protocol qualification. The existing
`--gpu-radial-tables` option selects this compact path.

The Jacobi solver now retains each round's paired matrix ordering, reducing
full-array rearrangements from six per round to two plus sweep-boundary work.
Three FePt16 x/z repetitions measure **1.017× incremental gain**, with pair
medians **13.104 → 12.886 s**, and all four magnetic/relaxation regressions pass.
[Integration and limits](../results/paired-block-layout-v126-20261008T213300Z/production-integration.json)
record the measured source. A single larger FePt128 z pair improved
**539.6 → 391.9 s (provisional 1.377×)** with direct numerical agreement, but
its initial upstream GPU reference timed out. The later v133 pristine CPU
reference validates the retained v126 z result; current larger-capacity
qualification is described below. This adds no new verified speedup over
upstream GPU SIESTA.

The nonlocal-force gather now distributes each sparse row's entries across a
CUDA block, preserving duplicate-column summation and SOC selection semantics.
Three matched nine-geometry Fe2 relaxations measure **1.479× incremental gain**:
median end-to-end time falls from **10.415 to 7.043 s** on one A100-SXM4-80GB.
All three direct comparisons, five magnetic regressions, FePt MAE and the
existing 734 compact nonlocal checks pass. This is integrated into the existing
force path without a new setting. [Measurement and integration](../results/nonlocal-row-gather-v130-20261008T233135Z/production-integration.json)
retain the exact tested source. The candidate SCF intervals retain zero
explicitly accounted copies. This incremental relaxation result does not change
the separately measured upstream-GPU speedups above. A preceding atomic-force
warp-reduction trial showed no end-to-end gain and was not integrated.

Density construction now shares occupation-weighted contractions across periodic
copies of an orbital pair, then applies each image's phase on the GPU. Three
matched FePt16 full-SOC x/z pairs measure **1.068× incremental gain**, with
median end-to-end times **11.365 → 10.644 s**. All three pairs improve; upstream
scientific/export/MAE checks and four magnetic/relaxation regressions pass.
FePt uses 20,736 contractions per k/spin job in place of 217,840 image entries,
adding 2.53 MiB of cycle scratch and about 0.99 MiB of geometry-map staging.
[Measurement and integration](../results/unique-density-pairs-v140-20261009T091807Z/production-integration.json)
retain the exact sources and per-repeat data. The whole device-SCF path selects
this layout when repeated pairs exist; `REMAG_DENSITY_UNIQUE_PAIRS=0` selects
the previous path. Explicitly accounted SCF copies remain zero. Capacity and
scientific tolerances are unchanged; this adds no new upstream-GPU speedup claim.

The optional shared-memory local eigensolver keeps each 32-state real or complex
Hermitian problem in one GPU thread block. With unchanged FP64 convergence
criteria, three FePt16 x/z pairs improve **11.062 → 9.643 s** median:
**1.147× over v141**. Existing solver checks, magnetic/relaxation regressions,
requested outputs and z/x/z sweep checks pass.
Use `REMAG_BLOCK_SHARED_JACOBI=1` with the 32-state Jacobi path, or
`--shared-local-solver` in the ordinary Modal launcher.
[Results, measured sources and limits](../results/shared-local-eigensolver-v142-20261009T095832Z/production-integration.json)
retain all repetitions. The cuSOLVER fallback and production capacity are unchanged;
this is an incremental gain, not a refreshed upstream-GPU comparison.

GPU-produced initial mesh values, atomic reference density and Hartree energy
now remove CPU evaluation and setup uploads from the supported device-SCF path.
Three matched FePt16 x/z pairs improve **11.065 → 10.254 s** median, an additional
**1.079× speedup over v140**. Scientific/export checks, four magnetic/relaxation
regressions and a z/x/z geometry-reuse sweep pass. Enable `REMAG_INITIAL_MESH=1`
with the production GPU initialization and final-force paths, or use
`--gpu-initial-mesh` in the ordinary Modal launcher.
[Measured sources, repetitions and limits](../results/initial-mesh-integration-v141b-20261009T094041Z/production-integration.json)
are retained. This adds no new upstream-GPU ratio or capacity qualification.

The 32-wide eigensolver now fuses eigenvector gathering and rotation, removing
one full vector rearrangement per round and one full device copy per sweep.
Three fresh FePt16 x/z pairs measure **1.023× incremental gain**, with median
times **13.336 → 13.032 s**; all science/export/MAE checks and four magnetic or
relaxation regressions pass. [Measured source and integration](../results/fused-vector-rotation-v134-20261009T071946Z/production-integration.json)
retain the individual results. One FePt128 z pair improves **353.98 → 320.74 s
(provisional 1.104×)** with direct agreement between REMAG versions. That
single historical pair remains provisional; it is not a matched upstream GPU
speedup. The larger CPU reference has since completed, as described below.

The current GPU path now passes pristine upstream CPU validation on **FePt128
with 2,304 spinor states**, full off-site SOC and fully relativistic Fe/Pt
pseudopotentials. Energies, forces, stress, moments, densities, eigenvalues and
requested matrices pass the existing gates; maximum saved-DM error is 1.108e-8
against a 1e-7 tolerance. The exact measured capacity guards are integrated:
2,048 spatial orbitals / 4,096 combined spinor states. The largest qualified
case is 2,304 states; the ceiling is not a claim of complete size/material coverage.
[Capacity integration](../results/soc-current-profile-v150-20261009T114825Z/production-integration.json).
The v163 implementation also passes direct pristine **x/z and large-cell
MAE** checks: E(x)-E(z) is **0.138045 eV upstream versus 0.138046 eV REMAG**, a
**1 micro-eV** difference within the unchanged 5 micro-eV gate. Final moment
directions pass. [Direct results and scope](../results/large-soc-x-v167-20261009T163613Z/capability-qualification.json).
The subsequent v172 matched z comparison completes in **10,060.055 s upstream
GPU versus 233.256 s REMAG: a provisional 43.13× end-to-end speedup** on one
A100-SXM4-80GB. Upstream uses native ELPA two-stage/block-64 with eight MPI ranks
and one thread each sharing that GPU, the fastest previously screened layout.
Both contenders and their requested raw outputs pass direct pristine-CPU checks;
the GPU pair also passes locally. This is one matched pair, not repeated timing
qualification or an x/z timing pair. [Results and raw comparisons](../results/large-upstream-comparison-v172-20261009T172335Z/upstream-comparison.json).
Its diagnostic profile puts 90.16% of device-cycle time in assembly/eigensolves;
no timing ratio is inferred from the CPU reference on different hardware.

An opt-in tiled matrix rotation further improves the FePt128 SOC z case:
three matched pairs give **317.581 → 295.669 s** median, a **1.074× incremental
speedup over v150**, with identical SCF counts within each pair. All six large
results pass pristine upstream science/export checks, and the existing magnetic,
nonparallel SOC, relaxation and orientation-sweep regressions pass. The small
FePt16 x/z workload is slightly slower (**9.548 → 9.750 s** median), so the
default remains off. Enable `--fused-matrix-rotation` with width 32, device SCF
and vector cache, or `REMAG_BLOCK_FUSED_MATRIX=1` in a configured standalone
candidate. [Repetitions, measured source and integration](../results/fused-matrix-qualification-v152-20261009T123001Z/production-integration.json)
retain both outcomes. This adds no new upstream-GPU ratio or large-cell MAE claim.

The same opt-in path now reuses Hermitian symmetry: it computes each lower
matrix-rotation tile once and writes its conjugate partner through shared
memory. Three fresh FePt128 z pairs measure **331.808 → 268.647 s median**, a
**1.235× incremental gain over v152** on one A100-80GB PCIe. All three pairs
are faster, despite more candidate SCF iterations. All six large results pass
direct pristine-CPU comparisons; nonparallel SOC, collinear/noncollinear,
relaxation and small directional-energy regressions also pass. Maximum saved-DM
error is **8.087e-9** against 1e-7. The existing 2e-13 eigensolver tolerance is
unchanged; its maximum is increased from 20 to 32 sweeps because the optimized
cold solve needed 22. No persistent workspace is added.
The small x/z median is slightly worse (**8.195 → 8.295 s**), so the flag stays
off by default. [Measured source, repetitions and integration](../results/hermitian-qualification-v163-20261009T151110Z/production-integration.json)
retain both outcomes. This result is not multiplied into the upstream-GPU
comparison. The separate v167 comparison above qualifies its large-cell x/MAE;
other materials/settings require their own direct checks.

The vector-rotation opt-in, `--fp64-vector-rotation` / `REMAG_BLOCK_FP64_VECTORS=1`,
packs complex vector tiles into shared FP64 planes and uses double-precision
tensor-core products. Three matched FePt128 full-SOC z pairs on one A100-SXM4-80GB
measure **237.266 → 220.395 s median**, a **1.07655× incremental speedup over
v163**; all three pairs are faster. All six large z runs and the large x run
pass direct pristine-CPU comparisons. Large-cell MAE remains **0.138045 eV
upstream versus 0.138046 eV REMAG**, within the unchanged 5 micro-eV gate.
Existing nonparallel SOC, collinear/noncollinear, relaxation and directional
sweep regressions pass. Arithmetic remains FP64 and solver tolerances are unchanged.
The flag requires width 32, device SCF and vector cache; the measured configuration
also enables Hermitian matrix rotation. It stays **off by default**: small x/z
median time rises from 11.420 to 11.741 s, with only one of three faster pairs.
[Measured source, timings and integration](../results/fp64-vector-qualification-v175-20261009T205055Z/production-integration.json)
retain both outcomes. This incremental result is not multiplied into the measured
upstream-GPU speedups above.

The latest opt-in, `--fp64-matrix-rotation` / `REMAG_BLOCK_FP64_MATRIX=1`,
also evaluates the complex Hermitian matrix rotation with FP64 tensor-core
tiles, keeping the intermediate in shared memory. Three matched FePt128 full-SOC
z pairs on one A100-SXM4-80GB measure **203.275 → 190.832 s median**, a
**1.06520× incremental speedup over v175**; all three pairs are faster.
All six large z runs and the large x run pass direct pristine-CPU comparisons;
MAE is **0.138045 eV** in both implementations at printed precision. Existing
nonparallel SOC, collinear/noncollinear, relaxation and directional-sweep
regressions pass. No persistent workspace is added. The flag requires width 32,
device SCF, vector cache and fused matrix rotation; the measured configuration
also enables FP64 vector rotation. It remains **off by default** because small
x/z median time is slightly worse (**11.011 → 11.025 s**).
[Measured source, individual timings and integration](../results/fp64-matrix-qualification-v178-20261009T215709Z/production-integration.json)
retain both outcomes. This incremental gain is separate from the upstream-GPU
comparisons above.

Fixed-geometry magnetic sweeps now retain sparse topology, overlap, mesh setup
and GPU projector/H0 data between directions. With independent atomic guesses,
the FePt16 x/z workflow measures **1.31× faster than separate REMAG processes**:
three paired medians **12.83 → 9.78 s** on one A100-SXM4-80GB.
[Timing results](../results/magnetic-geometry-reuse-v136-20261009T082001Z/summary.json)
and [integration regressions](../results/magnetic-geometry-regression-v136-20261009T082632Z/production-integration.json)
pass scientific/export/MAE checks, nonparallel SOC, noncollinear and ordinary
magnetic/relaxation coverage. This is an incremental workflow gain, not a new
upstream-GPU ratio.

For a standalone candidate already configured for the production device-SCF,
force, initialization and history paths, add these FDF settings for a z/x/z
sweep relative to an initial z spin configuration:

```text
DM.AllowReuse false
DM.AllowExtrapolation false
REMAG.ReuseOrientationGeometry true
%block REMAG.SpinRotations
1 0 0 0
1 0 1 0
1 0 0 0
%endblock REMAG.SpinRotations
```

Rows are absolute quaternions `(w,x,y,z)`, normalized internally. Each direction
starts a fresh SCF cycle and writes requested outputs under `.o001`, `.o002`,
etc. This supports serial fixed-geometry NC/full-SOC; it does not constrain final
moments. Retained-density guesses and per-direction NetCDF are unsupported.
Atomic density uploads and eigensolver workspace reconstruction still occur
between directions. [Details and limits](../docs/PROGRESS.md#fixed-geometry-magnetic-sweeps-v136-integrated).

Implemented paths:

The combined production options in the first bullet compute the supported final
forces, stress, density and population observables on GPU. The component and
shadow milestones below retain their original measurement scopes; their older
references to remaining force consumers describe those intermediate stages.
See [current progress](../docs/PROGRESS.md) for the integrated capability and timings.

- `--integral-force --population`, with the resident force/basis options, makes kinetic, overlap, scalar/SOC nonlocal and grid forces plus requested spin/Mulliken observables authoritative on GPU. E is downloaded only for explicitly requested TSDE/NetCDF output on this route. Adding `--gpu-initialization --initial-populations --gpu-history` qualifies the initial density owner, population output, density history across geometries and conditional terminal D export. `--final-spectra --gate-h-export` retains spectra until final analysis, removes host occupations and omits H downloads when no supported consumer requests them. Eighteen no-matrix controls retain their other scientific outputs with zero H/D/E downloads. [Remaining consumers](../docs/GPU_RESIDENCY.md#latest-accepted-checkpoint) include CPU geometry/setup and ionic work, requested host exports and complete independent observation.

- Experimental `--device-scf --block-width 32 --block-local-solver jacobi` executes the bounded electronic cycle on the GPU and publishes diagnostics after completion. All nine magnetic cases and both MAE pairs pass. Independent larger-case CUDA-copy qualification remains open because of profiling faults, as recorded in [the integration record](../docs/DEVICE_SCF_INTEGRATION.md). The combined force/history/final-spectrum options above remove unrequested electronic-array exports; requested files retain explicit export boundaries. `--device-cycle-ablation` adds a sequential same-allocation comparison against the previous GPU path.

- `--atomic-grid-force --final-grid` keeps core/neutral/dfscf accumulation on device and bypasses final-force density/Hartree/XC recomputation. [Nine-case production qualification](../results/20261006T125358Z/REPORT.md) passes all controls and MAE checks. CPU derivative staging and integral/analysis consumers remain; requested density reuse is a separate option below. `--final-grid-ablation` supplies a sequential CPU-final-grid control.

- `--final-density` (with `--final-grid`) reuses retained GPU RHO/DRHO for requested final outputs and removes analysis recomputation and uploads. [Production/control validation](../results/20261006T134040Z/REPORT.md) passes all nine cases and MAE checks. `--final-density-ablation` restores original analysis as a sequential control. Requested output downloads and file writes remain explicit; other final consumers still need migration.

- `--orbital-basis` (with `--grid-force`) generates FP64 orbital values, gradients and displacements on the GPU, bypassing CPU generation and derivative uploads. [Nine-case production/control qualification](../results/20261006T150954Z/REPORT.md) removes net 317 MB of upload for Fe16 and 1.071 GB for Fe54 with no added derivative download. `--orbital-basis-ablation` supplies sequential CPU-derivative controls. The separate `--orbital-basis-shadow` diagnostic previously compared 250.9 million material elements with zero numerical difference; `--probe-orbital-basis` runs the isolated numerical, comparator and ownership checks. Core/neutral radial generation has a separate qualified option below; other final consumers still require migration.

- `--atomic-basis` (with `--atomic-grid-force`) generates core/neutral derivatives and selection masks on GPU, bypassing CPU radial loops and uploads. [Nine-case production/control qualification](../results/20261006T170248Z/REPORT.md) removes net 63.0 MB of upload for Fe16 and 213.6 MB for Fe54 with no added download. `--atomic-basis-ablation` supplies sequential CPU-radial controls. The separate `--atomic-basis-shadow` previously compared 87.67 million material elements with zero difference; `--probe-atomic-basis` retains isolated qualification. Kinetic/nonlocal/SOC/overlap and other analysis consumers still need migration.

- Final force preparation validates its entry count through retained geometry metadata, removing a four-byte readback per geometry. [Full-corpus validation](../results/20261006T173034Z/REPORT.md) passes nine cases and both MAE pairs. The next integral/projector migration has a [pristine two-center oracle and implementation plan](../docs/TWO_CENTER_RESIDENCY.md). Its isolated GPU evaluator passes [79 retained-array checks](../results/20261006T175529Z/REPORT.md) under `--probe-two-center`; material force integration remains pending.

- FP64 real and complex generalized eigensolvers through cuSOLVER, with reusable device buffers.
- Optional GPU sparse density and energy-density matrix construction, retaining eigenvectors and accumulating across k points. Real, non-collinear and SOC paths preserve upstream occupations and force-related energy weights.
- Optional device FD occupations with retained first-pass spectra and direct E/Q consumption by density construction, validated through Fe54. Temporary host exports remain.
- Optional GPU density and local-potential matrix contractions, including SIESTA's eight-component SOC density-matrix convention. Basis values and periodic orbital mappings remain on device with explicit geometry invalidation and a content-verification mode.
- Converged collinear, non-collinear and full off-site SOC regression inputs; displaced-atom force and structural-relaxation cases; paired x/z directional energy comparisons.
- `--two-center-shadow` (with `--grid-force`) compares actual S/T integral values and gradients on the GPU for kinetic, scalar/SOC nonlocal and overlap force consumers. [Full material qualification](../results/20261006T182622Z/REPORT.md) covers 30.61 million elements across 17 geometries with zero numerical difference. Tables stay on device across relaxation; CPU force results remain authoritative and diagnostic transfers are counted explicitly. The shadow does not qualify performance or remove legacy electronic exports.
- `--probe-pair-force` checks the isolated GPU kinetic/overlap contractions against literal pristine Fortran arithmetic. [All 202 scenarios pass independent reanalysis](../docs/PAIR_FORCE_RESIDENCY.md), including captured execution and failure propagation. `--pair-force-shadow` now passes all nine materials and both MAE pairs using retained device D/E. CPU outputs remain authoritative; geometry staging and shared final accumulation remain unfinished. No production speedup is claimed.
- `--probe-soc-projector` qualifies the isolated FP64 off-site SOC projector and full left derivative against pristine Fortran. [All 168 scenarios pass independent reanalysis](../results/20261006T202253Z/REPORT.md). Its shared arithmetic also passes the compact material nonlocal shadow described below.
- `--probe-nonlocal-force` checks both standalone and compact scalar/SOC nonlocal contractions. [The latest qualification](../results/20261006T232754Z/REPORT.md) passes 398 standalone, 734 compact and 40 actual-owner GPU scenarios with independent reanalysis. `--nonlocal-force-shadow` passes all nine materials and both MAE pairs over 17 geometries using compact pre-SCF geometry, GPU overlaps and retained D. All 204 extended handoff checks pass. Owner probes cover failure/recovery and stale leases across 45 geometries. CPU nonlocal outputs remain authoritative; shared production accumulation and final export removal remain unfinished.

- Isolated execution, checksummed inputs/pseudopotentials, upstream commit provenance, hierarchical timing, per-iteration timing, CPU/RSS and GPU sampling, explicit transfer/allocation accounting for the custom GPU code.
- Strict comparison of energies, eigenvalues, moments, atom charges, forces, stress, and every real-space density component. Unconverged, incomplete, malformed, missing or nonfinite results cannot pass.

The initial benchmark basis, mesh, smearing and k meshes are intended for **implementation regression**, not publication-quality materials predictions. FePt uses 1000 K electronic smearing to make the small regression case stable. Production MAE requires separate basis, mesh, k-point, smearing and SCF convergence studies. Its default 5 micro-eV comparison gate is provisional and close to stdout energy rounding (1 micro-eV per energy).

## Local baseline

Python 3.12+, CMake, a Fortran/C compiler and BLAS/LAPACK are required. A virtual environment is recommended.

```sh
python -m pip install -e '.[dev,modal]'
python scripts/build_upstream.py
remag prepare fe16_collinear --output benchmarks/generated/fe16
remag run benchmarks/generated/fe16 --executable build/upstream/Src/siesta --output runs/reference-fe16
remag compare runs/reference-fe16/result.json runs/candidate-fe16/result.json
```

`prepare` refuses an existing output directory; `run` copies only a clean prepared input into a new run directory. Raw `.RHO` files must be present beside each result for density comparisons. `upstream.lock.json` pins the source and its required Git submodules. Source downloads and builds remain under ignored `upstream/` and `build/` directories. The reference source stays unmodified; the GPU source patch is applied to a disposable copy inside the Modal image.

New inputs disable per-iteration DM checkpoints and retain the end-of-SCF-cycle DM artifact, along with the existing final scientific outputs. Use `remag prepare ... --io-policy checkpoint` or add `--io-policy checkpoint` to the Modal command to restore iterative checkpoints. The default `end-of-cycle` policy is recorded in each input manifest and applied equally to reference and candidate. It reduces checkpoint I/O; other I/O and host/device transfers remain.

## Modal: one GPU only

```sh
python scripts/run_modal.py --cases fe16_collinear,fe2_noncollinear,fept2_soc_z,fept2_soc_x,fe54_collinear --grid --dm
```

The sole GPU function requests **one A100 40 GB**, sets `max_containers=1`, uses no warm pool and no retries, and runs all cases sequentially. The launcher locks overlapping local invocations and refuses an already-active REMAG app. Do not run the app directly or launch from multiple machines simultaneously. The user's single-instance cap must be retained unless they explicitly authorize expansion. GPU image builds use CPU builders. No service is deployed and the ephemeral app stops after the run. The local entrypoint submits exactly one asynchronous call and prints its ID before waiting; recover retained artifacts by run ID if the client disconnects. The launcher continues to refuse another active allocation.

Reference and candidate run on the same CPU/GPU host, with identical inputs and one BLAS thread, including setup, SCF, final forces and output in measured wall time. GPU initialization is included in each candidate process. Cloud provisioning, image builds and artifact transfer are excluded. This measures against a serial upstream reference; it is not a claim against an optimized MPI/OpenMP CPU installation. Omit both `--grid` and `--dm` to isolate the eigensolver experiment. Add `--ablation` with `--dm` to compare directly with density construction disabled on the same host; `--candidate-repeats 3` repeats cold candidate processes against one CPU reference measurement. All variants run sequentially. GPU errors abort rather than quietly running a CPU fallback.

Full outputs are retained in the `siesta-remag-results` Modal Volume, under the printed run ID. Compact logs, inputs, profiles, comparisons and metadata are returned to `results/<run-id>/`. The JSON ledger reports speedup only after scientific regression passes on identical hardware. Allocation and transfer counters name their coverage; unmeasured library-internal kernel launches remain `null`. GPU memory/utilization are sampled with `nvidia-smi` at 200 ms; CPU and RSS at 100 ms, so very short peaks can be missed.

The grid cache now uses geometry epochs announced by the patched upstream initializer/reset. `--verify-epoch` rechecks all basis/topology bytes and fails on mutation inside a declared lifetime; use it for ownership validation. `--epoch-ablation` adds a same-host content-hashing control. These controls require `--grid`. The normal epoch path avoids repeated host scans; unversioned native callers retain content validation. `remag-residency.json` records explicit transfers separately for initialization, each SCF iteration, post-SCF work and analysis/export. The runner reconciles these counts with bridge totals. Library transfers, migrations and file I/O remain unmeasured, so this is partial accounting rather than proof of residency.

`--normalize` (requires `--dm`) moves charge normalization ahead of the existing D/E export, with `--normalization-ablation` providing a CPU-normalization control. Its [first validation](../docs/NORMALIZATION_VALIDATION.md) passes 16 upstream pairs, eight GPU controls and two MAE checks through Fe54. A subsequent untraced batch passes 20 pairs, five controls and FePt16 MAE through Fe128. Normalization has no demonstrated incremental end-to-end speed benefit; its purpose is consumer migration. `--probe-control` separately checks whether generalized divide-and-conquer or Jacobi solvers can execute in a device-controlled conditional graph. Neither feature establishes a fully resident SCF loop. New runs archive the exact implementation and harness in `implementation.tar.gz`, including native-check failures.

`--occupations` (requires `--dm`) computes FD occupations on device from retained spectra; density construction borrows both arrays without re-uploading them. `--occupation-ablation` adds a host-occupation GPU control. [Integration validation](../docs/OCCUPATION_VALIDATION.md) passes 16 upstream pairs, eight controls, both MAE checks and 16 CUDA traces, including forces, relaxation and Fe54. Eigenvalues, occupations and scalar results still download for remaining host consumers; total traffic is not uniformly lower. This does not establish strict residency or a speedup. Unsupported fixed-spin, alternative occupation and other diagonal routes fail explicitly when enabled.

`--resident-spectra` (requires `--occupations`) additionally removes eigenvalue and occupation downloads inside SCF, with explicit E/Q export after each geometry cycle. `--spectral-ablation` compares the per-iteration-export path. [Nine-case validation](../docs/SPECTRAL_RESIDENCY_VALIDATION.md) passes 18 upstream pairs, nine controls, both MAE checks and 36 CUDA/filesystem traces, including sampled-k non-collinear Fe. Scalar result/status downloads and the grid/H/S/D/E boundaries remain. Larger untraced validation also passes all 20 comparisons and FePt16 MAE; Fe128 takes 95.5 seconds at the three-run median (12.04× serial CPU), with no uniform incremental gain over the spectral-export GPU control.

The next experimental path, `--matrix-assembly` (requires `--resident-spectra`), constructs dense solver H/S directly on device from sparse upstream matrices. Immutable overlap, topology and k-point geometry are staged per geometry; the mutable sparse Hamiltonian is still uploaded each diagonalization. `--matrix-ablation` adds a control with CPU assembly. The [matrix/Hartree diagnostic](../docs/MATRIX_HARTREE_VALIDATION.md) passes 27 upstream comparisons, 18 controls, six MAE checks and 54 traces. Dense H/S uploads are zero in 351 candidate SCF iterations. Its ownership contract and remaining boundaries are in [the residency inventory](../docs/GPU_RESIDENCY.md#device-matrix-assembly-in-development).

`--hartree` (requires `--grid`) connects an experimental periodic Hartree component, with `--hartree-ablation` providing a CPU-Hartree control. The component keeps potential, energy, stress and status on device; its temporary SIESTA adapter still uploads density and exports those outputs for CPU consumers. All 19 ordinary/captured component scenarios, native ownership checks and the same end-to-end diagnostic pass. Temporary Hartree copies remain explicitly counted. See [component scope and validation](../docs/GPU_RESIDENCY.md#periodic-hartree-component-in-development).

`--resident-grid` (requires `--grid --hartree`) adds the combined density → Hartree/PBE → potential-matrix path. All **32 CUDA component checks and the 18-pair full SCF diagnostic pass**, including forces, relaxation and four MAE checks. Density and potential grids remain on device; a temporary 224-byte energy/stress/status export and sparse matrix transfers remain. `--grid-flow-ablation` compares the preceding host-grid path. The larger untraced batch also passes 20 upstream pairs, five controls and four MAE comparisons through Fe128. See [measured scope, repeated timings and remaining boundaries](../docs/RESIDENT_GRID_VALIDATION.md).

`--resident-scf` adds explicit GPU ownership of sparse H/D/E, Hamiltonian seeds/SOC, charge/energy/residual reductions and stable Pulay. The [nine-case validation](../docs/SPARSE_STATE_VALIDATION.md) passes 18 upstream comparisons, nine resident-grid controls, four MAE checks and 36 traces. All 349 SCF iterations retain these electronic arrays; steady uploads and selected file operations are zero, while scalar downloads and mappings remain. The 37 coupled ownership checks and 92 isolated convergence checks pass. Production still uses host convergence and CPU final observables. Larger validation and untraced performance are pending; this does not establish strict residency.

`--trace-io` adds a diagnostic syscall trace for each candidate, attributed to execution phases. Traced runs retain scientific comparisons but never qualify speedups; CPU/RSS telemetry is left unknown. Raw syscall logs and coverage errors are preserved. `--defer-scf-log` buffers routine stdout diagnostics until the SCF cycle ends; `--log-ablation` compares the original flush behavior. [Five-case validation](../docs/IO_VALIDATION.md) shows zero selected file operations during SCF with unchanged scientific results. Finite buffering alone does not guarantee this on other workloads. Mapping activity, page faults, asynchronous writeback and external observers remain separate limits.

Recover compact evidence after a completed or stopped run with `python scripts/fetch_evidence.py RUN_ID`; this reads the Volume without allocating compute. Add `--density` to download raw RHO/DRHO files and rerun `remag compare` locally. Generate a readable comparison with `python scripts/report_results.py results/RUN_ID`.

`--trace-cuda` enables a separate CUPTI observer for candidate processes, including CUDA-library copies and kernel activity. It buffers evidence in memory until explicit finalization, uses external correlation to associate activity with the launching API phase, and reconciles observed bytes against explicit bridge counters. Dropped records, timestamp gaps, missing SCF markers or inconsistent counts prevent complete trace coverage. The earlier [six-case validation](../docs/CUDA_ACTIVITY_VALIDATION.md) passes scientific regression and MAE, exposing three extra scalar downloads per eigensolver call. Its historical timestamp-only phase counts need requalification after a later known-copy calibration exposed clock-alignment errors; the correlation-based observer is under validation. Traced timings never qualify speedups. Unified-memory migrations and kernel access to mapped host memory remain unmeasured. Verify a completed run's archived source against its recorded hashes with `python scripts/verify_evidence.py results/RUN_ID`.

Both image builds disable SIESTA's host-specific CPU optimization. Modal CPU builders and GPU runtime hosts can expose different instruction sets; enabling `-march=native` in a cached build can otherwise produce an illegal-instruction failure on a later host.

## Current boundaries

- Serial single-process SIESTA; no MPI, multiple GPUs, reduced precision, or learned SCF accelerator.
- GPU grid path requires stored single-precision basis values and double-precision grids (upstream defaults). `DirectPhi`, other precision builds, and orbital lookup tables exceeding 512 MiB are unsupported and fail explicitly.
- Sparse Hamiltonian/density matrices are preserved. A bounded dense integer lookup is used for sparse pair indexing; it is not a dense electronic Hamiltonian. Its memory growth is a known optimization target.
- Supported density builders retain eigenvectors and optionally spectra/occupations; the qualified resident grid keeps density/potential arrays on device. The opt-in complete device cycle removes its scalar boundaries. The combined force/basis/integral/population route computes electronic forces and supported final populations on GPU. GPU density history and next-geometry electronic initialization are qualified on the combined route. CPU ionic terms and geometry/setup work remain. Occupations use upstream CPU code unless `--occupations` is enabled.
- Density-matrix acceleration covers serial collinear and non-collinear/SOC Gamma and sampled-k paths. File-backed eigenvector and spin-spiral routes retain their existing CPU density construction.
- Fixed-geometry directional sweeps reuse geometry and basis/grid mappings while starting each electronic cycle from atomic density. Final magnetization is checked against the requested axis; `DM.InitSpin` is an initialization, not a constraint. Retained electronic guesses, anisotropy fitting, more complicated material fixtures, broader nitride NC coverage, and downstream exchange workflows remain future milestones. Full-SOC Fe2B x/z and MAE and nonrelativistic Fe3C collinear/noncollinear regressions now pass direct upstream checks; see the [boride](../docs/PROGRESS.md#fully-relativistic-boride-v171v173-qualified) and [carbide evidence](../results/carbide-gpu-v169-20261009T170851Z/summary.json). These are implementation regressions, not production convergence studies.

The lean test suite is run with `python -m pytest`. Native GPU scenarios run inside Modal before DFT: generalized residuals, overlap orthogonality, real/complex eigenvalues, periodic grid folding, SOC components, basis-cache invalidation, sparse density/energy-density accumulation, guarded resident-state reuse, device charge normalization and invalidation/error handling. All 38 native scenarios, 27 isolated FD checks and 19 isolated Hartree checks pass. The [device PBE component](../docs/PBE_VALIDATION.md) passes 34 numerical/capture checks against full upstream GridXC and its resident-grid SCF integration is qualified. Isolated scalar (39), Pulay (24) and Hamiltonian (52) checks also pass; their coupled SCF adapter passes the nine-case sparse-state diagnostic. Upstream end-to-end regression remains the primary verifier.

References: [SIESTA source](https://gitlab.com/siesta-project/siesta/-/tree/5.4.2), [SIESTA manual](https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html), [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html), [Modal GPU configuration](https://modal.com/docs/guide/gpu).


`--probe-grid-force` checks the FP64 grid-force/stress contraction against extracted pinned upstream arithmetic (156 GPU scenarios passed). `--force-shadow` additionally compares a captured GPU contraction against unchanged CPU `dfscf` on actual geometries, using explicitly staged host inputs after SCF. It is a correctness diagnostic; it retains CPU force output and disqualifies timing. Final-state ownership and the remaining force terms are still being migrated.


`--force-shadow-resident` (with `--device-scf`) validates the force contraction using D/V/Vaux retained on the GPU across final publication. [All nine magnetic cases, both MAE comparisons and 17 geometries pass](../results/20261006T104613Z/REPORT.md). The handoff uses device-to-device copies and checks geometry/consumer lifetimes. Derivative staging, legacy exports for other consumers and CPU-authoritative force output remain; this diagnostic does not qualify full final-force residency or speedup.


`--grid-force` (requires `--device-scf`; incompatible with force shadows and device-cycle ablation) makes the retained-input GPU dfscf contraction authoritative. [All nine cases, nine CPU-force controls and both MAE pairs pass](../results/20261006T110416Z/REPORT.md). `--grid-force-ablation` runs that CPU-force control sequentially on the same GPU allocation. Exact FP64 derivative generation/staging, other force terms and legacy exports remain. This is a qualified contraction substitution, not full final-observable residency; see [the remaining boundaries](../docs/FINAL_OBSERVABLE_RESIDENCY.md).


`--probe-atomic-grid-force` validates the isolated partial-core/neutral-atom contraction against 27 extracted upstream fixtures. All 154 isolated GPU scenarios and the nine-case material shadow pass. This probe does not enable those terms in SIESTA.


[Repeated authoritative-force measurements](../results/20261006T111309Z/REPORT.md) pass ten candidates and two CPU-force controls. Fe16 median runtime is 44.40 s versus 49.74 s at the same iteration count; Fe54 is 24.68 s versus 29.84 s with differing SCF iteration counts. All direct scientific comparisons pass. These measurements include derivative staging and other remaining host work.


`--atomic-force-shadow` (requires `--grid-force`) compares exact upstream core/neutral contributions with the GPU contraction using retained device potentials/deformation density. It enforces ordered final consumers and excludes timing claims. [All nine materials, both MAE pairs and 34 material contractions pass](../results/20261006T114022Z/REPORT.md). The diagnostic preserves CPU core/neutral outputs.

`--atomic-grid-force` (requires `--grid-force`; incompatible with atomic shadows) makes core and neutral contributions authoritative and shares one device force/stress/status accumulator through dfscf. [All nine cases, nine sequential controls, four MAE checks and 42 ownership scenarios pass](../results/20261006T121029Z/REPORT.md). `--atomic-force-ablation` retains GPU dfscf but restores CPU core/neutral for a same-allocation control. No intermediate atomic-term exports or per-term files are produced. Geometry staging remains substantial (1.31 GB for Fe54), and final-force timings currently show overhead. Final CPU grid recomputation, derivative preparation, integral/SOC/overlap and analysis consumers still prevent complete final-observable residency.
