# Resident density-matrix validation — 5 October 2026

This stage moves sparse density and energy-density matrix construction onto the GPU, retaining eigensolver vectors and accumulating across k points. It also preserves the normalized density on device for the next real-space grid contraction when the host input still matches. The reference remains unmodified SIESTA 5.4.2. The earlier eigensolver/grid milestone is recorded separately in [VALIDATION.md](VALIDATION.md).

## Measurement method

The normalization-aware run is `20261005T170415Z`, on one Modal A100 SXM4 40 GB. Each case runs one serial CPU reference, three independent cold GPU candidate processes, and one GPU control with density construction disabled. All processes run sequentially on the same host with one BLAS thread. The control retains GPU eigensolvers and grid contractions, isolating the effect of the density implementation and its resident handoffs.

Wall times include initialization, SCF, final forces and output. Provisioning, builds and downloads are excluded. The median and range summarize only the three candidate timings; the CPU reference and control each have one measurement. Incremental speedup in the ablation record uses the first candidate, whose outputs are compared directly with the control. These results do not establish performance against an optimized parallel CPU installation.

All 36 reference/candidate comparisons, nine direct GPU ablations and two MAE comparisons passed. Each of the nine cases has three passing candidate repetitions and a passing density-disabled control. See the [complete report](../results/20261005T170415Z/REPORT.md), [comparison ledger](../results/20261005T170415Z/ledger.json) and [run metadata](../results/20261005T170415Z/metadata.json).

| Case | CPU seconds | GPU median seconds | GPU range seconds | CPU / GPU median | Previous GPU seconds | Previous / first new GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Displaced Fe2 forces | 6.305 | 4.063 | 3.966–4.067 | 1.55× | 4.374 | 1.10× |
| Fe2 CG relaxation | 27.637 | 16.943 | 16.851–17.061 | 1.63× | 18.273 | 1.07× |
| Fe2 non-collinear, Gamma | 3.353 | 2.036 | 1.935–3.237 | 1.65× | 1.935 | 0.95× |
| FePt2 SOC z, Gamma | 7.935 | 4.781 | 4.779–4.783 | 1.66× | 4.783 | 1.00× |
| FePt2 SOC x, Gamma | 8.374 | 4.887 | 4.776–4.989 | 1.71× | 4.882 | 0.98× |
| Fe16 non-collinear | 126.826 | 30.098 | 29.756–30.472 | 4.21× | 43.806 | 1.44× |
| FePt16 SOC z | 115.733 | 41.846 | 41.264–43.166 | 2.77× | 58.637 | 1.42× |
| FePt16 SOC x | 104.672 | 37.365 | 36.560–38.885 | 2.80× | 54.283 | 1.45× |
| Fe128 collinear | 1163.095 | 163.674 | 146.194–176.875 | 7.11× | 482.680 | 2.95× |

Small Gamma cases receive little benefit from this additional offload; the non-collinear case also has a visibly noisy repetition. The explicit `REMAG_DM` switch is retained for controlled comparisons. Larger non-collinear and SOC cases show a clear end-to-end gain.

SCF counts are unchanged for forces (25), relaxation (96 across geometries), Fe2 non-collinear Gamma (10), FePt2 Gamma z/x (36/38) and Fe16 non-collinear (45). FePt16 z uses 47 CPU iterations, 48/49/48 candidate iterations and 48 control iterations; x uses 42 CPU, 39/39/41 candidate and 43 control iterations. End-to-end speedups include these convergence differences. Mean iteration time for the first Fe16 non-collinear candidate is 0.474 s versus 0.753 s for the previous GPU path. The corresponding FePt16 z/x times are 0.559/0.582 s versus 0.919/0.925 s.

Fe128 uses 98 CPU iterations, 88/98/77 candidate iterations and 93 control iterations. Its candidate mean iteration times are 1.462/1.449/1.443 s, versus 11.358 s upstream and 4.815 s in the previous GPU path. The total-time spread largely tracks iteration count. The [direct ablation](../results/20261005T170415Z/ablation-fe128_collinear.json) validates the old/new GPU outputs against each other as well as against upstream.

## Scientific coverage

The same fixed [scientific gates](VALIDATION.md#scientific-gates) apply to all variants: energies, Fermi energy, eigenvalues, full real-space density components, total and atomic moments, charges, forces, stress and relaxed positions. Missing results, unconverged execution and mismatched input provenance cannot pass. Both real and complex collinear density paths and four/eight-component spinor paths are covered across the staged regression batches.

The final batch adds Gamma non-collinear/SOC cases and larger Fe16 non-collinear and FePt16 SOC cells. It repeats the displaced-force and geometry-relaxation checks with the normalization-aware bridge. Direct x-minus-z MAE comparisons and final moment directions are checked separately from each orientation's absolute energy. These remain implementation regressions at a small basis/grid and 1000 K FePt smearing, not physically converged anisotropy predictions.

Maximum absolute errors across all 36 comparisons are:

| Observable | Maximum error | Gate |
| --- | ---: | ---: |
| Energy per atom | 5.0e-7 eV | 1e-5 eV |
| Fermi energy | 1.0e-6 eV | 1e-5 eV |
| Eigenvalues | 2.862e-5 eV | 1e-4 eV |
| Whole-cell moment components | 5.0e-5 μB | 1e-4 μB |
| Atomic moment components | 5.0e-6 μB | 1e-4 μB |
| Atomic charges | 1.0e-6 electrons | 1e-4 electrons |
| Force components | 1.0e-6 eV/Å | 1e-4 eV/Å |
| Stress components | 2.0e-6 eV/Å³ | 1e-5 eV/Å³ |
| Density components | 5.484e-6 electrons/Bohr³ | 1e-5 electrons/Bohr³ |
| Relaxed positions | 1.1e-8 Bohr | 1e-3 Bohr |

The [FePt16 MAE](../results/20261005T170415Z/mae-fept16.json) is 0.056827 eV upstream and 0.056828 eV for the primary candidate, an error of 1 micro-eV. The [FePt2 Gamma MAE](../results/20261005T170415Z/mae-fept2-gamma.json) is −0.021465 eV for both. Both pass the fixed 5 micro-eV gate and final-axis checks. Stdout energies resolve 1 micro-eV; printed equality is not a claim of bitwise equality. Direct MAE records use the primary candidate pair; individual orientation regressions cover all repetitions.

Twelve local tests passed. Eighteen native numerical scenarios passed on the A100: six generalized eigensolver problems, six grid contractions and six sparse density/energy-density scenarios. The new scenarios cover real and complex coefficients, nonzero periodic phases, repeated orbital images, truncated spectra, signed/zero occupations, both real spin slots, optional host wavefunction export, stale-slot rejection, normalization, and a host mutation between grid calls. The mutation check verifies that the grid does not reuse stale density.

The [native check record](../results/20261005T170415Z/kernel-check.json) has maximum generalized residual 3.28e-15, overlap-orthogonality error 3.93e-14, sparse density error 3.06e-16 and energy-density error 1.46e-16. Native, integration-patch and Python harness hashes were verified against the source when this milestone completed. Subsequent harness changes to the [I/O policy](GPU_RESIDENCY.md#first-io-change) have separate local validation; the native numerical code and integration patch are unchanged. The pinned upstream checkout remains unmodified.

## Resident state and transfer evidence

For the first Fe16 non-collinear candidate, 360 eigensolver vector sets were retained, avoiding 228,413,952 bytes of eigenvector download. All 45 density builds received a verified normalization update. The grid reused resident density on 47 calls, avoiding 332,299,776 bytes of density upload. Sparse density topology uploaded once. The initial normalization has no preceding GPU density and is recorded as one cache miss.

The force case independently confirms 27 resident grid hits and 11,930,976 bytes of avoided density upload. Thus the solver → density matrix → normalization → grid handoff is exercised inside SIESTA, in addition to the native checks.

The first Fe128 candidate retains 176 vector sets and avoids 1,868,562,432 bytes of eigenvector download. Its 88 normalization updates enable 90 resident grid hits, avoiding another 1,533,173,760 bytes of density upload. Basis and sparse density topology each upload once. Across the custom solver, grid and density code, the run still transfers approximately 6.08 GB host-to-device and 5.08 GB device-to-host.

The full SCF state is still not resident. Each completed build downloads density and energy-density matrices for upstream CPU consumers; Fe16 non-collinear records 636,318,720 bytes for these new outputs. This can increase total device-to-host traffic even while eigenvector downloads and subsequent density uploads are removed. The evidence establishes specific avoided boundaries, not a universal reduction in total transfer volume.

The native `accumulation_wall_seconds` counter covers sparse accumulation calls only. It excludes vector capture, plan construction, finish/download and normalization. Fortran timers and end-to-end measurements include those costs. Custom kernel/transfer/allocation counters exclude CUDA library internals, whose launch counts remain null.

## Profiling and remaining work

The previous Fe128 profile attributed 399.746 s to CPU `r-buildD`, motivating this stage. In the new same-host control, CPU `r-buildD` consumes 318.897 s across 93 iterations. The first new candidate's `remag_dm` timer totals 6.828 s across 88 iterations, and end-to-end time falls from 482.680 s to 163.674 s. In spinor routines, upstream does not separately time the CPU density loop; much of that cost appears under miscellaneous. Direct ablations and complete SCF iteration timings provide the meaningful comparison.

The first Fe128 candidate's exclusive routine totals now place `gridxc@cellXC` at 36.478 s, `rhoofd` at 34.329 s, `vmat` at 27.148 s and `dfscf` at 19.238 s. Density-matrix construction (`remag_dm`) is 6.828 s, about 4.2% of total runtime. The next profiling pass should split the existing grid path into content validation, transfers and device computation, then prioritize that work alongside CPU XC and final force contractions. Grid wall time alone does not distinguish these costs. Mean sampled GPU utilization is 10.9%, peak sampled GPU memory is 889 MiB, and peak host RSS is approximately 2.42 GB.

Remaining CPU work includes Hartree/XC, occupations, mixing, sparse Fourier transforms, final forces, and consumers of sparse density/energy-density outputs. The next optimization should follow the updated component and routine profiles. Orientation/state reuse, the boride/carbide/nitride corpus, production convergence studies, complete CUDA activity tracing and learned acceleration remain unfinished.

## Earlier stage evidence

The [first density batch](../results/20261005T164854Z/REPORT.md) passed seven cases including forces, relaxation, Fe16 collinear, non-collinear Fe2, sampled-k SOC/MAE and Fe54. Its Fe54 pair measured 141.573 s CPU and 31.529 s GPU (4.49×), with 33/34 iterations. It precedes normalization-aware grid reuse and is retained as separate evidence; cross-host raw times are not used to claim an incremental improvement.

An [intermediate larger batch](../results/20261005T165851Z/REPORT.md) was intentionally stopped after ten completed passing comparisons to integrate the normalization handoff before the long Fe128 test. Its artifacts remain available. An earlier image build failed on generated fixed-format Fortran line length; the source patch was corrected before GPU execution.

Full raw outputs remain in the `siesta-remag-results` Modal Volume. Compact local evidence retains input manifests, comparisons, source hashes, raw logs, timers and telemetry. Use `scripts/fetch_evidence.py` for retrieval without allocating compute. Each stage obeyed the single-GPU cap, with no overlapping REMAG containers. The final app completed normally, and the Modal container inventory was empty after completion; no GPU instance was left running.
