# Validation record — 5 October 2026

This is the historical eigensolver/grid milestone. The subsequent density-matrix implementation and its validation are in [DENSITY_VALIDATION.md](DENSITY_VALIDATION.md); pending-work statements below describe the earlier stage.

This is a working GPU integration milestone against pinned SIESTA 5.4.2. It accelerates the real/complex generalized eigensolver and real-space density/local-potential contractions. It does not complete the full GPU-resident SCF and learned-acceleration program in the specification.

## Accepted measurements

Reference and candidate execute sequentially on the same Modal A100 40 GB host with one BLAS thread. Times cover a cold process, initialization, SCF, final forces and output; they exclude cloud provisioning, image builds and downloading artifacts. These are single observations against serial upstream, without repeated-run uncertainty estimates or comparisons to an optimized parallel CPU installation.

| Case | Upstream seconds | GPU seconds | End-to-end speedup | SCF iterations upstream / GPU | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Fe16 collinear | 59.079 | 28.354 | 2.08× | 26 / 26 | [Portable regression](../results/20261005T064711Z/REPORT.md) |
| Displaced Fe2 forces | 7.536 | 4.790 | 1.57× | 25 / 25 | [Portable regression](../results/20261005T064711Z/REPORT.md) |
| Fe2 CG relaxation | 33.193 | 20.336 | 1.63× | 96 / 96 across geometries | [Portable regression](../results/20261005T064711Z/REPORT.md) |
| Fe54 collinear | 304.408 | 74.313 | 4.10× | 61 / 57 | [Tighter SCF regression](../results/20261005T070118Z/REPORT.md) |
| Fe128 collinear | 1276.616 | 551.807 | 2.31× | 73 / 68 | [Tighter SCF regression](../results/20261005T070118Z/REPORT.md) |
| FePt SOC, z | 16.553 | 11.970 | 1.38× | 35 / 35 | [Portable SOC regression](../results/20261005T070118Z/REPORT.md) |
| FePt SOC, x | 15.430 | 11.251 | 1.37× | 32 / 32 | [Portable SOC regression](../results/20261005T070118Z/REPORT.md) |

The accepted Fe54 average SCF iteration fell from 4.641 s to 1.059 s. Its changed iteration count is included in the total speedup and must not be confused with per-iteration acceleration. Basis/topology uploaded once over 119 grid calls. Custom grid and solver accounting recorded approximately 1.10 GB host-to-device and 0.58 GB device-to-host traffic, excluding library internals. Mean sampled GPU utilization was only 7.4%, with peak sampled GPU memory 571 MiB. These measurements expose substantial remaining CPU work and transfer overhead.

Fe128 averaged 16.570 s per upstream SCF iteration and 7.444 s per candidate iteration. CPU density-matrix construction (`r-buildD`) consumed 399.746 s, approximately 72% of the candidate's total time. This is the next measured optimization priority. GPU utilization averaged 2.8%, with peak sampled memory 777 MiB; custom code transferred 5.92 GB to the GPU and 3.06 GB back. This CPU bottleneck limits the larger-cell speedup.

The [first run](../results/20261005T063026Z/REPORT.md) also passed tilted non-collinear Fe2 and both FePt SOC orientations. That run used the earlier host-optimized build and is retained as separate historical evidence.

## Scientific gates

Every qualified pair completed and converged with identical input and pseudopotential hashes. Required comparisons include energy per atom, Fermi energy, all exported eigenvalues, full real-space density components, total/atomic moments, atom charges, forces and stress. Missing, malformed, nonfinite or unconverged results fail. Relaxation additionally checks final coordinates and the requested component-wise force threshold.

| Observable | Absolute comparison gate |
| --- | ---: |
| Energy per atom | 1e-5 eV |
| Fermi energy | 1e-5 eV |
| Eigenvalues | 1e-4 eV |
| Total magnetic moment components, whole cell | 1e-4 μB |
| Atomic moment components | 1e-4 μB |
| Atomic net charges | 1e-4 electrons |
| Force components | 1e-4 eV/Å |
| Stress components | 1e-5 eV/Å³ |
| Density components | 1e-5 electrons/Bohr³ |
| Relaxed positions | 1e-3 Bohr |
| MAE difference between implementations | 5e-6 eV |

For the tightened Fe54 pair, errors were 1.85e-8 eV/atom, 1e-6 eV Fermi energy, 1.89e-6 eV eigenvalues, 1e-5 μB total moment and 5.96e-8 electrons/Bohr³ density. Forces and stress matched at the printed precision.

Fe128 matched total energy, Fermi energy, forces, stress and total/atomic moments at the printed precision; maximum eigenvalue and density errors were 5e-7 eV and 5.96e-8 electrons/Bohr³ respectively.

MAE is checked directly as E(x)−E(z), after both constituent calculations pass and the final moments remain within one degree of their requested axes. Full off-site SOC is included in the nonlocal energy, so an upstream `Eso(nil)=0` line is not treated as a separately measured SOC energy. Stdout total energies resolve only 1 micro-eV. The small basis, grid, k meshes and 1000 K FePt smearing establish implementation regression, not physically converged anisotropy.

The final portable FePt pair gave E(x)−E(z) = −0.007489 eV for both implementations, with no difference at the printed precision, passing the 5e-6 eV gate. The signed value is a regression result for these settings, not a production materials prediction. See the [direct MAE record](../results/20261005T070118Z/mae-fept2.json).

## Numerical checks and reproducibility

The native checks cover six real/complex generalized eigenproblems and six periodic-grid cases spanning two, four and eight density-matrix components, including basis-cache invalidation after geometry-like changes. All twelve passed on the A100. The six eigensolver residuals were below 4e-15 and overlap-orthogonality errors below 4e-14. All twelve local tests passed; the suite covers hierarchical timing, scientific-output parsing, spinor band counts, binary density records, convergence rejection, MAE checks, stale-input rejection and source patching. The final run's native and integration-patch hashes match the delivered source.

Each run stores its machine-readable ledger, raw stdout/stderr, upstream timers, sampled telemetry, input manifests and native counters. The full DM, RHO/DRHO, HSX, EIG and geometry artifacts are retained under the run ID in the `siesta-remag-results` Modal Volume. Compact evidence is in `results/`; raw RHO files are required to rerun the density comparison locally and can be fetched with `python scripts/fetch_evidence.py RUN_ID --density` without allocating compute. Latest metadata records source hashes, build caches, upstream commit and host details.

Component reports are regenerated from raw hierarchical timers using the current categorization. This exposes `c/r-buildD` as density-matrix work, `c/r-buildHS` as sparse-to-k-space work and SIESTA's `POISON` timer as Hartree work; original result files remain unchanged. GPU counters cover custom code only. Library-internal launches and allocations remain unmeasured, represented by null rather than inferred zeros.

## Failures retained

- [Initial run](../results/20261005T063026Z/REPORT.md): the Fe16 candidate did not converge within 200 iterations with eight Pulay history entries. The iron fixtures now use six; the accepted Fe16 pair converged in 26 iterations each.
- [Portability failure](../results/20261005T064211Z/REPORT.md): a cached CPU reference built with host-specific instructions raised SIGILL on a different Modal runtime host. Both builds now disable host-specific optimization, and the batch controller stops if the reference terminates by signal.
- [Initial portable Fe54](../results/20261005T064711Z/REPORT.md): both runs converged, but total moments differed by 2.3e-4 μB, exceeding the fixed 1e-4 μB gate. Large-cell SCF thresholds were tightened tenfold. The accepted rerun passed the unchanged gate. The unfinished Fe128 reference in that batch was cancelled before rerunning with the tighter inputs.

Failed comparisons have no qualified speedup. Inputs and evidence from earlier settings remain available instead of being overwritten.

## Runtime policy

Each REMAG run used one A100 instance, with sequential reference/candidate execution, no warm pool and no retries. The launcher refuses both running and stopping REMAG apps. The final app completed normally, and its container was absent from the Modal container inventory after completion. No SIESTA GPU instance was left running.

## Remaining work

At this 5 October checkpoint, Hartree/XC, occupations, sparse Fourier assembly,
density-matrix construction, mixing and final forces remained CPU operations.
Later stages moved the supported operations to the GPU. See the
[current capabilities](../README.md#capabilities) and
[residency record](GPU_RESIDENCY.md) for the implemented path and remaining
boundaries.
