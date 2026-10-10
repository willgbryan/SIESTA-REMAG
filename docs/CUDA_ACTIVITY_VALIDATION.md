# CUDA-library activity and device-control validation

**Phase-attribution correction (6 October):** This report retains the historical schema-1 results. Later known-copy calibration `20261006T033337Z` found host/device timestamp misalignment. Whole-process totals and scientific results remain valid; phase-specific zero-transfer claims require requalification with the schema-2 launch-correlation observer. See [the residency record](GPU_RESIDENCY.md).

Run [20261005T212723Z](../results/20261005T212723Z/REPORT.md) passes all six upstream scientific comparisons, MAE and six CUPTI coverage checks. It covers displaced Fe forces, nine-geometry relaxation, Gamma non-collinear Fe, sampled-k FePt SOC in both directions, and Fe16 non-collinear. The single Modal GPU was released after completion. All 27 archived source hashes verify. The usual 27 native solver/density/grid checks, 25 isolated occupation scenarios and the known-copy instrumentation check pass. All 26 local tests pass.

The prior run [20261005T211820Z](../results/20261005T211820Z/REPORT.md) also passes six scientific comparisons and MAE, but its traces correctly fail coverage. Final synchronization from a C++ destructor occurred after CUDA began unloading. Those failed traces are retained unchanged. The fix calls optional trace finalization explicitly in `siesta_end`, after scientific work and FDF shutdown, while the CUDA runtime is alive. Its Fortran seam compiled locally before the corrected GPU run. The previous container/app was released before the rerun launched.

## What the trace establishes

The optional CUPTI observer records memcpy and concurrent-kernel activity, including CUDA-library calls, from process initialization through explicit finalization. It keeps data in bounded host memory until finalization. Native phase hooks use CUPTI timestamps directly. Completion requires successful process execution, activity enablement, zero dropped/overflowed records, valid timestamps, full SCF marker coverage and observed transfer totals no smaller than explicit bridge totals. No boundary overlaps occur in these six traces.

Before any traced DFT, a separate executable stages a known array, performs two GPU increments, one array device-to-device copy, two four-byte status downloads, and a final array export. The observer detects exactly the expected two SCF kernels, 32,768-byte device-to-device copy, eight all-SCF status bytes and four steady-state status bytes. This validates the observer against operations independent of SIESTA's counters.

| Case | SCF iterations | Solver calls | Additional downloads beyond explicit copies | Additional downloaded bytes | Recorded kernels, whole process |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fe2 forces | 25 | 1,400 | 4,200 | 22,400 | 45,630 |
| Fe2 relaxation | 96 | 5,376 | 16,128 | 86,016 | 175,237 |
| Fe2 NC Gamma | 10 | 10 | 30 | 160 | 1,155 |
| FePt2 SOC z | 35 | 1,890 | 5,670 | 30,240 | 136,260 |
| FePt2 SOC x | 32 | 1,728 | 5,184 | 27,648 | 124,581 |
| Fe16 NC | 45 | 720 | 2,160 | 11,520 | 92,390 |

Observed host-to-device bytes exactly match explicit bridge bytes in every case. Each has three additional downloads and 16 additional bytes per solver call. After subtracting the bridge's four-byte solver and normalization status copies, the SCF size counts are consistent with two four-byte and one eight-byte library readback per solve. This is aggregate/size evidence, not an API-stack attribution. The raw trace preserves stream, context and correlation identifiers for further attribution.

The first Fe2 force candidate records 10,920 copies after the first SCF iteration, including 34,275,936 downloaded bytes. Fe16 non-collinear records 5,940 steady-state copies, 2,246,119,328 uploaded bytes and 1,000,595,376 downloaded bytes. Bulk array boundaries remain dominant in byte volume, while these extra scalar readbacks expose additional host control that bridge-only counters missed.

Maximum scientific differences are 1.0e-6 eV/Å for forces, 3.0e-7 eV for eigenvalues, and 1.1921e-7 e/Bohr³ for density; reported total energies, Fermi energies and moments agree at stored precision. MAE error is zero at stored precision. All existing gates are unchanged. Every traced speedup is deliberately null.

Coverage is limited to requested memcpy/kernel activities. Unified-memory migration, kernels accessing mapped host memory, CPU-library computation/synchronization, and activity after trace finalization remain unmeasured. The observer does not certify strict residency or filesystem behavior. Separate syscall evidence is in [I/O validation](IO_VALIDATION.md).

## Solver control experiment

The isolated probes in `20261005T211820Z` use synthetic tridiagonal Hamiltonians and diagonal positive overlap matrices. The generic batched solver, with batch size one after Cholesky/triangular reduction, passes two changing-Hamiltonian iterations in a conditional device graph at size 64 for both real and complex matrices. Each graph has 170 nodes with no visible host, host-to-device, device-to-host or default-copy nodes. Residuals are below 1.2e-15 and overlap-orthogonality errors below 2.8e-15. Requested host workspace is zero.

Size 576 and 1152 warmups succeed but graph capture fails at the standard batched eigensolve with cuSOLVER status 6. Their requested host workspace is also zero: that fact alone does not establish device-only execution. All 18 generalized/Jacobi variants still reject capture. The small graph result does not qualify a replacement for the production solver or establish full-library residency. Its exact source and all 24 probe results are retained with 27 matching source hashes.

## Device occupation component

The independent FD occupation kernel passes 25 scenarios in both diagnostic runs. Nine fixtures compare with compiled, unmodified pinned upstream `fermid.F`; each is checked with ordinary and captured execution. They cover scalar/collinear/spinor layouts, weighted k points, up to 1,152 bands, padding, degeneracy, the temperature floor and full bands. Seven additional captured cases verify invalid-input, insufficient-band and iteration-limit statuses.

Maximum differences from upstream are 1.388e-17 in Fermi energy, 1.422e-13 in entropy and 1.777e-15 in occupations. Charge error stays below upstream's 1e-10 bisection tolerance. The kernel retains bisection state, termination and error status on device; its test adapter explicitly stages inputs and exports results. It was not integrated into SIESTA in these two diagnostic runs. The later [occupation integration](OCCUPATION_VALIDATION.md) passes 16 upstream comparisons, eight controls, both MAE checks and 16 CUDA traces with retained E/Q consumption. Host consumers and final scalar transfers remain unfinished work.
