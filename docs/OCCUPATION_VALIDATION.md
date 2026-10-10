# Device occupations and retained spectrum integration

The opt-in `--occupations` path passes all 16 upstream comparisons, eight direct GPU controls, both Gamma/sampled-k MAE checks and all 16 CUDA trace-coverage checks in [20261005T221744Z](../results/20261005T221744Z/REPORT.md). It requires `--dm`; `--occupation-ablation` runs an otherwise identical GPU control using upstream host occupations. All 26 local tests pass, and the patched Fortran interfaces compile successfully. The run's 28 native checks, 27 isolated FD scenarios and known-copy CUPTI check pass. All 28 source hashes verify, and the single Modal GPU was released. Traced timings do not qualify speedups.

The existing device FD kernel reproduces upstream's bisection, charge tolerance, temperature floor and entropy convention. Ten fixtures pass both ordinary and CUDA-graph execution against compiled, unmodified `fermid.F`, plus seven error scenarios. The tenth fixture covers upstream Gamma non-collinear's explicit half-weight convention.

The preceding run `20261005T221646Z` stopped before DFT in the native checker. Its terminal bad-normalization test deliberately leaves a failed density plan active; the new occupation test, appended afterward, refused to overwrite that active plan. Moving the terminal-error check last resolves the test-order issue. The failed log and source archive are preserved with all 28 hashes verified. Its GPU was released before the retry.

## Integrated evidence

All 306 candidate SCF iterations use device occupations and all 5,349 density consumers borrow resident E/Q blocks. No candidate SCF uploads an occupation or energy-weight array for density construction. Setup is staged once per geometry, including nine updates across relaxation. Debug ownership verification is enabled throughout. Counts below describe explicit copies in each candidate, not a whole-pipeline byte reduction against runs with differing iteration counts.

| Case | Device FD calls | Retained E/Q borrows | Setup uploads | Avoided E/Q upload bytes | Temporary FD/Q download bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fe2 forces | 25 | 700 | 1 | 94,800 | 101,600 |
| Fe2 relaxation | 96 | 2,688 | 9 | 364,032 | 390,144 |
| Fe2 NC Gamma | 10 | 10 | 1 | 5,760 | 3,200 |
| FePt2 SOC Gamma z | 36 | 36 | 1 | 20,736 | 11,520 |
| FePt2 SOC Gamma x | 38 | 38 | 1 | 21,888 | 12,160 |
| FePt2 SOC z | 35 | 945 | 1 | 327,360 | 273,280 |
| FePt2 SOC x | 32 | 864 | 1 | 299,504 | 249,856 |
| Fe54 collinear | 34 | 68 | 1 | 528,768 | 265,472 |

The new bridge stages 2,224 bytes in total across these cases. The remaining downloads can exceed the avoided uploads, as in the force and relaxation cases, because all occupations are exported while the old density builder uploaded only its occupied prefix. This step therefore does not uniformly reduce transfer volume. Fe54 uses 34 iterations versus 33 for upstream and 31 for the host-occupation control; no timing improvement is inferred.

Maximum upstream-comparison errors are 1.852e-8 eV/atom in energy, 1.0e-6 eV/Å in forces, 1.0e-6 eV/Å³ in stress, 1.0e-5 μB in total moment, 7.6e-7 eV in eigenvalues, 8.941e-8 e/Bohr³ in density and 9.0e-9 Bohr in relaxed coordinates. Both MAE errors are zero at stored precision. All original tolerances remain unchanged. Isolated FD maximum errors remain 1.388e-17 in Fermi energy, 1.422e-13 in entropy and 1.777e-15 in occupations, with charge error below 1e-10.

CUPTI coverage is complete with uploads matching explicit counters. Each candidate still has 16 additional downloaded bytes per eigensolver call, consistent with the previously measured library scalar boundary. No strict residency claim follows. Fe54 alone records 452,285,856 uploaded and 442,821,324 downloaded explicit bytes after its first SCF iteration. This run did not request filesystem tracing; earlier I/O evidence cannot certify these new executions.

## Ownership and numerical contract

Each supported `diagon` call declares an eigenvalue/occupation bank with its shape, host-array identity, geometry epoch, k-point weights, temperature and charge. Successful eigensolver producers copy their device spectrum directly into that bank. Missing, foreign or duplicate producers fail explicitly. Once occupations have been computed, the bank is frozen until the density builder consumes its blocks.

This distinction matters for sampled k points: upstream first solves for eigenvalues in `eo`, computes occupations, then solves again for eigenvectors with a temporary eigenvalue output. Its density and energy-density weights use the first `eo` spectrum. The GPU integration preserves that convention; a second solve must not overwrite the retained first spectrum.

The device FD kernel writes occupations, Fermi energy, entropy and error state. Density construction borrows matching retained spectrum/occupation blocks without uploading those arrays. Each block has a single consumer. The debug epoch mode additionally compares host E/Q and immutable weights with snapshots and rejects modifications to owned data. Normal operation trusts the explicit lifetime contract and does not scan the host E/Q arrays.

Native checks exercise the actual eigensolver → FD → sparse D/E path, including first-pass spectrum preservation, missing/duplicate/foreign producers, incorrect output identity, repeated consumers, host E/Q mutation, changed weights and geometry invalidation. The numerical check uses independent scalar root finding and analytic diagonal density/energy-density matrices. These checks complement, rather than replace, full upstream physics comparisons.

The integration supports serial variable-spin FD occupations in scalar, collinear, non-collinear and SOC Gamma/sampled-k density builders. Fixed spin, spin spirals, file-backed wavefunctions, LDOS energy windows, alternative occupation functions and single excitation are rejected when enabled. The kernel's current contract requires positive finite k weights; upstream Gamma non-collinear's half weight is supported explicitly.

## Remaining boundaries

This stage deliberately retains eigenvalue exports, occupation-array exports for existing host consumers and a 32-byte result/status download per FD call. Configuration and weights are uploaded when their declared lifetime changes. `remag-occupations.json` and phase counters record these copies separately; direct device spectrum copies are also counted. Device D/E normalization still reads back its error status, and D/E still return to CPU consumers.

Consequently this is a consumer migration, not a zero-transfer loop. Host band selection, electronic bookkeeping, Hamiltonian/potential construction, mixing/convergence and final observables still need migration. CUPTI has also exposed additional scalar transfers inside the production eigensolver. Learned acceleration remains deferred.

The next stage, deferred spectral export, now passes a [separate nine-case regression](SPECTRAL_RESIDENCY_VALIDATION.md) behind `--resident-spectra`: 18 upstream comparisons, nine controls, both MAE checks and all 36 CUDA/filesystem trace checks. It requires `--occupations`; `--spectral-ablation` supplies the earlier per-iteration-export control. Larger untraced validation is still in progress.

Sampled-k density builders retain all requested vectors from the cuSOLVER full-spectrum solve. The device density kernel applies the original occupation cutoff, avoiding both a host Q scan and a new band-count download. Gamma builders already use all requested vectors. Host E/Q buffers remain unchanged throughout SCF; the bank may replace unpublished results only for the same owner/layout/geometry. An explicit call immediately after the post-SCF phase marker exports the latest spectrum and occupations for analysis and subsequent host workflows. All spectrum blocks must have been consumed before the bank closes.

The runner rejects missing exports or any SCF eigenvalue/occupation download in this mode. Debug guards additionally reject host-buffer changes before consumers, subsequent iterations and export. Native checks cover multiple iterations before export, changed spectra, nonzero occupations below the density cutoff, incorrect export shape/identity, duplicate export and missing consumers. All patched Fortran paths compile locally; all 26 local tests and 29 native scenarios pass. E/Q remain on device across 365 candidate SCF iterations, with 17 post-SCF exports. FD result/status and eigensolver status downloads remain.
