# REMAG discovery response — 23 November 2026

Native dense-k MFT, faster exchange, and an actual GPAC → REMAG → TWINS → GPAC feedback loop now work. The FePt example rejected the model's initial structural hypothesis and produced a different, precisely authored follow-up. A 96 → 72 spinor PAO contraction is useful provisionally for anisotropy; it does not preserve exchange adequately. Material-aware 4f work remains deferred.

## Dense frozen MFT

The native exporter now retains the **actual final sparse real-space generator**, its overlap and separate magnetic/SOC operators. GPU Fourier interpolation uses its 205 lattice translations, independent of the original SCF k mesh. Reconstructing the native source mesh differed by at most **3.11e−15 Ry**. This avoids extrapolating a Fourier fit to only 4³ matrices.

One converged 4³ SOC-free FePt reference supplies all the following evaluations. The reference includes the scalar part of the relativistic operator; screening restores its spin-traceless SOC part. Settings are TZDP with nonperturbative polarization, 400 Ry grid cutoff, PBE and Fermi–Dirac occupations at 300 K, fixed electron count and FP64 response arithmetic. The two-site primitive cell has a=b=2.73 Å and c=3.79 Å.

| Mesh | Fx−Fz, meV/cell | Directions | MFT time, s | Interpolation + MFT, s |
|---|---:|---:|---:|---:|
| 4³ | −0.680212 | 3 | 1.149 | 1.303 |
| 8³ | +2.994588 | 3 | 5.097 | 6.165 |
| 12³ | +2.722036 | 3 | 17.243 | 20.452 |
| 16³ | +2.737267 | 6 | 80.941 | 89.913 |

The sign becomes positive, favoring z. The 12³ → 16³ change is **0.01523 meV (0.56%)**. The six-direction surface retains z, 22.5°, 45°, 67.5°, x and y samples, an axial sin²θ/sin⁴θ fit, angle-sensitive near-Fermi band diagnostics and k-space energy contributions. This is stabilization of the **frozen response**, not dense-SCF, basis or physical MAE convergence. No directional SCF reruns were needed for these meshes or angles.

## Exchange throughput and interpretation

The default engine now evaluates complex-energy Green's functions, Fourier-transforms them once per pole, and contracts the requested real-space site pairs. This replaces O(nk²) spectral pair work with O(nk × poles) work, apart from eigensolution and orbital contractions. It retains the onsite rigid-field approximation. The pole expansion follows [Ozaki's continued-fraction representation](https://www.openmx-square.org/tech_notes/CF_Fermi.pdf).

| 12 interaction responses | Time, s |
|---|---:|
| Previous spectral-pair algorithm, 4³ | 52.240 |
| New algorithm, 4³ first call | 16.826 |
| New algorithm, 4³ warm call | 3.600 |
| New algorithm, 8³ warm call | 7.473 |

The first-call improvement is **3.10×**. The warm result is 14.51× faster than the saved old measurement; that comparison includes a warm-up advantage. These are single observations, not timing distributions. Old/new 4³ interactions agree within **1.73e−10 eV**. The first 8³ call in the child process took 18.529 s, making startup/JIT costs explicit.

Sites now carry local-moment, induced, nonmagnetic or unassigned roles. Only pairs of declared local moments enter the independent-spin network. Pt channels remain available as electronic responses. The fallback spectral algorithm follows the same interpretation.

Exchange physics remains mesh-sensitive: the parent nearest Fe–Fe response changes from **28.927 meV at 4³ to 4.150 meV at 8³**. Faster execution does not make that interaction converged. No ordering temperature or robust exchange enhancement is inferred.

## Completed authoring feedback loop

GPAC authored exact parent and child words, preserving Fe/Pt site correspondence. The adapter checks the evaluated geometry against SIESTA's actual geometry and binds the native PAO quantum numbers. It reuses the electronic template and pseudopotentials. All predictions, outcomes and interpretations were written through **TWINS' existing `Notebook`**, with no new registry. The initial Python integration is now exposed through TWINS direct chat/JSON-lines tools; the browser IDE is unchanged.

Before observing the child, the model predicted that **c+2% at fixed volume** would weaken Fe-d/Pt-d near-Fermi mixing and reduce anisotropy. GPAC applied c×51/50 and a,b÷√(51/50), yielding a=b=2.7031027923 Å and c=3.8658 Å. Both reference SCFs converged: parent 15.634 s; child 16.621 s.

| Descriptor | Parent | Child |
|---|---:|---:|
| 16³ Fx−Fz, meV/cell | 2.737267 | 3.322907 |
| 16³ z spin moment, μB/cell | 3.359908 | 3.276966 |
| Fe d projected Fermi DOS, states/eV/cell | 0.955264 | 0.863355 |
| Pt d projected Fermi DOS, states/eV/cell | 0.310945 | 0.323126 |
| Fe d/Pt d near-Fermi co-weight | 0.401946 | 0.431753 |
| 8³ nearest Fe–Fe response, meV | 4.150248 | 31.423609 |

Anisotropy **increased by 0.585640 meV (21.4%)**, and the mixing diagnostic also increased. Both outcomes fell outside the prospectively recorded uncertainty intervals: 2.4±0.6 meV and 0.38±0.04 respectively. TWINS retains the rejected predictions. The child 8³ anisotropy was 3.165634 meV, so its remaining mesh dependence is explicit.

The model can now inspect named s/p/d projected DOS, near-Fermi centroids, intersite band co-weights, common-angle energy changes, k-space contributions and angle-sensitive band-index shifts. Some strongly shifting near-Fermi states have substantial Fe/Pt **p** weight as well as d weight. These diagnostics do not establish a unique orbital contribution to MAE: Mulliken partitions depend on representation, and sorted band indices can cross.

The measured changes motivate an authored follow-up: **restore a=b=2.73 Å while keeping c=3.8658 Å**, separating the original in-plane compression from the c stretch. The new hypothesis predicts reduced mixing and anisotropy relative to the child, with explicit alternatives involving band crossings and p/d redistribution. The exact follow-up word and prospective predictions were saved before execution. The [completed follow-up](../followup-response-20261123/REPORT.md) now measures 2.918684 meV/cell and 0.401144 co-weight, both within those intervals.

Exact GPAC source identities:

- Parent: `00f64fc152db148830f1bf26de347a68122ff055156cf912da90bd179a1a20c1`
- Child: `2655ff8b059af4ee36a671ce59b4a20a4cea135c0d0dc58f2d6cec556095d827`
- Follow-up: `b6d55f549171abbbc5c27cfd73b87432bd0f80087029ccbbd12174cd995fc106`

These are validated exact P1 source words; no highest-space-group canonicality is claimed. [Notebook and feedback receipt](authoring/feedback-receipt.json), [follow-up structure](authoring/followup/structure.json), and [paired electronic changes](child/parent-child-k16.json) retain the full example.

## Genuine PAO compression

`contract_radial_manifolds` learns two local radial combinations per occupied shell from occupied and nearby SOC states, preserving complete angular manifolds. One constant local map projects **all** real-space H/S/magnetic/SOC operators. Construction took 0.950 s and reduced 96 spinor states to **72**.

At 8³, three-direction MFT fell from **5.097 to 3.871 s (1.32×)** with maximum anisotropy error **0.05122 meV**. This is a provisional screening option at the tested settings. Exchange was faster but its nearest Fe–Fe response changed from 4.150 to 7.080 meV, a **70.6% error**: retain the full PAO model for exchange. The [observable-specific assessment](contraction-assessment.json) supersedes the raw harness's MFT-only `screening_use` label. No model is automatically promoted.

## Evidence and remaining limits

- [Machine-readable summary](summary.json) preserves individual times, identities, software/source hashes, failed attempts and unknown telemetry as null. [Parent records](parent/summary.json) and [child records](child/summary.json) retain measured implementation hashes. Frozen harness copies are in `harness/`; their original staged directories and source bundles retain execution context.
- Raw matrices, sparse generator, spectra, contracted coefficients, input/pseudopotential provenance and source archives remain in Modal volume `siesta-remag-magnetic-results`, under `discovery-response-20261123T175305Z` and `discovery-child-20261123T180908Z`. Compact results are also in the matching local `results/` directories. The parent app was stopped after scientific completion because packaging large retained orbital arrays kept it allocated. Full raw orbital-weight retention is now opt-in.
- Both successful runs used one A100-SXM4-40GB, driver 610.57.04, CUDA 12.6.3 and CuPy 14.2.0. The child scientific workflow, including contraction and warm exchange, completed in 158.301 s. GPU allocations were sequential and are now released. Times exclude image building, provisioning and final evidence packaging; no matched upstream-GPU speedup is claimed.
- Two attempts crashed before SCF on driver 620.70.06. The successful child also enabled `CUDA_MODULE_LOADING=EAGER`, but ran on driver 610.57.04, so a driver-620 fix is **not established**. The [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html) describes that setting; failed attempts and logs remain retained. Input-format/preflight failures were fixed and retained separately.
- 17 focused existing tests passed, plus the same-composition authoring handoff check. Pristine upstream remains unchanged at its pinned commit. No independent CPU solver was added. The later, specifically requested [focused TB2J comparison](../exchange-comparison-20261123/REPORT.md) reuses the same saved parent generator. The earlier high-energy all-band regression issue remains documented in the [initial report](../magnetic-response-20261123/REPORT.md).
- Dense frozen MFT still inherits the 4³ reference density; exchange and compact-model accuracy need care at new settings. Structure-derived 4f crystal fields remain unimplemented. These limits do not prevent the demonstrated electronic feedback loop, but they restrict claims of quantitative materials prediction.
