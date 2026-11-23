# Completed FePt follow-up and model tools — 23 November 2026

The existing follow-up experiment is complete. Restoring a=b=2.73 Å while
keeping c=3.8658 Å reduced frozen 16³ anisotropy and the Fe-d/Pt-d mixing
diagnostic, as prospectively predicted. TWINS' actual tool host prepared and ran
the screen, inspected and compared the results, and recorded outcomes against
the existing exact-source experiment. No new notebook or model service was added.

## Measured structural response

| Observable | Parent | Fixed-volume c+2% child | Follow-up: restore a,b |
|---|---:|---:|---:|
| a=b, Å | 2.730000 | 2.703103 | 2.730000 |
| c, Å | 3.790000 | 3.865800 | 3.865800 |
| 16³ Fx−Fz, meV/cell | 2.737267 | 3.322907 | **2.918684** |
| 16³ z spin moment, μB/cell | 3.359908 | 3.276966 | 3.346931 |
| Fe d Fermi DOS, states/eV/cell | 0.955264 | 0.863355 | 0.932939 |
| Pt d Fermi DOS, states/eV/cell | 0.310945 | 0.323126 | 0.316825 |
| Fe d/Pt d near-Fermi co-weight | 0.401946 | 0.431753 | **0.401144** |
| 8³ nearest Fe–Fe response, meV | 4.150248 | 31.423609 | 8.731044 |

The follow-up's 8³ anisotropy is 2.963484 meV/cell; changing to 16³ shifts it by
−0.044800 meV. Both meshes retain six directions (z, 22.5°, 45°, 67.5°, x, y),
with z preferred. The fixed-density reference is still the same 4³ SOC-free,
scalar-preserving PBE/TZDP, nonperturbative polarization, 400 Ry, 300 K setup.
Full PAOs are used; the earlier compact Hamiltonian was rejected for exchange.

The unchanged predictions were **3.0±0.6 meV/cell** and **0.410±0.040** co-weight.
Both measurements lie within their intervals, with errors −0.081316 meV and
−0.008856 respectively. The [tool-bound notebook receipt](feedback/feedback-receipt.json)
retains experiment `32975c19a675482090cc916910646f61`, outcome events and the
explicitly inconclusive mechanistic interpretation. The exact source remains
`b6d55f549171abbbc5c27cfd73b87432bd0f80087029ccbbd12174cd995fc106`.
Historical predictions and the earlier rejected child hypothesis are unchanged.

Restoring a,b removes 0.404223 meV of the child's anisotropy increase, supporting
the predicted direction of the in-plane response. However, the c-only follow-up
still exceeds the original parent by 0.181418 meV while d/d co-weight is nearly
unchanged. That scalar mixing diagnostic is therefore insufficient as a causal
explanation. Specific near-Fermi p/d band shifts are a useful next hypothesis;
the retained k/state contributions can be inspected before another calculation.
No additional calculation is scheduled. Exchange remains mesh-sensitive, and
Pt channels remain induced responses rather than independent local spins.

## Direct authoring-model use

TWINS registers five physics tools: `prepare_magnetic_screen`,
`run_magnetic_screen`, `inspect_magnetic_screen`, `compare_magnetic_screens`,
and `record_magnetic_outcome`. The catalog, direct-chat `tool-call`, and
JSON-lines `tools` transport share the implementation. The browser IDE is
unchanged. An author can supply the normal GPAC physical export; exact source,
Cartesian geometry, site order and physical-to-orbit site mapping are checked.

The run tool is exposed only when the operator selects a trusted runtime. The
model chooses a prepared request, meshes, directions and optional exchange mesh;
the operator chooses the native image/template and resource bounds. REMAG's
existing serialized launcher enforces the one-GPU limit and clean upstream pin.
Failures remain evidence and no scientific retry is automatic. All magnetic
tools are absent in the scalar profile, and held-out sources remain inaccessible.

The model receives named s/p/d Fermi DOS and centroids, intersite band co-weights,
near-Fermi angular band-index shifts, angular free energies and fitted surface,
magnetization, and local-moment versus induced exchange networks. Comparisons
preserve exact identities and flag unmatched settings. Recording an outcome
checks the experiment's source and binds the immutable screen hash. The existing
TWINS Notebook supplies prediction ordering and duplicate-outcome protection.

In this workspace the installed command is
`build/authoring-env/bin/twins`. The configured runtime is
`build/magnetic-runtime.json`; use it with `--magnetic-runtime` on `tool-catalog`
or `tool-call`. [Runtime snapshot](runtime.json), [prepared request](request.json),
[prepare/run receipts](tool-receipts/), and [inspection/comparison/outcome receipts](feedback/)
retain the actual round trip. Workspace-local absolute paths are provenance,
not portable defaults. See [the interface documentation](../../../docs/MAGNETIC_RESPONSE.md)
for the runtime fields and call arguments.

## Timing, evidence and checks

| Stage | Wall time, s |
|---|---:|
| Converged reference SCF | 17.031397 |
| 8³ six-direction MFT | 10.099121 |
| 8³ exchange | 18.778816 |
| Entire 8³ screen including interpolation | 31.996589 |
| 16³ six-direction MFT | 79.834563 |
| Entire 16³ screen including interpolation | 85.271275 |
| Whole remote scientific workflow | 141.283737 |

The successful run used one A100-SXM4-40GB with driver 610.57.04. It reused the
previously measured native image `im-v2SJzLdYwhBNWghlBeQgMq`; the native executable
hash and pseudopotential/input hashes are in [the native result](completed/reference/result.json).
The current Python response source is hashed in [the completed manifest](completed/magnetic-run.json)
and archived on the result volume. The native executable was not rebuilt from
the merged discovery checkout. Local compact evidence and ledger are retained;
raw generator matrices and spectra remain in `siesta-remag-magnetic-results` at
`magnetic-screen-20261123T195200Z-94ee7cd6`.

The [prior failed attempt](failed-driver620/magnetic-run.json) crashed before SCF
on driver 620.70.06 with eager CUDA loading. The successful host used driver 610;
no general driver-620 fix is established. All cloud applications were sequential.
The separately requested [TB2J comparison](../exchange-comparison-20261123/REPORT.md)
measures a 5.18× first-call advantage on the existing parent generator.

TWINS' required offline suite yielded 499 passing tests and four optional skips.
Initial failures from sandboxed loopback servers and an inherited old Ouro SDK
were resolved by using local-server permission and the declared SDK pin; only
those 53 failed checks were rerun. Its wheel/source build passed. Focused
cross-engine source/outcome/tool checks and magnetic-state regressions passed
(14 checks), followed by the four tool tests after the final coordinate-order
check. Earlier merge-focused REMAG checks passed (89, one optional skip).
Pristine upstream is unchanged. Frozen-density/basis convergence, robust exchange
trends, a general noncollinear formulation and structure-derived 4f crystal
fields remain outside this delivery.
