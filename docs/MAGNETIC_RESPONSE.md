# Magnetic response screening

One converged FePt reference supplies angular free energies, near-Fermi orbital
weights and exchange interactions at multiple k meshes. The native exporter
captures the final generator Hamiltonian, overlap and SOC operator before the
SCF workspace is released. The sparse real-space generator supports interpolation
beyond the SCF mesh. The screen retains site identities, source hashes, reference
settings and numerical checks. Energy differences are in eV per evaluated cell.

## Running a screen

Build the native candidate in a disposable source copy with
`-DREMAG_BUILD_FROZEN_POTENTIAL=ON`. Build the `remag_magnetic` target and point
`REMAG_MAGNETIC_LIBRARY` at the resulting `libremag_magnetic.so`. Install the
`magnetic` Python extra on the CUDA host. The TWINS physics catalog accepts an
exact GPAC structure through `prepare_magnetic_screen`. An operator enables
`run_magnetic_screen` by passing `--magnetic-runtime` to `twins tool-call` or
`twins tools`. One run performs a single fixed-geometry SOC-free reference SCF,
then evaluates the requested meshes and directions. The existing
`scripts/run_modal.py` serializes the A100 allocation.

The runtime file has format `REMAG-MAGNETIC-RUNTIME/1`. It specifies absolute
`repository` and `python` paths, a pinned native `image`, `volume`,
`template_volume`, `template_relative_path`, a native `executable`, its string
`settings`, scientific `electronic_settings`, and `timeout_seconds` and
`scf_timeout_seconds` bounded to 900 seconds. The template must have the same
species and atom order as the structure. The retained FePt run includes a
[complete runtime example][runtime] and its [prepared request][request].

Prepare takes `structure` and a fresh `output` directory. Supply `meshes`,
Cartesian `directions` and optionally `angular_model`, `exchange_mesh` and
`exchange_radius_angstrom`. Run takes the prepared `request` path and a fresh
`output`. `inspect_magnetic_screen` and `compare_magnetic_screens` read completed
`magnetic-run.json` manifests. `record_magnetic_outcome` binds the selected
observable and screen hash to the exact source in the existing TWINS notebook.

Saved native exports can also be used directly. `identity.json` binds the
actual generator, geometry, reference direction, magnetic terms and PAO site
mapping. The authored workflow creates this metadata from GPAC and SIESTA.

```sh
remag magnetic-import reference/ --metadata identity.json --output state/
remag mft state/ --generator reference/ --mesh 16 16 16 \
  --directions directions.json --angular-model axial4 --output angles/
remag exchange state/ --generator reference/ --mesh 9 9 9 \
  --radius 2.8 --output exchange.json
```

## Interpreting results

Angular MFT rotates a collinear exchange field relative to fixed SOC. The free
energy uses a fixed electron count and electronic temperature. Named Mulliken
orbital weights and band shifts help locate near-Fermi changes. The exchange
calculation uses onsite rigid fields and a finite-temperature Green function.
Positive isotropic J favors parallel unit spins and each pair is counted once.
Only pairs between declared local moments enter the independent-spin network.
Induced Pt sites remain visible as electronic response channels.

The PAO reduction API projects H, S, magnetic and SOC operators together. On
FePt, a 96-to-72-state contraction accelerated 8³ MFT by 1.32× but shifted
nearest Fe–Fe exchange by 70.6%. Use the full PAO basis for that exchange
comparison. The optional 4f API requires supplied crystal-field and atomic
parameters. The [method reference][methods] records conventions and input fields.

## Measured FePt results

Restoring a=b=2.73 Å at fixed c=3.8658 Å reduced 16³ frozen anisotropy from
3.323 to 2.919 meV/cell and Fe-d/Pt-d co-weight from 0.432 to 0.401. Both
fell within their prospective intervals. The 9³ matched exchange run took
29.16 s on one A100 including first-call interpolation, compared with 150.96 s
for full-basis TB2J on one CPU worker and 158.04 s on eight. The largest error
across 12 interactions was 4.50e−14 eV against the single-worker result.
Exchange strength remains sensitive to the k mesh.

The archived [follow-up report][followup], [TB2J report and individual pair values][tb2j],
[parent and child evidence][discovery] and [initial response evidence][initial]
retain the original execution sources, measurements and failed attempts. Raw
generator matrices remain on the Modal volume recorded in those reports.

[runtime]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/benchmarks/evidence/followup-response-20261123/runtime.json
[request]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/benchmarks/evidence/followup-response-20261123/request.json
[methods]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/docs/MAGNETIC_RESPONSE.md
[followup]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/benchmarks/evidence/followup-response-20261123/REPORT.md
[tb2j]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/benchmarks/evidence/exchange-comparison-20261123/REPORT.md
[discovery]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/benchmarks/evidence/discovery-response-20261123/REPORT.md
[initial]: https://github.com/willgbryan/SIESTA-REMAG/blob/86a0a4bedb03fcd95d7310475e52b21c086410bd/benchmarks/evidence/magnetic-response-20261123/REPORT.md
