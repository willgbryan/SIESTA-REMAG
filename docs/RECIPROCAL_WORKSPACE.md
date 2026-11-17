# Saved bands and reciprocal regions

The electronic workspace links saved k points, band states, atom/orbit/motif
projections and signed DOS. It consumes the existing native-checked spectra;
viewing and exporting a path do not invoke a solver or reserve compute.

The live registered IDE initially loads the DOS and bounded state preview.
**Load saved k-space** retrieves one spectrum's complete exported arrays on
demand. The read binds the displayed report and spectrum metadata hashes, then
rechecks source, native artifacts and arrays; changed evidence requires a
workspace refresh. Original weights, signed populations and full k/band
resolution are retained. Static snapshots remain self-contained. This changes
transport/loading only, not the scientific sampling or acceptance thresholds.

## Periodic regions

Select a fractional reciprocal center and three half widths in `[0, 0.5]`.
Membership uses the componentwise periodic distance, with a `1e-12` boundary
tolerance. This is a box on the reciprocal torus, not a Cartesian sphere or a
Wigner–Seitz first Brillouin zone. Integer translations of the center select the
same samples. The scatter plot offers three coordinate projections; overlapping
points can be distinguished with the k-point selector and full coordinates.

The region filters the complete exported sample's DOS and near-Fermi states.
Original k weights are retained, without renormalizing a partial region. Empty
regions produce no states and zero DOS. Signed Mulliken contributions remain
signed. State details include the selected atom/orbit/motif weight, reconstructed
occupation and spin expectation when available.

The same selection works with `inspect_electronic_structure`, `projected_dos`
and `compare_electronic_states`:

```json
{"atom_indices":[0],"k_region":{"center":[0,0,0],"half_width":[0.126,0.126,0.126]}}
```

Choose either `k_region` or explicit `k_indices`. Comparisons require the same
selected sample indices after sampling correspondence; a rounded coordinate
crossing the region boundary is rejected, rather than silently changing the
comparison population.

## Ordered band paths

Enter 2–256 ordered saved k indices. The initial preview uses up to sixteen
points in stored order, shown explicitly in the path field. These are not
automatically identified high-symmetry paths. Repeated endpoints are allowed.
The viewer shows up to 32 energy-ranked branches intersecting ±3 eV, with the
displayed/available count stated. Clicking a point, or activating it with the
keyboard, selects that same k/state in the diagnostic controls. Point size shows
the absolute selected Mulliken weight; hollow points identify negative weights.

The view's path distance uses the exact authored source's cell converted to
numerical Cartesian coordinates. It continues to use that calculation source
when an independently relaxed geometry is displayed. Connecting lines guide
the eye: no intermediate eigenvalues or wavefunction-based band tracking are
inferred. An energy-rank crossing need not preserve an orbital identity.

For a reproducible artifact using the **native HSX cell**, run:

```sh
python -m siesta_remag.research bands collected/report.json --stage soc-z \
  --collection collected/archive-manifest.json \
  --spectrum electronic-z/electronic.json --native-hsx recovered/soc-z/remag.HSX \
  --k-indices 0 1 2 3 --output band-path
```

The `export_band_path` agent tool accepts the same paths and indices, optional
atom/orbital/l `selection`, and `breaks` giving indices at which a new segment
starts. Across a break no connecting distance is added. The tool is excluded
from scalar profiles and respects known held-out source restrictions.

Export verifies the accepted report/stage and HSX/KP/EIG hashes, reproduces the
saved fractional coordinates, native weights and eigenvalue acceptance, and
checks basis/energy-reference metadata. A mutually consistent replacement
NPZ/metadata pair cannot bypass these native checks. The saved result retains
the native residual and the explicit Bohr conversion convention.

`bands.npz` holds coordinates, cumulative distance, energies, selected signed
projections, occupations and spin expectations. `bands.json` binds it to the
report, source, native Hamiltonian and spectrum. Arrays stay out of the tool
response. Path samples have no quadrature definition; they must not be used as
a DOS integration mesh. The exporter performs no eigensolve.

## Bounds and evidence

The complete browser sample is available up to 4096 k points, 2048 states per
point and two million numerical values. Larger exports retain the older bounded
near-Fermi preview and the artifact tools; the mesh is never silently reduced.
Shared array objects are serialized once through `sampled_arrays` references,
while Python datasets retain their existing aliases. Cached selections bind
the full spectrum metadata, including its energy reference, as well as the
array bytes. A changed source/stage/definition resets the region and path.

Analytic tests cover periodic boundaries, empty regions, original weights,
skew-cell metrics and path breaks. Native-format fixtures exercise Python, CLI
and agent export, changed native bindings and failed branches. Shipped client
code is tested with DOM/canvas doubles for linked selection, keyboard controls,
regional DOS and stale-state clearing. These are not browser visual acceptance.

The saved 8³ DZP FePt example uses 512 existing k points, 60 bands, an eight-point
path and three directional spectra. A region of half widths 0.126 selects
weight 0.052734375. All 422 original files remain unchanged. This coarse
configuration is uncalibrated; neither the visualization nor agreement with
printed native eigenvalues establishes physical accuracy or a mechanism.
