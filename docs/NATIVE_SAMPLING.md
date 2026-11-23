# Native reciprocal-sampling correspondence

Structural spectral responses require comparable reciprocal sampling. The
original SIESTA KP file stores Cartesian inverse-Bohr coordinates at six
significant decimal digits. Conversion through different strained cells can
therefore produce different stored fractional arrays even for the same intended
Monkhorst-Pack grid. Treating every such difference as a changed mesh blocks
valid saved-response analysis; a blanket numerical tolerance could instead hide
a genuinely changed mesh.

`research.sampling.compare_sampling` first retains the existing literal ordered
array check (`1e-12` absolute tolerance). If it fails, the native fallback:

1. Requires the original hash-bound `input.fdf` and `remag.KP` for both accepted
   report stages. Metadata must bind the same KP file. Indirect inputs, duplicate
   settings and missing files are rejected.
2. Checks the explicit input cell against the fixed authored geometry, with
   `LatticeConstant 1 Ang`, and a diagonal zero-displacement Monkhorst-Pack block
   identical to the report's declared mesh. Skew real-space cells are supported.
3. Reproduces every stored fractional coordinate from the printed Cartesian KP
   values and the input cell. The candidate length conversions are precisely
   the CODATA-2018 and legacy conventions in pinned SIESTA 5.4. A convention is
   admitted only when it reproduces the saved arrays within a floating-arithmetic
   allowance. Both endpoints must admit a common convention; this is evidence
   of consistent conversion, not a new executable-build certificate.
4. Uses half of each printed decimal last-place unit as the Cartesian rounding
   interval. Transforming that box into fractional coordinates must identify one
   unique declared grid label per native sample. The corresponding grid point
   must also lie inside the original Cartesian box, with only a small floating
   arithmetic allowance. Unstable conversions whose allowance exceeds 1% of a
   recorded half-unit are rejected. Duplicate labels and unresolved samples are
   rejected. Coordinates equivalent modulo a reciprocal lattice vector share
   a label; sample order is never changed.
5. Checks every stored weight against its original normalized native weight.
   The printed weights must also identify positive integer mesh multiplicities
   that sum to the full grid size within their own decimal intervals. Reduced
   native sample sets retain their multiplicities. Corresponding endpoints must
   have exactly the same ordered labels and multiplicities.

The fractional conversion is `k_cart @ cell_bohr.T / (2*pi)`, matching the
existing HSX reconstruction. Bounds follow each file's actual printed precision;
there is no user-tunable matching tolerance. The fallback is bounded to 100,000
full-grid points and explicit prepared input syntax. Shifted or non-diagonal
Monkhorst-Pack matrices require an additional explicit implementation and are
currently rejected. Literal matching remains available without native context.

Both `record_response_outcome` and `analyze_response_series` use this matcher
without argument or stored-spectrum-format changes. The standalone
`compare_electronic_states` agent tool and Python `compare_electronic` also
accept optional `before_native` and `after_native` records, each with `report`,
`stage` and optional `collection`. Both must be supplied together. Their native
acceptance and spectrum-stage bindings are checked, and agent paths/held-out
source restrictions still apply. Omitting them retains literal-array matching.
Native matches record input/KP
hashes, sample/grid counts, admitted unit conventions, maximum rounding bounds,
actual stored-coordinate discrepancy and a digest of ordered mesh labels and
multiplicities. The previous failed historical comparison remains evidence;
later analysis is saved in a new output directory.

This changes **correspondence**, not the data: no coordinate snapping, weight
replacement, state interpolation, band reassignment or Hamiltonian calculation
occurs. Original reconstructed spectra retain their printed-coordinate effects.
Their energy error and the response/finite-difference uncertainty remain unknown.
Matching orbital labels also does not establish radial-basis identity or a
unique decomposition of anisotropy.

Tests generate known skew-cell grids independently, round them to native text
precision, and verify both supported length conventions and reduced weights.
Negative controls cover off-grid points, ambiguous coarse coordinates, reordered
or duplicated points, changed weights, missing samples, changed hashes, cells,
offsets, meshes and unsupported units. Synthetic series and prediction-outcome
adapters use the same proof. Saved FePt spectral responses separately demonstrate
the real archived path, with no new native calculations or modified source data.

Producer references in the pinned source: `Src/iokp.f` (KP coordinates/format),
`Src/libunits/units.F90` (build-time convention selection), and
`Src/libunits/units_codata2018.f90` / `units_legacy.f90` (length constants).
