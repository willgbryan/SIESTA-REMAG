# Native-density orbital moments

The research API exposes per-orbital, per-atom and cell orbital moments from an
accepted saved SOC calculation. This fills the orbital-moment inspection part
of spec §5.1; it does not isolate the SOC Hamiltonian or establish a mechanism.
No solver, reference calculation or paid submission runs during this operation.

```sh
python -m siesta_remag.research orbital-moments saved/report.json \
  --stage soc-z --native-directory recovered/soc-z --output moments-z
```

For a compact collection, also supply its original `--collection
archive-manifest.json`. The JSON-lines/agent tool `export_orbital_moments` takes
`report`, `stage`, `native_directory`, `output` and optional `collection`. Paths
must stay inside the research workspace. Output directories must be new.

The exporter verifies the report, accepted stage, magnetic branch, and original
HSX/DM/log hashes. It reads the native double-precision real-space DM, verifies
its sparse topology against HSX and reproduces the final native orbital, atom
and total printed tables within their three-decimal rounding. These are native
density values integrated over the calculation's k sampling, not reconstructed
Fermi occupations from a selected-k spectrum. The new DM reader also supports
1/2/4 spin-component files, but this observable requires eight SOC components.

`orbital-moments.json` records provenance, labels, vectors in μB and the native
comparison. Its hashed NPZ contains the eight native zero-cell density
components, the real contraction kernels, orbital vectors and atom vectors.
The density arrays retain native component/row/column order; components 5+6
(one-based) supply the imaginary spin trace. They are not silently reinterpreted
as a conventional dense spinor density matrix. The Python `read_dm` API retains
all sparse cells and components when the full real-space density is needed.

The approximation and sign follow the pinned `Src/moments.F` and
`Src/spinorbit.f::int_so_ang`: same atom, same l and zeta, zero-cell pairs.
Cross-n pairs are retained when those native conditions hold. Intersite and
cross-zeta terms are excluded, and no extra overlap multiplication is added.
This is a basis-dependent onsite observable, not bulk orbital magnetization or
a general nonorthogonal angular-momentum operator. The exported kernels are
contraction weights; within an isolated complete shell, `i*kernel` satisfies
the angular-momentum algebra. That algebra is not asserted for an arbitrary
multiple-zeta nonorthogonal basis.

HSX stores n/l/zeta rather than m. Complete shell runs recover m=-l..l from the
pinned producer's `atom.F::comBasis/comPOL` real-harmonic ordering. Incomplete,
ambiguous or higher-than-f shells are rejected. This derivation is recorded in
metadata instead of claiming m was read directly from HSX. New producer
conventions require a new adapter, not an assumption of compatibility.

Workspace calculation entries can register
`"orbital_moments": {"soc-z": "moments-z/orbital-moments.json"}`. The viewer
checks the calculation/HSX/DM bindings and recomputes the displayed contractions
from the hashed arrays. Atom/orbit selection changes the moment vector and its
orbital contributions; switching direction or structure clears absent data.

Validation includes analytic circular p states, s/p/d/f spectra and commutators,
atom/zeta/translated-cell exclusions, both binary byte orders, malformed-file
controls and saved-artifact/branch failures. A component test compiles only the
pinned angular arithmetic and compares all 83 p/d/f matrix-element calls. It
does not build or run SIESTA. Saved FePt z/x/y density contractions also reproduce
all native table entries. These checks verify implementation, not physical
accuracy, basis completeness or a new-query latency target.
