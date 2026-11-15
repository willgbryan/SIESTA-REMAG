# Native SOC operator inspection

The optional `REMAG.WriteSOCMatrix true` FDF consumer exports the
**spin-dependent part of the off-site nonlocal pseudopotential** in serial
REMAG builds (§§5.1–5.2 of the development specification). It does not subtract
two self-consistent total Hamiltonians or label exchange splitting as SOC.

The pinned `setup_hamiltonian.F` adds four complex nonlocal spin blocks to
eight real native Hamiltonian components. With `SOC.Split.SR.SO` disabled,
those blocks also contain a spin-independent nonlocal part. The export forms

`V_SOC = V_nonlocal − ½ Tr_spin(V_nonlocal) ⊗ I_spin`

before applying the pinned component encoding. This definition is independent
of the native scalar-relativistic/SOC energy bookkeeping split. It excludes
exchange, local potential, Hubbard and other total-H terms. It is not a unique
atom partition of anisotropy or a physical-accuracy statement.

The producer writes double-precision `remag.SOC.HSX` and a small
`remag.SOC.json` definition record. Its temporary matrix uses the ordinary
HSX sparse topology, overlap, geometry, orbital metadata, chemical potential
and k-point descriptor; it does not modify H, S or the nonlocal matrix. The
hook runs after analysis restores the final electronic geometry and is inert
unless explicitly requested.

The first producer supports off-site SOC with retained host nonlocal H0. It
rejects multi-process execution, onsite-only/collinear modes, external
`Read-H-from-file`, device-owned nonlocal H0 and poisoned host H0. Device SCF
alone does not invalidate a retained host nonlocal matrix. Device-owned H0
needs an explicit copy/ownership contract; stale host values are never used.

## Prepare and inspect

```sh
python -m siesta_remag.research prepare structure.json \
  --pseudos pseudos --query directions.json --soc-matrix --output query
```

The equivalent `prepare_query` tool accepts `soc_matrix: true`. This prepares
work without authorizing execution or reserving funds. Only requested SOC
stages receive the FDF flag; the scalar seed is unchanged. Scalar-only queries
reject the option. Historical requests omit it and retain their original
inputs. Execution verifies the requested native operator pair before another
direction, so an older binary silently ignoring the flag cannot return a
measured diagnostic result.

After a separately authorized execution has produced the artifacts:

```sh
python -m siesta_remag.research soc-matrices collected/report.json \
  --collection collected/archive-manifest.json --stage soc-z \
  --native-directory recovered/soc-z --k-indices 0 20 --output inspection
```

`export_soc_matrices` exposes the same operation to physics-profile research
agents. Scalar-profile tools exclude it; known held-out restrictions apply.
The existing checked total-H exporter reconstructs at most eight k points and
256 spatial orbitals. Native HSX/KP/EIG, actual PAO arrays, accepted stages and
magnetic branches retain their checks. Both SOC artifacts must match original
report hashes. SOC and total-H metadata, geometry, topology and overlap must
agree, and the SOC spin trace must vanish.

The hashed NPZ contains the covariant operator, its eigenstate and symmetric
Löwdin representations, occupations, eigenvalues and selected k weights. For
`C† S C = I`, matrix elements are `C† V_SOC C`, without an extra S. The Löwdin
operator is `S^(-1/2) V_SOC S^(-1/2)`. The occupation contraction is checked
against `Tr(D V_SOC)` using reconstructed `D = C f C†`. It is an operator
expectation, not a DFT total energy or perturbative estimate. Selected k
weights are retained without renormalization.

Individual entries depend on phases and eigenvectors within degenerate
subspaces; fully occupied subspace traces survive internal unitary rotations.
Atom/l Löwdin blocks are basis-dependent. No force-theorem MAE, second-order
SOC energy or isolated operator is inferred from historical total-H exports.

## Validation scope

Tests compile the component routine and adapter against small interface
doubles, checking spin-independent cancellation, native encoding, nonmutation
and unsupported-owner rejection. Native-format synthetic fixtures exercise
Python, CLI and agent handoffs, nonorthogonal contractions, phase/subspace
behavior, evidence tampering, total-H substitution, failed branches and native
eigenvalue mismatch. The integration patch is checked against pinned source.
No solver calculation, GPU job or pristine reference is run. A full native
application build and measured production export remain unverified for this
increment. Saved scientific results are not rewritten.
