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

The producer supports off-site SOC with retained host nonlocal H0 or the
production device H0 owner. It rejects multi-process execution,
onsite-only/collinear modes and external `Read-H-from-file`. Device SCF alone
does not invalidate a retained host nonlocal matrix.

For device H0, capture runs after converged SCF releases its lease, before
forces borrow the owner. The C++ owner checks the geometry epoch, dimensions,
sparse topology and production lifetime, then copies four complex components
into a separate snapshot. Forces may subsequently release the device owner.
The analysis writer verifies the snapshot's epoch, restored final geometry,
species and topology before writing it. Neither capture nor output reads or
overwrites the poisoned host H0 array. A stale, missing, repeated, nonfinite or
failed copy is rejected without a host fallback.

This explicit diagnostic costs `64 × sparse capacity` device-to-host bytes
per electronic cycle. The `soc_operator_output` boundary records it in the
post-SCF phase, separately from the normal H0 initialization counters; the
audit includes both in aggregate traffic. This is a requested diagnostic copy,
not a zero-copy or strict-residency claim. Without the option there is no new
copy. Device requests require the complete production owner configuration:
`REMAG_NONLOCAL_H0`, `REMAG_DEVICE_SCF`, `REMAG_GPU_INITIALIZATION` and
`REMAG_INTEGRAL_FORCE` set to `1`, with H0 shadow mode disabled.

Host outputs retain the `REMAG-NATIVE-SOC/1` metadata format. Device outputs
use `/2`, recording the snapshot epoch, dimensions and `post_scf` /
`before_forces` capture point. Python binds these fields to the native owner
audit and HSX dimensions. Selected-state exports retain this ownership record.

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
doubles, checking spin-independent cancellation, native encoding, nonmutation,
snapshot lifetime and poisoned-host independence. The real C++ ownership/copy
function is compiled with a CPU transport double, exercising lease rejection,
transfer accounting and copy failure. These tests do not exercise CUDA hardware.
Native-format synthetic fixtures exercise
Python, CLI and agent handoffs, nonorthogonal contractions, phase/subspace
behavior, evidence tampering, total-H substitution, failed branches and native
eigenvalue mismatch. The integration patch is checked against pinned source.

The compile-only check below builds all patched Fortran library sources and
the native main-program object against real pinned modules and dependencies:

```sh
python scripts/check_native_fortran.py --dependencies /path/to/pinned/External \
  --output /path/to/new/check-directory --blas /path/to/blas-library
```

The four dependency repositories must already be clean at their pinned
revisions. The check fetches nothing, applies the patch in a separate archived
source tree, and replaces only that tree's C/CUDA link dependency with an
unresolved interface target. It preserves compiler logs, input hashes and
library/object hashes. The configured translation-unit count includes optional
targets; the compiled archive member count describes the built library.

This validates real Fortran module interfaces, including the optional-NetCDF
guard in the orientation adapter. It does not link a runnable CUDA application
or execute a solver. Full CUDA linking and a measured production SOC export
remain unverified. No GPU job or pristine reference calculation is run, and
saved scientific results are not rewritten.
