# Electronic matrix and mechanism inspection

These tools implement selected-k access to wavefunctions, reconstructed density,
orbital coupling and endpoint state correspondence (§§5.1–5.3 of the development
specification). They operate on saved accepted native stages. They do not run
SCF, submit jobs, choose a convergence threshold or certify a mechanism.

```sh
python -m siesta_remag.research matrices collected/candidate/report.json \
  --collection collected/archive-manifest.json --stage soc-z \
  --native-directory recovered/candidate/soc-z \
  --k-indices 0 20 --output inspection/z
python -m siesta_remag.research compare-matrices \
  inspection/z/matrices.json inspection/x/matrices.json \
  --energy-window-ev -1 1 --output inspection/z-to-x
```

The equivalent agent tools are `export_electronic_matrices` and
`compare_electronic_matrices`. Both persist large arrays in hashed NPZ artifacts
and return bounded JSON summaries. Output directories must be new. At most
eight k points and 256 spatial orbitals are reconstructed per export. A complete
local result does not need `--collection`; recovered HSX and PAO files must
always match the original report's hashes. Selected eigenvalues must reproduce
native EIG within the existing 1 meV reconstruction check. Per-state residuals
are retained: this is a check against printed EIG/KP, not physical uncertainty.

## Matrix conventions

The existing HSX v2 reader and `bloch_matrices` define spatial indices, lattice
gauge, units and the interleaved Cartesian up/down spinor convention. The pinned
upstream `Src/io_hsx.F90` writes Hamiltonian and overlap, not an isolated SOC
operator; see also the [SIESTA spin-orbit documentation](https://docs.siesta-project.org/projects/siesta/en/latest/tutorials/advanced/spin-orbit/index.html).

At each selected k the generalized eigenproblem is

`H C = S C ε`, with `C† S C = I`.

The export contains H (eV), S, C, eigenvalues, occupations, a reconstructed
density matrix `D = C f C†`, and overlap eigenvalues. Occupations use the HSX
chemical potential and temperature, with a factor of two for spin-degenerate
states. Thus `Tr(D S) = Σ f` and `Tr(D H) = Σ f ε`. D is not the saved native
SCF density matrix, and band sums are not DFT total or free energies. Each k
matrix is unweighted; native k weights are stored separately. No selected-k
renormalization, overlap clipping or basis regularization is performed.

For a symmetric Löwdin representation, `H_L = S^(-1/2) H S^(-1/2)`. Grouping
these orthonormalized coordinates by the original atom and angular-momentum l
labels gives off-diagonal block Frobenius norms, largest singular values and
on-group eigenvalue lists. All zetas and spin components are included. These
are explicit basis-dependent coupling and level proxies. Symmetric
orthogonalization delocalizes the coordinates; the blocks are not unique
hybridization strengths, atomic energies or crystal-field observables. A shift
`H → H + α S` changes diagonal levels by α and leaves off-group couplings
unchanged. These matrix exports do not infer real-harmonic m labels or orbital
angular-momentum operators. The separate [native orbital-moment adapter](ORBITAL_MOMENTS.md)
uses the pinned producer's complete-shell ordering and native density, with an
explicit onsite approximation and a final native-table comparison.

## Endpoint correspondence

Comparison requires identical source, executable, orbital inventory, spin
frame, temperature and native k sampling. The actual PAO radial functions,
geometry and overlap must agree within `1e-10` in their stored units. Matching
orbital labels alone is insufficient. Structural changes or changed radial
bases require a cross-basis overlap implementation and are rejected here.

For common S, `M = C_before† S C_after`. Maximum-weight one-to-one assignment
uses `|M|²`; it can follow a state through a change of energy order. It is an
endpoint association, not evidence of an adiabatic trajectory. Spinors remain
in the fixed laboratory frame; no automatic spin rotation is applied.

Adjacent eigenvalues within the explicitly recorded degeneracy tolerance form
subspaces. Individual assignments in either degenerate subspace are marked
nonunique. For each source subspace, the tool returns the strongest target
subspace projector traces and principal-overlap singular values, plus omitted
trace. Those quantities survive arbitrary phase and unitary rotations within
the chosen subspaces. The tolerance defines diagnostic groups; it is not an
inferred uncertainty or a proof of exact degeneracy.

Assignment ambiguity also flags overlap margins of at most `1e-6` against the
next row or column candidate, including nondegenerate ties. Reported margins
are confidence diagnostics, not physical correspondence certificates.

`C_before† (H_after − H_before) C_before` is exported as a total Hamiltonian
change. It includes self-consistent exchange and charge-response changes. It
must not be interpreted as pure SOC, second-order SOC perturbation theory or
an atom-resolved MAE. Raw and own-Fermi-relative band-sum differences are
reported with electron-count and chemical-potential shifts. Their energy-zero
dependence and incomplete-k scope remain explicit.

## Validation and remaining work

Analytic controls exercise nonorthogonal normalization, density/band traces,
energy-gauge invariance, level crossings, complex phases, degenerate-subspace
rotation and rejection of indefinite overlap. Native-format fixtures exercise
agent export/comparison and rejection of changed PAOs, failed branches,
tampering and native eigenvalue disagreement. A saved 62-orbital FePt example
uses three of 13,824 SOC k points and both z/x directions, without solver work.

The separate [native SOC adapter](NATIVE_SOC.md) supplies opt-in spin-dependent
nonlocal operators when retained host H0 is available. Historical total-H
exports do not contain that isolated operator. Full-application SOC validation,
general orbital-magnetization operators, changed-basis correspondence and
perturbative/force-theorem comparisons remain separate work. No orbital mechanism or empirical research advantage is
established by this diagnostic interface alone.
