# Scalar-preserving SOC-free reference validation — awaiting new budget

Requested ceiling: 1,200 seconds on one exact H100, with 90 seconds reserved
for evidence. Authorization is zero; importing the app refuses execution until
this new allowance is approved. Use only `scripts/run_modal.py`. No restart,
overlapping app, extra allocation or deadline extension is included.

Nine sequential calculations, all at 300 K electronic smearing:

1. FePt z, scalar-preserving SOC-free SCF plus SOC-restored z/x: fail early if the corrected operator or moments are invalid.
2. FePt z, ordinary full-SOC SCF plus frozen x: unchanged-path regression.
3. FePt x, full-SOC SCF anchor.
4–6. B2 FeCo [001], [110], [111], full-SOC SCF anchors.
7–8. FeCo SOC-free [001] and [110] references, each with SOC-restored
     [001]/[110]/[111] targets.
9. FeCo SOC-free [001] repeat with eigenvalues-only targets.

FePt retains 48 spatial orbitals / 96 spinors, 4³ = 64 k points,
400 Ry requested grid and 0.01 Ry confinement. FeCo retains 56 spatial
orbitals / 112 spinors, 16³ = 4,096 k points, 2,000 Ry requested grid and
0.00125 Ry confinement. Both retain TZDP, non-perturbative polarization,
SplitNorm 0.25, PBE, zero mesh offsets, fixed geometry/pseudopotentials, FP64,
eight SCF lanes, DM 1e-9, H 1e-8 eV and 300-SCF cap. Frozen solves use one
streamed dense lane. The three staged 32,768 admission guards are inherited;
production remains 128. Neither guard is a capacity qualification.

The accepted base image is im-qnL5ciyPWdAwnkfIpjGXOm. Rebuild verifies the
previous executable, library and complete native-source hashes, then overlays
only the six declared native files. The executable is unchanged. No local
unrelated native edits enter this image. Current Python runner/checkers accompany
it and the implementation archive preserves the exact executed sources.

Implementation gates match each rotation to its requested target input axis
and retain convergence, signed final direction ≤1 degree, actual
stdout k counts / eigensystem jobs, raw electron counts and band/free-energy
reductions, exports and full-SCF E/F comparisons. New SOC-free references have
an extra, explicitly counted SOC-free identity solve (1e-7 eV spectrum gate).
Selected matrices at first/middle/last k retain the existing 1e-10 Ry matrix
and 1e-7 eV spectral tolerances. SOC is restored exactly once after rotating
the exchange field. Ordinary full-SOC and SOC-free SCF results cannot pass an
ordinary equivalence/performance comparison. Their final exports retain their
actual Hamiltonian treatment.

The small full-SOC FePt controls use retained pristine-SIESTA results from
mae-energy-only-c-20261025T200237Z, exact completed attempt
47cbf904d51b4be8b5ff8549d53ef3e4. The default frozen regression uses
mae-frozen-reference-20261025T221016Z, attempt
5b5324782a4f47d9bbf17d6730430ba7. No new full CPU SCF timing run is included.
An ordinary noncollinear input regenerates scalar-relativistic projectors and
is not an exact oracle for the fixed-projector spin-trace decomposition of the unsplit offsite operator.

All five full-SOC directional anchors are fresh. Approximation bias versus
those anchors and spread between the two SOC-free references are reported
separately for total-band and band-free differences, with the unchanged 5 μeV
per formula-unit diagnostic target. The two SOC-free SCFs must also agree in
E/F within 5 μeV and spin magnitude within 1e-4 μB. An approximation miss is
retained even if implementation checks pass. No zero-temperature MAE, physical
convergence, full angular anisotropy or useful/upstream-GPU speedup is claimed.

After completion: confirm app STOPPED and zero account containers, save
allocation-release.json, fetch existing outputs using the exact completed
attempt with `scripts/fetch_evidence.py --outputs-only --density --matrices
--workers 1`, run `scripts/check_frozen_matrices.py` into
`local-independent-matrices.json`, then run this directory's `analyze_local.py`
with the run name. Preserve partial records and do not create the final results
directory while its launcher is active. Update progress and commit compact
findings after local checks. No automatic promotion follows this diagnostic.

The six native overlays are frozen at d08b6c501ef17e3e57b4d84ec8637ca3c50ef737. The scalar/spin-traceless split runs after the input-staging barrier. Current runtime gates require spin-trace decomposition receipts and spin-traceless restored SOC. The failed original stage remains closed and unchanged.
