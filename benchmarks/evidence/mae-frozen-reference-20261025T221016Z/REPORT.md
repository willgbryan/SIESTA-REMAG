# Frozen-reference MAE diagnostic — 25 October 2026

All five planned calculations and all 20 executed remote comparisons passed.
The local science/export, convergence, raw occupation and spin checks pass after
fetching 45 output files from the exact completed attempt. Independent local
matrix checks also pass for all 42 captures. FeCo's
previously missing eigenvalues-only control passes. The stopping reason is
**completion of the prepared diagnostic**, with the approximation still
unqualified: changing the converged reference direction changes the sign of
FeCo's directional energies. The same 5 µeV/f.u. method target remains unmet.

The calculation attempt is `5b5324782a4f47d9bbf17d6730430ba7`, run
`mae-frozen-reference-20261025T221016Z`. No new full CPU-SCF timing was run.
Historical pristine-SIESTA evidence is retained; small independent CPU matrix
calculations diagnose the GPU Hamiltonian construction and diagonalization.

## What the diagnostic establishes

The experimental capture path compiled on sm_90 and records the exact final
SCF generator H, S, fixed lattice SOC and rotated H. Independent NumPy SU(2)
multiplication and SciPy/LAPACK generalized diagonalization agree with the GPU
at the first, middle and last k index of every evaluated direction and mode.

| Local check | Largest absolute error |
| --- | ---: |
| Rotated Hamiltonian | 6.46705e-15 Ry |
| Generator spectrum | 2.64984e-10 eV |
| Rotated spectrum | 4.08980e-10 eV |
| Canonical-lower-triangle rotation spectrum | 4.17101e-10 eV |
| Rotation invariance after removing SOC | 1.15659e-10 eV |

Matrix and spectral gates remain 1e-10 Ry and 1e-7 eV. Full-mesh identity
spectra agree with their SCF generators within 1.29e-10 eV. FeCo eigenvalues-only
and eigenvector controls agree within 5.36e-8 eV per eigenvalue and 3.79e-11 eV
in directional band/free differences. In `eigenvalues-only-comparisons.json`,
all error values are eV, including legacy field names ending `_ry`.

These are selected-k construction checks and full-mesh recorded probe gates,
not independent qualification of every rotated matrix, the complete SOC model,
or physical accuracy. The measured errors do not account for the observed
method biases. The original failures and previous incomplete control remain
preserved in their original attempts; this is a separate completed attempt.

## Reference-direction sensitivity

All energies below are **µeV/f.u. at 300 K electronic smearing**. Full-SCF E/F
are total/free energy; frozen E/F are band/band-free energy. Each frozen row
uses one independently converged SOC reference and is normalized to [001]
within that reference. These estimators must not be interchanged.

| FeCo estimator / reference | [110] − [001], E / F | [111] − [001], E / F |
| --- | ---: | ---: |
| Fresh full SCF | 7.446113 / 8.561012 | 20.732781 / 22.214647 |
| Frozen from [001] | 103.605354 / 105.007912 | 42.037277 / 43.677332 |
| Frozen from [110] | −107.327315 / −106.512299 | −88.607853 / −87.353812 |
| Frozen from [111] | −27.049677 / −26.011498 | −47.009536 / −45.647057 |
| Spread between references | 210.932669 / 211.520211 | 130.645130 / 131.031144 |

Every reference predicts itself lower than its two rotated targets. This
reference dependence and sign reversal prevent using this approximation for
screening. The largest direct pairwise frozen/full-SCF free-energy bias is
115.073311 µeV/f.u.; even the smallest FeCo pairwise bias misses the unchanged
5 µeV target. Do not select the most favorable reference or average them to
claim convergence.

FePt x − z full-SCF E/F is −1357.583585 / −541.785467 µeV/f.u.; frozen E/F
is −1357.944434 / −579.581773. The biases are −0.360849 / −37.796306 µeV/f.u.
FePt x is the retained historical full-SCF anchor; its z generator is fresh.
Agreement in total energy alone again hides the free-energy discrepancy.

## Numerical scope and moments

Both systems use two atoms, PBE, full offsite SOC, TZDP with non-perturbative
polarization, global SplitNorm 0.25, fixed geometry/pseudopotentials, zero k
shift and FP64. SCF uses eight solver lanes, DM 1e-9, H 1e-8 eV and a 300-SCF
cap. The frozen evaluator uses one dense solver lane.

| System | Spatial / spinor basis | Mesh / actual k and GPU job count | Requested / actual grid | Confinement |
| --- | --- | --- | --- | --- |
| FePt | 48 / 96 | 4³ / 64 | 400 / 443.309 Ry | 0.01 Ry |
| B2 FeCo, a = 2.8562 Å | 56 / 112 | 16³ / 4096 | 2000 / 2168.227 Ry | 0.00125 Ry |

Full-SCF spin magnitudes are 3.378620 / 3.382100 µB/f.u. for FePt z/x
(x retained), and 4.592510 / 4.592502980 / 4.592498075 for FeCo [001]/[110]/[111].
Frozen z/[001]-reference magnitudes are 3.378622441 / 3.381366104 for FePt and
4.592512972 / 4.592582591 / 4.592536218 for FeCo. All measured target directions
pass the unchanged 1° gate. The other reference vectors and raw occupation/spin
checks are retained in `local-qualification.json`. These are spin moments;
orbital contributions and total saturation magnetization are not established.

Explicit plan storage is 17,020,708 bytes for FePt spins, 30,329,508 for FeCo
spins, and 18,516,644 for FeCo eigenvalues-only. Selected matrix capture adds
589,824 / 802,816 bytes respectively. Borrowed SCF storage and library-internal
allocations are excluded. Production admission remains 128 k points; staged
32768 remains an admission ceiling, not validated general capacity.

## Allocation, evidence and next step

The single H100 app `ap-kGZs7NXNYvmJQzEj99Jqzo` is STOPPED with zero account
containers. Its conservative app lifetime, including CPU image building, was
856.537442 seconds (14.275624 minutes), within the approved 900 seconds. It
stopped at 22:24:07.205 UTC, before the unchanged 22:25:16.131 UTC deadline.
Function work before packaging took 674.829048 seconds. Actual billed GPU
seconds are unavailable. No follow-on allocation was launched.

Raw spectra, capture binaries, density/matrix outputs, implementation archive,
logs and every original record remain in the local results directory and
Modal volume. Compact records retain source identities and raw-file hashes.
The five process times include different SCF/probe work and diagnostic I/O;
there is no scientifically equivalent speedup or upstream-GPU speedup claim.

The next useful experiment is a common **SOC-free, spin-polarized reference**
(with scalar-relativistic pseudopotential terms retained), followed by the same
fixed-lattice-SOC directional solves and fresh full-SCF anchors. This is a
hypothesis to test, not a validated fix. A common reference would remove the
choice among orientation-specific self-consistent SOC potentials; its bias
must still meet the same E/F and direction targets. No such calculation has
been run or added to this allowance. Resolve that method question before
optimizing frozen-solve throughput or introducing interpolation.

The usable path remains energy-only full SCF. Basis/confinement, grid and
k/smearing convergence, geometry, independent pseudopotential/semicore,
held-out angles, electronic torque and orbital work remain open. This
fixed-setting diagnostic is neither zero-temperature MAE nor full physical
qualification or cross-package agreement.
