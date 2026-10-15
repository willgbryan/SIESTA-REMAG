# Tetrahedron occupations

A self-consistent calculation of a metal has to decide, at every k point, how full each band is. SIESTA normally does this with a smeared step: each state is occupied according to a Fermi-Dirac (or similar) function of width set by `ElectronicTemperature`. Quantities that come from a few bands crossing the Fermi level, such as the magnetic anisotropy of cobalt, then need either a large width, which changes the answer, or a very dense k mesh. The tetrahedron method instead interpolates each band between the k points of the mesh and integrates the interpolated bands exactly, so it needs no width and converges faster with the mesh.

REMAG adds three tetrahedron methods to SIESTA's self-consistent loop. They are selected with `OccupationFunction`:

| Value | Method |
| --- | --- |
| `Tetrahedra.Linear` | Bands linear inside each tetrahedron of the mesh |
| `Tetrahedra.Bloechl` | Linear, plus the correction of Blöchl, Jepsen and Andersen, *Phys. Rev. B* **49**, 16223 (1994) |
| `Tetrahedra.Optimized` | Kawamura, Gohda and Tsuneyuki, *Phys. Rev. B* **89**, 094515 (2014) |

Upstream SIESTA stops on these values, so an input written for REMAG does not run silently with smearing elsewhere. `ElectronicTemperature` is ignored and the entropy term is zero, so SIESTA's free energy equals its total energy.

## How the weights are found

The k points must be a `kgrid.MonkhorstPack` mesh with a diagonal supercell; any displacement is allowed. SIESTA may list only one of each pair k and −k when time reversal holds. REMAG maps every point of the full mesh back to a listed point and stops if a point is missing, listed twice, or carries the wrong weight. Each cell of the mesh is cut into six tetrahedra that share its shortest main diagonal.

For a band that is linear inside a tetrahedron, the occupied part below a trial Fermi level is a polyhedron whose volume and corner weights have closed forms. The Fermi level is found by bisection on the electron count, which rises steadily with the trial level for all three methods. Blöchl's correction adds to each corner a term proportional to the density of states at the Fermi level. Within a tetrahedron the added terms sum to zero, so the correction moves weight between corners without changing the count. The optimized method first replaces each corner energy by a fixed combination of the energies at 20 mesh points around the tetrahedron. That combination is a least-squares fit that cancels the leading error of linear interpolation. It then hands the linear weights back to those 20 points by the same combination.

Weights from the correction and from the optimized method can be negative or exceed a full state. Only their sum over the zone is constrained. SIESTA's density matrix keeps any state whose weight is larger than `DM.OccupancyTolerance` in magnitude, so negative weights are kept. After the weights are found, states at the same k whose energies agree within 10⁻⁶ Ry share their total weight equally. Otherwise the density built from a degenerate set of states would depend on how the eigensolver chose to rotate them. Quantum ESPRESSO does the same for its linear and optimized methods.

The weights do not vary smoothly with the band energies, and the corrected and optimized energies are not variational. Forces and stress are therefore not consistent with the energy, as in other codes that offer tetrahedra. The SCF also needs more iterations than with smearing: in hcp cobalt, 20 to 50 instead of about 20.

On the CPU the weights are computed in `native/tetrahedra.cpp`, split over `REMAG.HostK.Threads` threads when that is set. In the device SCF cycle they are computed on the GPU by `native/tetrahedra_device.cu`, from the same grid, which is built once on the CPU when the cycle is prepared. The GPU code follows the CPU code step by step. The single-tetrahedron formulas in `native/tetrahedron_math.hpp` are compiled for both. The Fermi-level bisection runs in one thread block, so its sums have a fixed order, and each listed k point gathers its weights in the order the CPU adds them. The older GPU occupation route outside the device cycle (`REMAG_OCCUPATIONS=1` alone) accepts only Fermi-Dirac occupations and stops with a message. Fixed total spin (`FixSpin`) and `SingleExcitation` are not supported with tetrahedra.

## How it was checked

`scripts/check_tetrahedra.py` runs on the CPU in a few seconds, and as part of `pytest`. It checks the implementation at four levels.

The first level is a single tetrahedron. For 3000 random tetrahedra and Fermi levels, including tetrahedra with equal corner energies, the occupied part is clipped out exactly. The integral of each corner's linear weight over it is then compared with the code. The two agree to 2×10⁻¹⁶. Blöchl's correction agrees with its formula, using a numerical derivative of the occupied volume for the density of states, to 2×10⁻¹⁰.

The second level is whole meshes of a tight-binding model on the hcp cell. The checks cover a time-reversal-reduced k list against the full mesh, a half-shifted mesh, two spin channels sharing one Fermi level, and the factor two for spinless states. They also confirm that five kinds of bad input are refused: a point off the mesh, a missing point, wrong weights, an occupied top band and too few states. On the same model, the band-energy error at 24×24×14 is 1.7×10⁻³ for linear tetrahedra, 6×10⁻⁶ with Blöchl's correction and 2×10⁻⁵ with the optimized method, in units of the model's hopping.

The third level is Quantum ESPRESSO's own code. Its tetrahedron routines in `PW/src/tetra.f90` (version 7.3.1) were transcribed line by line into numpy and run on random band structures at a common Fermi level. They were run in REMAG's QE-compatible settings, described below. The two agree to 2×10⁻¹⁷.

The fourth level is finished pw.x runs. Nine calculations were run with pw.x 7.3.1 on full meshes, one per combination of three systems and the three methods. The systems are aluminium without spin, collinear hcp cobalt, and hcp cobalt with spin-orbit coupling. Their eigenvalues, occupations and Fermi energies are kept in `benchmarks/tetrahedra-qe/reference/qe-tetrahedra.json`. From QE's eigenvalues alone, REMAG finds QE's Fermi energy to 2×10⁻¹¹ Ry and its occupations at that Fermi energy to 8×10⁻¹⁵. `benchmarks/tetrahedra-qe/` holds the input generator and the script that extracts the records.

For Blöchl's method QE always cuts the cell along one fixed diagonal and does not share weights between degenerate states. REMAG uses the shortest diagonal and shares the weights, as for the other two methods. The third and fourth checks reproduce QE's choice through a setting used only by the check script.

`scripts/check_tetrahedra_gpu.py` compares the GPU weights with the CPU code on the tight-binding model of the second level, for the three methods. It covers full and time-reversal-reduced meshes, the shifted mesh with two spin channels, spinless states and the two refusals that depend on the bands, run directly and as a captured CUDA graph. In all 24 comparisons the Fermi levels are identical and the occupations agree to 2.5×10⁻¹⁷. Both refusals give the same status codes as the CPU.

