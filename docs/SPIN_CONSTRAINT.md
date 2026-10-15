# Constrained site moments

REMAG, the GPU-accelerated build of SIESTA in this repository, can hold the spin moment of chosen atoms, or groups of atoms, along directions you specify. This works in noncollinear and full spin-orbit calculations. It builds on the Lagrange-multiplier scheme of Cuadrado, Pruneda, García and Ordejón, *J. Phys. Mater.* **1**, 015010 (2018). Each constrained site feels an extra magnetic field perpendicular to its target direction. REMAG adjusts these fields until every site moment points where it was asked to. The resulting energy is the energy of that magnetic configuration, which is what magnetic anisotropy and exchange studies need.

How it was checked is described at the end. Complete calculations of hcp cobalt, compared with Quantum ESPRESSO and with experiment, are in [the hcp cobalt benchmark](../benchmarks/spin-constraint-hcp-co/README.md).

## Requesting constraints

Constraints go in a `%block REMAG.SpinConstraint` input block. Each row is one site: a polar angle and an azimuthal angle in degrees, then one or more atom indices. A row with several atoms constrains their summed moment as one site.

```
%block REMAG.SpinConstraint
  35.  20.   1        # atom 1 alone
  80. 110.   2 3      # atoms 2 and 3 share one direction
%endblock
```

Start the density with moments near the targets, for example with SIESTA's `DM.InitSpin` block of initial atomic moments. The calculation must keep the geometry fixed, run serially, and use noncollinear or spin-orbit spin. It needs `DM.AllowReuse true` with `DM.AllowExtrapolation false`, so that each new trial starts from the previous converged density, and `SCF.MustConverge true`. It cannot be combined with `REMAG.SpinRotations`, REMAG's sweep that turns the whole magnetization through a list of directions. With `REMAG_RESIDENT_SCF=1` the constraint works only through the GPU SCF cycle; the CPU route for the Hamiltonian stops with a message.

REMAG runs the SCF either on the GPU (the device path) or on the CPU (the host path, `REMAG_GPU=0`). Both use the same constraint arithmetic and the same search. The device SCF cycle accepts at most 128 k points unless `REMAG_SCF_K_LOOP=1` is set (see [dense k meshes](RUNNING.md#dense-k-meshes)).

The other options and their defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `REMAG.SpinConstraint.Direction` | `multipliers` | How the fields are updated: `multipliers`, `steepest` or `bfgs` |
| `REMAG.SpinConstraint.Penalty` | 0.02 Ry | Stiffness added inside every SCF, per Bohr magneton squared; see below. Must be positive for `multipliers`; 0 otherwise |
| `REMAG.SpinConstraint.LineSearch` | `golden` | One-dimensional search for `steepest` and `bfgs`: `golden` or `secant` |
| `REMAG.SpinConstraint.Criterion` | `angle` | Stop on `angle`, `torque` or `both` |
| `REMAG.SpinConstraint.AngleTolerance` | 0.1 | Largest allowed angle between a site moment and its target, in degrees |
| `REMAG.SpinConstraint.TorqueTolerance` | 1e-3 | Largest allowed moment component perpendicular to the target, in Bohr magnetons |
| `REMAG.SpinConstraint.InitialField` | 1 meV | Largest field change in the first line-search trial, per Bohr magneton |
| `REMAG.SpinConstraint.EnergyTolerance` | 1e-5 eV | Energy spread that ends a golden-section search |
| `REMAG.SpinConstraint.SlopeRatio` | 0.1 | Secant search ends when the slope falls to this fraction of its starting value |
| `REMAG.SpinConstraint.MaximumField` | 5 eV | Stop if any site field exceeds this, per Bohr magneton |
| `REMAG.SpinConstraint.MinimumMoment` | 0.01 | Smallest site moment that counts as magnetic, in Bohr magnetons |
| `REMAG.SpinConstraint.MaxEvaluations` | 60 | Most SCF runs allowed for the whole search |
| `REMAG.SpinConstraint.ReuseGeometry` | false | Keep the GPU's geometry-dependent arrays (overlaps, sparsity, mesh setup) between trials instead of rebuilding them. Requires the device SCF cycle |
| `REMAG.SpinConstraint.CoarseSCFScale` | 1 | Factor applied to the SCF tolerances during early trials; see below |
| `REMAG.SpinConstraint.CoarseAngleTolerance` | 0 | Angle tolerance that ends the early trials, in degrees |
| `REMAG.SpinConstraint.CoarseTorqueTolerance` | 0 | Torque tolerance that ends the early trials, in Bohr magnetons |

## How the fields are found

A site's moment is its Mulliken spin moment. The overlap between two orbitals is counted fully when both belong to the site and half when only one does. That moment is a linear function of the density matrix. For a given set of fields the constraint therefore adds a fixed matrix to the Hamiltonian. Every entry gets the overlap times the average of the two orbitals' site fields, coupled to the Pauli matrices. The field on each site is kept perpendicular to its target, so two numbers per site describe it.

Every trial set of fields costs one complete SCF calculation. The search maximizes the free energy plus the sum of each site's field dotted into its moment. For fixed fields the SCF finds the density that minimizes this sum. For any one density the sum is a linear function of the fields, because the moment depends only on the density. Taking the minimum over densities of linear functions gives a concave function, so the quantity being maximized is concave in the fields. Its slope with respect to each field component is the moment component it acts on. At the maximum the perpendicular components vanish, so every moment lies along its target, unless it collapsed to zero or points backwards. Both of those stop the run with a message.

The paper starts from zero field and searches along the direction of steepest increase, finding the best point on each line by golden section on the energy. In hcp cobalt that first zero-field trial did not converge: with the moment started at 45 degrees and nothing holding it, the SCF spent over 400 iterations slowly turning the moment toward the c axis. The energy that resists this turn is only micro-electronvolts, so the SCF has almost no restoring force to work with. A small self-consistent test model, three atoms with p orbitals described at the end, shows the same stall: 1387 SCF iterations for the paper's first trial.

The default search therefore adds a stiffness inside every SCF. Each site's energy gains the penalty strength times half the squared perpendicular moment, so the SCF itself pushes the moment back toward its target. Between SCF runs the field is updated by the penalty strength times the remaining perpendicular moment. This is the method of multipliers for an augmented Lagrangian (Bertsekas). It has the same answer as the paper's scheme, because at the answer the perpendicular moment, and with it the penalty, is zero. If an SCF does not reduce the perpendicular moment fourfold, the penalty strength grows fourfold, up to a thousand times its starting value. On hcp cobalt every angle met the default tolerances in the first SCF run, and a thousand times tighter torque tolerance in the second. On the self-consistent test model the stalled case above needed 2 SCF runs of 21 and 13 iterations.

The paper's steepest-ascent search remains available with `Direction steepest` and `Penalty 0`, together with two variants that use the slope the search already computes. The secant line search finds where the slope along the line reaches zero. BFGS directions build an estimate of the curvature as the search proceeds. The same test model can also be run with a fixed exchange field instead of self-consistent interactions. There, a strongly noncollinear case, with one atom held at 35 degrees and the other two together at 80 degrees, needed 224 SCF runs with the paper's scheme, 18 with BFGS and the secant search, and 9 with the default. Holding all three atoms together at 60 degrees took 34, 11 and 7. All reached the same fields to five digits.

The final field on a site has a physical meaning whenever the SCF potential is the derivative of the energy. It is the torque needed to hold the moment in place, so the derivative of the constrained energy with angle follows with no further calculation: minus the field component along the direction of increasing θ, times the moment along the target. On the test models, finite differences of separately constrained energies agree with this to about one part in a hundred thousand.

In SIESTA the identity holds with the local-density approximation, and with gradient-corrected functionals only if they are evaluated consistently for noncollinear spin. For hcp cobalt at a 6×6×3 k mesh, a central difference over 40 to 50 degrees, compared with the field at 45 degrees, gives a ratio of 1.005 in LDA, which is the finite-difference truncation of a sin²θ curve. It gives 2.38 with PBE as upstream SIESTA evaluates it. The libgridxc version pinned by SIESTA 5.4.2 (`ggaxc.F90`, `old_scheme = .true.`) leaves out of its potential how the local spin direction depends on the density, so that potential is not the derivative of the energy. With `REMAG.XC.NoncollinearGGA local-frame` or `multicollinear` the ratio is 1.004 or 1.005. [Noncollinear GGA](NONCOLLINEAR_GGA.md) describes the choices. Read torques off the fields only in LDA or with one of those two settings.

With `local-frame`, the holding torques of hcp cobalt at 15 to 75 degrees are within 6 to 10% of the slope of Quantum ESPRESSO's energies, in PBE and in LDA; [the hcp cobalt benchmark](../benchmarks/spin-constraint-hcp-co/README.md) traces the remainder to the basis.

Large trial fields in the line searches can make a moment jump to another stable state, as a magnet does when it switches. The next SCF then starts from the switched density and stays there, and the search can no longer improve. The run stops with a message suggesting a smaller initial field. The small default, and the secant search's extrapolation of the zero slope, both keep trials close to the answer for this reason.

## Cheaper early trials

Early trials only need to point the search in roughly the right direction. With `CoarseSCFScale` above 1, they run with SCF tolerances multiplied by that factor. They continue until the looser coarse tolerances hold. The last coarse fields are then evaluated again with the normal SCF settings, and the search finishes from there. The function being maximized is the same in both stages, so nothing learned is thrown away. The coarse stage cannot by itself declare success.

This requires the default `multipliers` direction or `LineSearch secant`. Golden section compares energies along a line, and an SCF converged to loose tolerances gives energies too noisy for that comparison. In a test where coarse trials carried density errors of one part in ten thousand, the golden-section search never left the coarse stage. With the secant search and BFGS, the noncollinear test case took 13 coarse and 8 full trials, compared with 18 full trials without the coarse stage.

REMAG's SCF has no single-precision path yet. Every SCF stage computes in double precision. Only the stored basis values are single precision. So the coarse stage currently differs only in its looser tolerances. A single-precision solver could later be used for the same coarse trials without changing the search.

## What the output reports

Each trial prints a `REMAG SPIN CONSTRAINT STEP` line with the free energy, the maximized quantity and the worst site's angle, torque and field. Then come one `REMAG SPIN CONSTRAINT SITE` line per site, with its moment, its angle from the target and its field in eV per Bohr magneton. With the default search the printed field is the multiplier only. Inside each SCF the applied field also includes the penalty strength times the perpendicular moment, which vanishes as the constraint is met. At shutdown `remag-spin-constraint.json` records every trial. The final SCF's ordinary outputs describe the constrained state.

The total and free energies printed by SIESTA exclude the constraint and penalty terms, so they are the energies of the constrained magnetic configuration. The eigenvalues, the Fermi level, the band-structure energy and the Harris energy include the constraining fields, because they come from the Hamiltonian that was diagonalized. Forces and stress omit the term that comes from the overlap's dependence on atomic positions. That is one reason the geometry must stay fixed.

## How it was checked

`scripts/check_spin_constraint_math.py` runs on the CPU in about twenty seconds, and as part of `pytest`. It compares the packing of the constraint into SIESTA's eight-component spinor storage with an operator built independently from the paper's weights. It compares the Mulliken moments the same way, and checks that the constraint's band-energy contribution equals the sum of field times moment. All inputs are fractions with small powers of two in the denominator, so every floating-point operation is exact and the comparisons are exact equality. Twelve fixtures cover noncollinear and spin-orbit storage with and without periodic images, plus one for the penalty field. A deliberate defect in one Hermitian pair breaks the identity, which shows the test can fail.

The same script drives the production host code end to end, through every search option, on a model of three atoms with p orbitals. The model has spin-orbit coupling, a crystal field, a nonorthogonal basis and an exchange field. Each trial passes through the same entry points SIESTA calls. A self-consistent version of the model, with on-site interactions, confirms that the maximized quantity is concave in the fields and that its slope equals the perpendicular moments to about one part in a hundred billion. Separate tests check every failure message's trigger.

`scripts/check_spin_constraint_gpu.py` compares the device kernels with the same arithmetic compiled for the CPU. On an H100 all 64 cases agree bit for bit, both when launched directly and when replayed inside a CUDA graph, with and without the penalty.

In complete calculations of hcp cobalt the device and host paths give free energies equal to 1×10⁻⁸ eV; the details are in the benchmark.
