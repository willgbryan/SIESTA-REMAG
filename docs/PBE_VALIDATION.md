# Resident PBE mesh component

The standalone device component passes all 34 numerical/capture scenarios against the complete, unmodified pinned GridXC `cellXC` routine. [Retained component evidence](../results/pbe-component-20261006T001018Z/pbe-check.json) comes from run `20261006T001018Z`, whose [matrix/Hartree benchmark also completed](MATRIX_HARTREE_VALIDATION.md#larger-untraced-validation). All 41 recorded hashes verify against its retained implementation archive. That image does not connect PBE to SIESTA's density/potential path; the combined integration is in the subsequent diagnostic. No full-SCF residency or PBE speed claim follows from the component checks.

`native/pbe_math.hpp` translates the original built-in PBE, PW92 correlation and nonrelativistic exchange arithmetic. It preserves GridXC's active legacy non-collinear rotation, density/gradient floors, spin clipping and off-diagonal factors. Attribution and upstream redistribution terms are retained in `native/GRIDXC_NOTICE`. Supported input is a full serial periodic FP64 grid with 1, 2 or 4 spin components, built-in PBE and `irel=0`.

`native/pbe.cu` implements the original seven-point periodic gradient stencil, an adjoint gather for the potential, four energy/double-counting reductions and all nine stress components. It preserves separate contributions from wrapped duplicate neighbors on small grids and the nonzero adjoint contribution that low-density points can receive from nearby active points. Geometry and output status live in device records. Fixed-tree reductions replace serial summation; floating-point atomics are not used for potentials or reductions. FMA contraction is disabled for this translation unit.

## Numerical evidence

Before CUDA execution, shared algebra passed 900 pointwise comparisons against pristine `ggaxc` and nine full-mesh comparisons with independently assembled Python stencils. [CPU evidence](../results/local-pbe-math-20261005/check.json) records source, oracle and library hashes. This includes scalar/collinear/NC densities, zero gradients, full polarization, low densities, and negative collinear-channel clipping. The full-mesh oracle itself passed independent collinear finite-variation checks for potential and stress.

The CUDA suite runs each of these 13 fixtures both ordinarily and through a captured graph:

| Fixtures | Required behavior |
| --- | --- |
| Constant scalar, collinear and spinor | Uniform potential and diagonal fixed-density stress |
| Smooth scalar and collinear | Periodic gradients, potential adjoint, all energy reductions and stress |
| NC texture | Pinned legacy spin-density rotation and gradient derivative convention |
| 2×3×4 spinor mesh | Repeated wrapped neighbors within the seven-point stencil |
| Fully polarized collinear | Original spin and gradient floors |
| Sub-threshold density | Original zero point-functional contribution |
| Vacuum interface | Neighbor-derived potential at skipped low-density points |
| Nearly unpolarized NC | Original polarization-angle threshold |
| Left-handed spinor cell | Reciprocal mesh orientation and positive volume |
| 40×32×28 collinear grid | Larger reduction and scratch layout |

Each scenario reuses the same device buffers for two executions. Between them a device kernel multiplies density by 0.75. The final result is compared with a fresh upstream evaluation of that changed density; nonlinear PBE results are not assumed to scale linearly. Maximum errors across all ordinary/captured cases are 2.19e-12 Ry for the four energy reductions, 6.67e-15 Ry for the potential grid and 2.84e-15 Ry/Bohr³ for stress. All checks use unchanged component tolerances of 1e-9 absolute and 1e-10 relative.

Eight additional captured scenarios retain device error status for NaN density, density overflow, NaN gradient geometry, singular gradient geometry, negative/NaN volume, mesh mismatch and spin-count mismatch. Outputs are invalid on failure; integrating consumers must honor the status. The interface rejects identical input/output pointers because adjoint assembly reads density while writing potential.

## Residency contract and remaining integration

Every captured scenario has four kernel nodes and zero forbidden host or non-device-copy nodes. The largest fixture uses 9,763,040 bytes of reusable workspace. `PbePlan::apply()` performs no explicit host/device copies, allocations, synchronization or file operations. Initial staging and final exports occur in the test-only adapter outside capture. This graph inventory does not establish whole-SCF behavior, unmeasured mapped-host access or caller-side transfers.

The [combined integration](GPU_RESIDENCY.md#resident-grid-integration) now joins retained density, sequential/clustered grid reorderings, reference-density subtraction, partial-core correction, Hartree, PBE, neutral-atom potential and the existing device potential-matrix contraction. All 32 combined CUDA numerical/capture/ownership scenarios pass. The subsequent [full grid-stage diagnostic](RESIDENT_GRID_VALIDATION.md) passes 18 upstream pairs, nine controls, four MAE checks and all 36 CUDA/filesystem traces. The DHSCF energy contract includes `Uscf`, `DUscf`, `Enaatm`, `Enascf`, XC double counting and both Hartree/XC stress contributions. Unsupported external-field, doping and other paths are rejected explicitly. Host consumers and scalar loop control remain separate residency work. Learned SCF stays deferred.
