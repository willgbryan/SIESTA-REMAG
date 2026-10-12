# Hopper support

REMAG builds native SM90 code for NVIDIA H100 alongside its existing SM80
A100 target. Select one H100 through the serialized Modal launcher with
`--gpu-model H100`; the GPU count remains one. See the
[integrated execution command](RUNNING.md#single-gpu-execution).

The previous SM80 CMake setting also emits forward-compatible PTX, so older
binaries may already execute on H100 through driver JIT compilation. This
release qualifies native SM90 execution and exposes H100 in the launcher;
it does not establish that earlier binaries were incompatible with Hopper.

This change preserves the numerical kernels and their tolerances. It adds a
shared CMake architecture default, `80;90`, and removes the per-target SM80
settings that previously overrode architecture selection. Explicit
`CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS` settings remain available for custom
builds. Use a fresh build directory or explicitly update an existing cache.

## Qualification

Qualified on 12 October 2026 using one **NVIDIA H100 80GB HBM3**
(compute capability 9.0), CUDA 12.6.85 and driver 610.57.04. All **72 existing
solver checks** passed: 36 in the default mode and 36 with optional FP64 tiles.
All **10 material calculations across eight inputs** passed direct comparisons
with retained pristine CPU SIESTA outputs. Both x/z directional-energy checks
passed. No scientific tolerance was relaxed.

The native library was rebuilt from the release's CUDA/C++ sources with SM80
and SM90 code. It was linked at runtime to the unchanged, previously validated
SIESTA integration executable; both binary hashes are recorded. The run used
`CUDA_DISABLE_PTX_JIT=1`, and `cuobjdump` confirmed native SM90 cubins. This
qualifies native Hopper execution, including the device SCF cycle.

The cases cover collinear Fe54, noncollinear Fe16, nonparallel full-SOC Fe3,
full-SOC FePt16 x/z and FePt128 z, and a DZP/400 Ry/4³/300 K two-atom FePt x/z
pair. FePt128 also passes with optional FP64 tiles; FePt16 z passes the optional
GPU radial-table mode. All other material checks use upstream radial tables.
Energies, eigenvalues, densities, density matrices, forces, stress, moments,
convergence and requested exports are checked. The nonparallel case also
verifies its final magnetic state and full-SOC matrix content.

| Input / mode | Seconds | SCF iterations | Numerical checks |
| --- | ---: | ---: | --- |
| fe54_collinear / candidate | 6.691 | 77 | Pass |
| fe16_noncollinear / candidate | 3.668 | 72 | Pass |
| fe3_linear_soc_k2_xyz / candidate | 2.239 | 242 | Pass |
| fept16_soc_z / candidate | 5.496 | 80 | Pass |
| fept16_soc_x / candidate | 5.397 | 70 | Pass |
| dzp_m400_k4_t300_z / candidate | 8.000 | 64 | Pass |
| dzp_m400_k4_t300_x / candidate | 7.726 | 51 | Pass |
| fept128_soc_z / candidate | 188.996 | 128 | Pass |
| fept128_soc_z / fp64-tiles | 107.553 | 102 | Pass |
| fept16_soc_z / gpu-radial | 3.977 | 73 | Pass |

Times include process startup, setup, SCF and requested final outputs. They
exclude image building and evidence retrieval and are single observations.
CPU references were reused; no new upstream calculation or upstream-GPU timing
comparison was performed. The H100 allocation is confirmed released.

The [qualification report](../benchmarks/evidence/hopper-qualification-v187-20261012T214738Z/qualification.json),
[inputs and source identities](../benchmarks/evidence/hopper-qualification-v187-20261012T214738Z/metadata.json),
[individual timings](../benchmarks/evidence/hopper-qualification-v187-20261012T214738Z/timings.csv) and
[solver results](../benchmarks/evidence/hopper-qualification-v187-20261012T214738Z/kernel-checks.json)
are retained in the public evidence set. Full raw outputs and implementation
archives remain in the local evidence archive. The local test suite also
passed all 116 tests.

## Scope

The qualification is for one H100 and the recorded configurations. H200, GH200,
multi-GPU execution and diagnostic tracing are not qualified by this run.
Both architecture targets compile, but the H100 qualification does not rerun
the earlier A100 measurements. Timings are individual observations, not a
repeated comparison with upstream GPU SIESTA or with A100.

The standard configuration keeps upstream radial tables for one-time setup
and retains GPU SCF. The optional GPU radial-table path has a known DZP
numerical-equivalence failure from the earlier A100 property study; its
selected SZP regressions do not establish general larger-basis support.
Optional FP64 tile modes remain opt-in. No mixed-precision experiment is
included in this release.

Numerical agreement on these inputs does not establish physically converged
magnetic properties. The stronger DZP pair still uses a coarse k mesh and
finite smearing. Basis, grid, k-point, smearing, pseudopotential and geometry
convergence require separate studies. Production capacity limits are unchanged;
the staged 2048-k-point experiment is not part of this change.

## Build references

NVIDIA distinguishes native Hopper cubins from forward-compatible PTX;
shipping SM90 code avoids relying on JIT compilation for this target.
[NVIDIA Hopper compatibility guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html).
The architecture list initializes all CUDA targets through the standard
[CMake architecture variable](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).
