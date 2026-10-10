# SIESTA-REMAG

GPU-first SIESTA for magnetic materials. REMAG keeps electronic state and
supported SCF operations on the GPU, using data structures designed for that
workload. Fully relativistic, noncollinear spin–orbit calculations are the
primary target.

This is an **experimental source release** built against SIESTA 5.4.2, pinned
in [upstream.lock.json](upstream.lock.json). The integration patch applies to a
disposable copy; pristine upstream SIESTA remains the numerical reference.
REMAG is an independent derivative, not an official SIESTA release.

## What works

- Collinear, noncollinear and full off-site SOC regression calculations,
  including nonparallel spins and Gamma/sampled-k paths.
- GPU SCF state, eigensolver reuse, sparse density construction, grid operations,
  mixing and supported force/population operations.
- Upstream comparisons of energies, densities, magnetic moments, forces,
  stress, relaxation and directional magnetic energy differences.
- Validated FePt workloads through 128 atoms / 2,304 spinor states at the recorded
  regression settings. Fe, Fe3C, Fe2B and Fe5PB2 have additional numerical coverage.

The candidate runs in one process on one NVIDIA GPU. CPU geometry/setup and
ionic work remain, and requested output files retain explicit export boundaries.
Numerical regression coverage does not establish production materials accuracy.

## Measured performance

These are complete converged calculations compared with **upstream GPU SIESTA**,
including initialization, SCF, final observables and requested exports. Each row
uses three fresh repetitions on the same allocation; values are medians.
Builds, provisioning, diagnostic tracing and artifact transfer are excluded.

| Workload | Upstream GPU | REMAG | Speedup | Measured REMAG build |
| --- | ---: | ---: | ---: | --- |
| FePt, 16 atoms, full SOC x/z pair | 573.31 s | 9.84 s | 58.27× | v142 |
| Fe, 16 atoms, noncollinear | 177.81 s | 5.98 s | 29.73× | v127 |
| Fe, 54 atoms, collinear | 79.63 s | 9.51 s | 8.37× | v127 |

FePt used an A100-SXM4-80GB; the two Fe comparisons used A100 80GB PCIe.
Upstream GPU execution was checked separately. Fe54's comparator is ELSI/ELPA
with eight MPI ranks; Fe16 and FePt use native ELPA with their recorded thread
settings. Inputs and stopping criteria match within each comparison; SCF counts
can differ. These are workload-specific SZP/100 Ry regression results, not a
universal speedup or a comparison of every row on one common software version.

[Versioned benchmark records](benchmarks/evidence/README.md) include individual
repetitions, settings, scientific checks and source identities. They also record
an explicitly **provisional, single-pair** FePt128 upstream-GPU comparison and
the separately measured v178 incremental optimization. Historical gains are
never multiplied into a current speedup. Broader benchmarking is ongoing.

## Get started

Use a source checkout, Python 3.12+, Git, CMake, C/C++ and Fortran compilers,
and BLAS/LAPACK. The GPU implementation uses CUDA 12.6 and targets A100 (SM80).
The Modal image supplies the Linux/CUDA build dependencies.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,netcdf,modal]'
python scripts/build_upstream.py
python -m pytest -q
```

The build script fetches the pinned upstream source and required submodules,
refuses a mismatched or modified reference, and builds a serial CPU oracle.
Upstream and build products live in ignored directories.

Prepare and run a small CPU reference:

```sh
remag prepare fe2_collinear --output benchmarks/generated/fe2
remag run benchmarks/generated/fe2 \
  --executable build/upstream/Src/siesta --output runs/reference-fe2
```

Each preparation and run needs a fresh output directory. See
[benchmark inputs](benchmarks/README.md) for default settings and their limits.

## Run on Modal

Authenticate Modal with your own account. Always launch through the serialized
launcher, which refuses an existing REMAG app or another active/unknown GPU
allocation. It submits one bounded GPU function, with no GPU warm pool or
retries. Do not launch from multiple machines simultaneously.

A small eigensolver/grid/density integration run:

```sh
python scripts/run_modal.py --cases fe2_collinear --grid --dm
```

This default harness compares the candidate with a **CPU upstream oracle** on
the same host. It does not recreate the upstream-GPU timing table above.
The full device-SCF configuration and evidence-recovery instructions are in
[docs/RUNNING.md](docs/RUNNING.md). Defaults use one A100 40 GB;
`--gpu-model A100-80GB` selects one 80 GB instance. These commands incur cloud
compute charges; do not run them while another GPU allocation is active.

## Scope and limits

- Validated hardware is A100. The candidate has no MPI or multi-GPU execution.
- Existing evidence primarily uses small bases/coarse sampling. Production
  MAE needs basis, mesh, k-point, smearing and SCF convergence studies.
- Fe4N's tested difficult noncollinear settings remain unqualified. Initial
  spin directions are guesses, not constraints; final directions are checked.
- The block solver is bounded to 4,096 states, with the larger magnetic
  qualification currently at 2,304. Sparse lookup and other storage limits
  can constrain larger bases/cells before GPU memory is exhausted.
- The grid path expects stored single-precision basis values and
  double-precision grids; `DirectPhi` and other precision builds are unsupported.
- Learned SCF is deferred. Zero host I/O or zero total device/host traffic is
  not claimed.

## Repository guide

| Path | Contents |
| --- | --- |
| `native/` | CUDA/C++ kernels, owners and Fortran integration templates |
| `src/siesta_remag/` | Input generation, execution, comparison and evidence tools |
| `scripts/` | Upstream patch/build, single-GPU launcher and existing checks |
| `tests/` | Local unit and integration-patch tests |
| `benchmarks/` | Corpus notes, solver regression fixture and selected evidence |
| `docs/` | Architecture, scientific gates, specification and development records |

Raw outputs, generated charts, downloaded datasets, upstream checkouts and build
products remain local and are excluded from Git. Historical documents link to
that local `results/` archive; those links are not bundled public downloads.
The source distribution includes the native sources and build scripts; the
Python wheel alone is only the harness.

See [CONTRIBUTING.md](CONTRIBUTING.md), [the specification](docs/SPEC.md),
[scientific tolerances](docs/VALIDATION.md), and
[historical development notes](docs/DEVELOPMENT_NOTES.md).

## License

[GPL-3.0-or-later](LICENSE). Preserve the upstream and third-party notices in
[NOTICE](NOTICE), including the [libGridXC notice](native/GRIDXC_NOTICE).
Dependencies and externally obtained pseudopotentials retain their own terms.
