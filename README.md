# SIESTA-REMAG

GPU acceleration of SIESTA for magnetic materials. REMAG uses persistent
electronic state, sparse data structures and custom CUDA kernels to reduce the time
to a converged calculation, with fully relativistic noncollinear magnetism
and spin–orbit coupling (SOC) as its primary targets.

REMAG is an independent derivative of **SIESTA 5.4.2** and an experimental
source release. It applies an integration patch to a disposable copy of the
[pinned upstream source](upstream.lock.json), keeping pristine SIESTA as the
numerical reference. Execution currently targets one NVIDIA A100 GPU.

## Capabilities

- Collinear, noncollinear and full off-site SOC calculations, with Gamma and
  sampled k-point regression coverage, including nonparallel spins.
- Device-resident self-consistent field (SCF) state, eigenbasis reuse, sparse
  density construction, real-space grid operations and mixing.
- GPU evaluation of supported forces, stress and population observables,
  checked against upstream alongside energies, densities and magnetic moments.
- Directional magnetic-energy comparisons and FePt regression coverage through
  128 atoms / 2,304 spinor states, with additional Fe, Fe3C, Fe2B and Fe5PB2 cases.

CPU geometry/setup and ionic work remain. Requested scientific outputs are
exported explicitly. See [execution and configuration](docs/RUNNING.md) for the
integrated GPU path and its supported settings.

## Benchmarks

The latest repeated comparisons use the same A100-SXM4-80GB allocation and
equivalent inputs for REMAG and upstream GPU SIESTA. Values below are medians
of three fresh process runs per implementation, including initialization, SCF,
final observables and requested exports. Build, provisioning, diagnostic
tracing and evidence retrieval are excluded.

| Workload | Upstream GPU solver | Upstream | REMAG | Speedup |
| --- | --- | ---: | ---: | ---: |
| Fe, 54 atoms, collinear | ELSI / ELPA | 104.24 s | 7.43 s | 14.04× |
| Fe3C, 16 atoms, collinear | Native ELPA | 144.06 s | 2.84 s | 50.74× |
| Fe3C, 16 atoms, noncollinear | Native ELPA | 360.35 s | 3.26 s | 110.69× |

All timed runs pass direct pristine-CPU checks of scientific outputs, requested
exports and convergence. Upstream GPU execution was verified separately. The
Fe54 row uses the faster passing upstream solver family. Fe3C collinear is a
**native-only comparison**: its ELSI screens converged but did not produce the
required eigenvalue output. The Fe3C fixtures use nonrelativistic pseudopotentials.

These measure time to equivalent converged solutions; SCF iteration counts can
differ. The seven-workflow campaign ended at its compute budget before all
workflows completed, so **no full-suite average is reported**. These coarse
regression inputs do not establish universal speedups or production materials
accuracy.

A separate earlier full-SOC FePt16 x/z comparison measured **573.31 → 9.84 s
(58.27×)** against native ELPA GPU SIESTA, with three repetitions and direct
directional-energy checks. It uses an earlier REMAG build and is kept separate
from the table above. The [benchmark records](benchmarks/evidence/README.md)
include exact inputs, repetitions, solver settings, source identities, numerical
checks and the latest campaign's failed screens and incomplete coverage.

## Installation

Use a source checkout with Python 3.12+, Git, CMake, C/C++ and Fortran compilers,
and BLAS/LAPACK. The GPU implementation uses CUDA 12.6 and targets A100 (SM80).
The Modal image provides the Linux/CUDA build environment. The Python package
is a harness; it does not include a compiled SIESTA executable.

From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[netcdf,modal]'
python scripts/build_upstream.py
```

The build script fetches the pinned upstream source and required submodules,
checks that the reference is unmodified, and builds the serial CPU executable.
Prepare and run a small reference calculation:

```sh
remag prepare fe2_collinear --output benchmarks/generated/fe2
remag run benchmarks/generated/fe2 \
  --executable build/upstream/Src/siesta --output runs/reference-fe2
```

Use a fresh preparation and run directory each time. The [benchmark input
guide](benchmarks/README.md) describes the generated fixtures and their defaults.

## GPU execution on Modal

Authenticate Modal with your own account, then launch a small
eigensolver/grid/density integration run:

```sh
python scripts/run_modal.py --cases fe2_collinear --grid --dm
```

The launcher serializes execution and refuses an existing REMAG app or another
active or unknown GPU allocation. Each run uses one GPU, a bounded timeout,
no warm pool and no retries. It defaults to an A100 40 GB;
`--gpu-model A100-80GB` selects the 80 GB model. Launches incur cloud charges.

This small integration run compares against upstream **CPU** SIESTA. See
[the full device-SCF configuration](docs/RUNNING.md#single-gpu-execution) to
exercise the integrated implementation. Reproducing the performance table
also requires the recorded GPU-enabled upstream build and solver settings.

## Limitations

- The candidate runs in one process on one A100; MPI and multi-GPU candidate
  execution are not supported.
- Current evidence uses regression settings. Production magnetic anisotropy
  calculations require basis, mesh, k-point, smearing and SCF convergence studies.
  Fe4N's tested noncollinear settings remain unqualified. Initial moment
  directions are guesses; final directions must be checked.
- The block solver supports at most 4,096 states; magnetic qualification reaches
  2,304. Sparse lookup and other storage limits can constrain larger inputs.
- The grid path requires stored single-precision basis values and
  double-precision grids. `DirectPhi` and other precision builds are unsupported.
  Complete removal of host/device traffic and host I/O is not established.

## Development

The [contributor guide](CONTRIBUTING.md) covers local tests and numerical
validation. Native kernels and Fortran integration live in `native/`; the
Python harness is in `src/siesta_remag/`. Build, patch and execution tools are
in `scripts/`, with local tests in `tests/`.

See [architecture and residency](docs/GPU_RESIDENCY.md),
[scientific comparison tolerances](docs/VALIDATION.md#scientific-gates), and
[selected benchmark evidence](benchmarks/evidence/README.md). Dated engineering
records describe their measured implementation stages. Links into `results/`
refer to the separate local raw-evidence archive, which is not bundled.

## License

[GPL-3.0-or-later](LICENSE). Preserve the upstream and third-party notices in
[NOTICE](NOTICE), including the [libGridXC notice](native/GRIDXC_NOTICE).
Dependencies and externally obtained pseudopotentials retain their own terms.
