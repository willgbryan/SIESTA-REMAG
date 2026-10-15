# Running the source release

Start with the installation and pinned CPU build in the [README](../README.md).
Use the repository root as the working directory. The Python package supplies
the harness; CUDA/Fortran sources and build scripts come from the source checkout
or source distribution. A Python wheel alone does not contain a SIESTA binary.

## Single-GPU execution

Configure Modal credentials outside the repository. `scripts/run_modal.py`
checks the pristine upstream revision, serializes local launches, and refuses
an already-active REMAG app or another active/unknown GPU allocation. All
contenders within one invocation execute sequentially on the same GPU.

This command selects the integrated device-SCF and supported observable path
on small regression cases. It runs upstream CPU references and candidates;
it is not an upstream-GPU performance comparison or an exact replay of the
historical benchmark table.

```sh
python scripts/run_modal.py \
  --cases fe16_noncollinear,fept16_soc_z,fept16_soc_x \
  --gpu-model A100-80GB --candidate-repeats 3 \
  --grid --dm --normalize --occupations --resident-spectra \
  --matrix-assembly --hartree --resident-grid --resident-scf \
  --device-scf --vector-cache --block-width 32 --block-local-solver jacobi \
  --shared-local-solver --solver-lanes 8 --solver-warm-start \
  --grid-force --atomic-grid-force --final-grid --final-density \
  --orbital-basis --atomic-basis --integral-force --population \
  --gpu-initialization --initial-populations --gpu-history \
  --final-spectra --gate-h-export --nonlocal-h0 \
  --gpu-initial-mesh --grid-potential-spins --defer-scf-log
```

Use `--gpu-model H100` in the same command to select one Hopper GPU.
The [H100 qualification record](HOPPER.md) lists tested inputs, numerical gates
and hardware. The historical A100 timing table does not establish H100 speedups.
The example keeps radial-table setup on the CPU: the optional
`--gpu-radial-tables` path passed selected SZP regressions but failed a stronger
DZP numerical comparison. GPU SCF remains enabled with upstream radial tables.

The app uses a bounded timeout, one GPU, no warm pool and no retries. Image
builds use CPU builders. Launching incurs Modal charges; do not start another
allocation from another machine or bypass the launcher. The local client prints
the run and function IDs. A disconnected client does not imply that the
detached allocation has stopped.

For a lightweight initial integration run, use the smaller command in the
README. The many flags above expose previously measured implementation stages;
their presence does not imply every combination or material is qualified.
FP64 vector/matrix tile experiments remain optional and are intentionally
omitted here because their gains are workload-dependent.

## CUDA build targets

A fresh native CMake build defaults to `CMAKE_CUDA_ARCHITECTURES="80;90"`.
Both the production library and CUDA check targets inherit this selection;
CUDA 12.6 emits native code and PTX for each architecture. To build only SM90:

```sh
cmake -S native -B build/native-h100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/native-h100 --target remag_gpu --parallel 8
```

This builds the native library; a runnable SIESTA executable also requires the
integration patch and Fortran build. The Modal image builds both together.
The same CMake option works when the native project is embedded in a patched
SIESTA build. Explicit CMake settings and the `CUDAARCHS` environment variable
override the default. Reconfigure an existing build explicitly to change its
cached architecture list. H100 qualification uses native SM90 with PTX JIT
disabled; it does not establish support for every GPU in the Hopper family.

## Inputs and numerical checks

`remag prepare` generates the coarse regression inputs documented in
[benchmarks/README.md](../benchmarks/README.md). It does not silently substitute
the tighter input variants used by later experiments. Use the exact FDF,
pseudopotential hashes and runtime settings attached to a reported measurement
when reproducing it. A new input or source revision is a new measurement.

The default I/O policy disables per-iteration DM checkpoints and retains the
final cycle's requested scientific outputs. `--io-policy checkpoint` restores
iterative checkpointing, equally for the reference and candidate. Required
outputs and stopping criteria must remain equivalent in a comparison.

```sh
remag compare runs/reference/result.json runs/candidate/result.json
remag mae runs/reference-x/result.json runs/reference-z/result.json \
  runs/candidate-x/result.json runs/candidate-z/result.json
```

These paths are examples of completed run directories, not files shipped in
the repository. Keep the raw densities and requested matrix outputs beside
their result records. MAE checks compare the directional difference directly
and verify final moment directions; an initial spin direction is not a
constraint. See [the scientific gates](VALIDATION.md). Explicit per-site
direction constraints are described in [constrained site moments](SPIN_CONSTRAINT.md).
Their GPU kernels match the host bit for bit, and
[the hcp cobalt benchmark](../benchmarks/spin-constraint-hcp-co/README.md)
compares constrained energies with Quantum ESPRESSO. The consistent gradient
corrections for noncollinear spin (`REMAG.XC.NoncollinearGGA local-frame` or
`multicollinear`, see [noncollinear GGA](NONCOLLINEAR_GGA.md)) run on both
paths. Tetrahedron occupations (`OccupationFunction Tetrahedra.Optimized`, see
[tetrahedron occupations](TETRAHEDRA.md)) run on the host path and in the
device SCF cycle; the older GPU occupation route on its own stops with a
message when they are requested.

## Dense k meshes

The device SCF cycle builds one CUDA graph for a whole SCF iteration. By
default that graph holds a separate solve and density step for every k point,
which limits it to 128 k points. `REMAG_SCF_K_LOOP=1` replaces those copies
with a loop on the GPU over groups of k points, so the graph no longer grows
with the mesh. The number of k points solved at once is
`REMAG_SCF_SOLVER_LANES`, which may then be 1, 2, 4, 8, 16, 32, 64 or 128, and no longer
requires `REMAG_DEVICE_VECTOR_CACHE=1`. With the cache, every k point's
eigenvectors stay on the GPU between the two passes of an iteration (n² complex
numbers per k point for n spinor states); without it, each k point is solved
again in the second pass, as SIESTA does. `scripts/check_scf_k_loop.py` runs
the composed SCF check with five k points under each setting. The looped cycle
passes the same CPU oracle and agrees with the unrolled cycle to rounding
(at most 2×10⁻¹³, in the Fermi level; 10⁻¹⁴ or less in the matrices). The unrolled cycle differs from itself between processes
by a similar amount, because the mesh density is summed with atomic additions.

`REMAG_SCF_BATCHED_SOLVER=1` (with the loop) replaces REMAG's block-Jacobi
eigensolver for those groups. Each group of k points is assembled by one
kernel and solved by batched library calls: a Cholesky factorization of every
overlap, two triangular solves that reduce each problem to standard form,
cuSOLVER's `XsyevBatched`, and one triangular solve back. The density of the
group is then built in one pass that adds the k points in order, as the
per-k-point kernels do. Groups may then hold up to 1024 k points; 256 is the
default. It has no warm start. In `check_scf_k_loop.py` it passes the CPU
oracle and differs from the unrolled block-Jacobi cycle by at most 6×10⁻¹⁴.
A separate probe of the batched solve on random Hermitian problems (dimension
32 to 336) found residuals of at most 1.5×10⁻¹³.

On hcp cobalt (two atoms, 84 spinor states, 20×20×11 = 4400 k points with
spin-orbit coupling, PBE local frame, one constrained site moment at 45°,
2000 K smearing, 25 SCF iterations), single runs on one H100 gave these wall
times for the whole calculation:

| Path | Wall time |
| --- | ---: |
| Host path, REMAG's CPU k loop on 16 threads | 322 s |
| Device SCF loop, block Jacobi, 16 k points at a time | 31.6 s |
| Device SCF loop, block Jacobi, 64 k points at a time | 22.9 s |
| Device SCF loop, batched solver, 256 k points at a time | 14.7 s |

All four converged to the same total energy, free energy, Fermi level and site
moment at every printed digit. Of the last run's 14.7 s, 6.3 s is the SCF loop
itself; most of the rest is SIESTA's CPU setup of the mesh and basis
(`REMESH`, `PHION`). These compare REMAG's own paths with each other; they are
not a comparison with upstream GPU SIESTA. The same calculation with
`multicollinear` took 835 s on the CPU path and 31.2 s on the GPU (block
Jacobi, 16 k points at a time), and with `Tetrahedra.Optimized` 404 s and 33.3 s, again
with identical printed results.

## Threaded CPU k loop

On the host path, `REMAG.HostK.Threads N` (N > 0) replaces SIESTA's
spin-orbit k-point loop (`diag3k`) with REMAG's own (`native/host_kloop.cpp`).
It runs in one process, needs no GPU, and is not used when wavefunctions are
written; when it cannot run, SIESTA's loop is used and the output says so. It
splits the k points among N threads and sums their contributions in a fixed
order, so a given thread count always gives the same result. Each thread forms
the Bloch phases once per orbital rather than once per matrix element, and
builds the density with one matrix product per k point. The eigenvectors of
the first pass are kept for the density when they fit in
`REMAG.HostK.VectorMemoryMB` (default 8192); otherwise each k point is solved
again. `REMAG.HostK.Solver` selects LAPACK's divide-and-conquer solver
(`divide`, the default), MRRR (`mrrr`, which in the second pass computes only
the occupied states) or QR (`qr`). The tetrahedron weights use the same thread
count. `scripts/check_host_kloop.py` compares the loop with a line-by-line
NumPy port of `diag3k` for every solver, thread count and vector setting.

On hcp cobalt (20×20×11 k points, off-site spin-orbit coupling, PBE local
frame), four SCF iterations gave the same energies at every step, to the
printed 10⁻⁶ eV, as SIESTA's loop. The time per diagonalization was 36.6 s for
SIESTA's loop and 14.5, 5.9 and 3.5 s for 1, 4 and 16 threads. These were
single runs on a 28-core node shared with other jobs, a CPU-to-CPU comparison
rather than a GPU result.

## Existing evidence and recovery

Compact results return to `results/<run-id>/`; full artifacts stay in the
account's `siesta-remag-results` Modal Volume. Both are outside Git. The existing
read-only downloader allocates no compute:

```sh
python scripts/fetch_evidence.py RUN_ID --density --matrices --workers 1 \
  --attempt-id ATTEMPT_ID
```

Replace the placeholders with the exact run ID and the 32-character attempt ID
from its `run-attempt.json`. A live attempt mismatch invalidates a fetch; use a
new destination rather than combining outputs from different attempts. To
recover only raw binaries omitted from an already returned archive, also pass
`--outputs-only`. Incomplete block downloads are rejected. The downloader uses
Modal's internal block-reading API, so the optional dependency is pinned to the
tested SDK version in `pyproject.toml`.

The [versioned evidence directory](../benchmarks/evidence/README.md) contains
selected historical records and exact FDF inputs. Full raw scientific outputs,
source archives, pseudopotentials and historical cached Modal images are not
bundled. Replaying the historical upstream-GPU comparisons additionally requires
their recorded GPU-enabled upstream build and solver configuration; the default
Modal harness builds the serial CPU oracle. Do not label its CPU comparison as
an upstream-GPU speedup.
