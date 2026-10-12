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
constraint. See [the scientific gates](VALIDATION.md).

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
