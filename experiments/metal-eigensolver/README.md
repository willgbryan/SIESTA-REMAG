# Metal eigensolver experiment

Isolated complex FP32 cyclic-Jacobi seed solver for even dimensions 2–128,
with one threadgroup per matrix and a reusable shared-buffer batch plan.
The 512-matrix allocation bound is not validated scientific capacity.
No production build, runtime option, or CUDA source uses this code.

The [M3 Max benchmark](../../benchmarks/evidence/apple-metal-solver-20261110/REPORT.md)
passes the saved-matrix accuracy checks but is slower than native CPU FP64.
This prototype is not a production acceleration option or an MAE workflow.

Build on macOS with Command Line Tools (the shader is compiled at runtime):

```sh
mkdir -p build/metal-eigensolver
clang++ -O3 -std=c++17 -fobjc-arc -dynamiclib -framework Foundation -framework Metal -framework Accelerate \
  experiments/metal-eigensolver/bridge.mm experiments/metal-eigensolver/cpu_fp64.cpp \
  -o build/metal-eigensolver/libmetal_seed.dylib
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q experiments/metal-eigensolver/test_solver.py
```

The process needs local Metal device access. Plan instances are for serial use
and must be closed (prefer a `with MetalSeed(...)` block). Inputs are finite
Hermitian standard matrices, not generalized H/S pairs. Outputs are unsorted
eigenpairs; per-matrix status 0 means seed convergence, 1 means nonfinite GPU
arithmetic, and 2 means the bounded sweep count was exhausted. Failed seeds
are not accepted solutions. FP32 seed convergence does not qualify MAE.

Fast math is disabled. Timings distinguish command-buffer GPU execution from
host packing, shared-buffer copies, submission and completion. Shader
compilation and persistent workspace setup are separate one-time costs.

`hybrid.py` retains original FP64 H/S, standardizes with CPU Cholesky/HEGST,
corrects the Metal eigenvectors in FP64 with near-degenerate block treatment,
and checks the result against original H/S after back-transformation. It
requires generalized residual and S-orthogonality errors <=1e-10. Bounded
seed/correction failure triggers a full CPU FP64 solve; the reason and cost
remain in the result. This is not an SCF or MAE qualification gate.

The benchmark uses the previously hashed 39 FePt/FeCo captures in `results/`:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/metal-eigensolver/benchmark.py \
  --output build/metal-eigensolver/benchmark-01.json
```

It compares fresh and cached-overlap solves with CPU FP64 Accelerate, using
both one and four CPU job workers, one untimed warmup and five alternating
timing pairs. Both methods receive the same H/S and cache opportunity.
The default CPU comparator calls the new Accelerate LP64 LAPACK C interface
directly, with persistent per-worker workspaces and native task dispatch;
`--cpu-backend scipy` retains the initial Python baseline for comparison.
Independent spectrum/residual/orthogonality/density checks run outside timing;
the hybrid's required acceptance checks and fallback run inside timing.
The eightfold batches repeat saved matrices to measure amortization; they
are not new k meshes or evidence of additional scientific capacity. Existing
output paths are rejected, and partial/failed benchmark results are saved.

For the prepared cross-hardware comparison with REMAG's H100 solver:

```sh
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/metal-eigensolver/compare_cuda.py prepare \
  --folder build/metal-cuda-comparison
# Requires a new, explicitly approved GPU allowance; no default budget.
.venv/bin/python scripts/run_modal.py --module-file experiments/metal-eigensolver/cuda_app.py \
  --budget-seconds 600
```

The adapter compiles the unchanged production `BlockJacobiPlan`, with eight
independent lanes, FP64 and the existing 32-sweep/2e-13 limits. It replays the
same 39 matrices and four batch sizes used in the native four-worker CPU
benchmark, retaining all five CPU timing repetitions. GPU graph time,
resident submission/wait time and host input/output time remain separate.
The identical-input warm-basis control is an optimistic bound on reuse;
it does not measure SCF Hamiltonian evolution. This comparison excludes
assembly, occupations and density construction. Passing it alone does not
qualify a production backend, full SCF, MAE or an upstream-GPU speedup.
The single H100 function has a 540-second ceiling, a shared fixed deadline,
a 60-second evidence reserve and no automatic retry.
