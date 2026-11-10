# Metal eigensolver experiment

Isolated complex FP32 cyclic-Jacobi seed solver for even dimensions 2–128,
with one threadgroup per matrix and a reusable shared-buffer batch plan.
The 512-matrix allocation bound is not validated scientific capacity.
No production build, runtime option, or CUDA source uses this code.

Build on macOS with Command Line Tools (the shader is compiled at runtime):

```sh
mkdir -p build/metal-eigensolver
clang++ -O3 -std=c++17 -fobjc-arc -dynamiclib -framework Foundation -framework Metal \
  experiments/metal-eigensolver/bridge.mm -o build/metal-eigensolver/libmetal_seed.dylib
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
