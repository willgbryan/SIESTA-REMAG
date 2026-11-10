# Apple Silicon / Metal feasibility

Exploration dated 9 November 2026, following the user's pause of the MAE goal.
The user subsequently authorized investigating FP32 and mixed precision for
acceleration. This supersedes the earlier deferral for this qualification work;
production remains FP64. An isolated Metal eigensolver prototype now runs and
passes selected-matrix checks, but is slower than native CPU FP64 on this Mac.
There is no integrated Apple SIESTA backend or full-SCF qualification. The
initial CPU replay and subsequent GPU benchmark are separate experiments.
Those completed experiments used no cloud allocation.

## Finding

Apple support is feasible, but a direct Metal translation cannot preserve the
current FP64 arithmetic. The local M3 Max runtime accepts a simple `float`
compute shader and rejects its otherwise identical `double` version. Apple's
[MPS data types](https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype)
provide no FP64/complex-FP64 substitute. This affects much more than final
energy summation: Hamiltonian assembly, density construction, eigensolving,
Hartree FFTs and grid contractions currently depend on double precision.

The implemented candidate uses an FP32 eigensolver seed with FP64 overlap
treatment and correction against the original operators. It establishes
component accuracy, but fails the measured performance gate. Native Mac
execution, Metal GPU execution and scientifically equivalent speedup remain
separate claims.

## Observed local environment

| Item | Observation |
| --- | --- |
| Device | Apple M3 Max |
| System memory | 51,539,607,552 bytes (48 GiB) |
| OS | macOS 26.5, build 25F71 |
| Metal unified memory | Available |
| Recommended working set returned by Metal | 40,200,896,512 bytes |
| Maximum individual buffer returned by Metal | 30,150,672,384 bytes |
| Runtime FP32 shader compilation | Passed |
| Runtime native FP64 shader compilation | Rejected: `double` unsupported |
| Standalone `metal` compiler in selected CLT installation | Not found |
| Host clang / gfortran | Available |
| Accelerate new-LAPACK `zhegvd_` interface | Native CPU comparator passes all 39 saved matrix captures |

The working-set recommendation is an observed API value, not a demonstrated
problem-size capacity. No large allocation was attempted. Runtime shader
compilation works despite the missing standalone tool, so that absence does
not block an initial local prototype. The exact probe and diagnostic are in
[capability evidence](../benchmarks/evidence/apple-metal-capability-20261109/capability.json).

Apple's [shared-buffer documentation](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)
explains the useful memory model: CPU and GPU can access the same allocation,
but producer/consumer synchronization is still required. Shared memory does
not provide native FP64 arithmetic or make all transfers and synchronization
costs disappear.

## Current REMAG inventory

The native build requires CUDA 12.6 and links cuSOLVER, cuBLAS and cuFFT.
There are 40 non-check/non-probe `.cu` source files, including experimental
paths, plus C++ state owners and Fortran integration templates. This is a
substantial backend project, not a build-option change.

| Area | Current implementation | Apple route and difficulty |
| --- | --- | --- |
| SIESTA integration and scientific checks | C bindings, pinned upstream patch, Python comparison tools | Reuse input/output contracts, reference cases and tolerances. Keep changes out of pinned upstream. |
| Generalized Hermitian eigensolver | `native/solver.cpp`: FP64 `Dsygvd` / complex-FP64 `Zhegvd`; custom device-SCF solver path | CPU Accelerate LAPACK is a candidate. Preserve lower-triangle semantics, layout, integer ABI, eigenvector normalization and status. This does not by itself port device SCF. |
| Buffers and ownership | CUDA allocation, pointers, streams, persistent state | Metal buffers plus explicit byte offsets/ownership. Retain accepted-publication and lifetime rules; avoid exporting CUDA types through a portable boundary. |
| SCF control | `native/scf_cycle.cu`: CUDA conditional graphs, device convergence and mixing | Start with host orchestration and Metal command buffers where applicable. Do not assume Metal is a direct replacement for CUDA's conditional graph semantics. |
| Hartree | `native/hartree.cu`: complex-double 3D cuFFT | Keep a CPU FP64 FFT implementation initially; FFTW is a practical candidate. Accelerate's transform APIs require an explicit layout/size/3D composition assessment. |
| Sparse H/S, density, grid, SOC, populations | Custom kernels, mostly double arithmetic and complex spinors | A native FP32 port changes the numerical method. Integer topology/permutation or bit-preserving data work is a possible narrower GPU entry point, only if profiling shows worthwhile cost. |
| Forces and stress | Coupled final-state owners and derivative kernels | Defer until an energy-only path is qualified; never report omitted observables as computed zeros. |

[Accelerate](https://developer.apple.com/documentation/accelerate) supplies CPU
linear algebra and signal-processing routines. Its existence should not be
reported as Metal acceleration. The local SDK includes the needed complex
generalized eigensolver declaration. The initial precision replay uses SciPy's
CPU LAPACK; the later benchmark also validates a direct C++ Accelerate
comparator. REMAG/SIESTA native Mac integration and end-to-end Mac SCF remain
untested.

The pinned upstream explicitly marks
`cmake/toolchains/mac-accelerate-test-only.cmake` as test-only because of
Fortran/Accelerate return-value conventions. Its linear-algebra checks must
remain enabled. Use a compatible OpenBLAS configuration for an initial
gfortran reference, or validate a deliberate ABI bridge. A native C/C++
Accelerate solver adapter should use the new LAPACK interface and explicitly
match integer and complex layouts; merely adding `-framework Accelerate` to
the SIESTA link line is insufficient.

## Precision choices

| Route | Scientific implication | Assessment |
| --- | --- | --- |
| CPU FP64 physics with selective exact Metal work | Can preserve the current numerical contract after validation | Useful baseline, but exact data movement alone is unlikely to address the measured eigensolver bottleneck. |
| FP32 eigensolver seed, FP64 overlap and corrections against original H/S | New mixed-precision algorithm; small residuals alone do not establish accurate directional E/F | Recommended first acceleration experiment; local component evidence below. |
| Early FP32 SCF, followed by full FP64 convergence | Must reach the same magnetic solution and unchanged convergence criteria | Second candidate, especially with an existing FP64 GPU backend; on Apple the final FP64 stages would initially run on CPU. |
| Software FP64 or paired-float Metal arithmetic | Requires numerical qualification and custom solver/FFT support; paired floats are not IEEE binary64 | Research option, not a ready production backend. |

For scale, storing a total energy around −2224 eV in FP32 gives adjacent values
about 244 µeV apart, versus our 5 µeV/f.u. implementation gate. This illustrates
why a blanket type substitution is unsuitable; it does not prove every mixed
method impossible. FP64 final accumulation cannot recover errors already
introduced in a lower-precision Hamiltonian or density.

The inspected [metal-float64 project](https://github.com/philipturner/metal-float64)
describes itself as unfinished and lists unresolved precision behavior. Its
performance tables are theoretical. Do not treat it as a qualified dependency
or transfer its throughput estimates to REMAG. MPS/MLX cannot bypass the
underlying absence of native Metal FP64.

## Local precision replay

[Replay script](../benchmarks/evidence/apple-mixed-precision-20261109/replay.py)
and [all results, including failures](../benchmarks/evidence/apple-mixed-precision-20261109/replay.json)
use the 39 independently checked complex H/S captures from
`mae-soc-free-scalar-20261026T210526Z`, completed attempt
`677395ac67de45dea1d3e7b75dd8adc9`. There are 12 FePt 96-state matrices and
27 FeCo 112-state matrices, sampling three k points per rotation/reference
from 64- and 4096-point parent meshes. Some captures differ only through their
reference/direction or separate solver controls; these are not 39 independent
materials or full SCF runs. Original files remain in that run's results folder;
the replay records their hashes and retains every attempted method.

All solves here use CPU SciPy/LAPACK, including the complex64 solves. The lower
Hermitian triangle is authoritative, matching the original solver. Captured
overlap condition numbers span approximately 54,882–366,827. The current
component spectrum gate is 1e-7 eV; additional replay gates require relative
generalized residual <=1e-10 and maximum S-orthogonality error <=1e-10.

| Arithmetic | FePt maximum eigenvalue error (eV) | FeCo maximum eigenvalue error (eV) | Captures passing all three component gates |
| --- | ---: | ---: | ---: |
| FP32 H/S rounding, then FP64 generalized solve | 1.0406e-2 | 1.2832e-1 | 0/39 |
| FP32 generalized solve | 2.7451e-2 | 1.8538e-1 | 0/39 |
| FP64 overlap/standardization, FP32 standard solve | 6.6423e-5 | 1.0423e-4 | 0/39 |
| Above, with FP64 Rayleigh quotients only | 4.2066e-5 | 5.9543e-5 | 0/39 |
| Above FP32 seed, with FP64 vector correction | 1.0385e-10 | 2.1781e-10 | 39/39 |

The first row isolates information lost by rounding H/S; a high-precision solver
cannot recover it. Large all-band errors include unoccupied states. For the
uncorrected FP32 generalized solve, errors among states with reference
occupation >=1e-8 reach **53.51 micro-eV in FePt and 120.20 micro-eV in FeCo**.
These are per-eigenvalue errors, not measured MAE errors.

The successful correction keeps original H/S in FP64, reorthogonalizes the
seed with the original overlap, projects H into that basis, diagonalizes
near-degenerate blocks, and applies an off-diagonal vector correction. It
needed at most two correction steps and clusters of at most two states in
these captures. The reference spectrum is used only for measurement, not for
computing the correction or deciding when to stop. Worst corrected residual is
9.47e-11 and S-orthogonality error 1.64e-12. At the saved 300 K electronic
smearing and fixed source chemical potential, the relative density-operator
error in an orthonormal basis is <=1.44e-9. This is a selected-k density
diagnostic; it neither re-solves the full-mesh electron count nor validates SCF.

**This replay deliberately retains substantial O(n^3) CPU FP64 work** in the
projection products and Cholesky steps. It proves an accuracy route exists for
these samples, not that this implementation is cheaper. It is a small
experimental correction, not a production implementation of a published
refinement algorithm or a convergence proof for arbitrary spectra. No Metal
timing, SCF, directional E/F, or acceleration claim follows from it. The source
frozen-potential approximation still fails its full-SCF bias gates; improving
its matrix arithmetic does not repair that physical approximation.

## Actual Metal prototype and benchmark

The [isolated implementation](../experiments/metal-eigensolver/README.md) uses
a batched complex FP32 cyclic-Jacobi kernel, CPU FP64 standardization and
bounded corrections, original-H/S acceptance checks, and explicit FP64
fallback. Production has no dependency on it. Eleven focused tests pass.

The [measured report](../benchmarks/evidence/apple-metal-solver-20261110/REPORT.md)
retains 16 configurations and 80 timing pairs on the M3 Max. All corrected
matrix checks pass with zero timed fallbacks; maximum eigenvalue error is
1.93e-10 eV. With four CPU workers, the fresh 12-matrix FePt batch takes
**7.05 ms on native CPU FP64 versus 40.46 ms hybrid**, and the 27-matrix FeCo
batch **23.73 versus 107.77 ms**. Cached factors and eightfold repeated batches
also fail to produce a speedup. Repeated batches are throughput controls, not
additional scientific k-point coverage.

The initial Python CPU comparator was replaced by direct native Accelerate
task dispatch to avoid judging against a weak baseline. The hybrid correction
still uses Python/NumPy/SciPy and accounts for the largest measured stage.
Its cost and the Metal seed kernel must improve before expanding this backend.
Native correction and tiled/block-Jacobi seeds are the next specific candidates;
neither has measured gains yet. Emulated FP64 remains unqualified and unused.

## Techniques worth adapting

- [ELPA-AEO mixed-precision SCF study](https://arxiv.org/html/1806.01036v1),
  section 4.3: use lower precision in appropriate early eigensolver steps,
  then restore precision as SCF accuracy demands. This is directly relevant
  to atom-centered generalized eigenproblems. Its zirconia evidence does not
  qualify metallic SOC anisotropy. A fixed iteration-number switch is weaker
  than monitoring error/stagnation and completing the original FP64 criteria.
- [Residual-based Chebyshev filtering](https://arxiv.org/html/2503.22652v5):
  applies approximate matrix products to corrections that shrink as the
  solution improves, while retaining accurate residual information. It covers
  complex generalized DFT problems. It is more attractive for larger problems
  requiring a subset of states than our current 96/112-state, full-spectrum
  dense solves; its published speedups do not transfer automatically.
- [Ogita–Aishima clustered-eigenvalue refinement](https://link.springer.com/article/10.1007/s13160-019-00348-4):
  relevant because naïve corrections divided by small eigenvalue gaps are
  unstable. Preserve degenerate subspaces rather than demanding individual
  eigenvector agreement. This motivates cluster treatment; the local script
  is not a reproduction of the paper's exact algorithm.
- [Mixed-precision block Davidson implementation](https://github.com/jeheon1905/mp_davidson):
  another candidate for a future occupied-subspace solver. Adapting its
  real-space workflow to nonorthogonal full-SOC SIESTA requires explicit
  overlap, occupation, near-Fermi-state and final-density treatment.

Paired-float or split-matrix products may later move some accurate correction
work onto Metal. A pair of floats is not automatically IEEE FP64: exponent
range, cancellation, rounding, fused operations and reductions still need
qualification. Do not expand to a software-double backend before measuring
the narrower hybrid path.

## Qualification sequence and stopping decisions

1. **Component feasibility, done:** preserve original FP64 matrices; compare
   rounding, FP32 solves, and correction on the existing captures. Blanket
   FP32 and Rayleigh-quotient-only correction fail this sample set.
2. **Prototype done; acceleration gate failed:** the batched FP32 Metal seed
   plus CPU FP64 correction is accurate on the captures but slower than the
   matched native CPU baseline. Preserve fresh/cached-factor timings and all
   failures. Optimize the measured correction and seed costs before broadening
   the backend; raw FP32 throughput does not establish an end-to-end gain.
3. **Full SCF:** build the pinned pristine Mac reference with compatible
   BLAS/LAPACK and keep its ABI tests. Validate matched FePt x/z and FeCo
   001/110/111 runs, retaining density, charge, spectra, final moments and
   directions, exports, and convergence checks. Use original DM 1e-9 and
   H 1e-8 eV criteria for the final accepted state. Adaptive precision and
   bounded correction must fall back to FP64 on failure; retain the failure
   and include the fallback time. No automatic tolerance relaxation.
4. **MAE and end-to-end gain:** compare direct total/free directional-energy
   differences with the unchanged 5 micro-eV/f.u. implementation gate, first
   at matched 300 K settings and then at the denser meshes/lower smearing that
   matter scientifically. Treat metallic near-degeneracy as a required case.
   Repeat matched timings with identical requested outputs. An energy-only
   path makes no force/stress claim. Separate implementation equivalence from
   physical k/smearing/basis/pseudopotential convergence.

The Mac baseline answers whether Metal accelerates Apple execution. It does
not establish an upstream-GPU competitive gain or H100 parity; those require
their own matched evidence. Production CUDA stays FP64 and the prior MAE goal
stays paused. No new cloud budget is implied.

## Prepared comparison with REMAG on H100

At the user's request, `experiments/metal-eigensolver/cuda_app.py` now stages
a direct comparison with the unchanged CUDA `BlockJacobiPlan`. It uses the
same 39 hashed H/S captures, four batch sizes and existing five-repetition
native CPU timings. Local preparation verified exact input preservation and
reran the CPU accuracy checks successfully on all 39 pairs. The CUDA adapter
still requires compilation and execution on H100; no comparison result or
shipping decision is available yet.

Eight solver lanes, cold bases and identical-input warm-basis controls retain
the production FP64 settings. Graph execution and host-transfer boundaries
are reported separately. Repeated matrices are labelled exact replays, and
warm-basis replay does not establish performance on changing SCF operators.
The requested new budget is at most ten H100-minutes; it is awaiting approval.
The launcher defaults to no allowance and enforces one GPU at a time.

A competitive component result would support proceeding with Mac CPU backend
integration. The current comparator has a separate batch ABI and supports
only the small experiment's dimensions; it is not a drop-in SIESTA backend.
The full-SCF and directional-energy qualification above remains the release
gate before advertising a supported Mac solver.
