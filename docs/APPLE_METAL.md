# Apple Silicon / Metal feasibility

Exploration dated 9 November 2026, following the user's pause of the MAE goal.
No Apple backend is implemented or scientifically qualified. No cloud allocation
was requested. The local experiment queries capabilities and compiles shaders;
it dispatches no GPU kernels and measures no performance.

## Finding

Apple support is feasible, but a direct Metal translation cannot preserve the
current FP64 arithmetic. The local M3 Max runtime accepts a simple `float`
compute shader and rejects its otherwise identical `double` version. Apple's
[MPS data types](https://developer.apple.com/documentation/metalperformanceshaders/mpsdatatype)
provide no FP64/complex-FP64 substitute. This affects much more than final
energy summation: Hamiltonian assembly, density construction, eigensolving,
Hartree FFTs and grid contractions currently depend on double precision.

The recommendation is an accurate Mac baseline followed by a narrowly scoped
Metal feasibility experiment. Native Mac execution, Metal GPU execution and
scientifically equivalent speedup must be reported separately. Mixed precision
remains deferred under the existing project instructions; this exploration
does not silently enable it.

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
| Accelerate SDK `zhegvd_` interface | Present; execution not tested |

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
generalized eigensolver declaration, but no solver replay or end-to-end Mac
SCF has yet been performed here.

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
| CPU FP64 physics with selective exact Metal work | Can preserve the current numerical contract after validation | Best initial path under the existing precision constraint; actual GPU benefit is unknown. |
| FP32 Metal operators with CPU FP64 corrections/refinement | New mixed-precision algorithm; small residuals alone do not establish accurate directional E/F | Potential later research, requiring an explicit precision-policy change and end-to-end validation. |
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

## Proposed first implementation milestone

1. Build a pinned upstream Mac reference with compatible BLAS/LAPACK. Preserve
   all upstream library checks and establish one small full-SOC FePt case.
2. Introduce only the backend boundary needed for one selected path: explicit
   buffers, synchronization and solver ABI. Keep the CUDA default unchanged;
   avoid a broad abstraction rewrite before a measured Apple result exists.
3. Replay saved complex H/S matrices through a CPU FP64 solver adapter. Check
   generalized residuals, S-orthogonality and spectra, then run the actual SCF
   x/z pair. Existing density, moment, convergence and direct E/F gates remain.
4. Profile that local calculation. Select one meaningful exact Metal task, or
   document that substantive GPU acceleration requires a separately approved
   precision experiment. Compare equivalent requested outputs against the
   same Mac's FP64 baseline, including synchronization and CPU fallback time.

Success means a reproducible, accurate Apple execution path and a measured
decision about its first GPU operation. It does not mean H100 parity, a
complete Metal backend, physical MAE convergence or renewed cloud spending.
