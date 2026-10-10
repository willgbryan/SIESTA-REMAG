# GPU-First SIESTA for Magnetic Materials

> **Current goal — user revision, 8 October 2026:** build a GPU-first SIESTA implementation with purpose-built GPU data structures that delivers a measurable end-to-end speedup over upstream GPU SIESTA on magnetic workloads. Fully relativistic, spinful noncollinear/SOC systems remain the north star. Use base SIESTA directly for numerical validation, with equivalent inputs and unchanged scientific tolerances. Prioritize implementation and measured speedup; keep preparation and validation small and directly useful. Exhaustive zero-transfer/zero-I/O certification and additional observer/audit infrastructure are not prerequisites. Preserve requested scientific outputs and the single-Modal-GPU limit. Establish the conventional performance baseline before learned SCF work.

> This revision supersedes the 5 October sequencing that made fully certified zero-transfer/zero-I/O residency a prerequisite. Keeping electronic state on the GPU and avoiding costly transfers remain architectural priorities. Historical residency evidence and remaining boundaries are retained in [GPU_RESIDENCY.md](GPU_RESIDENCY.md); its older audit gates do not override the current goal.

> Target clarified on 7 October 2026: fully relativistic, spinful noncollinear magnetic systems are the primary correctness, residency and performance target. Compare with pristine upstream SIESTA's GPU path on magnetic systems; Gold–water is withdrawn. Exercise full off-site SOC, general moment directions, directional energy differences, forces/stress and repeated geometries. Retain simpler spin layouts as regressions and measure their performance separately.

## 1. Objective

Build a GPU-first derivative of SIESTA optimized specifically for high-throughput first-principles calculations of magnetic materials, with particular emphasis on:

* spin-polarized DFT;
* non-collinear magnetism;
* spin-orbit coupling (SOC);
* magnetic anisotropy energy (MAE);
* magnetocrystalline anisotropy constants;
* magnetic moments and spin textures;
* exchange-relevant electronic structure;
* forces and stresses needed for structural relaxation;
* quantities needed by downstream magnetic-property workflows.

The objective is not to redesign DFT or initially reproduce every capability in SIESTA.

The objective is:

> Make the subset of SIESTA relevant to computational permanent-magnet discovery dramatically faster on modern GPUs while preserving numerical agreement with upstream SIESTA.

Upstream SIESTA is the reference implementation and primary verifier.

The fundamental performance metric is therefore:

[
S = \frac{T_{\mathrm{upstream\ SIESTA}}}
{T_{\mathrm{GPU\ implementation}}}
]

for equivalent calculations producing equivalent physical results.

The project should optimize **end-to-end time to a scientifically usable result**, rather than isolated kernel throughput.

---

# 2. Engineering Philosophy

## 2.1 Upstream SIESTA is the executable specification

Do not construct an independent, enormous verification framework.

For representative calculations:

1. run upstream SIESTA;
2. run the GPU implementation from equivalent inputs;
3. compare physically relevant outputs;
4. investigate meaningful discrepancies.

The reference implementation already embodies decades of numerical and physical validation.

Testing should therefore concentrate on regression against upstream rather than duplicating its internal validation effort.

---

## 2.2 Performance work must be empirical

Do not assume which parts of SIESTA are expensive.

Instrument first.

For every representative workload, collect:

* wall-clock time;
* time per SCF iteration;
* number of SCF iterations;
* GPU utilization;
* CPU utilization;
* CPU↔GPU transfer volume;
* kernel launch counts;
* memory allocation activity;
* peak memory;
* major routine/kernel timings.

Maintain a continually updated performance breakdown.

Optimization priority should approximately follow:

[
\text{priority}
\propto
\text{fraction of runtime}
\times
\text{credible acceleration}
\times
\text{importance to target workloads}.
]

---

## 2.3 Optimize workflows, not benchmarks

A 20× faster matrix operation that accounts for 2% of runtime is not important.

A 2× acceleration of something responsible for 60% of runtime is.

Every optimization should eventually be evaluated against:

[
T_{\mathrm{solution}}
]

rather than microbenchmark performance alone.

---

## 2.4 Avoid premature generality

The initial implementation does not need to support:

* every SIESTA feature;
* every XC functional;
* every obscure solver;
* every CPU architecture;
* every historical input configuration;
* every accelerator vendor.

Start with the computational path required for permanent-magnet calculations.

Generalize later where doing so is inexpensive.

---

# 3. Initial Scientific Target

The initial target workload is approximately:

> periodic metallic magnetic crystals containing transition metals, evaluated with spin polarization, non-collinear magnetism, and spin-orbit coupling.

Representative systems should span approximately:

* 10–30 atoms;
* 30–100 atoms;
* 100–300 atoms;
* optionally one larger sparse-system benchmark.

Include systems representative of:

* Simple Fe based permanent magnet systems
* magnetic transition metal borides
* magnetic transition metal carbides
* magnetic transition metal nitrides
* complicated candidate permanent magnets.

The benchmark corpus should contain both relatively easy and difficult SCF convergence cases.

---

# 4. Required Physics

## Phase-1 required capabilities

Support the shortest path needed to reproduce upstream calculations involving:

### Electronic structure

* Kohn-Sham DFT;
* periodic boundary conditions;
* SIESTA numerical atomic orbitals;
* pseudopotentials;
* k-point sampling;
* metallic occupations/smearing.

### Magnetism

* collinear spin polarization;
* non-collinear spin;
* spinor wavefunctions/density matrices;
* spin-orbit coupling.

### Outputs

At minimum reproduce:

* total energy;
* band/eigenvalue information needed by the workflow;
* charge density;
* spin density / magnetic moments;
* forces;
* stress where required;
* SOC total energies.

These outputs enable the important downstream magnetic calculations.

---

# 5. Permanent-Magnet Workloads

The implementation should explicitly support efficient calculation of:

## 5.1 Magnetic moments

Total and atom-resolved magnetic moments.

## 5.2 Magnetocrystalline anisotropy energy

For magnetization directions (a) and (b),

[
\mathrm{MAE}=E_a-E_b.
]

This is particularly important because the physically relevant quantity can be tiny compared with the absolute electronic energy.

Therefore MAE validation must compare the **energy difference itself**, not merely whether both total energies independently appear close.

The implementation should make repeated calculations at different magnetization orientations inexpensive.

## 5.3 Anisotropy constants

Provide the electronic-structure outputs required to fit quantities such as (K_1) from directional energy calculations.

## 5.4 Structural relaxation

Forces must be sufficiently accurate to allow candidate structures to be relaxed before magnetic-property evaluation.

## 5.5 Downstream exchange calculations

Preserve or generate whatever electronic information is required by downstream tools used to estimate exchange parameters, Curie-temperature proxies, or related quantities.

Do not initially rewrite those downstream packages unless profiling later demonstrates that doing so matters.

---

# 6. Phase 0 — Establish the Baseline

Before substantial rewriting, construct a benchmark harness around upstream SIESTA.

For each benchmark store:

* input;
* pseudopotentials;
* basis configuration;
* k-point mesh;
* magnetic initialization;
* SOC/non-collinear settings;
* convergence parameters;
* hardware;
* SIESTA version/commit.

Capture:

* initialization time;
* Hamiltonian construction time;
* density construction time;
* Poisson/Hartree work;
* XC work;
* eigensolver/diagonalization time;
* sparse matrix operations;
* mixing time;
* communication time;
* SCF iteration count;
* total runtime.

If instrumentation requires modifying SIESTA, keep the changes minimal.

### Deliverable

Produce a machine-readable profile such as:

| Component                   | Wall Time | % Runtime | Calls | Notes |
| --------------------------- | --------: | --------: | ----: | ----- |
| Hamiltonian construction    |           |           |       |       |
| diagonalization             |           |           |       |       |
| density update              |           |           |       |       |
| mixing                      |           |           |       |       |
| Poisson/Hartree             |           |           |       |       |
| XC                          |           |           |       |       |
| SOC/noncollinear operations |           |           |       |       |
| communication               |           |           |       |       |
| miscellaneous               |           |           |       |       |

This profile determines subsequent engineering order.

Do not spend weeks perfecting the profiler.

---

# 7. GPU Architecture Principle

The desired architecture is not:

> CPU SIESTA with individual expensive functions occasionally sent to CUDA.

The desired eventual architecture is:

> Keep the electronic state resident on the GPU throughout the SCF loop whenever practical.

Repeated CPU/GPU movement should be considered an architectural defect unless profiling demonstrates that it is negligible.

Conceptually:

```text
Input / initialization
        ↓
GPU state construction
        ↓
┌─────────────────────────────────┐
│           SCF LOOP              │
│                                 │
│ density                         │
│   ↓                             │
│ potentials                      │
│   ↓                             │
│ Hamiltonian                     │
│   ↓                             │
│ electronic solve               │
│   ↓                             │
│ density matrix / density        │
│   ↓                             │
│ mixing                          │
│   ↓                             │
│ convergence                     │
└─────────────────────────────────┘
        ↓
forces / energies / observables
        ↓
output
```

Intermediate representations should be selected with this residency model in mind.

---

# 8. Data Structures

Treat data layout as a first-class optimization problem.

SIESTA's localized orbitals provide sparsity and locality that should be exploited rather than automatically converting everything into generic dense matrices.

Investigate GPU-native representations for:

* orbital metadata;
* neighbor lists;
* sparse Hamiltonians;
* overlap matrices;
* density matrices;
* spinor blocks;
* real-space grids;
* k-point batches.

For non-collinear SOC calculations, important quantities naturally contain (2\times2) spin blocks:

[
H=
\begin{pmatrix}
H_{\uparrow\uparrow} & H_{\uparrow\downarrow}\
H_{\downarrow\uparrow} & H_{\downarrow\downarrow}
\end{pmatrix}.
]

The representation should make these operations efficient rather than treating spinor support as an afterthought.

Experiment with:

* structure-of-arrays versus array-of-structures;
* block sparse formats;
* BSR;
* CSR where appropriate;
* custom orbital-pair layouts;
* packed spinor blocks;
* contiguous neighbor/orbital interaction lists.

Prefer representations that permit kernel fusion and coalesced memory access.

---

# 9. Kernel Strategy

Do not immediately write custom kernels for every operation.

Use optimized GPU libraries where they perform well:

* dense BLAS;
* batched BLAS;
* sparse BLAS;
* eigensolvers;
* reductions;
* FFTs where applicable.

Custom kernels are justified when SIESTA's structure provides information that generic libraries cannot exploit efficiently.

Likely candidates include:

* orbital-pair Hamiltonian assembly;
* sparse spinor operations;
* density-matrix accumulation;
* localized basis evaluation;
* grid deposition/gather;
* SOC matrix construction;
* non-collinear spin transformations;
* fused potential/Hamiltonian operations;
* small repeated matrix operations.

A custom kernel should generally eliminate one or more of:

* intermediate memory traffic;
* unnecessary dense representations;
* repeated launches;
* synchronization;
* host/device transfers;
* generic-library overhead.

---

# 10. SCF Loop

Treat SCF as the central computational object.

For iteration (n),

[
\rho_n
\rightarrow
V_{\mathrm{eff}}[\rho_n]
\rightarrow
H[\rho_n]
\rightarrow
{\psi_i,\epsilon_i}
\rightarrow
\rho_{n+1}
\rightarrow
\text{mix}
\rightarrow
\rho_{n+1}^{\mathrm{mixed}}.
]

Measure both:

[
T_{\mathrm{iteration}}
]

and

[
N_{\mathrm{iterations}}.
]

Then:

[
T_{\mathrm{SCF}}
\approx
N_{\mathrm{iterations}}
T_{\mathrm{iteration}}.
]

This distinction becomes critical when learned acceleration is introduced.

---

# 11. Electronic Solver

Do not assume diagonalization must remain architecturally unchanged.

Establish exactly how much runtime the existing solution strategy consumes for target systems.

Then evaluate, in increasing order of invasiveness:

1. GPU acceleration of the existing algorithm;
2. better GPU eigensolver libraries;
3. batched k-point solution;
4. iterative eigensolvers;
5. subspace methods;
6. spectrum slicing where appropriate;
7. sparse/localized approaches;
8. alternative density-matrix methods for sufficiently large systems.

The target is not necessarily:

> fastest eigensolver.

It is:

> fastest route to the density and energies required for the next SCF iteration.

That distinction should remain explicit.

---

# 12. K-Point Parallelism

K-points provide obvious parallel structure.

Support GPU scheduling that can exploit:

* multiple k-points concurrently;
* multiple CUDA streams;
* batching of similarly sized operations;
* multiple GPUs where appropriate.

Avoid architectures where the GPU repeatedly becomes idle while the CPU prepares the next k-point.

For small crystals with dense k meshes, this may provide substantial throughput gains independently of deeper solver work.

---

# 13. SOC and Non-Collinear Magnetism

This is a primary target rather than an optional compatibility feature.

Profile explicitly:

* complex arithmetic;
* spinor Hamiltonian assembly;
* SOC matrix elements;
* non-collinear density construction;
* spin rotations;
* spin-density reductions;
* increased eigensolver cost.

Determine whether spinor operations can be fused into larger kernels.

Where repeated small complex matrices occur, benchmark custom batched implementations against generic BLAS.

The implementation should assume from the beginning that complex-valued spinor quantities are normal.

Do not build a scalar-spin architecture and bolt SOC onto it later.

---

# 14. Real-Space/Grid Operations

Profile:

* basis evaluation on the grid;
* charge deposition;
* density accumulation;
* Hartree potential;
* XC evaluation;
* grid interpolation;
* force-related grid operations.

These operations may be bandwidth-bound rather than compute-bound.

Optimize accordingly:

* contiguous access;
* reduced intermediate storage;
* kernel fusion;
* shared-memory reuse where useful;
* fewer global-memory round trips.

Do not pursue FLOP optimization where memory movement dominates.

---

# 15. Precision

Do not assume FP64 everywhere is necessary.

Experiment with mixed precision while retaining upstream SIESTA as verifier.

Potential strategy:

```text
early SCF:
    lower/mixed precision

late SCF:
    increasingly high precision

final electronic solution:
    FP64 where required
```

Potential candidates for reduced precision include:

* intermediate density operations;
* mixing history;
* some sparse operations;
* early eigensolver stages;
* neural inference.

However, SOC/MAE calculations can depend on very small energy differences.

Therefore precision decisions must be validated against **magnetic observables**, particularly MAE, rather than total energy alone.

---

# 16. Magnetic-Orientation Reuse

Permanent-magnet calculations often solve nearly identical systems repeatedly for different magnetization orientations.

Exploit this deliberately.

For example:

[
E(\mathbf M\parallel z)
]

followed by

[
E(\mathbf M\parallel x).
]

Reuse where physically and numerically appropriate:

* converged charge density;
* orbital state;
* eigensolver subspace;
* Hamiltonian components independent of orientation;
* neighbor lists;
* basis/grid mappings;
* sparse structures.

Create an explicit workflow for directional energy sweeps rather than treating each direction as an unrelated cold-start calculation.

This is potentially as important as kernel optimization for MAE throughput.

---

# 17. Learned SCF Acceleration

This is a second major workstream, but should begin after the conventional GPU path is measurable.

The learned model is not initially trusted as the final electronic-structure solver.

Instead:

[
\text{initial state}
\xrightarrow{\text{learned operator}}
\tilde{\rho},\tilde{V},\text{ or electronic state}
\xrightarrow{\text{SIESTA}}
\text{converged DFT}.
]

The neural model predicts a state close to the fixed point.

DFT remains the oracle.

Possible learned targets include:

[
\rho_0\rightarrow\rho_\ast
]

or

[
(\rho_n,V_n,H_n,\text{history})
\rightarrow
\rho_{n+k}.
]

The first useful milestone is simply:

> reduce the number of expensive conventional SCF iterations.

---

# 18. Neural Operator Inference

If the learned model becomes important enough that inference appears in the runtime profile, treat it like any other solver kernel.

Avoid constructing a heavyweight inference service around it.

Investigate:

* static tensor shapes where possible;
* fused inference kernels;
* CUDA graphs;
* reduced precision;
* persistent GPU weights;
* zero-copy handoff between DFT state and model input;
* direct output into the representation consumed by the next DFT operation.

The desired path is approximately:

```text
GPU-resident DFT state
        ↓
learned inference
        ↓
GPU-resident predicted state
        ↓
DFT correction
```

rather than:

```text
GPU
 ↓
CPU serialization
 ↓
Python model framework
 ↓
GPU
 ↓
CPU
 ↓
Fortran
 ↓
GPU
```

---

# 19. Learned Solver Safety Model

Do not build an elaborate uncertainty framework initially.

Use DFT itself as the verifier.

A useful pattern is:

```text
learned prediction
       ↓
DFT iteration
       ↓
residual / energy / density check
       ↓
continue conventional SCF if needed
```

If the model makes a poor prediction, the calculation should degrade toward ordinary SCF rather than silently producing an unverified result.

This gives the learned accelerator a natural fallback path.

---

# 20. Benchmark Metrics

For every optimization report:

### Runtime

[
T_{\mathrm{total}}
]

### SCF runtime

[
T_{\mathrm{SCF}}
]

### Average iteration time

[
T_{\mathrm{iter}}
]

### Iteration count

[
N_{\mathrm{SCF}}
]

### End-to-end speedup

[
S=
\frac{T_{\mathrm{SIESTA}}}
{T_{\mathrm{new}}}.
]

Also record:

* GPU utilization;
* GPU memory;
* CPU utilization;
* host/device transfer;
* kernel launch count.

---

# 21. Scientific Comparison

For representative benchmark calculations compare against upstream:

* total energy;
* energy/atom;
* eigenvalues near the Fermi level;
* Fermi energy;
* charge;
* total magnetic moment;
* atom-resolved moments;
* forces;
* stress when relevant;
* SOC energies;
* directional energy differences;
* MAE.

Do not require bitwise equivalence.

Floating-point ordering and different linear algebra implementations make that counterproductive.

Require scientifically meaningful numerical agreement.

---

# 22. Testing Policy

Testing should remain deliberately lean.

Required tests are:

### Kernel-level sanity tests

Use these only where they make development materially easier.

Examples:

* sparse spinor multiplication;
* reductions;
* density accumulation;
* SOC block operations.

### Upstream regression tests

These are the principal tests.

For each benchmark:

```text
reference = upstream_siesta(input)
candidate = gpu_siesta(input)

compare(reference, candidate)
```

### End-to-end magnetic tests

Maintain a small set covering:

* collinear magnetic calculation;
* non-collinear calculation;
* SOC calculation;
* MAE calculation;
* force calculation;
* structural relaxation.

That is sufficient initially.

Do not construct a giant synthetic test suite before building the solver.

---

# 23. Optimization Ledger

Maintain a simple machine-readable ledger.

Example:

```text
optimization:
    gpu_hamiltonian_assembly

benchmark:
    Fe16_SOC

baseline:
    31.2 s

candidate:
    17.4 s

component_speedup:
    4.8x

end_to_end_speedup:
    1.79x
```
