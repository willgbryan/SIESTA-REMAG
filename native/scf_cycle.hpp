// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "hamiltonian.hpp"
#include "density_matrix.hpp"
#include "grid.hpp"
#include "grid_potential.hpp"
#include "hamiltonian_seeds.hpp"
#include "occupations.hpp"
#include "scf_control.hpp"
#include "block_jacobi.hpp"
#include "initial_density.hpp"
#include "spin_constraint.hpp"
#include "tetrahedra_device.hpp"
#include <array>
#include <memory>
#include <vector>

namespace remag {
struct ScfCycleShape {
    HamiltonianAssemblyView assembly;
    DensityMatrixView density;
    GridContractionView grid;
    int mesh[3];
    DensityNormalizationConfig normalization;
    // Cartesian k points, in exactly the assembly/occupation order.
    std::vector<std::array<double,3>> kpoints;
    double occupation_cutoff;
};
struct ScfCycleBindings {
    const FermiProblem* fermi;
    const double* weights;
    const HamiltonianSeedsProblem* hamiltonian;
    const double *kinetic,*nonlocal_h,*onsite;
    const cuDoubleComplex* offsite;
    const GridPotentialProblem* grid;
    const double *atomic_density,*core_density,*neutral_potential;
    const ScfScalarsProblem* scalars;
    const ScfEnergySeeds* constants;
    const int *num,*ptr;
    const double* overlap;
    // Exclusive mutable ownership for the complete cycle. These arrays may
    // never be passed to a legacy per-step adapter while this plan exists.
    double *h,*density,*energy_density;
    // Optional site spin constraint (spin_constraint.hpp), added to every H
    // after the grid and offsite SOC; with a penalty its field follows the density.
    SpinConstraintDevice constraint{};
    // Optional tetrahedron occupations (OccupationFunction Tetrahedra.*) in
    // place of FD, owned by the cycle state; entropy is then zero.
    TetraDevicePlan* tetra=nullptr;
};
struct ScfCycleResult {
    // Sticky status identifies the failed producer: 1 solver, 2 FD,
    // 3 normalization, 4 Hamiltonian seeds, 5 grid, 6 scalars, 7 control, 8 configuration.
    // producer_status preserves its original code. stage records the last
    // entered producer; block/pass identify the failed spectral solve.
    int status,producer_status,stage,block,pass,iterations,finished,reserved;
    ScfControlResult control;
};
static_assert(sizeof(ScfCycleResult)==160,"Unexpected device SCF cycle ABI");

struct ScfCycleStep {
    ScfControlStep step;
    FermiResult fermi;
    DensityNormalizationResult normalization;
    HamiltonianSeedsResult hamiltonian;
    GridPotentialResult grid;
};
static_assert(sizeof(ScfCycleStep)==560,"Unexpected complete SCF diagnostic ABI");

// Optional coarse performance timings. One serial graph owns this storage;
// it is exported once after SCF, never read by the convergence algorithm.
struct ScfStageProfile {
    unsigned long long start=0,nanoseconds[7]{},calls[7]{};
};

// Experimental whole-cycle executor. All bindings/views are exclusively
// leased device storage, prepared before construction, immutable except for
// H/D/E. They and the stream must outlive this plan and all outstanding work.
// This interface is independent of legacy adapter lifetimes/counters; it must
// not be connected to those owners without an explicit bridge handoff.
// Construction allocates, warms libraries on private scratch and builds the
// graph. launch enqueues the entire bounded cycle without any host numerical
// branch, transfer, allocation, synchronization or I/O. Replaying a completed
// cycle is inert. A fresh cycle/geometry requires a new plan and staged inputs.
// Callers synchronize once after launch before exporting results. A failure
// leaves partially updated state: only status/history are valid in that case.
class ScfCyclePlan {
public:
    ScfCyclePlan(const ScfCycleShape&,const ScfCycleBindings&,const ScfControlProblem&,cudaStream_t);
    ~ScfCyclePlan();
    ScfCyclePlan(const ScfCyclePlan&)=delete;
    ScfCyclePlan& operator=(const ScfCyclePlan&)=delete;
    cudaError_t launch();
    // Optional pre-step, before the first launch: construct the first H from
    // device D, including Hartree/PBE and SOC, and carry first-normalization's
    // mixing decision into the device control problem. No H/D/grid readback.
    // This is geometry initialization, not an accepted SCF iteration.
    cudaError_t initialize_hamiltonian(const InitialDensityResult*);
    cudaGraphExec_t executable() const;
    const ScfCycleResult* result() const;
    // Post-cycle failure diagnostics only; pass the failed job from ScfCycleResult.
    // H/S and local failures come from the private workspace for that job.
    const BlockJacobiResult* solver_diagnostic(int job=0) const;
    const BlockJacobiLocalFailure* solver_local_failure(int job=0) const;
    const void* solver_failed_local_input(int job=0) const;
    int solver_local_dimension() const;
    const void* assembled_hamiltonian(int job=0) const;
    const void* assembled_overlap(int job=0) const;
    const ScfControlStep* history() const;
    const ScfCycleStep* diagnostics() const;
    const ScfStageProfile* stage_profile() const;
    const double* old_hamiltonian() const;
    const double* old_density() const;
    const double* old_energy_density() const;
    const double* mesh_density() const;
    const double* mesh_potential() const;
    const double* mesh_auxiliary_potential() const;
    const double* mesh_deformation_density() const;
    const double* energies() const;
    const double* occupations() const;
    const FermiResult* fermi_result() const;
    size_t workspace_bytes() const;
    size_t setup_upload_bytes() const;
    // Layout chosen before capture. Sampled-k jobs can retain every vector
    // block on device and avoid solving the same H/S again after occupations.
    size_t vector_bank_bytes() const;
    int eigensystem_jobs() const;
    int solver_lanes() const;
    bool warm_start() const;
    // REMAG_SCF_K_LOOP=1: k points run in a device loop over solver waves
    // instead of one unrolled subgraph each, which lifts the 128-k limit.
    bool k_loop() const;
    // REMAG_SCF_BATCHED_SOLVER=1 (with the k loop): each wave of k points is solved by batched
    // cuSOLVER/cuBLAS calls (Cholesky, triangular solves, XsyevBatched) instead of block Jacobi.
    bool batched_solver() const;
    int solves_per_iteration() const;
    bool retains_all_vectors() const;
    int graph_nodes() const;
    int forbidden_nodes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
