// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "grid.hpp"
#include <cstdint>
#include <vector>

namespace remag {
struct FinalGridStateView {
    GridContractionShape shape{};
    int h_spins=0,energy_spins=0;
    uint64_t epoch=0;
    double dvol=0.,volume=0.;
    const double *density=nullptr,*energy_density=nullptr,*potential=nullptr,*auxiliary=nullptr,*deformation=nullptr;
    size_t bytes=0;
    int mesh[3]{},nsm=0,partial_core=0;
    const double* hartree_stress=nullptr;
    const int* convergence=nullptr;
};
bool final_grid_state_enabled();
bool atomic_grid_force_enabled();
bool integral_force_enabled();
bool pair_force_active();
struct AtomicForceRecord {
    int kind=0,points=0,entries=0,diagonal_spins=0,graph_nodes=0;
    size_t setup_bytes=0;
    size_t derivative_upload_bytes=0,host_derivative_storage_bytes=0;
};
struct FinalForceAccumulatorView {
    cudaStream_t stream=nullptr;
    double *force=nullptr,*stress=nullptr;
    int* status=nullptr;
    size_t staged_bytes=0; // common initial values plus completed atomic geometry
    double* energy=nullptr; // kinetic, scalar nonlocal, four real/four imaginary SOC terms
};
struct IntegralForceRecord {
    int consumer=0,queries=0,graph_nodes=0;
    size_t setup_bytes=0,workspace_bytes=0;
    bool retained_integrals=false;
};
struct FinalForcePublication {
    int status=-1;
    size_t exported_bytes=0;
    std::vector<double> energy,force,stress;
};
// No intermediate readback. The initial host contribution must stay unchanged
// across every consumer. Device status is checked only at final publication.
FinalForceAccumulatorView final_force_accumulator(uint64_t epoch,int atoms,int forces,int stress,
    const double* initial_force,const double* initial_stress);
void record_final_atomic_force(uint64_t epoch,const AtomicForceRecord&);
const std::vector<AtomicForceRecord>& final_atomic_force_records(uint64_t epoch);
void record_final_integral_force(uint64_t epoch,const IntegralForceRecord&);
void add_final_hartree_stress(uint64_t epoch);
// Collection checks the sticky device status and finite outputs once. A failed
// collection is terminal. Analysis/reset stay forbidden until publication is
// completed after all derivative/geometry owners accept the same result.
FinalForcePublication collect_final_forces(uint64_t epoch,int atoms,
    const double* initial_force,const double* initial_stress);
void complete_final_force_publication(uint64_t epoch);
const std::vector<IntegralForceRecord>& final_integral_force_records(uint64_t epoch);
cudaError_t add_final_stress_device(const double*,double*,int*,cudaStream_t);
// Immutable post-cycle snapshot; copies are device-to-device only. The cycle
// must have completed successfully, and source pointers must outlive this call.
void retain_final_grid_state(const FinalGridStateView&,cudaStream_t);
// Consumer bits: 1 partial core, 2 neutral atom, 4 dfscf. Required consumers
// execute once, in that order. A new epoch cannot discard an unfinished one.
FinalGridStateView borrow_final_grid_state(uint64_t epoch,int consumer=4);
void complete_final_grid_force(uint64_t epoch,int consumer=4);
void reset_final_grid_state();
// Read-only ownership check for a requested analysis export after all forces.
bool final_grid_forces_complete(uint64_t epoch);
// A single history lease after accepted final forces. The caller must complete
// every queued use before returning it; reset cannot discard an active lease.
FinalGridStateView borrow_final_history_state(uint64_t epoch);
void complete_final_history_state(uint64_t epoch);
// Population may join status only during the guarded final collection.
bool final_population_window(uint64_t epoch,const double* density);
// Diagnostic pair consumers share the immutable D/E snapshot, without changing
// the grid accumulator. Kinetic must finish before the first grid consumer;
// overlap must finish after the last. A failed lease cannot be discarded.
bool pair_force_shadow_enabled();
FinalGridStateView borrow_final_pair_state(uint64_t epoch,int consumer);
void complete_final_pair_shadow(uint64_t epoch,int consumer);
}
