// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "hamiltonian.hpp"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>
namespace remag {
// Add the site constraint term to sparse H in place, after the grid potential
// and offsite SOC. orbital_field holds the field (Ry per muB) for each
// unit-cell orbital, three values per orbital and zero off-site. Device
// pointers only; one kernel on the supplied stream, no allocation, copy,
// synchronization or I/O.
cudaError_t spin_constraint_field_device(const HamiltonianAssemblyView& view,int spins,
    const double* orbital_field,double* h,cudaStream_t stream);
// Row Mulliken spin moments (x,y,z per unit-cell orbital), the arithmetic of
// mulliken.F option 1 and spnvec. Deterministic row order, no atomics.
cudaError_t spin_constraint_moments_device(int rows,int spins,size_t capacity,const int* num,const int* ptr,
    const double* overlap,const double* density,double* orbital_moments,cudaStream_t stream);
// Device storage of one SCF's constraint. sites=0 means none. With penalty 0,
// orbital_field is the staged trial field; otherwise it is rebuilt from the
// density before every H as b_s + rho M_perp,s (spin_constraint_math.hpp).
struct SpinConstraintDevice {
    int sites=0;double penalty=0.;
    const int* orbital_site=nullptr;const double* site_fields=nullptr;const double* targets=nullptr;
    double *orbital_moments=nullptr,*orbital_field=nullptr;
};
// Rebuild orbital_field from density (penalty > 0): orbital moments, then one
// block reduces site moments in orbital order and writes the field.
cudaError_t spin_constraint_penalty_field_device(int rows,int spins,size_t capacity,const int* num,const int* ptr,
    const double* overlap,const double* density,const SpinConstraintDevice& constraint,cudaStream_t stream);

// Host owner between Fortran and the SCF producers (spin_constraint_state.cpp).
struct SpinConstraintTrial {
    int sites=0;double penalty=0.;
    const std::vector<int>* orbital_site=nullptr;
    const std::vector<double> *site_fields=nullptr,*targets=nullptr,*orbital_field=nullptr;
};
bool spin_constraint_active();
// Pending trial for an SCF with this many rows, or sites=0 without constraints.
SpinConstraintTrial spin_constraint_trial(int rows);
// Device-cycle producer: final-density orbital moments for the pending trial.
void spin_constraint_record_moments(uint64_t epoch,const double* orbital_moments,int rows);
}
