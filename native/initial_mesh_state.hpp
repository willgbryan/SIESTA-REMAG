// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "grid_potential.hpp"
#include <cstdint>
namespace remag {
bool initial_mesh_enabled();
// Compact topology is checked on host; orbital values are never scanned there.
const float* initial_mesh_values(uint64_t epoch,int no,int np,int nsp,int entries,
    const int* ends,const int* orbitals);
// Preparation completes its stream before returning. Bind the Hartree energy
// on the consumer stream. The owner survives SCF, requested exports and forces.
const double* borrow_initial_mesh_density(uint64_t epoch,const GridPotentialProblem&,
    GridPotentialProblem* device_problem,cudaStream_t);
void complete_initial_mesh_forces(uint64_t epoch);
void release_initial_mesh_density(uint64_t epoch,int cycle_status,int producer_status,int converged);
}
extern "C" int remag_initial_mesh_enabled();
extern "C" int remag_initial_mesh_prepare(const int*,const double*,double,double,
    const int*,const int*,const int*,const double*);
