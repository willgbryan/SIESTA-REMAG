// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "atomic_basis.hpp"
#include "atomic_grid_force.hpp"
#include <cstdint>
namespace remag {
bool atomic_basis_shadow_enabled();
bool atomic_basis_device_enabled();
bool atomic_basis_active();
void begin_atomic_basis(uint64_t epoch,int kind,int points,int nsp,int atoms,const int* mesh,int nsm,int core);
AtomicGridForceView prepare_atomic_basis(uint64_t epoch,int kind,size_t selected,cudaStream_t);
void enqueue_atomic_basis(uint64_t epoch,int kind,int* status,cudaStream_t);
// Consumes the existing terminal force status. Only shadow comparison records
// download here, after both terms and dfscf have completed; no per-term export.
void finish_atomic_basis(uint64_t epoch,int force_status);
}
extern "C" int remag_atomic_basis_stage(const int*,const int*,const int*,const double*,const double*,const double*,
    const int*,const int*,const int*,const int*,const int*,const double*,const double*,const double*);
extern "C" int remag_atomic_basis_point(int,int,int,int,int,int,double,const double*,const double*);
