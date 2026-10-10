// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "nonlocal_h0.hpp"
#include "nonlocal_query.hpp"
#include <cstdint>
namespace remag {
bool nonlocal_h0_shadow_enabled();
bool nonlocal_h0_enabled();
bool nonlocal_h0_active();
bool nonlocal_h0_poison_enabled();
void record_nonlocal_h0_host_poison(int capacity,const double* h,const H0Complex* z);
struct NonlocalH0Matrices {const double* scalar;const H0Complex* soc;};
NonlocalH0Matrices borrow_nonlocal_h0_matrices(uint64_t epoch,int rows,int columns,int capacity,int soc);
void release_nonlocal_h0_matrices(uint64_t epoch);
void validate_nonlocal_h0_topology(int rows,int columns,int capacity,const int* num,const int* ptr,const int* list);
void begin_nonlocal_h0_shadow(int rows,int columns,int capacity,int consumer,int split,
    const double* initial,const H0Complex* initial_soc);
void evaluate_nonlocal_h0_shadow(NonlocalH0View view,const NonlocalNeighbor* host_neighbors,
    const int* device_pairs,const double* device_xyz,const double* reference,const H0Complex* reference_soc,
    const NonlocalQueryView* compact=nullptr);
void bind_nonlocal_h0_shadow(uint64_t epoch);
const double* borrow_nonlocal_h0_integrals(uint64_t epoch,int consumer,int queries);
void finish_nonlocal_h0_shadow(uint64_t epoch);
}
extern "C" int remag_nonlocal_h0_shadow_enabled();
extern "C" int remag_nonlocal_h0_enabled();
extern "C" int remag_nonlocal_h0_active();
extern "C" int remag_nonlocal_h0_poison_enabled();
extern "C" int remag_nonlocal_h0_neighbor(int orbital,const double* xyz);
extern "C" int remag_nonlocal_h0_projectors(int neighbors,int projectors,const int* registry);
extern "C" int remag_nonlocal_h0_split_warning();
extern "C" int remag_nonlocal_h0_matrix(int phase,int capacity,const double*,const remag::H0Complex*);
