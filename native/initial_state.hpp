// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace remag {
bool initial_state_enabled();
bool initial_population_window();
void read_initial_population(int kind,int atoms,int rows,int spins,int count,double* output);
// Initial population output precedes normalization. Only after that window
// closes may SCF borrow and mutate D. No host density equality/readback is used.
double* borrow_initial_density(uint64_t epoch,int rows,int capacity,int spins,
    const int* num,const int* ptr,const double* overlap,const double* host_identity);
// Caller must first finish every use of the borrowed pointer on its stream.
void release_initial_density(uint64_t epoch);
}
extern "C" int remag_initial_state_enabled();
extern "C" int remag_initial_state_begin(int,int,int,int,int,int,const int*,const int*,
    const double*,const double*,const int*,const int*,const double*);
extern "C" int remag_initial_state_end();
