// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "final_grid_state.hpp"

namespace remag {
bool population_enabled();
// The atom partition is staged before SCF. Sparse topology and overlap are
// retained from device buffers before the cycle releases those buffers.
void prepare_population_state(uint64_t epoch,int atoms,int rows,int capacity,int spins,int species,const int* lasto,const int* atom_species,const double* valence);
void bind_population_topology(uint64_t epoch,const int* num,const int* ptr,const double* overlap,cudaStream_t);
// Runs after all force consumers, on their stream. Failure joins the existing
// shared status; no population readback or host polling occurs here.
void evaluate_final_population(const FinalGridStateView&,const FinalForceAccumulatorView&);
void complete_population_publication(uint64_t epoch);
void reset_population_state();
// Kind 1/2/3: raw orbital/atomic/total components. Kind 4/5/6/7:
// derived orbital/atomic/cumulative-species views and summary scalars.
// This is an explicitly requested terminal export, cached after its first use.
void read_final_population(uint64_t epoch,int kind,int atoms,int rows,int spins,int output_count,double* output);
cudaError_t merge_population_status_device(const int*,int*,cudaStream_t);
}

extern "C" int remag_population_enabled();
extern "C" int remag_population_begin(int,int,int,int,int,const int*,const int*,const double*);
extern "C" int remag_population_read(int,int,int,int,int,double*);
