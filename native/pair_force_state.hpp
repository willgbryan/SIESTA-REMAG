// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include "pair_geometry_state.hpp"
namespace remag {
bool pair_force_shadow_pending(int consumer);
PairGeometryView stage_pair_force_shadow(uint64_t epoch,int consumer,int queries,const int* pairs,const double* xyz,const double* integrals,cudaStream_t);
void apply_pair_force_shadow(int* status,cudaStream_t);
void finish_pair_force_shadow(int status);
}
extern "C" int remag_pair_force_shadow_enabled();
extern "C" int remag_pair_force_begin(int,int,int,int,int,int,double,const int*,const int*,const int*,const double*,const double*);
extern "C" int remag_pair_force_observe(int,int,int,int);
extern "C" int remag_pair_force_reference(int,double,const double*,const double*);
