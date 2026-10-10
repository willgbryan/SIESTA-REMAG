// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include "nonlocal_compact.hpp"
namespace remag {
struct FinalGridStateView;
struct FinalForceAccumulatorView;
void observe_nonlocal_geometry(int table,int a,int b,const double* xyz);
bool nonlocal_force_shadow_pending(int consumer);
struct NonlocalIntegralGeometry {const int* pairs;const double* xyz;};
NonlocalIntegralGeometry stage_nonlocal_force_shadow(int consumer,size_t queries,const int* pairs,const double* xyz,const double* integrals,cudaStream_t);
void apply_nonlocal_force_shadow(int* status,cudaStream_t);
void finish_nonlocal_force_shadow(int status);
struct PreparedNonlocalForceView {NonlocalCompactView view;const int* pairs;};
PreparedNonlocalForceView borrow_integral_nonlocal(int consumer,int atoms,const FinalGridStateView&,cudaStream_t);
void apply_integral_nonlocal(const double* integrals,double* weights,const FinalForceAccumulatorView&);
void finish_integral_nonlocal(int status);
}
extern "C" int remag_nonlocal_force_shadow_enabled();
extern "C" int remag_nonlocal_force_active();
extern "C" int remag_nonlocal_begin(int,int,int,int,int,int,int,double,const int*,const int*,const int*,const double*,const double*);
extern "C" int remag_nonlocal_center(int,int,int,int,const int*,const int*,const int*,const int*,const int*,const double*);
extern "C" int remag_nonlocal_end(int,const double*,const double*,const double*);
extern "C" int remag_nonlocal_bind_geometry();
