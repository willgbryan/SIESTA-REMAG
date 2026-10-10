// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "two_center.hpp"
#include <cstddef>
namespace remag {
struct TwoCenterComparison {
    unsigned long long compared,mismatches,nonfinite;
    double absolute[2],scaled[2];
};
static_assert(sizeof(TwoCenterComparison)==56,"Unexpected two-center comparison ABI");
cudaError_t compare_two_center_device(size_t count,const double* actual,const double* reference,
    TwoCenterComparison*,int* status,cudaStream_t);
size_t two_center_query_count(int consumer);
TwoCenterView borrow_integral_two_center(int consumer);
}
extern "C" int remag_two_center_shadow_enabled();
extern "C" int remag_two_center_active();
extern "C" int remag_two_center_stage(int,const int*,double,double,const int*,const int*,const int*,const int*,const double*,const double*,const double*);
// Consumers: 1 kinetic (T), 2 scalar nonlocal (S), 3 SOC nonlocal (S), 4 overlap (S).
extern "C" int remag_two_center_begin(int);
extern "C" int remag_two_center_observe(int,int,int,const double*,double,const double*);
extern "C" int remag_two_center_finish(int);
