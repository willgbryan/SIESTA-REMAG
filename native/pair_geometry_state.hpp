// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace remag {
struct FinalGridStateView;
struct PairGeometryView {
    int queries=0;
    const int *row=nullptr,*column=nullptr,*atoms=nullptr;
    const int *num=nullptr,*ptr=nullptr,*list=nullptr,*pairs=nullptr;
    const double* xyz=nullptr;
};
// Kinetic and overlap traverse the same orbital-pair geometry. Prepare once
// during H0, bind after the grid epoch advances, and consume in that order.
void validate_pair_geometry(int consumer,int atoms,int rows,int columns,int capacity,
    double volume,const int* num,const int* ptr,const int* list);
void begin_pair_geometry(int consumer);
void observe_pair_geometry(int consumer,int row,int column,int atom_a,int atom_b);
PairGeometryView stage_pair_geometry(int consumer,size_t queries,const int* pairs,const double* xyz);
void complete_pair_geometry(int consumer);
PairGeometryView borrow_integral_pair_geometry(int consumer,int atoms,const FinalGridStateView&);
void finish_integral_pair_geometry(uint64_t epoch,int status);
}
extern "C" int remag_pair_force_active();
extern "C" int remag_pair_geometry_prepare(int,int,int,int,double,const int*,const int*,const int*);
extern "C" int remag_pair_geometry_add(int,int,int,int,int,int,const double*);
extern "C" int remag_pair_geometry_finish();
extern "C" int remag_pair_geometry_bind();
