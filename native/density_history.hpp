// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace remag {
struct DensityHistoryConfig {int rows,orbitals,atoms,spins,capacity,depth;};
struct DensityHistoryCommand {
    uint64_t generation,distribution,epoch;
    int capacity,nsc[3],mode;
};
struct DensityHistoryResult {
    int status,count,rank,fallback;
    uint64_t generation,epoch,last_push;
};
// Process-lifetime, ordered density/geometry storage. Constructor capacity is
// an allocation bound, not the current sparse capacity. All operations enqueue
// only kernels; command/input/result pointers are device-owned. Geometry and
// orbital distribution must be stable within a generation. Reset is explicit,
// requires a strictly increasing generation and is the only way to change the
// distribution. A fresh topology requires a strictly increasing epoch.
//
// Status 1: invalid shape/topology/configuration; 2: nonfinite data; 3: solve
// failure; 4: stale generation/distribution/epoch; 5: unconverged or duplicate
// push; 6: no saved density. Invalid operations preserve state and outputs.
// Depth zero disables retention. No silent host fallback. Buffers must be
// disjoint from plan storage and from one another. One ordered CUDA stream.
class DensityHistoryPlan {
public:
    DensityHistoryPlan(DensityHistoryConfig,cudaStream_t);
    ~DensityHistoryPlan();
    DensityHistoryPlan(const DensityHistoryPlan&)=delete;
    DensityHistoryPlan& operator=(const DensityHistoryPlan&)=delete;
    int reset(const DensityHistoryCommand*,DensityHistoryResult*);
    int advance(const DensityHistoryCommand*,const int* num,const int* ptr,
                const int* col,DensityHistoryResult*);
    // convergence==1 is the device convergence predicate. Padding in the
    // external capacity is ignored; retained padding is deterministically zero.
    int push(const DensityHistoryCommand*,const double* density,const double* xyz,
             const int* convergence,DensityHistoryResult*);
    int extrapolate(const DensityHistoryCommand*,const double* xyz,double* density,
                    double* coefficients,DensityHistoryResult*);
    // Increase the allocation bound at a geometry boundary, retaining all
    // electronic/history/topology state on device. Does not change the current
    // sparse capacity or epoch. Smaller positive requests are no-ops. Actual
    // growth allocates, launches device copies and synchronizes this stream;
    // it is rejected during capture. Destroy graphs referencing this owner
    // before growing it. Allocation/validation failures preserve the old owner.
    int reserve(int capacity);
    int allocated_capacity() const {return c_.capacity;}
    // Explicit device-to-device diagnostic view; no export/readback. Density
    // shape is [depth,spins,allocation_capacity], coordinates [depth,3*atoms].
    // Optional output capacity pads each density spin and column array to a
    // larger stride; zero selects the current allocation bound.
    int inspect(double* density,double* xyz,int* num,int* ptr,int* col,DensityHistoryResult*,int output_capacity=0);
    size_t workspace_bytes() const;
private:
    DensityHistoryConfig c_;
    cudaStream_t stream_;
    void *state_=nullptr,*math_=nullptr;
    double *density_=nullptr,*xyz_=nullptr,*trial_=nullptr,*coefficients_=nullptr;
    int *num_=nullptr,*ptr_=nullptr,*col_=nullptr,*map_=nullptr,*translated_=nullptr;
    void release();
};
}
