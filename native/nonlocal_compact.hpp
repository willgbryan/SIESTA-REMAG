// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
struct NonlocalCenter {int neighbor,neighbors,block,blocks,projectors,query,atom;};
struct NonlocalNeighbor {int row,column,atom,center;}; // row==0 is not a local left orbital
struct NonlocalBlock {int offset,l;double epsilon[2];}; // offset is the actual Ski index minus one
static_assert(sizeof(NonlocalCenter)==28&&sizeof(NonlocalNeighbor)==16&&sizeof(NonlocalBlock)==24,"Nonlocal geometry ABI");
struct NonlocalCompactView {
    int neighbors,centers,blocks,queries,atoms,rows,columns,capacity,components,spinor,soc,split;
    double volume;
    const NonlocalCenter* center;
    const NonlocalNeighbor* neighbor;
    const NonlocalBlock* block;
    const int *num,*ptr,*list,*lookup;
    const double *integrals,*xyz;
};
// Geometry and caller-owned scratch are staged outside the electronic cycle.
// Scratch holds one scalar weight per sparse storage entry, never a dense D.
// CSR row spans must be disjoint; the geometry owner validates this before staging.
// Enqueues two kernels; no allocation, transfers, polling or I/O.
cudaError_t add_nonlocal_compact_device(const NonlocalCompactView&,const double* density,double* weights,
    double* energy,double* force,double* stress,int* status,cudaStream_t);
}
