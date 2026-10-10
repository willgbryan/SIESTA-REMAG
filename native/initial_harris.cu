// SPDX-License-Identifier: GPL-3.0-or-later
#include "initial_harris.hpp"
#include <cmath>
namespace remag {
namespace {
__global__ void bind(const HartreeResult* h,GridPotentialProblem* grid,int* status) {
    grid->harris_energy=NAN;
    if(*status)return;
    if(h->status){atomicMax(status,h->status==1?1:2);return;}
    if(!isfinite(h->energy)){atomicMax(status,2);return;}
    grid->harris_energy=h->energy;
}
}
cudaError_t bind_initial_harris_device(const HartreeResult* h,GridPotentialProblem* grid,int* status,cudaStream_t stream) {
    if(!h||!grid||!status||!stream)return cudaErrorInvalidValue;
    bind<<<1,1,0,stream>>>(h,grid,status);return cudaGetLastError();
}
}
