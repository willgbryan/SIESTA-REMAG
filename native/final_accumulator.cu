// SPDX-License-Identifier: GPL-3.0-or-later
#include "final_grid_state.hpp"
#include <cuda_runtime.h>
#include <cmath>
namespace {
__global__ void add_stress(const double* source,double* target,int* status){
    const int i=int(threadIdx.x);
    if(i<9&&!*status){
        const double value=source[i],sum=target[i]+value;
        if(!isfinite(value)||!isfinite(sum))atomicMax(status,2);
        else target[i]=sum;
    }
}
}
cudaError_t remag::add_final_stress_device(const double* source,double* target,int* status,cudaStream_t stream){
    if(!source||!target||!status||!stream)return cudaErrorInvalidValue;
    add_stress<<<1,32,0,stream>>>(source,target,status);return cudaGetLastError();
}
