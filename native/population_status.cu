// SPDX-License-Identifier: GPL-3.0-or-later
#include "population_state.hpp"
namespace remag {
namespace {
__global__ void merge(const int* population,int* shared) {
    if(*population)atomicMax(shared,16);
}
}
cudaError_t merge_population_status_device(const int* population,int* shared,cudaStream_t stream) {
    if(!population||!shared||population==shared)return cudaErrorInvalidValue;
    merge<<<1,1,0,stream>>>(population,shared);
    return cudaGetLastError();
}
}
