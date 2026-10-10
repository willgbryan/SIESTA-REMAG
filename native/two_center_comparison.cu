// SPDX-License-Identifier: GPL-3.0-or-later
#include "two_center_state.hpp"
#include <cmath>
#include <climits>
namespace remag {
namespace {
__global__ void compare(size_t n,const double* actual,const double* reference,TwoCenterComparison* out,int* status) {
    unsigned long long compared=0,mismatches=0,nonfinite=0;double absolute[2]{},scaled[2]{};
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<4*n;i+=size_t(blockDim.x)*gridDim.x) {
        compared++;const int kind=i%4==0?0:1;const double a=actual[i],b=reference[i],error=fabs(a-b);
        if(!isfinite(a)||!isfinite(b)||!isfinite(error)){nonfinite++;continue;}
        const double ratio=error/(1.e-10+1.e-10*fabs(b));absolute[kind]=fmax(absolute[kind],error);scaled[kind]=fmax(scaled[kind],ratio);
        if(ratio>1.)mismatches++;
    }
    if(compared) {
        atomicAdd(&out->compared,compared);
        if(mismatches){atomicAdd(&out->mismatches,mismatches);atomicMax(status,3);}
        if(nonfinite){atomicAdd(&out->nonfinite,nonfinite);atomicMax(status,2);}
        for(int k=0;k<2;k++) {
            atomicMax(reinterpret_cast<unsigned long long*>(out->absolute+k),__double_as_longlong(absolute[k]));
            atomicMax(reinterpret_cast<unsigned long long*>(out->scaled+k),__double_as_longlong(scaled[k]));
        }
    }
}
}
cudaError_t compare_two_center_device(size_t n,const double* actual,const double* reference,TwoCenterComparison* out,int* status,cudaStream_t stream) {
    if(n<1||n>size_t(INT_MAX)/4||!actual||!reference||!out||!status||!stream)return cudaErrorInvalidValue;
    const int blocks=int((4*n+255)/256);compare<<<blocks>1024?1024:blocks,256,0,stream>>>(n,actual,reference,out,status);return cudaGetLastError();
}
}
