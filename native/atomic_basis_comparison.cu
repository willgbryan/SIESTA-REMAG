// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_basis.hpp"
#include <cmath>
#include <climits>
namespace remag {
namespace {
__device__ void numeric(double a,double b,int kind,AtomicBasisComparison& c){
    c.compared++;const double error=fabs(a-b);
    if(!isfinite(a)||!isfinite(b)||!isfinite(error)){c.nonfinite++;return;}
    const double scaled=kind==2?error:error/(1.e-10+1.e-10*fabs(b));
    c.absolute[kind]=fmax(c.absolute[kind],error);c.scaled[kind]=fmax(c.scaled[kind],scaled);
    if(kind==2?error!=0.:scaled>1.)c.mismatches++;
}
__global__ void compare(size_t n,AtomicBasisValues a,AtomicBasisValues b,AtomicBasisComparison* out,int* status){
    AtomicBasisComparison c{};
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x){
        c.points++;c.selected+=a.active[i]==1;c.reference_selected+=b.active[i]==1;
        if((a.active[i]!=0&&a.active[i]!=1)||(b.active[i]!=0&&b.active[i]!=1)||a.active[i]!=b.active[i])c.mask_mismatches++;
        numeric(a.value[i],b.value[i],0,c);
        for(int k=0;k<3;k++)numeric(a.gradient[3*i+k],b.gradient[3*i+k],1,c);
        if(b.active[i]==1){
            c.index_compared+=2;c.index_mismatches+=a.point[i]!=b.point[i];c.index_mismatches+=a.atom[i]!=b.atom[i];
            for(int k=0;k<3;k++)numeric(a.displacement[3*i+k],b.displacement[3*i+k],2,c);
        }
    }
    if(!c.points)return;
    atomicAdd(&out->points,c.points);atomicAdd(&out->compared,c.compared);
    if(c.selected)atomicAdd(&out->selected,c.selected);
    if(c.reference_selected)atomicAdd(&out->reference_selected,c.reference_selected);
    if(c.index_compared)atomicAdd(&out->index_compared,c.index_compared);
    if(c.mismatches){atomicAdd(&out->mismatches,c.mismatches);atomicMax(status,3);}
    if(c.nonfinite){atomicAdd(&out->nonfinite,c.nonfinite);atomicMax(status,2);}
    if(c.mask_mismatches){atomicAdd(&out->mask_mismatches,c.mask_mismatches);atomicMax(status,3);}
    if(c.index_mismatches){atomicAdd(&out->index_mismatches,c.index_mismatches);atomicMax(status,3);}
    for(int k=0;k<3;k++){
        atomicMax(reinterpret_cast<unsigned long long*>(out->absolute+k),__double_as_longlong(c.absolute[k]));
        atomicMax(reinterpret_cast<unsigned long long*>(out->scaled+k),__double_as_longlong(c.scaled[k]));
    }
}
bool valid(const AtomicBasisValues& v){return v.value&&v.gradient&&v.displacement&&v.active&&v.point&&v.atom;}
}
cudaError_t compare_atomic_geometry_device(size_t n,const AtomicBasisValues& a,const AtomicBasisValues& b,
        AtomicBasisComparison* out,int* status,cudaStream_t stream){
    if(n<1||n>size_t(INT_MAX)/7||!valid(a)||!valid(b)||!out||!status||!stream)return cudaErrorInvalidValue;
    const int blocks=int((n+127)/128);
    compare<<<blocks>1024?1024:blocks,128,0,stream>>>(n,a,b,out,status);return cudaGetLastError();
}
}
