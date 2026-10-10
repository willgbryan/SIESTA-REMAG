// SPDX-License-Identifier: GPL-3.0-or-later
#include "initial_density.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
namespace remag {
namespace {
__global__ void prepare(int capacity,int spins,const InitialDensityProblem* p,const int* dependency,
        const double* overlap,const double* input,InitialDensityResult* r) {
    *r=InitialDensityResult{};
    if(*dependency){r->status=4;return;}
    if(p->capacity!=capacity||p->spins!=spins||p->spinor!=min(spins,2)||
       (p->mix_first!=0&&p->mix_first!=1)||!isfinite(p->target_charge)||p->target_charge<=0.){r->status=1;return;}
    double charge=0.;
    for(int spin=0;spin<p->spinor;spin++)for(int i=0;i<capacity;i++)charge+=input[size_t(spin)*capacity+i]*overlap[i];
    if(!isfinite(charge)||charge==0.){r->status=2;return;}
    r->charge=charge;r->scale=p->target_charge/charge;
    if(!isfinite(r->scale)){r->status=2;return;}
    r->mix_first=p->mix_first&&fabs(charge-p->target_charge)<=1.e-4;
    r->note=fabs(charge/p->target_charge-1.)>1.e-3;
}
__global__ void validate(int capacity,int spins,const double* input,InitialDensityResult* r) {
    if(r->status)return;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(capacity)*spins;i+=size_t(gridDim.x)*blockDim.x)
        if(!isfinite(input[i])||!isfinite(input[i]*r->scale))atomicMax(&r->status,2);
}
__global__ void commit(int capacity,int spins,const double* input,double* density,double* energy,const InitialDensityResult* r) {
    if(r->status)return;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(capacity)*spins;i+=size_t(gridDim.x)*blockDim.x) {
        density[i]=input[i]*r->scale;
        if(i<size_t(capacity)*min(spins,4))energy[i]=0.;
    }
}
}
cudaError_t initial_density_device(int capacity,int spins,const InitialDensityProblem* p,const int* dependency,
        const double* overlap,const double* input,double* density,double* energy,InitialDensityResult* r,cudaStream_t stream) {
    if(capacity<1||capacity>INT_MAX/8||(spins!=1&&spins!=2&&spins!=4&&spins!=8)||!p||!dependency||
       !overlap||!input||!density||!energy||!r||density==energy||overlap==density||overlap==energy||input==energy)return cudaErrorInvalidValue;
    prepare<<<1,1,0,stream>>>(capacity,spins,p,dependency,overlap,input,r);
    const int blocks=std::min(256,(capacity*spins-1)/256+1);
    validate<<<blocks,256,0,stream>>>(capacity,spins,input,r);
    commit<<<blocks,256,0,stream>>>(capacity,spins,input,density,energy,r);
    return cudaGetLastError();
}
}
