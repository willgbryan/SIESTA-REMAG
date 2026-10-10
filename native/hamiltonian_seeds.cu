// SPDX-License-Identifier: GPL-3.0-or-later
// Pinned setup_hamiltonian.F seed, energy and post-grid SOC arithmetic.
#include "hamiltonian_seeds.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace remag {
namespace {
constexpr int threads=256;
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
__device__ bool shape(const HamiltonianSeedsProblem& p,int n,int spins) {
    return p.capacity==n&&p.spins==spins&&!p.reserved&&
        (spins==8?(p.soc==1||p.soc==2):p.soc==0);
}
__global__ void initialize(int n,int spins,const HamiltonianSeedsProblem* p,
                           const double* on,const cuDoubleComplex* off,HamiltonianSeedsResult* r,int post) {
    if(!post) *r=HamiltonianSeedsResult{};
    r->active=0;
    if(!shape(*p,n,spins)||(p->soc==2&&!off)||(!post&&p->soc==1&&!on)) {r->status=1;return;}
    if(r->status) return;
    if(post&&(r->stage!=1||p->soc!=r->prepared_soc)) {r->status=1;return;}
    if(!post) r->prepared_soc=p->soc;
    r->active=1;
}
__device__ double finite_value(double value,HamiltonianSeedsResult* r) {
    if(!isfinite(value)) {atomicMax(&r->status,2);return 0.;}return value;
}
__global__ void seed(int n,int spins,const HamiltonianSeedsProblem* p,const double* kin,const double* vkb,
        const double* on,const cuDoubleComplex* off,const double* density,double* h,double* partial,
        HamiltonianSeedsResult* r) {
    __shared__ double sums[3][threads];double value[3]{};
    if(r->active) for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        const double k=finite_value(kin[i],r),v=finite_value(vkb[i],r);double d[8]{},s[6]{};
        for(int c=0;c<spins;c++) d[c]=finite_value(density[i+n*c],r);
        for(int c=0;c<spins;c++) {
            double element=0.;
            if(c<2) element=k+v;
            else if(p->soc==1) {s[c-2]=finite_value(on[i+n*(c-2)],r);element=s[c-2];}
            h[i+n*c]=finite_value(element,r);
        }
        for(int c=0;c<min(spins,2);c++) {value[0]+=k*d[c];value[1]+=v*d[c];}
        if(p->soc==1) value[2]+=s[0]*d[6]+s[1]*d[7]+s[4]*d[2]+s[5]*d[3]-s[2]*d[4]-s[3]*d[5];
        if(p->soc==2) {
            cuDoubleComplex z[4];for(int c=0;c<4;c++)
                z[c]=make_cuDoubleComplex(finite_value(off[i+n*c].x,r),finite_value(off[i+n*c].y,r));
            value[2]+=z[0].x*d[0]-z[0].y*d[4];
            value[2]+=z[1].x*d[1]-z[1].y*d[5];
            value[2]+=z[3].x*d[2]-z[3].y*d[3];
            value[2]+=z[2].x*d[6]+z[2].y*d[7];
        }
    }
    for(int c=0;c<3;c++) sums[c][threadIdx.x]=value[c];__syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int c=0;c<3;c++) sums[c][threadIdx.x]+=sums[c][threadIdx.x+step];
        __syncthreads();
    }
    if(!threadIdx.x) for(int c=0;c<3;c++) partial[3*blockIdx.x+c]=sums[c][0];
}
__global__ void energies(int blocks,const double* partial,HamiltonianSeedsResult* r) {
    if(r->status) return;
    double values[3]{};for(int i=0;i<blocks;i++) for(int c=0;c<3;c++) values[c]+=partial[3*i+c];
    for(double value:values) if(!isfinite(value)) {r->status=2;return;}
    r->kinetic=values[0];r->nonlocal_energy=values[1];r->soc_energy=values[2];
}
__global__ void post_grid(int n,int spins,const HamiltonianSeedsProblem* p,const cuDoubleComplex* off,
                          const double* h,double* trial,HamiltonianSeedsResult* r) {
    if(!r->active) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        double value[8]{};for(int c=0;c<spins;c++) value[c]=finite_value(h[i+n*c],r);
        if(p->soc==2) {
            cuDoubleComplex z[4];for(int c=0;c<4;c++)
                z[c]=make_cuDoubleComplex(finite_value(off[i+n*c].x,r),finite_value(off[i+n*c].y,r));
            value[0]+=z[0].x;value[4]+=z[0].y;value[1]+=z[1].x;value[5]+=z[1].y;
            value[2]+=z[2].x;value[3]+=z[2].y;value[6]+=z[3].x;value[7]-=z[3].y;
        }
        for(int c=0;c<spins;c++) trial[i+n*c]=finite_value(value[c],r);
    }
}
__global__ void commit(int size,int stage,const double* trial,double* h,HamiltonianSeedsResult* r) {
    if(r->status) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) h[i]=trial[i];
    if(!blockIdx.x&&!threadIdx.x) r->stage=stage;
}
}

HamiltonianSeedsPlan::HamiltonianSeedsPlan(int n,int spins,cudaStream_t stream):
    capacity_(n),spins_(spins),stream_(stream) {
    if(n<1||(spins!=1&&spins!=2&&spins!=4&&spins!=8)||n>INT_MAX/spins)
        throw std::runtime_error("Invalid Hamiltonian seed dimensions");
    blocks_=std::min(1024,(n-1)/threads+1);
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&trial_),size_t(n)*spins*sizeof(double)));
        check(cudaMalloc(reinterpret_cast<void**>(&partial_),size_t(blocks_)*3*sizeof(double)));
    } catch(...) {release();throw;}
}
void HamiltonianSeedsPlan::release() {if(trial_) cudaFree(trial_);if(partial_) cudaFree(partial_);}
HamiltonianSeedsPlan::~HamiltonianSeedsPlan() {release();}
size_t HamiltonianSeedsPlan::workspace_bytes() const {return (size_t(capacity_)*spins_+3*blocks_)*sizeof(double);}
int HamiltonianSeedsPlan::prepare(const HamiltonianSeedsProblem* p,const double* kin,const double* vkb,
        const double* on,const cuDoubleComplex* off,const double* density,double* h,HamiltonianSeedsResult* r) {
    if(!p||!kin||!vkb||!density||!h||!r||h==density||h==kin||h==vkb||h==on||
       static_cast<const void*>(h)==static_cast<const void*>(off)) return int(cudaErrorInvalidValue);
    initialize<<<1,1,0,stream_>>>(capacity_,spins_,p,on,off,r,0);
    seed<<<blocks_,threads,0,stream_>>>(capacity_,spins_,p,kin,vkb,on,off,density,trial_,partial_,r);
    energies<<<1,1,0,stream_>>>(blocks_,partial_,r);
    commit<<<blocks_,threads,0,stream_>>>(capacity_*spins_,1,trial_,h,r);
    return int(cudaGetLastError());
}
int HamiltonianSeedsPlan::finish_grid(const HamiltonianSeedsProblem* p,const cuDoubleComplex* off,
                                      double* h,HamiltonianSeedsResult* r) {
    if(!p||!h||!r||static_cast<const void*>(h)==static_cast<const void*>(off)) return int(cudaErrorInvalidValue);
    initialize<<<1,1,0,stream_>>>(capacity_,spins_,p,nullptr,off,r,1);
    post_grid<<<blocks_,threads,0,stream_>>>(capacity_,spins_,p,off,h,trial_,r);
    commit<<<blocks_,threads,0,stream_>>>(capacity_*spins_,2,trial_,h,r);
    return int(cudaGetLastError());
}
}
