// SPDX-License-Identifier: GPL-3.0-or-later
// Device consumers of pinned compute_energies, dm_charge and compute_max_diff.
#include "scf_scalars.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace remag {
namespace {
constexpr int threads=256;
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
__global__ void initialize(int capacity,int spins,int rows,const ScfScalarsProblem* p,
                           const ScfEnergySeeds* seeds,ScfScalarsResult* r) {
    *r=ScfScalarsResult{};
    if(p->capacity!=capacity||p->spins!=spins||p->rows!=rows||p->reserved) {r->status=1;return;}
    const double values[]={seeds->ena,seeds->kinetic,seeds->nonlocal_energy,seeds->soc,seeds->ions,
        seeds->enascf,seeds->enaatm,seeds->hartree,seeds->xc,seeds->entropy,seeds->temperature,seeds->madelung};
    for(double value:values) if(!isfinite(value)) {r->status=2;return;}
    if(seeds->temperature<0.) {r->status=1;return;}
    r->active=1;
}
__device__ double finite_value(double value,ScfScalarsResult* r) {
    if(!isfinite(value)) {atomicMax(&r->status,2);return 0.;}return value;
}
__global__ void capacity_reduce(int n,int spins,const double* h,const double* hold,
        const double* density,const double* old_density,const double* energy,const double* old_energy,
        double* residual,double* partial,ScfScalarsResult* r) {
    __shared__ double sums[5][threads];double value[5]{};
    if(r->active) for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        for(int spin=0;spin<spins;spin++) {
            const int j=i+n*spin;
            const double a=finite_value(h[j],r),b=finite_value(hold[j],r);
            const double d=finite_value(density[j],r),old=finite_value(old_density[j],r);
            const double dd=finite_value(d-old,r),dh=finite_value(a-b,r);
            residual[j]=dh;
            const double weight=spins==4&&spin>=2?2.:1.;
            value[0]+=weight*a*d;value[1]+=weight*a*dd;
            value[2]=fmax(value[2],fabs(dd));value[3]=fmax(value[3],fabs(dh));
            if(spin<4) {
                const double e=finite_value(energy[j],r),eo=finite_value(old_energy[j],r);
                value[4]=fmax(value[4],fabs(finite_value(e-eo,r)));
            }
        }
    }
    for(int f=0;f<5;f++) sums[f][threadIdx.x]=value[f];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int f=0;f<5;f++)
            sums[f][threadIdx.x]=f<2?sums[f][threadIdx.x]+sums[f][threadIdx.x+step]:
                fmax(sums[f][threadIdx.x],sums[f][threadIdx.x+step]);
        __syncthreads();
    }
    if(!threadIdx.x) for(int f=0;f<5;f++) partial[5*blockIdx.x+f]=sums[f][0];
}
// Cooperate across the entries of each sparse row. A block owns one row
// (or a grid-stride group for very large row counts), retaining only its four
// spin-charge partials. Padding outside num/ptr never enters the contraction.
__global__ void charge_reduce(int n,int spins,int rows,const int* num,const int* ptr,
        const double* overlap,const double* density,double* partial,ScfScalarsResult* r) {
    __shared__ double sums[4][threads];double value[4]{};
    if(r->active) for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
        const int count=num[row],start=ptr[row];
        if(start<0||start>n||count<0||count>n-start) {
            if(!threadIdx.x) atomicMax(&r->status,2);
            continue;
        }
        // Offset iteration avoids start+threadIdx.x overflowing for a short
        // row located near the end of a large, valid capacity.
        for(long long offset=threadIdx.x;offset<count;offset+=blockDim.x) {
            const int i=start+int(offset);
            const double s=finite_value(overlap[i],r);
            for(int spin=0;spin<min(spins,4);spin++) {
                double d=finite_value(density[i+n*spin],r);
                if(spins==8&&spin>=2) d=.5*(d+finite_value(density[i+n*(spin+4)],r));
                value[spin]+=d*s;
            }
        }
    }
    for(int f=0;f<4;f++) sums[f][threadIdx.x]=value[f];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int f=0;f<4;f++) sums[f][threadIdx.x]+=sums[f][threadIdx.x+step];
        __syncthreads();
    }
    if(!threadIdx.x) for(int f=0;f<4;f++) partial[4*blockIdx.x+f]=sums[f][0];
}
__global__ void finish(int blocks,int row_blocks,const ScfEnergySeeds* seeds,
                       const double* partial,const double* charge,ScfScalarsResult* r) {
    r->active=0;if(r->status) return;
    double reduced[9]{};
    for(int i=0;i<blocks;i++) for(int f=0;f<5;f++) {
        const double value=partial[5*i+f];
        if(!isfinite(value)) {r->status=2;return;}
        reduced[f]=f<2?reduced[f]+value:fmax(reduced[f],value);
    }
    for(int i=0;i<row_blocks;i++) for(int f=0;f<4;f++) reduced[5+f]+=charge[4*i+f];
    const auto s=*seeds;double out[13]{};
    out[0]=reduced[0];out[1]=reduced[1];out[5]=s.enascf-s.enaatm;
    out[2]=s.ena+s.kinetic+s.nonlocal_energy+s.soc-s.ions+out[5]+s.hartree+s.xc+s.madelung;
    out[3]=out[2]+out[1];out[4]=out[2]-s.temperature*s.entropy;
    for(int f=0;f<3;f++) out[6+f]=reduced[2+f];
    for(int f=0;f<4;f++) out[9+f]=reduced[5+f];
    for(double value:out) if(!isfinite(value)) {r->status=2;return;}
    for(int f=0;f<13;f++) r->values[f]=out[f];r->active=1;
}
__global__ void commit(int size,const double* residual,double* output,const ScfScalarsResult* r) {
    if(!r->active) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) output[i]=residual[i];
}
}

ScfScalarsPlan::ScfScalarsPlan(int capacity,int spins,int rows,cudaStream_t stream):
    capacity_(capacity),spins_(spins),rows_(rows),stream_(stream) {
    if(capacity<1||(spins!=1&&spins!=2&&spins!=4&&spins!=8)||capacity>INT_MAX/spins||rows<1)
        throw std::runtime_error("Invalid SCF scalar plan dimensions");
    blocks_=std::min(1024,(capacity-1)/threads+1);row_blocks_=std::min(1024,rows);
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&partial_),size_t(blocks_)*5*sizeof(double)));
        check(cudaMalloc(reinterpret_cast<void**>(&charge_partial_),size_t(row_blocks_)*4*sizeof(double)));
        check(cudaMalloc(reinterpret_cast<void**>(&residual_),size_t(capacity)*spins*sizeof(double)));
    } catch(...) {release();throw;}
}
void ScfScalarsPlan::release() {
    if(partial_) cudaFree(partial_);if(charge_partial_) cudaFree(charge_partial_);if(residual_) cudaFree(residual_);
}
ScfScalarsPlan::~ScfScalarsPlan() {release();}
size_t ScfScalarsPlan::workspace_bytes() const {
    return (size_t(blocks_)*5+size_t(row_blocks_)*4+size_t(capacity_)*spins_)*sizeof(double);
}
int ScfScalarsPlan::apply(const ScfScalarsProblem* p,const ScfEnergySeeds* seeds,const int* num,const int* ptr,
        const double* overlap,const double* h,const double* hold,const double* d,const double* old,
        const double* e,const double* eold,double* residual,ScfScalarsResult* r) {
    if(!p||!seeds||!num||!ptr||!overlap||!h||!hold||!d||!old||!e||!eold||!residual||!r||
       residual==h||residual==hold||residual==d||residual==old||residual==e||residual==eold||residual==overlap)
        return int(cudaErrorInvalidValue);
    initialize<<<1,1,0,stream_>>>(capacity_,spins_,rows_,p,seeds,r);
    capacity_reduce<<<blocks_,threads,0,stream_>>>(capacity_,spins_,h,hold,d,old,e,eold,residual_,partial_,r);
    charge_reduce<<<row_blocks_,threads,0,stream_>>>(capacity_,spins_,rows_,num,ptr,overlap,d,charge_partial_,r);
    finish<<<1,1,0,stream_>>>(blocks_,row_blocks_,seeds,partial_,charge_partial_,r);
    commit<<<blocks_,threads,0,stream_>>>(capacity_*spins_,residual_,residual,r);
    return int(cudaGetLastError());
}
}
