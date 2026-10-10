// SPDX-License-Identifier: GPL-3.0-or-later
// Periodic Poisson potential, energy and stress from upstream poison.F.
#include "hartree.hpp"
#include <cuComplex.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace remag {
namespace {
constexpr int threads=256,fields=10;
void check(cudaError_t error) {
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(cufftResult error) {
    if(error!=CUFFT_SUCCESS) throw std::runtime_error("cuFFT status "+std::to_string(int(error)));
}
__global__ void initialize(const HartreeProblem* p,HartreeResult* result,int n1,int n2,int n3) {
    *result=HartreeResult{};
    if(p->mesh[0]!=n1||p->mesh[1]!=n2||p->mesh[2]!=n3) result->status=1;
    if(!isfinite(p->cutoff)||p->cutoff<=0.||!isfinite(p->volume)||p->volume<=0.) result->status=2;
    for(int i=0;i<9;i++) if(!isfinite(p->reciprocal[i])) result->status=2;
}
__global__ void pack(int n,const double* density,cufftDoubleComplex* spectrum,HartreeResult* result) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        double value=density[i];
        if(!isfinite(value)) {atomicMax(&result->status,2);value=0.;}
        spectrum[i]=make_cuDoubleComplex(value,0.);
    }
}
__global__ void filter(int n,const HartreeProblem* problem,cufftDoubleComplex* spectrum,
                       double* partial,HartreeResult* result) {
    __shared__ double sums[fields][threads];
    double values[fields]={};const auto p=*problem;
    bool shape_ok=p.mesh[0]>0&&p.mesh[1]>0&&p.mesh[2]>0&&
        static_cast<long long>(p.mesh[0])*p.mesh[1]<=INT_MAX/p.mesh[2]&&
        p.mesh[0]*p.mesh[1]*p.mesh[2]==n;
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<n;index+=gridDim.x*blockDim.x) {
        if(!shape_ok) {spectrum[index]=make_cuDoubleComplex(0.,0.);continue;}
        int j[3]={index%p.mesh[0],(index/p.mesh[0])%p.mesh[1],index/(p.mesh[0]*p.mesh[1])};
        for(int a=0;a<3;a++) if(j[a]>p.mesh[a]/2) j[a]-=p.mesh[a];
        double g[3];
        for(int a=0;a<3;a++) g[a]=(p.reciprocal[a]*j[0]+p.reciprocal[a+3]*j[1])+p.reciprocal[a+6]*j[2];
        double g2=(g[0]*g[0]+g[1]*g[1])+g[2]*g[2];
        auto rho=spectrum[index];
        if(!isfinite(g2)||!isfinite(rho.x)||!isfinite(rho.y)) atomicMax(&result->status,2);
        if(g2<p.cutoff&&g2>1.e-15) {
            double vg=(8.*3.141592653589793238462643383279502884)/g2;
            double du=vg*(rho.x*rho.x+rho.y*rho.y),c=2.*du/g2;
            values[0]+=du;
            for(int col=0;col<3;col++) for(int row=0;row<3;row++) values[1+row+3*col]+=c*g[col]*g[row];
            spectrum[index]=make_cuDoubleComplex(vg*rho.x,vg*rho.y);
        } else spectrum[index]=make_cuDoubleComplex(0.,0.);
    }
    for(int field=0;field<fields;field++) sums[field][threadIdx.x]=values[field];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int field=0;field<fields;field++) sums[field][threadIdx.x]+=sums[field][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int field=0;field<fields;field++) partial[fields*blockIdx.x+field]=sums[field][0];
}
__global__ void reduce(int n,int blocks,const HartreeProblem* p,const double* partial,HartreeResult* result) {
    __shared__ double sums[fields][threads];
    for(int field=0;field<fields;field++) {
        double value=0.;
        for(int i=threadIdx.x;i<blocks;i+=threads) value+=partial[fields*i+field];
        sums[field][threadIdx.x]=value;
    }
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int field=0;field<fields;field++) sums[field][threadIdx.x]+=sums[field][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        for(int field=0;field<fields;field++) if(!isfinite(sums[field][0])) result->status=2;
        if(result->status) return;
        double norm=.5/(double(n)*n);
        result->energy=.5*sums[0][0]*p->volume/(double(n)*n);
        for(int i=0;i<9;i++) result->stress[i]=norm*sums[1+i][0];
        for(int i=0;i<3;i++) result->stress[4*i]+=result->energy/p->volume;
        if(!isfinite(result->energy)) result->status=2;
        for(int i=0;i<9;i++) if(!isfinite(result->stress[i])) result->status=2;
    }
}
__global__ void unpack(int n,const cufftDoubleComplex* spectrum,double* potential,HartreeResult* result) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        double value=spectrum[i].x/double(n);
        if(!isfinite(value)) atomicMax(&result->status,2);
        potential[i]=value;
    }
}
}

HartreePlan::HartreePlan(int n1,int n2,int n3,cudaStream_t stream):mesh_{n1,n2,n3},stream_(stream) {
    if(n1<1||n2<1||n3<1||static_cast<long long>(n1)*n2>INT_MAX/n3)
        throw std::runtime_error("Invalid periodic Hartree mesh");
    size_=n1*n2*n3;blocks_=std::min(1024,(size_-1)/threads+1);
    try {
        check(cufftCreate(&plan_));check(cufftSetAutoAllocation(plan_,0));
        // Fortran's first index is contiguous; cuFFT's last dimension is.
        check(cufftMakePlan3d(plan_,n3,n2,n1,CUFFT_Z2Z,&work_bytes_));
        if(work_bytes_) {
            check(cudaMalloc(&work_,work_bytes_));check(cufftSetWorkArea(plan_,work_));
        }
        check(cufftSetStream(plan_,stream_));
        check(cudaMalloc(reinterpret_cast<void**>(&spectrum_),size_t(size_)*sizeof(*spectrum_)));
        check(cudaMalloc(reinterpret_cast<void**>(&partial_),size_t(blocks_)*fields*sizeof(double)));
    } catch(...) {release();throw;}
}
void HartreePlan::release() {
    if(plan_) cufftDestroy(plan_);
    if(work_) cudaFree(work_);
    if(spectrum_) cudaFree(spectrum_);
    if(partial_) cudaFree(partial_);
}
HartreePlan::~HartreePlan() {release();}
size_t HartreePlan::resident_bytes() const {
    return work_bytes_+size_t(size_)*sizeof(*spectrum_)+size_t(blocks_)*fields*sizeof(double);
}
int HartreePlan::apply(const HartreeProblem* problem,const double* density,double* potential,HartreeResult* result) {
    initialize<<<1,1,0,stream_>>>(problem,result,mesh_[0],mesh_[1],mesh_[2]);
    pack<<<blocks_,threads,0,stream_>>>(size_,density,spectrum_,result);
    auto error=cudaGetLastError();if(error!=cudaSuccess) return int(error);
    // GPFA uses a positive exponent for poison.F's isn=-1 forward transform.
    auto fft=cufftExecZ2Z(plan_,spectrum_,spectrum_,CUFFT_INVERSE);
    if(fft!=CUFFT_SUCCESS) return -int(fft);
    filter<<<blocks_,threads,0,stream_>>>(size_,problem,spectrum_,partial_,result);
    reduce<<<1,threads,0,stream_>>>(size_,blocks_,problem,partial_,result);
    error=cudaGetLastError();if(error!=cudaSuccess) return int(error);
    fft=cufftExecZ2Z(plan_,spectrum_,spectrum_,CUFFT_FORWARD);
    if(fft!=CUFFT_SUCCESS) return -int(fft);
    unpack<<<blocks_,threads,0,stream_>>>(size_,spectrum_,potential,result);
    return int(cudaGetLastError());
}
}
