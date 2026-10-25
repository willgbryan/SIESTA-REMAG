// SPDX-License-Identifier: GPL-3.0-or-later
#include "frozen_potential.hpp"
#include "frozen_potential_math.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace remag {
namespace {
void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void check(cusolverStatus_t e) {
    if(e!=CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("Frozen-potential cuSOLVER status "+std::to_string(e));
}
void check(cublasStatus_t e) {
    if(e!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Frozen-potential cuBLAS status "+std::to_string(e));
}
__global__ void initialize(FrozenSpectrumStatus* result) {*result=FrozenSpectrumStatus{};}

// Padded sparse entries are not necessarily initialized. Visit only active
// topology indices (each appears in exactly one grouped assembly pair).
__global__ void rotate(const HamiltonianAssemblyConfig* config, const HamiltonianPair* pairs,
        const int* indices, mae::SpinRotation rotation, const double* reference,
        const cuDoubleComplex* offsite, double* output, FrozenSpectrumStatus* result) {
    const auto p=*config;
    for(int pair=blockIdx.x*blockDim.x+threadIdx.x;pair<p.pairs;pair+=gridDim.x*blockDim.x) {
        for(int j=pairs[pair].begin;j<pairs[pair].end;j++) {
            const size_t index=indices[j], stride=p.ldh;
            double h[8], off[8], soc[8], rotated[8];
            for(int c=0;c<8;c++) h[c]=reference[index+stride*c];
            for(int c=0;c<4;c++) {
                const auto z=offsite[index+stride*c];off[2*c]=z.x;off[2*c+1]=z.y;
            }
            mae::offsite_soc(off,soc);
            if(!mae::rotate_exchange(rotation,h,soc,rotated)) {
                atomicExch(&result->status,1);
                // Keep the subsequent library solve well-defined; failed data
                // remain explicitly unusable through the sticky status.
                for(int c=0;c<8;c++) rotated[c]=0.;
            }
            for(int c=0;c<8;c++) output[index+stride*c]=rotated[c];
        }
    }
}

__global__ void inspect_spectrum(int n,int k,const int* info,const double* values,
                                 FrozenSpectrumStatus* result) {
    if(result->status) return;
    if(*info) {*result={2,k,*info,0};return;}
    for(int i=0;i<n;i++) if(!isfinite(values[i])||(i&&values[i]<values[i-1])) {
        *result={3,k,0,0};return;
    }
}

// cuSOLVER's generalized type-1 eigenvectors are S-orthonormal. Form S*C with
// the ORIGINAL (pre-factorization) overlap, then evaluate Pauli expectations.
// One thread owns a band. No cross-band or cross-k atomic reductions.
__global__ void spin_per_band(int n,int k,const cuDoubleComplex* c,const cuDoubleComplex* sc,
        double* output,FrozenSpectrumStatus* result) {
    if(result->status) return;
    for(int band=blockIdx.x*blockDim.x+threadIdx.x;band<n;band+=gridDim.x*blockDim.x) {
        mae::SpinBandSum sum;
        for(int i=0;i<n;i+=2) {
            const auto up=c[i+size_t(n)*band], down=c[i+1+size_t(n)*band];
            const auto sup=sc[i+size_t(n)*band], sdown=sc[i+1+size_t(n)*band];
            const double value[4]={up.x,up.y,down.x,down.y},
                         svalue[4]={sup.x,sup.y,sdown.x,sdown.y};
            sum.add(value,svalue);
        }
        double spin[3];
        if(!sum.finish(spin)) {
            if(atomicCAS(&result->status,0,3)==0) result->k=k;
            continue;
        }
        for(int component=0;component<3;component++) output[3*size_t(band)+component]=spin[component];
    }
}
}

struct FrozenPotentialPlan::Impl {
    HamiltonianAssemblyView view;
    const double* reference;
    const cuDoubleComplex* soc;
    cudaStream_t stream;
    cusolverDnHandle_t solver=nullptr;
    cublasHandle_t blas=nullptr;
    std::vector<void*> allocations;
    size_t bytes=0,limit=0;
    int n=0,lwork=0,jobs=0;
    bool moments=false;
    double *rotated=nullptr,*values=nullptr,*spin=nullptr;
    cuDoubleComplex *h=nullptr,*s=nullptr,*original_s=nullptr,*sc=nullptr,*work=nullptr;
    int* info=nullptr;
    FrozenSpectrumStatus* result=nullptr;

    template<class T> T* allocate(size_t count) {
        if(!count||count>SIZE_MAX/sizeof(T)||bytes>limit||count*sizeof(T)>limit-bytes)
            throw std::runtime_error("Frozen-potential explicit workspace limit exceeded");
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));
        // Keep ownership even if the host vector allocation throws.
        try {allocations.push_back(p);} catch(...) {cudaFree(p);throw;}
        bytes+=count*sizeof(T);return p;
    }
    Impl(const HamiltonianAssemblyView& v,const double* ref,const cuDoubleComplex* off,
         bool with_moments,size_t maximum,cudaStream_t queue):
        view(v),reference(ref),soc(off),stream(queue),limit(maximum),moments(with_moments) {
        const auto& p=v.shape;
        if((p.kind!=2&&p.kind!=3)||p.nh!=8||p.n<1||p.n>INT_MAX/2||p.nnz<1||
           p.ldh<p.nnz||p.nk<1||(p.kind==2&&p.nk!=1)||p.pairs<1||p.pairs>INT_MAX-127||
           !v.config||!v.pairs||!v.indices||!v.overlap||
           (p.kind==3&&(!v.kpoints||!v.xij))||!ref||!off||!maximum)
            throw std::runtime_error("Frozen-potential evaluator requires a full SOC assembly view");
        n=2*p.n;
        if(size_t(n)>SIZE_MAX/size_t(n)||size_t(n)>SIZE_MAX/size_t(p.nk))
            throw std::runtime_error("Frozen-potential dimensions overflow");
        const size_t matrix=size_t(n)*n,bank=size_t(n)*p.nk;
        try {
            check(cusolverDnCreate(&solver));check(cusolverDnSetStream(solver,stream));
            h=allocate<cuDoubleComplex>(matrix);s=allocate<cuDoubleComplex>(matrix);
            rotated=allocate<double>(size_t(p.ldh)*8);values=allocate<double>(bank);
            info=allocate<int>(1);result=allocate<FrozenSpectrumStatus>(1);
            if(moments) {
                check(cublasCreate(&blas));check(cublasSetStream(blas,stream));
                original_s=allocate<cuDoubleComplex>(matrix);sc=allocate<cuDoubleComplex>(matrix);
                if(bank>SIZE_MAX/3) throw std::runtime_error("Frozen spin bank dimensions overflow");
                spin=allocate<double>(bank*3);
            }
            const auto mode=moments?CUSOLVER_EIG_MODE_VECTOR:CUSOLVER_EIG_MODE_NOVECTOR;
            check(cusolverDnZhegvd_bufferSize(solver,CUSOLVER_EIG_TYPE_1,mode,CUBLAS_FILL_MODE_LOWER,
                n,h,n,s,n,values,&lwork));
            if(lwork<1) throw std::runtime_error("Invalid frozen-potential solver workspace");
            work=allocate<cuDoubleComplex>(size_t(lwork));
        } catch(...) {release();throw;}
    }
    void release() noexcept {
        // Asynchronous consumers may still reference these buffers on errors
        // or early destruction. Caller retains the borrowed stream until here.
        cudaStreamSynchronize(stream);
        if(blas) {cublasDestroy(blas);blas=nullptr;}
        if(solver) {cusolverDnDestroy(solver);solver=nullptr;}
        for(void* p:allocations) cudaFree(p);
        allocations.clear();
    }
    ~Impl() {release();}
    void solve(const double quaternion[4]) {
        mae::SpinRotation rotation;
        if(!quaternion||!mae::spin_rotation(quaternion,rotation))
            throw std::runtime_error("Invalid frozen-potential spin rotation");
        const auto& p=view.shape;jobs=0;
        initialize<<<1,1,0,stream>>>(result);check(cudaGetLastError());
        rotate<<<std::min(1024,(p.pairs+127)/128),128,0,stream>>>(view.config,view.pairs,
            view.indices,rotation,reference,soc,rotated,result);check(cudaGetLastError());
        const auto mode=moments?CUSOLVER_EIG_MODE_VECTOR:CUSOLVER_EIG_MODE_NOVECTOR;
        const cuDoubleComplex one=make_cuDoubleComplex(1.,0.),zero=make_cuDoubleComplex(0.,0.);
        for(int k=0;k<p.nk;k++) {
            check(assemble_hamiltonian_device(view,rotated,0,k,h,s,stream));
            if(moments) check(cudaMemcpyAsync(original_s,s,size_t(n)*n*sizeof(*s),cudaMemcpyDeviceToDevice,stream));
            check(cusolverDnZhegvd(solver,CUSOLVER_EIG_TYPE_1,mode,CUBLAS_FILL_MODE_LOWER,
                n,h,n,s,n,values+size_t(n)*k,work,lwork,info));
            ++jobs;
            inspect_spectrum<<<1,1,0,stream>>>(n,k,info,values+size_t(n)*k,result);check(cudaGetLastError());
            if(moments) {
                check(cublasZgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,original_s,n,h,n,&zero,sc,n));
                spin_per_band<<<std::min(1024,(n+127)/128),128,0,stream>>>(n,k,h,sc,spin+3*size_t(n)*k,result);
                check(cudaGetLastError());
            }
        }
    }
};

FrozenPotentialPlan::FrozenPotentialPlan(const HamiltonianAssemblyView& v,const double* h,
        const cuDoubleComplex* soc,bool moments,size_t maximum,cudaStream_t stream):
    impl_(std::make_unique<Impl>(v,h,soc,moments,maximum,stream)) {}
FrozenPotentialPlan::~FrozenPotentialPlan()=default;
void FrozenPotentialPlan::solve(const double q[4]) {impl_->solve(q);}
const FrozenSpectrumStatus* FrozenPotentialPlan::status() const {return impl_->result;}
const double* FrozenPotentialPlan::energies() const {return impl_->values;}
const double* FrozenPotentialPlan::spin_expectations() const {return impl_->spin;}
int FrozenPotentialPlan::bands() const {return impl_->n;}
int FrozenPotentialPlan::kpoints() const {return impl_->view.shape.nk;}
int FrozenPotentialPlan::eigensystem_jobs() const {return impl_->jobs;}
size_t FrozenPotentialPlan::workspace_bytes() const {return impl_->bytes;}
}
