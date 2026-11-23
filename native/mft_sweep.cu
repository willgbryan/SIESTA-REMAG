// SPDX-License-Identifier: GPL-3.0-or-later
// Standalone frozen-state screening, sharing REMAG's FD and paired reductions.
#include "magnetic_response.h"
#include "frozen_potential_math.hpp"
#include "occupations.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Z=cuDoubleComplex;
thread_local std::string error;
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
void check(cublasStatus_t code) {if(code!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Magnetic cuBLAS failure "+std::to_string(code));}
void check(cusolverStatus_t code) {if(code!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("Magnetic cuSOLVER failure "+std::to_string(code));}
void require(bool yes,const char* why) {if(!yes) throw std::runtime_error(why);}
__device__ Z add(Z a,Z b) {return cuCadd(a,b);}
__device__ Z mul(Z a,Z b) {return cuCmul(a,b);}
__device__ Z scale(Z a,double b) {return make_cuDoubleComplex(a.x*b,a.y*b);}
__device__ double real_inner(Z a,Z b) {return a.x*b.x+a.y*b.y;}

struct Rotation {Z u[4];}; // row-major SU(2)
__global__ void rotate(int n,const Z* base,const Z* mag,const Z* soc,Rotation r,
                       double soc_scale,Z* rotated_mag,Z* h) {
    const size_t count=size_t(n)*n;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        const int row=i%n,col=i/n,a=row%2,b=col%2;
        Z value=make_cuDoubleComplex(0.,0.);
        for(int s=0;s<2;s++) for(int t=0;t<2;t++)
            value=add(value,mul(mul(r.u[2*a+s],mag[row-a+s+size_t(n)*(col-b+t)]),cuConj(r.u[2*b+t])));
        rotated_mag[i]=value;h[i]=add(add(base[i],value),scale(soc[i],soc_scale));
    }
}
__global__ void hermitize(int n,Z* a) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(n)*n;i+=gridDim.x*blockDim.x) {
        const int row=i%n,col=i/n;
        if(row<col) {Z v=scale(add(a[i],cuConj(a[col+size_t(n)*row])),.5);a[i]=v;a[col+size_t(n)*row]=cuConj(v);}
        else if(row==col) a[i].y=0.;
    }
}
// Derivative w.r.t. active spin rotation theta_a: dH/dtheta=-i[sigma_a/2,Hmag].
// Physical torque is minus the occupied expectation of this derivative.
__global__ void derivative(int n,int axis,const Z* mag,Z* output) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(n)*n;i+=gridDim.x*blockDim.x) {
        const int row=i%n,col=i/n,a=row%2,b=col%2;
        Z left,right;
        if(axis==2) {left=scale(mag[i],a?-1.:1.);right=scale(mag[i],b?-1.:1.);}
        else {
            left=mag[row+(a?-1:1)+size_t(n)*col];
            right=mag[row+size_t(n)*(col+(b?-1:1))];
            if(axis==1) {
                left=mul(make_cuDoubleComplex(0.,a?1.:-1.),left);
                right=mul(right,make_cuDoubleComplex(0.,b?-1.:1.));
            }
        }
        output[i]=mul(make_cuDoubleComplex(0.,-.5),cuCsub(left,right));
    }
}
__global__ void expectation(int n,int axis,const Z* c,const Z* dc,double* torque) {
    for(int b=blockIdx.x*blockDim.x+threadIdx.x;b<n;b+=gridDim.x*blockDim.x) {
        remag::mae::CompensatedSum sum;
        for(int i=0;i<n;i++) sum.add(real_inner(c[i+size_t(n)*b],dc[i+size_t(n)*b]));
        torque[3*b+axis]=-sum.value();
    }
}
// Retain orbital Mulliken weights and spin per band, not the full eigenbank.
__global__ void inspect(int n,const Z* h,const Z* s,const Z* c,const Z* hc,const Z* sc,
    const Z* gram,const double* e,double* spin,double* population,double* diagnostics) {
    for(int b=blockIdx.x*blockDim.x+threadIdx.x;b<n;b+=gridDim.x*blockDim.x) {
        double rr=0.,cn=0.,hn=0.,sn=0.,ortho=0.;
        remag::mae::SpinBandSum moment;
        for(int i=0;i<n;i++) {
            const size_t p=i+size_t(n)*b;
            Z residual=cuCsub(hc[p],scale(sc[p],e[b]));
            rr+=real_inner(residual,residual);cn+=real_inner(c[p],c[p]);
            Z g=gram[p];if(i==b) g.x-=1.;ortho=fmax(ortho,cuCabs(g));
            if(i%2==0) {
                const Z u=c[p],d=c[p+1],su=sc[p],sd=sc[p+1];
                const double cv[]={u.x,u.y,d.x,d.y},sv[]={su.x,su.y,sd.x,sd.y};
                moment.add(cv,sv);
                population[(size_t(b)*n+i)/2]=real_inner(u,su)+real_inner(d,sd);
            }
        }
        // Norms are shared by all bands, but small representative states make
        // this simple reduction inexpensive relative to the eigensolve.
        for(size_t i=0;i<size_t(n)*n;i++) {hn+=real_inner(h[i],h[i]);sn+=real_inner(s[i],s[i]);}
        double value[3];bool ok=moment.finish(value);
        for(int a=0;a<3;a++) spin[3*b+a]=ok?value[a]:NAN;
        diagnostics[2*b]=sqrt(rr)/(fmax(1.e-300,(sqrt(hn)+fabs(e[b])*sqrt(sn))*sqrt(cn)));
        diagnostics[2*b+1]=ortho;
    }
}
__global__ void reduce_result(int n,int nk,double temperature,double electrons,
    const double* w,const double* e,const double* q,const double* ref_e,const double* ref_q,
    const remag::FermiResult* fd,const remag::FermiResult* ref_fd,const double* spin,
    const double* torque,const double* diagnostics,RemagMftResult* out) {
    if(blockIdx.x||threadIdx.x) return;
    RemagMftResult r{};
    r.chemical_potential=fd->fermi;r.charge=fd->charge;
    remag::mae::CompensatedSum energy,entropy,moment[3],t[3];
    remag::mae::BandPairSum pair(ref_fd->fermi);
    for(int i=0;i<n*nk;i++) {
        energy.add(q[i]*e[i]);entropy.add(w[i/n]*remag::mae::fd_entropy(q[i]/w[i/n]));
        pair.add(ref_e[i],e[i],ref_q[i],q[i],w[i/n]);
        for(int a=0;a<3;a++) {moment[a].add(q[i]*spin[3*i+a]);t[a].add(q[i]*torque[3*i+a]);}
        if(!isfinite(diagnostics[2*i])||!isfinite(diagnostics[2*i+1])) {r.max_residual=NAN;break;}
        r.max_residual=fmax(r.max_residual,diagnostics[2*i]);
        r.max_orthogonality_error=fmax(r.max_orthogonality_error,diagnostics[2*i+1]);
    }
    const auto d=pair.finish(temperature,ref_fd->fermi,fd->fermi,electrons);
    r.band_energy=energy.value();r.free_energy=r.band_energy-temperature*entropy.value();
    r.delta_band=d.band_energy;r.delta_free=d.band_free_energy;r.delta_canonical_free=d.canonical_free_energy;
    if(!d.electron_count_passed) r.delta_free=NAN;
    for(int a=0;a<3;a++) {r.spin[a]=moment[a].value();r.torque[a]=t[a].value();}
    *out=r;
}

struct Plan {
    int n,nk,lwork=0;size_t matrix,bank,bytes=0,limit;
    cudaStream_t stream=nullptr;cublasHandle_t blas=nullptr;cusolverDnHandle_t solver=nullptr;
    std::vector<void*> owned;
    Z *base,*mag,*soc,*s,*factor,*h,*rotmag,*c,*hc,*sc,*gram,*d,*dc,*work;
    double *weights,*energies,*occupation,*ref_e,*ref_q,*spin,*torque,*population,*diagnostics;
    remag::FermiProblem* problem;remag::FermiResult *fd,*ref_fd;
    RemagMftResult* result;int* info;double temperature,electrons;bool has_reference=false;
    template<class T> T* alloc(size_t count,const T* source=nullptr) {
        require(count&&count<=SIZE_MAX/sizeof(T)&&bytes<=limit&&count*sizeof(T)<=limit-bytes,"Magnetic GPU workspace exceeds memory limit");
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));
        try {owned.push_back(p);} catch(...) {cudaFree(p);throw;}
        bytes+=count*sizeof(T);
        if(source) check(cudaMemcpyAsync(p,source,count*sizeof(T),cudaMemcpyHostToDevice,stream));
        return p;
    }
    void solver_ok() {int code;check(cudaMemcpyAsync(&code,info,sizeof(int),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));require(code==0,"Magnetic factorization/eigensolve failed");}
    Plan(int dimension,int points,const Z* hb,const Z* hm,const Z* hs,const Z* overlap,
         const double* w,double temp,double count,size_t maximum):n(dimension),nk(points),
         matrix(size_t(n)*n),bank(size_t(n)*nk),limit(maximum),temperature(temp),electrons(count) {
        require(n>=2&&n%2==0&&n<=4096&&nk>0&&bank<=INT_MAX&&hb&&hm&&hs&&overlap&&w,
            "Invalid magnetic dimensions or matrices");
        require(std::isfinite(temp)&&temp>=1.e-6&&std::isfinite(count)&&count>0.&&count<n,"Unsupported magnetic occupations");
        try {
            check(cudaStreamCreate(&stream));check(cublasCreate(&blas));check(cublasSetStream(blas,stream));
            check(cusolverDnCreate(&solver));check(cusolverDnSetStream(solver,stream));
            base=alloc<Z>(matrix*nk,hb);mag=alloc<Z>(matrix*nk,hm);soc=alloc<Z>(matrix*nk,hs);
            s=alloc<Z>(matrix*nk,overlap);factor=alloc<Z>(matrix*nk,overlap);
            h=alloc<Z>(matrix);rotmag=alloc<Z>(matrix);c=alloc<Z>(matrix);hc=alloc<Z>(matrix);
            sc=alloc<Z>(matrix);gram=alloc<Z>(matrix);d=alloc<Z>(matrix);dc=alloc<Z>(matrix);
            weights=alloc<double>(nk,w);energies=alloc<double>(bank);occupation=alloc<double>(bank);
            ref_e=alloc<double>(bank);ref_q=alloc<double>(bank);
            spin=alloc<double>(3*bank);torque=alloc<double>(3*bank);population=alloc<double>(bank*(n/2));
            diagnostics=alloc<double>(2*bank);info=alloc<int>(1);result=alloc<RemagMftResult>(1);
            const remag::FermiProblem p{2,1,nk,n,n,temp,count};problem=alloc<remag::FermiProblem>(1,&p);
            fd=alloc<remag::FermiResult>(1);ref_fd=alloc<remag::FermiResult>(1);
            int potrf_work=0;
            check(cusolverDnZpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,n,factor,n,&potrf_work));
            check(cusolverDnZheevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,c,n,energies,&lwork));
            work=alloc<Z>(std::max(lwork,potrf_work));
            for(int k=0;k<nk;k++) {
                check(cusolverDnZpotrf(solver,CUBLAS_FILL_MODE_LOWER,n,factor+matrix*k,n,work,potrf_work,info));solver_ok();
            }
        } catch(...) {release();throw;}
    }
    void release() noexcept {
        if(stream) cudaStreamSynchronize(stream);
        if(solver) {cusolverDnDestroy(solver);solver=nullptr;}
        if(blas) {cublasDestroy(blas);blas=nullptr;}
        for(void* p:owned) cudaFree(p);owned.clear();
        if(stream) {cudaStreamDestroy(stream);stream=nullptr;}
    }
    ~Plan() {release();}
    void gemm(const Z* a,const Z* b,Z* out,cublasOperation_t op=CUBLAS_OP_N) {
        const Z one=make_cuDoubleComplex(1.,0.),zero=make_cuDoubleComplex(0.,0.);
        check(cublasZgemm(blas,op,CUBLAS_OP_N,n,n,n,&one,a,n,b,n,&zero,out,n));
    }
    void solve(const double* quaternion,double soc_scale,bool reference,RemagMftResult* output,
               double* values,double* occ,double* populations) {
        require(quaternion&&output&&values&&occ&&populations,"Missing MFT output buffer");
        require(reference||has_reference,"First MFT solve must establish the energy reference");
        require(soc_scale==0.||soc_scale==1.,"SOC scale must be 0 or 1");
        remag::mae::SpinRotation validated;require(remag::mae::spin_rotation(quaternion,validated),"Invalid spin rotation");
        double scale_q=0.,norm=0.,q[4];for(int i=0;i<4;i++) scale_q=std::max(scale_q,std::abs(quaternion[i]));
        for(int i=0;i<4;i++) {q[i]=quaternion[i]/scale_q;norm+=q[i]*q[i];}
        for(double& x:q) x/=std::sqrt(norm);
        Rotation r{{make_cuDoubleComplex(q[0],-q[3]),make_cuDoubleComplex(-q[2],-q[1]),
                    make_cuDoubleComplex(q[2],-q[1]),make_cuDoubleComplex(q[0],q[3])}};
        auto start=std::chrono::steady_clock::now();const Z one=make_cuDoubleComplex(1.,0.);
        for(int k=0;k<nk;k++) {
            const Z* l=factor+matrix*k;
            rotate<<<std::min(size_t(1024),(matrix+255)/256),256,0,stream>>>(n,base+matrix*k,mag+matrix*k,soc+matrix*k,r,soc_scale,rotmag,h);
            check(cudaGetLastError());check(cudaMemcpyAsync(c,h,matrix*sizeof(Z),cudaMemcpyDeviceToDevice,stream));
            // A=L^-1 H L^-H; factors remain immutable across every direction.
            check(cublasZtrsm(blas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,n,n,&one,l,n,c,n));
            check(cublasZtrsm(blas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_C,CUBLAS_DIAG_NON_UNIT,n,n,&one,l,n,c,n));
            hermitize<<<std::min(size_t(1024),(matrix+255)/256),256,0,stream>>>(n,c);check(cudaGetLastError());
            check(cusolverDnZheevd(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,c,n,energies+size_t(n)*k,work,lwork,info));solver_ok();
            check(cublasZtrsm(blas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_C,CUBLAS_DIAG_NON_UNIT,n,n,&one,l,n,c,n));
            gemm(h,c,hc);gemm(s+matrix*k,c,sc);gemm(c,sc,gram,CUBLAS_OP_C);
            inspect<<<(n+127)/128,128,0,stream>>>(n,h,s+matrix*k,c,hc,sc,gram,energies+size_t(n)*k,
                spin+3*size_t(n)*k,population+size_t(n)*(n/2)*k,diagnostics+2*size_t(n)*k);check(cudaGetLastError());
            for(int a=0;a<3;a++) {
                derivative<<<std::min(size_t(1024),(matrix+255)/256),256,0,stream>>>(n,a,rotmag,d);check(cudaGetLastError());
                gemm(d,c,dc);expectation<<<(n+127)/128,128,0,stream>>>(n,a,c,dc,torque+3*size_t(n)*k);check(cudaGetLastError());
            }
        }
        check(remag::fermi_device(problem,energies,weights,occupation,fd,stream));
        remag::FermiResult f;check(cudaMemcpyAsync(&f,fd,sizeof(f),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
        require(!f.status&&std::isfinite(f.charge)&&std::abs(f.charge-electrons)<=1.e-10,"Magnetic occupation solver failed");
        if(reference) {
            check(cudaMemcpyAsync(ref_e,energies,bank*sizeof(double),cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(ref_q,occupation,bank*sizeof(double),cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(ref_fd,fd,sizeof(f),cudaMemcpyDeviceToDevice,stream));has_reference=true;
        }
        reduce_result<<<1,1,0,stream>>>(n,nk,temperature,electrons,weights,energies,occupation,
            ref_e,ref_q,fd,ref_fd,spin,torque,diagnostics,result);check(cudaGetLastError());
        check(cudaMemcpyAsync(output,result,sizeof(*output),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(values,energies,bank*sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(occ,occupation,bank*sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(populations,population,bank*(n/2)*sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        output->seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();output->workspace_bytes=bytes;
        require(std::isfinite(output->delta_free)&&std::isfinite(output->band_energy)&&
            std::isfinite(output->max_residual)&&output->max_residual<=1.e-8&&output->max_orthogonality_error<=1.e-8,
            "Magnetic spectrum/residual/orthogonality checks failed");
        for(int a=0;a<3;a++) require(std::isfinite(output->spin[a])&&std::isfinite(output->torque[a]),"Invalid magnetic spin/torque");
        for(size_t i=0;i<bank;i++) require(std::isfinite(values[i]),"Nonfinite magnetic spectrum");
    }
};
}
extern "C" const char* remag_magnetic_error() {return error.c_str();}
extern "C" int remag_mft_create(int n,int nk,const void* hb,const void* hm,const void* hs,const void* s,
    const double* w,double temperature,double electrons,size_t limit,void** handle) {
    if(!handle) return 1;*handle=nullptr;
    try {*handle=new Plan(n,nk,static_cast<const Z*>(hb),static_cast<const Z*>(hm),static_cast<const Z*>(hs),
        static_cast<const Z*>(s),w,temperature,electrons,limit);error.clear();return 0;}
    catch(const std::exception& e) {error=e.what();return 1;}
}
extern "C" int remag_mft_sweep(void* handle,const double* q,double scale,int reference,
    RemagMftResult* result,double* e,double* occ,double* population) {
    try {require(handle,"Missing magnetic plan");static_cast<Plan*>(handle)->solve(q,scale,reference!=0,result,e,occ,population);error.clear();return 0;}
    catch(const std::exception& e) {error=e.what();return 1;}
}
extern "C" void remag_mft_destroy(void* handle) {delete static_cast<Plan*>(handle);}
