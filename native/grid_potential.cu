// SPDX-License-Identifier: GPL-3.0-or-later
// Device composition of the periodic PBE path in pinned dhscf.F and reord.f.
#include "grid_potential.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace remag {
namespace {
constexpr int threads=256;
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
__device__ bool shape_ok(const GridPotentialProblem& p,int size,int spins) {
    const auto m=p.pbe.mesh;
    if(m[0]<1||m[1]<1||m[2]<1||p.pbe.spins!=spins||p.nsm<1||
       static_cast<long long>(m[0])*m[1]>INT_MAX/m[2]||m[0]*m[1]*m[2]!=size) return false;
    for(int i=0;i<3;i++) if(m[i]!=p.hartree.mesh[i]||m[i]%p.nsm!=0) return false;
    return true;
}
__device__ int clustered(int seq,const GridPotentialProblem& p) {
    const int* m=p.pbe.mesh;const int s=p.nsm;
    int x=seq%m[0],y=(seq/m[0])%m[1],z=seq/(m[0]*m[1]);
    return x%s+s*(y%s+s*(z%s))+s*s*s*(x/s+(m[0]/s)*(y/s+(m[1]/s)*(z/s)));
}
__global__ void initialize(const GridPotentialProblem* p,GridPotentialResult* out,int n1,int n2,int n3,int spins) {
    *out=GridPotentialResult{};
    if(p->pbe.mesh[0]!=n1||p->pbe.mesh[1]!=n2||p->pbe.mesh[2]!=n3||p->pbe.spins!=spins) {out->status=1;return;}
    if(!shape_ok(*p,n1*n2*n3,spins)||!isfinite(p->harris_energy)||
       !isfinite(p->dvol)||p->dvol<=0.||(p->partial_core!=0&&p->partial_core!=1)||
       p->pbe.volume!=p->hartree.volume) out->status=2;
}
__device__ double finite_value(double value,GridPotentialResult* out) {
    if(!isfinite(value)) {atomicMax(&out->status,2);return 0.;}return value;
}
__global__ void prepare(int size,int spins,const GridPotentialProblem* problem,const double* rho,
                        const double* atom,const double* core,double* charge,double* xc,GridPotentialResult* out) {
    const auto p=*problem;const bool valid=shape_ok(p,size,spins);
    const double fraction=spins==1?1.:.5;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        if(!valid) {charge[i]=0.;for(int s=0;s<spins;s++) xc[i+size*s]=0.;continue;}
        const int c=clustered(i,p);const double a=finite_value(atom[i],out);
        const double pc=p.partial_core?finite_value(core[i],out):0.;
        double d[4]{};for(int s=0;s<spins;s++) d[s]=finite_value(rho[c+size*s],out);
        // Preserve upstream subtraction, spin sum/difference, reconstruction
        // and reference/core addition in the same arithmetic sequence.
        d[0]-=a*fraction;if(spins>1) d[1]-=a*fraction;
        if(spins>1) {
            double total=d[0]+d[1],difference=d[1]-d[0];charge[i]=total;
            d[0]=.5*(total-difference);d[1]=.5*(total+difference);
        } else charge[i]=d[0];
        const double restore=p.partial_core?(pc+a)*fraction:a*fraction;
        d[0]+=restore;if(spins>1) d[1]+=restore;
        for(int s=0;s<spins;s++) {
            xc[i+size*s]=d[s];if(!isfinite(d[s])) atomicMax(&out->status,2);
        }
        if(!isfinite(charge[i])) atomicMax(&out->status,2);
    }
}
__global__ void combine(int size,int spins,const GridPotentialProblem* problem,const double* atom,const double* neutral,
                        const double* charge,const double* vh,const double* vx,double* potential,double* auxiliary,double* partial,
                        GridPotentialResult* out) {
    __shared__ double sums[3][threads];double value[3]{};
    const auto p=*problem;const bool valid=shape_ok(p,size,spins);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        if(!valid) {auxiliary[i]=0.;for(int s=0;s<spins;s++) potential[i+size*s]=0.;continue;}
        const int c=clustered(i,p); // a bijection, so output has a single writer
        const double a=finite_value(atom[i],out),v=finite_value(neutral[i],out);
        const double h=finite_value(vh[i],out),d=finite_value(charge[i],out);
        auxiliary[c]=h+v;
        if(!isfinite(auxiliary[c])) atomicMax(&out->status,2);
        value[0]+=h*a;value[1]+=v*a;value[2]+=v*d;
        for(int s=0;s<spins;s++) {
            const double x=finite_value(vx[i+size*s],out);
            potential[c+size*s]=s<(spins==1?1:2)?x+(h+v):x;
            if(!isfinite(potential[c+size*s])) atomicMax(&out->status,2);
        }
    }
    for(int f=0;f<3;f++) sums[f][threadIdx.x]=value[f];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int f=0;f<3;f++) sums[f][threadIdx.x]+=sums[f][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int f=0;f<3;f++) partial[3*blockIdx.x+f]=sums[f][0];
}
__global__ void finish(int blocks,const GridPotentialProblem* p,const double* partial,
                       const HartreeResult* h,const PbeResult* x,GridPotentialResult* out) {
    __shared__ double sums[3][threads];
    for(int f=0;f<3;f++) {double value=0.;for(int i=threadIdx.x;i<blocks;i+=threads) value+=partial[3*i+f];sums[f][threadIdx.x]=value;}
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int f=0;f<3;f++) sums[f][threadIdx.x]+=sums[f][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        if(!out->status&&(h->status||x->status)) out->status=2;
        for(int f=0;f<3;f++) if(!isfinite(sums[f][0])) out->status=2;
        if(out->status) return;
        out->energies[0]=sums[0][0]*p->dvol+p->harris_energy+h->energy;
        out->energies[1]=h->energy;out->energies[2]=p->harris_energy;
        out->energies[3]=sums[1][0]*p->dvol;
        out->energies[4]=out->energies[3]+sums[2][0]*p->dvol;
        for(int i=0;i<4;i++) out->energies[5+i]=x->energies[i];
        for(int i=0;i<9;i++) {
            out->hartree_stress[i]=h->stress[i];out->xc_stress[i]=x->stress[i];
            if(!isfinite(out->energies[i])) out->status=2;
        }
    }
}
}

GridPotentialPlan::GridPotentialPlan(int n1,int n2,int n3,int spins,cudaStream_t stream):
    mesh_{n1,n2,n3},spins_(spins),stream_(stream) {
    if((spins!=1&&spins!=2&&spins!=4)||n1<1||n2<1||n3<1||
       static_cast<long long>(n1)*n2>INT_MAX/n3||n1*n2*n3>INT_MAX/spins)
        throw std::runtime_error("Invalid grid-potential shape");
    size_=n1*n2*n3;blocks_=std::min(1024,(size_-1)/threads+1);
    try {
        hartree_=std::make_unique<HartreePlan>(n1,n2,n3,stream);
        pbe_=std::make_unique<PbePlan>(n1,n2,n3,spins,stream);
        auto alloc=[](auto** p,size_t bytes) {check(cudaMalloc(reinterpret_cast<void**>(p),bytes));};
        const size_t bytes=size_t(size_)*sizeof(double);
        alloc(&charge_,bytes);alloc(&hartree_potential_,bytes);alloc(&auxiliary_,bytes);
        alloc(&xc_density_,bytes*spins);alloc(&xc_potential_,bytes*spins);
        alloc(&partial_,size_t(blocks_)*3*sizeof(double));
        alloc(&hartree_result_,sizeof(HartreeResult));alloc(&pbe_result_,sizeof(PbeResult));
    } catch(...) {release();throw;}
}
void GridPotentialPlan::release() {
    void* buffers[]={charge_,hartree_potential_,xc_density_,xc_potential_,auxiliary_,partial_,hartree_result_,pbe_result_};
    for(void* p:buffers) if(p) cudaFree(p);
}
GridPotentialPlan::~GridPotentialPlan() {release();}
size_t GridPotentialPlan::workspace_bytes() const {
    return hartree_->resident_bytes()+pbe_->workspace_bytes()+
           size_t(size_)*(3+2*spins_)*sizeof(double)+size_t(blocks_)*3*sizeof(double)+sizeof(HartreeResult)+sizeof(PbeResult);
}
const double* GridPotentialPlan::auxiliary_potential() const {return auxiliary_;}
const double* GridPotentialPlan::deformation_density() const {return charge_;}
int GridPotentialPlan::apply(const GridPotentialProblem* problem,const double* rho,const double* atom,
                             const double* core,const double* neutral,double* potential,GridPotentialResult* result) {
    if(!problem||!rho||!atom||!core||!neutral||!potential||!result||rho==potential) return int(cudaErrorInvalidValue);
    initialize<<<1,1,0,stream_>>>(problem,result,mesh_[0],mesh_[1],mesh_[2],spins_);
    prepare<<<blocks_,threads,0,stream_>>>(size_,spins_,problem,rho,atom,core,charge_,xc_density_,result);
    int status=int(cudaGetLastError());if(status) return status;
    status=hartree_->apply(&problem->hartree,charge_,hartree_potential_,hartree_result_);if(status) return status;
    status=pbe_->apply(&problem->pbe,xc_density_,xc_potential_,pbe_result_);if(status) return status;
    combine<<<blocks_,threads,0,stream_>>>(size_,spins_,problem,atom,neutral,charge_,hartree_potential_,xc_potential_,potential,auxiliary_,partial_,result);
    finish<<<1,threads,0,stream_>>>(blocks_,problem,partial_,hartree_result_,pbe_result_,result);
    return int(cudaGetLastError());
}
}
