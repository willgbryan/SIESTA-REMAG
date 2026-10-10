// SPDX-License-Identifier: GPL-3.0-or-later
// Pinned kinefsm.F and overfsm.f90 sparse diagonal-spin contraction.
#include "pair_force.hpp"
#include <climits>
#include <cmath>
namespace remag {
namespace {
__device__ void fail(int* status,int code){atomicMax(status,code);}
__device__ void add(double* output,double x,int* status){const double old=atomicAdd(output,x);if(!isfinite(old+x))fail(status,2);}
__global__ void contract(PairForceView v,const PairForceProblem* problem,const double* density,
        double* energy,double* force,double* stress,int* status) {
    const auto p=*problem;
    if((p.kind!=1&&p.kind!=4)||!isfinite(p.volume)||p.volume<=0.){fail(status,1);return;}
    for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<v.entries;q+=blockDim.x*gridDim.x) {
        const int qr=v.query_rows[q],col=v.query_columns[q],a=v.atom_pairs[2*size_t(q)],b=v.atom_pairs[2*size_t(q)+1];
        if(qr<1||qr>v.rows||col<1||col>v.columns||a<1||a>v.atoms||b<1||b>v.atoms){fail(status,1);continue;}
        const int row=qr-1,ia=a-1,ja=b-1;
        const int offset=v.ptr[row],count=v.num[row];
        if(offset<0||offset>v.capacity||count<0||count>v.capacity-offset){fail(status,1);continue;}
        double weight=0.;bool valid=true;
        for(int j=0;j<count;j++) {
            const int index=offset+j,column=v.list[index];
            if(column<1||column>v.columns){fail(status,1);valid=false;break;}
            if(column==col)for(int spin=0;spin<v.spinor;spin++)weight=weight+density[size_t(spin)*v.capacity+index];
        }
        if(!valid)continue;
        double df[3],ds[9],de=0.;const double* integral=v.integrals+4*size_t(q);const double* xyz=v.xyz+3*size_t(q);
        valid=isfinite(weight);
        if(p.kind==1){de=weight*integral[0];valid=valid&&isfinite(de);}
        for(int i=0;i<3;i++) {
            df[i]=(p.kind==1?weight:-weight)*integral[i+1];valid=valid&&isfinite(df[i]);
            for(int j=0;j<3;j++){ds[3*i+j]=xyz[j]*df[i]/p.volume;valid=valid&&isfinite(ds[3*i+j]);}
        }
        if(!valid){fail(status,2);continue;}
        if(p.kind==1)add(energy,de,status);
        for(int i=0;i<3;i++){add(force+3*ia+i,df[i],status);add(force+3*ja+i,-df[i],status);}
        for(int i=0;i<9;i++)add(stress+i,ds[i],status);
    }
}
}
cudaError_t add_pair_force_device(const PairForceView& v,const PairForceProblem* p,const double* density,
        double* energy,double* force,double* stress,int* status,cudaStream_t stream) {
    if(v.entries<0||v.entries>INT_MAX/4||v.atoms<1||v.rows<1||v.columns<1||v.capacity<1||(v.spinor!=1&&v.spinor!=2)||
       !v.query_rows||!v.query_columns||!v.atom_pairs||!v.num||!v.ptr||!v.list||!v.integrals||!v.xyz||!p||!density||!energy||!force||!stress||!status||!stream)
        return cudaErrorInvalidValue;
    const int blocks=v.entries?(v.entries+127)/128:1;
    contract<<<blocks>1024?1024:blocks,128,0,stream>>>(v,p,density,energy,force,stress,status);return cudaGetLastError();
}
}
