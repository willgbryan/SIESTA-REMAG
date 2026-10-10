// SPDX-License-Identifier: GPL-3.0-or-later
#include "final_grid.hpp"
#include <climits>
namespace remag {
namespace {
__global__ void density_order(int n1,int n2,int points,int nsm,int spins,const double* rho,const double* atom,double* out,double* deformation) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(points)*spins;i+=size_t(blockDim.x)*gridDim.x) {
        int p=i%points,x=p%n1,y=(p/n1)%n2,z=p/(n1*n2);
        int c=x%nsm+nsm*(y%nsm+nsm*(z%nsm))+nsm*nsm*nsm*(x/nsm+(n1/nsm)*(y/nsm+(n2/nsm)*(z/nsm)));
        const int spin=i/points,nsd=spins<2?1:2;
        const double value=rho[c+size_t(points)*spin];
        if(out) out[i]=value;
        if(deformation) deformation[i]=spin<nsd?value-atom[p]*(1./nsd):value;
    }
}
__global__ void seed(int n,int nh,int soc,const double* kinetic,const double* nonlocal_h,
        const double* onsite,double* h) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(n)*nh;i+=size_t(blockDim.x)*gridDim.x) {
        int spin=i/n,element=i%n;double value=0.;
        // Literal final_H_f_stress assembly: kinefsm has spin%spinor
        // diagonal channels. nlefsm_SO_off updates only the first H0
        // channel and its separate complex H0_off; final H_new does not
        // add that offsite matrix. Preserve this upstream convention.
        if(spin<2) value=kinetic[element];
        if(spin<(soc==2?1:2)) value+=nonlocal_h[element];
        if(soc==1&&spin>=2) value=onsite[element+size_t(n)*(spin-2)];
        h[i]=value;
    }
}
}
cudaError_t final_density_sequential_device(const int m[3],int nsm,int spins,
        const double* rho,const double* atom,double* out,double* deformation,cudaStream_t stream) {
    if(!m||!rho||(!out&&!deformation)||(deformation&&!atom)||rho==out||rho==deformation||(out&&out==deformation)||!stream||nsm<1||nsm>64||(spins!=1&&spins!=2&&spins!=4)) return cudaErrorInvalidValue;
    for(int i=0;i<3;i++) if(m[i]<1||m[i]%nsm) return cudaErrorInvalidValue;
    if(size_t(m[0])*m[1]>size_t(INT_MAX)/m[2]/spins) return cudaErrorInvalidValue;
    const int points=m[0]*m[1]*m[2];
    density_order<<<(size_t(points)*spins+255)/256,256,0,stream>>>(m[0],m[1],points,nsm,spins,rho,atom,out,deformation);
    return cudaGetLastError();
}
cudaError_t final_grid_hamiltonian_device(const GridContractionView& g,int nh,int soc,
        const double* kinetic,const double* nonlocal_h,const double* onsite,const double* potential,
        double dvol,double* h,cudaStream_t stream) {
    if(!stream||g.shape.nnz<1||(nh!=1&&nh!=2&&nh!=4&&nh!=8)||
        (nh==8?(soc!=1&&soc!=2):soc!=0)||!kinetic||!nonlocal_h||!potential||!h||(soc==1&&!onsite)) return cudaErrorInvalidValue;
    const size_t size=size_t(g.shape.nnz)*nh;
    seed<<<int((size+255)/256),256,0,stream>>>(g.shape.nnz,nh,soc,kinetic,nonlocal_h,onsite,h);
    auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    return add_grid_potential_device(g,nh<4?nh:4,nh,potential,dvol,nullptr,h,stream);
}
}
