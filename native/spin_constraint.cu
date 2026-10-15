// SPDX-License-Identifier: GPL-3.0-or-later
#include "spin_constraint.hpp"
#include "spin_constraint_math.hpp"

namespace {
__global__ void field(int pairs,int spins,size_t n,const remag::HamiltonianPair* pair,const int* indices,
        const double* overlap,const double* orbital_field,double* h) {
    for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<pairs;p+=gridDim.x*blockDim.x) {
        const auto q=pair[p];const double* a=orbital_field+3*size_t(q.col);const double* b=orbital_field+3*size_t(q.row);
        for(int j=q.begin;j<q.end;j++) {const int i=indices[j];remag::spin_constraint_add(spins,n,i,overlap[i],a,b,h);}
    }
}
__global__ void moments(int rows,int spins,size_t n,const int* num,const int* ptr,const double* overlap,
        const double* density,double* out) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=gridDim.x*blockDim.x) {
        double m[3]={0.,0.,0.};
        for(int j=0;j<num[row];j++) {const size_t i=size_t(ptr[row])+j;remag::spin_constraint_moment(spins,n,i,overlap[i],density,m);}
        out[3*size_t(row)]=m[0];out[3*size_t(row)+1]=m[1];out[3*size_t(row)+2]=m[2];
    }
}
// One block: every orbital reduces its own site's moment in orbital order,
// the same sum the host performs, then writes b_s + rho M_perp,s.
__global__ void penalized(int rows,remag::SpinConstraintDevice c) {
    for(int r=threadIdx.x;r<rows;r+=blockDim.x) {
        const int s=c.orbital_site[r];double* f=c.orbital_field+3*size_t(r);
        if(s<0) {f[0]=f[1]=f[2]=0.;continue;}
        double m[3];remag::spin_constraint_site_moment(s,rows,c.orbital_site,c.orbital_moments,m);
        remag::spin_constraint_penalized_field(c.site_fields+3*size_t(s),c.targets+3*size_t(s),c.penalty,m,f);
    }
}
int blocks(int work) {return work<1?1:(work+127)/128<1024?(work+127)/128:1024;}
}
namespace remag {
cudaError_t spin_constraint_field_device(const HamiltonianAssemblyView& view,int spins,
        const double* orbital_field,double* h,cudaStream_t stream) {
    const auto& c=view.shape;
    if(c.kind<2||(spins!=4&&spins!=8)||c.nh!=spins||c.pairs<1||c.ldh<1||!view.pairs||!view.indices||
       !view.overlap||!orbital_field||!h||!stream) return cudaErrorInvalidValue;
    field<<<blocks(c.pairs),128,0,stream>>>(c.pairs,spins,size_t(c.ldh),view.pairs,view.indices,view.overlap,orbital_field,h);
    return cudaGetLastError();
}
cudaError_t spin_constraint_moments_device(int rows,int spins,size_t capacity,const int* num,const int* ptr,
        const double* overlap,const double* density,double* orbital_moments,cudaStream_t stream) {
    if(rows<1||(spins!=4&&spins!=8)||capacity<1||!num||!ptr||!overlap||!density||!orbital_moments||!stream)
        return cudaErrorInvalidValue;
    moments<<<blocks(rows),128,0,stream>>>(rows,spins,capacity,num,ptr,overlap,density,orbital_moments);
    return cudaGetLastError();
}
cudaError_t spin_constraint_penalty_field_device(int rows,int spins,size_t capacity,const int* num,const int* ptr,
        const double* overlap,const double* density,const SpinConstraintDevice& c,cudaStream_t stream) {
    if(c.sites<1||!(c.penalty>0.)||!c.orbital_site||!c.site_fields||!c.targets||!c.orbital_moments||!c.orbital_field)
        return cudaErrorInvalidValue;
    const auto status=spin_constraint_moments_device(rows,spins,capacity,num,ptr,overlap,density,c.orbital_moments,stream);
    if(status!=cudaSuccess) return status;
    penalized<<<1,256,0,stream>>>(rows,c);
    return cudaGetLastError();
}
}
