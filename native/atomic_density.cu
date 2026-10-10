// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_density.hpp"
#include <climits>
#include <cmath>
namespace remag {
namespace {
__device__ double point_density(const AtomicDensityView& p,int seq,int* status) {
    const int* m=p.mesh;const int s=p.nsm;
    const int x=seq%m[0],y=(seq/m[0])%m[1],z=seq/(m[0]*m[1]);
    const int c=x%s+s*(y%s+s*(z%s))+s*s*s*(x/s+(m[0]/s)*(y/s+(m[1]/s)*(z/s)));
    const int point=c/p.subpoints,sub=c%p.subpoints;
    const int begin=p.ends[point],end=p.ends[point+1];
    if(p.ends[0]!=0||p.ends[p.coarse_points]!=p.entries||begin<0||end<begin||end>p.entries) {
        atomicMax(status,1);return 0.;
    }
    double value=0.;
    for(int entry=begin;entry<end;entry++) {
        const int orbital=p.entry_orbital[entry];
        if(orbital<1||orbital>p.orbitals) {atomicMax(status,1);return 0.;}
        const int mapped=p.unit_orbital[orbital-1];
        if(mapped<1||mapped>p.unit_orbitals) {atomicMax(status,1);return 0.;}
        const int unit=mapped-1;
        const double occupancy=p.occupation[unit],phi=p.phi[size_t(entry)*p.subpoints+sub];
        if(!isfinite(occupancy)||!isfinite(phi)) {atomicMax(status,2);return 0.;}
        // Upstream rhooda: rhoatm = rhoatm + Datm(iu) * Ci * Ci.
        value=value+(occupancy*phi)*phi;
    }
    if(!isfinite(value)) {atomicMax(status,2);return 0.;}
    return value;
}
__global__ void density(int points,AtomicDensityView p,double* output,int* status) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(points);i+=size_t(blockDim.x)*gridDim.x)
        output[i]=point_density(p,int(i),status);
}
}
cudaError_t initial_atomic_density_device(const AtomicDensityView& p,double* output,int* status,cudaStream_t stream) {
    if(p.nsm<1||p.nsm>16||p.subpoints!=p.nsm*p.nsm*p.nsm||p.coarse_points<1||p.entries<0||
       p.orbitals<1||p.unit_orbitals<1||p.unit_orbitals>p.orbitals||!p.ends||!p.entry_orbital||!p.unit_orbital||
       !p.occupation||!p.phi||!output||!status||!stream)return cudaErrorInvalidValue;
    for(int i=0;i<3;i++)if(p.mesh[i]<1||p.mesh[i]%p.nsm)return cudaErrorInvalidValue;
    if(size_t(p.mesh[0])*p.mesh[1]>size_t(INT_MAX)/p.mesh[2])return cudaErrorInvalidValue;
    const int points=p.mesh[0]*p.mesh[1]*p.mesh[2];
    if(points/p.subpoints!=p.coarse_points||size_t(p.entries)*p.subpoints>size_t(INT_MAX))return cudaErrorInvalidValue;
    const int blocks=(points-1)/256+1;
    density<<<blocks>1024?1024:blocks,256,0,stream>>>(points,p,output,status);return cudaGetLastError();
}
}
