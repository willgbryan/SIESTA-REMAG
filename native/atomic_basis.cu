// SPDX-License-Identifier: GPL-3.0-or-later
// Pinned atmfuncs:chcore_sub/phiatm(io=0), radial:rad_get and interpolation.
#include "atomic_basis.hpp"
#include <cmath>
#include <climits>
namespace remag {
namespace {
__device__ void fail(int* status,int code){atomicMax(status,code);}
__device__ void evaluate(const AtomicBasisView& b,int kind,int mesh_filter,int species,
        const double* r,double* out,int& active,int* status) {
    active=0;for(int i=0;i<4;i++)out[i]=0.;
    if(species<0||species>=b.species){fail(status,1);return;}
    if(!mesh_filter&&b.atomic_number[species]<0){active=1;return;}
    if(!isfinite(r[0])||!isfinite(r[1])||!isfinite(r[2])){fail(status,2);return;}
    const double size=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if(!isfinite(size)){fail(status,2);return;}
    const int index=2*(2*species+kind-1);
    const double cutoff=b.parameters[index],delta=b.parameters[index+1];
    if(!isfinite(cutoff)||cutoff<0.){fail(status,1);return;}
    if(mesh_filter&&((kind==1&&cutoff<=1.e-12)||!(size<cutoff)))return;
    active=1;
    if(b.atomic_number[species]<0)return;
    const double radius=size+1.e-20;
    // The mesh test above uses the unshifted radius and strict '<'. These
    // inner tests come from different upstream routines and use strict '>'.
    if(radius>(kind==1?cutoff:cutoff-1.e-12))return;
    const int n=b.index[index],begin=b.index[index+1];
    if(n<0||n==1||begin<0||begin>b.samples||n>b.samples-begin){fail(status,1);return;}
    if(!n)return; // rad_get does not inspect delta or sample pointers for n=0.
    if(!isfinite(delta)||delta<=0.){fail(status,1);return;}
    const double xmax=(n-1)*delta,tol=delta*1.e-6,q=radius/delta;
    if(!isfinite(xmax)||!isfinite(q)||q>INT_MAX-1||radius< -tol||radius>xmax+tol){fail(status,2);return;}
    const int kl=min(n-2,max(0,int(floor(q)))),kh=kl+1;
    const double xl=kl*delta,xh=kh*delta,dx=xh-xl;
    const double yl=b.values[begin+kl],yh=b.values[begin+kh],d2l=b.second[begin+kl],d2h=b.second[begin+kh];
    if(!isfinite(yl)||!isfinite(yh)||!isfinite(d2l)||!isfinite(d2h)||dx<=0.){fail(status,2);return;}
    const double a=(xh-radius)/dx,bb=(radius-xl)/dx;
    out[0]=a*yl+bb*yh+((a*a*a-a)*d2l+(bb*bb*bb-bb)*d2h)*(dx*dx)/6.;
    const double da=-1./dx,db=1./dx;
    const double derivative=da*yl+db*yh+(da*(3*(a*a)-1.)*d2l+db*(3*(bb*bb)-1.)*d2h)*(dx*dx)/6.;
    for(int k=0;k<3;k++)out[k+1]=derivative*r[k]/radius;
    for(int k=0;k<4;k++)if(!isfinite(out[k]))fail(status,2);
}
__global__ void queries(AtomicBasisView b,int kind,int mesh_filter,int count,const int* species,
        const double* xyz,double* result,int* active,int* status) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(count);i+=size_t(blockDim.x)*gridDim.x)
        evaluate(b,kind,mesh_filter,species[i],xyz+3*i,result+4*i,active[i],status);
}
__global__ void geometry(AtomicBasisView b,AtomicGeometryView g,int kind,double* value,double* gradient,
        double* displacement,int* active,int* point,int* unit_atom,int* status) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(g.pairs)*g.nsp;i+=size_t(blockDim.x)*gridDim.x) {
        value[i]=0.;active[i]=point[i]=unit_atom[i]=0;
        for(int k=0;k<3;k++)gradient[3*i+k]=displacement[3*i+k]=0.;
        const int pair=i/g.nsp,sub=i%g.nsp,atom=g.pair_atom[pair]-1,offset=g.pair_offset[pair]-1,cluster=g.pair_cluster[pair]-1;
        if(atom<0||atom>=g.atoms||offset<0||offset>=g.offsets||cluster<0||cluster>=g.points/g.nsp){fail(status,1);continue;}
        const int ua=g.unit_atom[atom];if(ua<1||ua>g.unit_atoms){fail(status,1);continue;}
        double r[3],out[4];for(int k=0;k<3;k++)r[k]=(g.offset[3*offset+k]+g.subpoint[3*sub+k])-g.atom_offset[3*atom+k];
        evaluate(b,kind,1,g.atom_species[atom]-1,r,out,active[i],status);
        value[i]=out[0];for(int k=0;k<3;k++){gradient[3*i+k]=out[k+1];displacement[3*i+k]=r[k];}
        int ip=cluster*g.nsp+sub;
        if(kind==2) {
            const int n=g.nsm,nx=g.mesh[0]/n,ny=g.mesh[1]/n;
            const int x=(cluster%nx)*n+sub%n,y=((cluster/nx)%ny)*n+(sub/n)%n,z=(cluster/(nx*ny))*n+sub/(n*n);
            ip=x+g.mesh[0]*(y+g.mesh[1]*z);
        }
        point[i]=ip+1;unit_atom[i]=ua;
    }
}
bool valid(const AtomicBasisView& b){return b.species>0&&b.species<=10000&&b.samples>=0&&
    b.atomic_number&&b.index&&b.parameters&&b.values&&b.second;}
int blocks(size_t n){return int(n>128*1024?1024:(n+127)/128);}
}
cudaError_t evaluate_atomic_queries_device(const AtomicBasisView& b,int kind,int mesh_filter,int count,
        const int* species,const double* xyz,double* result,int* active,int* status,cudaStream_t stream) {
    if(!valid(b)||(kind!=1&&kind!=2)||(mesh_filter!=0&&mesh_filter!=1)||count<1||count>INT_MAX/4||
        !species||!xyz||!result||!active||!status||!stream)return cudaErrorInvalidValue;
    queries<<<blocks(count),128,0,stream>>>(b,kind,mesh_filter,count,species,xyz,result,active,status);return cudaGetLastError();
}
cudaError_t evaluate_atomic_geometry_device(const AtomicBasisView& b,const AtomicGeometryView& g,int kind,
        double* value,double* gradient,double* displacement,int* active,int* point,int* unit_atom,int* status,cudaStream_t stream) {
    if(!valid(b)||(kind!=1&&kind!=2)||g.pairs<1||g.nsp<1||g.nsp>4096||g.atoms<1||g.offsets<1||g.unit_atoms<1||g.points<1||
        g.nsm<1||g.nsm>16||g.nsp!=g.nsm*g.nsm*g.nsm||size_t(g.pairs)*g.nsp>INT_MAX/4||
        g.mesh[0]<1||g.mesh[1]<1||g.mesh[2]<1||g.mesh[0]%g.nsm||g.mesh[1]%g.nsm||g.mesh[2]%g.nsm||
        size_t(g.mesh[0])*g.mesh[1]>INT_MAX||size_t(g.mesh[0])*g.mesh[1]*g.mesh[2]!=size_t(g.points)||!g.pair_atom||!g.pair_offset||!g.pair_cluster||
        !g.atom_species||!g.unit_atom||!g.offset||!g.subpoint||!g.atom_offset||!value||!gradient||!displacement||
        !active||!point||!unit_atom||!status||!stream)return cudaErrorInvalidValue;
    geometry<<<blocks(size_t(g.pairs)*g.nsp),128,0,stream>>>(b,g,kind,value,gradient,displacement,active,point,unit_atom,status);
    return cudaGetLastError();
}
}
