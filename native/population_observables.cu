// SPDX-License-Identifier: GPL-3.0-or-later
#include "population_observables.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
namespace remag {
namespace {
// Literal arithmetic convention of pristine mulliken.F:spnvec. Collinear
// spin is signed; only four-component spin takes the vector norm.
__device__ bool spin_view(int components,const double* q,double* v) {
    v[0]=q[0];v[1]=v[2]=v[3]=v[4]=0.;
    if(components>=2){v[0]+=q[1];v[1]=v[4]=q[0]-q[1];}
    if(components==4){v[2]=2.*q[2];v[3]=2.*q[3];v[1]=sqrt(v[2]*v[2]+v[3]*v[3]+v[4]*v[4]);}
    bool finite=true;for(int i=0;i<5;i++)finite=finite&&isfinite(v[i]);return finite;
}
__global__ void per_item(int rows,int atoms,int species,int components,const PopulationObservableProblem* p,
        const int* ids,const double* z,const double* orbital,const double* atomic,
        double* out_orb,double* out_atom,int* status) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<max(rows,atoms);i+=gridDim.x*blockDim.x){
        if(i<rows)for(int k=0;k<5;k++)out_orb[5*i+k]=0.;
        if(i<atoms)for(int k=0;k<6;k++)out_atom[6*i+k]=0.;
        if(p->rows!=rows||p->atoms!=atoms||p->species!=species||p->components!=components){atomicMax(status,1);continue;}
        if(*status)continue;
        double v[5];
        if(i<rows){
            if(spin_view(components,orbital+size_t(i)*components,v))for(int k=0;k<5;k++)out_orb[5*i+k]=v[k];
            else atomicMax(status,2);
        }
        if(i<atoms){
            const int id=ids[i]-1;
            if(id<0||id>=species){atomicMax(status,1);continue;}
            if(!spin_view(components,atomic+size_t(i)*components,v)||!isfinite(z[id])||!isfinite(z[id]-v[0])){atomicMax(status,2);continue;}
            out_atom[6*i]=z[id]-v[0];for(int k=0;k<5;k++)out_atom[6*i+k+1]=v[k];
        }
    }
}
__global__ void totals(int atoms,int species,int components,const int* ids,
        const double* atomic,const double* total,const double* atom_view,
        double* species_view,double* summary,int* status) {
    for(int k=0;k<species*5;k++)species_view[k]=0.;for(int k=0;k<10;k++)summary[k]=0.;
    if(*status)return;
    // Upstream qts is initialized before, and carried across, the species loop.
    double q[4]{},v[5];
    for(int species_id=1;species_id<=species;species_id++){
        for(int atom=0;atom<atoms;atom++)if(ids[atom]==species_id)
            for(int component=0;component<components;component++)q[component]+=atomic[size_t(atom)*components+component];
        if(!spin_view(components,q,v)){atomicMax(status,2);return;}
        for(int k=0;k<5;k++)species_view[5*(species_id-1)+k]=v[k];
    }
    if(!spin_view(components,total,v)){atomicMax(status,2);return;}
    for(int k=0;k<5;k++)summary[k]=v[k];
    double net=0.,s[3]{};
    for(int atom=0;atom<atoms;atom++){
        net+=atom_view[6*atom];for(int k=0;k<3;k++)s[k]+=atom_view[6*atom+3+k];
    }
    const double magnitude=components==4?sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]):s[2];
    if(!isfinite(net)||!isfinite(magnitude)||!isfinite(s[0])||!isfinite(s[1])||!isfinite(s[2])){atomicMax(status,2);return;}
    summary[5]=net;summary[6]=magnitude;for(int k=0;k<3;k++)summary[7+k]=s[k];
}
}
cudaError_t population_observables_device(int rows,int atoms,int species,int components,
        const PopulationObservableProblem* p,const int* ids,const double* valence,
        const double* orbital,const double* atomic,const double* total,
        double* orbital_view,double* atomic_view,double* species_view,double* summary,int* status,cudaStream_t stream) {
    if(rows<1||atoms<1||species<1||rows>INT_MAX/6||atoms>INT_MAX/6||species>INT_MAX/5||
       (components!=1&&components!=2&&components!=4)||!p||!ids||!valence||!orbital||!atomic||!total||
       !orbital_view||!atomic_view||!species_view||!summary||!status)return cudaErrorInvalidValue;
    const int blocks=std::min(1024,(std::max(rows,atoms)-1)/256+1);
    per_item<<<blocks,256,0,stream>>>(rows,atoms,species,components,p,ids,valence,orbital,atomic,orbital_view,atomic_view,status);
    totals<<<1,1,0,stream>>>(atoms,species,components,ids,atomic,total,atomic_view,species_view,summary,status);
    return cudaGetLastError();
}
}
