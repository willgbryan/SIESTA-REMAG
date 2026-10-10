// SPDX-License-Identifier: GPL-3.0-or-later
#include "grid_force.hpp"
#include <climits>
#include <cmath>

namespace remag {
namespace {
__global__ void contract(GridForceView v,int nh,const GridForceProblem* problem,
        const double* density,const double* potential,const double* auxiliary,double* force,double* stress,int* status) {
    if(*status) return;
    const auto p=*problem;const auto g=v.grid.shape;const int ng=nh<4?nh:4,ip=blockIdx.x;
    if((p.forces!=0&&p.forces!=1)||(p.stress!=0&&p.stress!=1)||!isfinite(p.dvol)||p.dvol<=0.||!isfinite(p.volume)||p.volume<=0.) {
        atomicCAS(status,0,1);return;
    }
    const int begin=v.grid.ends[ip],end=v.grid.ends[ip+1];
    if(begin<0||end<begin||end>v.entries) {atomicCAS(status,0,1);return;}
    for(int entry=begin+threadIdx.x;entry<end;entry+=blockDim.x) {
        const int i=v.grid.orbs[entry]-1;
        if(i<0||i>=g.no) {atomicCAS(status,0,1);continue;}
        const int atom=v.orbital_atoms[i]-1;
        if(atom<0||atom>=v.atoms) {atomicCAS(status,0,1);continue;}
        double df[3]={},ds[9]={};bool valid=true;
        for(int sub=0;sub<g.nsp;sub++) {
            double cdv=0.;
            for(int spin=0;spin<ng;spin++) {
                double cd=0.;
                for(int other=begin;other<end;other++) {
                    const int j=v.grid.orbs[other]-1;
                    if(j<0||j>=g.no) {atomicCAS(status,0,1);valid=false;continue;}
                    const int relative=v.grid.translations[(i/g.nuo)*g.cells+j/g.nuo];
                    if(relative<0||relative>=g.cells) {atomicCAS(status,0,1);valid=false;continue;}
                    const int index=v.grid.lookup[size_t(i%g.nuo)*g.no+relative*g.nuo+j%g.nuo];
                    if(index<0) continue;
                    if(index>=g.nnz) {atomicCAS(status,0,1);valid=false;continue;}
                    double d=density[size_t(spin)*g.nnz+index];
                    if(nh==8&&spin>=2) d=.5*(d+density[size_t(spin+4)*g.nnz+index]);
                    cd+=v.phi[size_t(other)*g.nsp+sub]*d;
                }
                const double pot=potential[(size_t(spin)*g.np+ip)*g.nsp+sub];
                cdv+=cd*(ng==4&&spin>=2?2.*pot:pot);
            }
            cdv-=v.phi[size_t(entry)*g.nsp+sub]*v.atomic_occupation[i%g.nuo]*auxiliary[size_t(ip)*g.nsp+sub];
            cdv*=2.*p.dvol;valid=valid&&isfinite(cdv);
            const size_t offset=3*(size_t(entry)*g.nsp+sub);
            for(int ix=0;ix<3;ix++) {
                const double grad=v.gradient[offset+ix];
                if(p.forces) df[ix]+=grad*cdv;
                if(p.stress) for(int iy=0;iy<3;iy++) ds[3*ix+iy]+=v.displacement[offset+iy]*grad/p.volume*cdv;
            }
        }
        for(double x:df) valid=valid&&isfinite(x);
        for(double x:ds) valid=valid&&isfinite(x);
        if(!valid) {atomicCAS(status,0,2);continue;}
        if(p.forces) for(int ix=0;ix<3;ix++) atomicAdd(force+3*atom+ix,df[ix]);
        if(p.stress) for(int i=0;i<9;i++) atomicAdd(stress+i,ds[i]);
    }
}
}
cudaError_t add_grid_force_device(const GridForceView& v,int nh,const GridForceProblem* problem,const double* density,
        const double* potential,const double* auxiliary,double* force,double* stress,int* status,cudaStream_t stream) {
    const auto& g=v.grid.shape;
    if(!stream||!problem||!density||!potential||!auxiliary||!force||!stress||!status||
        !v.orbital_atoms||!v.phi||!v.gradient||!v.displacement||!v.atomic_occupation||
        !v.grid.ends||!v.grid.orbs||!v.grid.translations||!v.grid.lookup||
        g.no<1||g.nuo<1||g.np<1||g.nsp<1||g.nnz<1||g.cells<1||g.no%g.nuo||g.no/g.nuo!=g.cells||
        v.atoms<1||v.entries<1||v.entries>INT_MAX/3/g.nsp||(nh!=1&&nh!=2&&nh!=4&&nh!=8)) return cudaErrorInvalidValue;
    contract<<<g.np,128,0,stream>>>(v,nh,problem,density,potential,auxiliary,force,stress,status);
    return cudaGetLastError();
}
}
