// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_grid_force.hpp"
#include <cmath>

namespace remag {
namespace {
__global__ void contract(AtomicGridForceView v,const AtomicGridForceProblem* problem,
        const double* potential,const double* auxiliary,const double* deformation,
        double* force,double* stress,int* status) {
    if(*status) return;
    const auto p=*problem;
    if((p.kind!=1&&p.kind!=2)||(p.diagonal_spins!=1&&p.diagonal_spins!=2)||
        (p.kind==2&&p.diagonal_spins!=1)||(p.forces!=0&&p.forces!=1)||
        (p.stress!=0&&p.stress!=1)||!isfinite(p.dvol)||p.dvol<=0.||!isfinite(p.volume)||p.volume<=0.) {
        atomicCAS(status,0,1);return;
    }
    for(size_t entry=size_t(blockIdx.x)*blockDim.x+threadIdx.x;entry<size_t(v.entries);entry+=size_t(blockDim.x)*gridDim.x) {
        if(v.active){const int selected=v.active[entry];
            if(selected!=0&&selected!=1){atomicCAS(status,0,1);continue;}
            if(!selected)continue;
        }
        const int ip=v.point_index[entry],ia=v.atom_index[entry];
        if(ip<1||ip>v.points||ia<1||ia>v.atoms) {atomicCAS(status,0,1);continue;}
        const int point=ip-1,atom=ia-1;
        double df[3]={},ds[9]={};bool valid=true;
        for(int spin=0;spin<p.diagonal_spins;spin++) {
            const double field=p.kind==1?potential[size_t(spin)*v.points+point]-auxiliary[point]:deformation[point];
            valid=valid&&isfinite(field);
            for(int i=0;i<3;i++) {
                const double g=v.gradient[3*entry+i];
                // Preserve each pinned formula's multiplication order.
                const double value=p.kind==1?p.dvol*field*g/p.diagonal_spins:p.dvol*g*field;
                valid=valid&&isfinite(value);
                if(p.forces) df[i]+=value;
                if(p.stress) for(int j=0;j<3;j++) ds[3*i+j]+=v.displacement[3*entry+j]*value/p.volume;
            }
        }
        for(double x:df) valid=valid&&isfinite(x);
        for(double x:ds) valid=valid&&isfinite(x);
        if(!valid) {atomicCAS(status,0,2);continue;}
        if(p.forces) for(int i=0;i<3;i++) atomicAdd(force+3*atom+i,df[i]);
        if(p.stress) for(int i=0;i<9;i++) atomicAdd(stress+i,ds[i]);
    }
}
}
cudaError_t add_atomic_grid_force_device(const AtomicGridForceView& v,const AtomicGridForceProblem* p,
        const double* potential,const double* auxiliary,const double* deformation,
        double* force,double* stress,int* status,cudaStream_t stream) {
    if(v.points<1||v.entries<0||v.atoms<1||!v.point_index||!v.atom_index||!v.gradient||!v.displacement||
        !p||!potential||!auxiliary||!deformation||!force||!stress||!status||!stream) return cudaErrorInvalidValue;
    const int blocks=v.entries?int((size_t(v.entries)+127)/128):1;
    contract<<<blocks,128,0,stream>>>(v,p,potential,auxiliary,deformation,force,stress,status);
    return cudaGetLastError();
}
}
