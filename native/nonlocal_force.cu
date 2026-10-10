// SPDX-License-Identifier: GPL-3.0-or-later
// Pinned nlefsm scalar and off-site SOC energy/force/stress conventions.
#include "nonlocal_force.hpp"
#include "nonlocal_force_detail.cuh"
#include <climits>
#include <cmath>
namespace remag {
namespace {
__device__ void fail(int* status,int code){atomicMax(status,code);}
__device__ void add(double* output,double value,int* status){const double previous=atomicAdd(output,value);if(!isfinite(previous+value))fail(status,2);}
__global__ void contract(NonlocalForceView view,const NonlocalForceProblem* problem,const double* density,
        double* energy,double* force,double* stress,int* status) {
    const auto p=*problem;
    if(p.kind<0||p.kind>2||(p.split!=0&&p.split!=1)||!isfinite(p.volume)||p.volume<=0.||
       (p.kind!=0&&(view.components!=8||view.spinor!=2))){fail(status,1);return;}
    for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<view.entries;q+=blockDim.x*gridDim.x) {
        const int query_row=view.query_rows[q],col=view.query_columns[q];
        const int query_atom=view.atom_pairs[2*size_t(q)],query_center=view.atom_pairs[2*size_t(q)+1];
        if(query_row<1||query_row>view.rows||col<1||col>view.columns||query_atom<1||query_atom>view.atoms||query_center<1||query_center>view.atoms){fail(status,1);continue;}
        const int row=query_row-1,atom=query_atom-1,center=query_center-1;
        const int start=view.ptr[row],count=view.num[row];
        if(start<0||start>view.capacity||count<0||count>view.capacity-start){fail(status,1);continue;}
        double diagonal=0.,d[8]={};bool valid=true;
        for(int offset=0;offset<count;offset++) {
            const int index=start+offset,column=view.list[index];
            if(column<1||column>view.columns){fail(status,1);valid=false;break;}
            if(column!=col)continue;
            if(p.kind==0) {
                for(int spin=0;spin<view.spinor;spin++)diagonal+=density[size_t(spin)*view.capacity+index];
            } else {
                // Pristine SOC scatter assigns rather than adding duplicate
                // columns. Only the final matching entry reaches contraction.
                for(int spin=0;spin<8;spin++)d[spin]=density[size_t(spin)*view.capacity+index];
                diagonal=d[0]+d[1];
            }
        }
        if(!valid)continue;
        double de[9]={},df[3],ds[9];
        nonlocal_detail::evaluate(p.kind,p.split,diagonal,d,view.scalar+6*size_t(q),view.operators+33*size_t(q),de,df);
        valid=isfinite(diagonal);
        for(int i=0;i<9;i++)valid=valid&&isfinite(de[i]);
        for(int axis=0;axis<3;axis++) {
            valid=valid&&isfinite(df[axis]);
            for(int j=0;j<3;j++) {
                ds[3*axis+j]=view.xyz[3*size_t(q)+j]*df[axis]/p.volume;
                valid=valid&&isfinite(ds[3*axis+j]);
            }
        }
        if(!valid){fail(status,2);continue;}
        for(int k=0;k<9;k++)add(energy+k,de[k],status);
        for(int axis=0;axis<3;axis++) {
            add(force+3*size_t(atom)+axis,-df[axis],status);
            add(force+3*size_t(center)+axis,df[axis],status);
        }
        for(int k=0;k<9;k++)add(stress+k,ds[k],status);
    }
}
}
cudaError_t add_nonlocal_force_device(const NonlocalForceView& view,const NonlocalForceProblem* problem,const double* density,
        double* energy,double* force,double* stress,int* status,cudaStream_t stream) {
    if(view.entries<0||view.entries>INT_MAX/4||view.atoms<1||view.rows<1||view.columns<1||view.capacity<1||
       (view.components!=1&&view.components!=2&&view.components!=4&&view.components!=8)||
       (view.spinor!=1&&view.spinor!=2)||view.components<view.spinor||
       !view.query_rows||!view.query_columns||!view.atom_pairs||!view.num||!view.ptr||!view.list||!view.scalar||!view.operators||!view.xyz||
       !problem||!density||!energy||!force||!stress||!status||!stream)return cudaErrorInvalidValue;
    const int blocks=view.entries?(view.entries+127)/128:1;
    contract<<<blocks>1024?1024:blocks,128,0,stream>>>(view,problem,density,energy,force,stress,status);
    return cudaGetLastError();
}
}
