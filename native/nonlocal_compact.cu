// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_compact.hpp"
#include "nonlocal_force_detail.cuh"
#include "soc_projector_detail.cuh"
#include <cmath>
#include <climits>
namespace remag {
namespace {
__device__ void fail(int* s,int c){atomicMax(s,c);}
__device__ void add(double* out,double v,int* s){double old=atomicAdd(out,v);if(!isfinite(old+v))fail(s,2);}
__global__ void gather(NonlocalCompactView v,const double* d,double* weights,int* status){
    // A block owns a sparse row. Each lane computes independent stored
    // entries; duplicate-column summation retains its original j/spin order.
    // The geometry owner guarantees disjoint row spans.
    for(size_t row=blockIdx.x;row<size_t(v.rows);row+=gridDim.x){
        int start=v.ptr[row],count=v.num[row];
        if(start<0||start>v.capacity||count<0||count>v.capacity-start){fail(status,1);continue;}
        for(size_t offset=threadIdx.x;offset<size_t(count);offset+=blockDim.x){
            const int i=start+int(offset);
            weights[i]=0.;
            int col=v.list[i];if(col<1||col>v.columns){fail(status,1);continue;}
            int last=v.lookup[size_t(row)*v.columns+col-1];
            if(last<start||last>=start+count||v.list[last]!=col){fail(status,1);continue;}
            if(last!=i)continue;
            double sum=0.;int actual_last=-1;
            for(int j=start;j<start+count;j++)if(v.list[j]==col){
                actual_last=j;
                if(!v.soc)for(int spin=0;spin<v.spinor;spin++)sum+=d[size_t(spin)*v.capacity+j];
            }
            if(actual_last!=last){fail(status,1);continue;}
            if(v.soc)sum=d[i]+d[size_t(v.capacity)+i];
            weights[i]=sum; // Nonfinite values fail only if actually consumed.
        }
    }
}
__global__ void contract(NonlocalCompactView v,const double* d,const double* weights,double* energy,double* force,double* stress,int* status){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<v.neighbors;i+=blockDim.x*gridDim.x){
        auto left=v.neighbor[i];if(left.row==0)continue;
        if(left.row<1||left.row>v.rows||left.atom<1||left.atom>v.atoms||left.center<0||left.center>=v.centers){fail(status,1);continue;}
        auto c=v.center[left.center];
        if(c.neighbor<0||c.neighbors<0||c.neighbor>v.neighbors-c.neighbors||i<c.neighbor||i>=c.neighbor+c.neighbors||
           c.block<0||c.blocks<0||c.block>v.blocks-c.blocks||c.projectors<1||c.query<0||
           size_t(c.query)+size_t(c.neighbors)*c.projectors>size_t(v.queries)||c.atom<1||c.atom>v.atoms){fail(status,1);continue;}
        const int row_start=v.ptr[left.row-1],row_count=v.num[left.row-1];
        if(row_start<0||row_start>v.capacity||row_count<0||row_count>v.capacity-row_start){fail(status,1);continue;}
        size_t first=size_t(c.query)+size_t(i-c.neighbor)*c.projectors;
        double total_e[9]={},total_f[3]={},total_s[9]={};
        for(int j=c.neighbor;j<c.neighbor+c.neighbors;j++){
            int col=v.neighbor[j].column;
            if(col<1||col>v.columns){fail(status,1);continue;}
            int index=v.lookup[size_t(left.row-1)*v.columns+col-1];
            if(index==-1)continue;
            if(index<row_start||index>=row_start+row_count||v.list[index]!=col){fail(status,1);continue;}
            double diagonal=weights[index],density[8]={};
            if(v.soc)for(int k=0;k<8;k++)density[k]=d[size_t(k)*v.capacity+index];
            for(int b=c.block;b<c.block+c.blocks;b++){
                auto block=v.block[b];
                if(block.l<0||block.l>5||(!v.soc&&block.l)){fail(status,1);continue;}
                const int width=block.l?2*(2*block.l+1):1;
                if(block.offset<0||block.offset>c.projectors-width){fail(status,1);continue;}
                const double* a=v.integrals+4*(first+block.offset);
                const double* z=v.integrals+4*(size_t(c.query)+size_t(j-c.neighbor)*c.projectors+block.offset);
                double scalar[6]={block.epsilon[0],a[0],z[0],a[1],a[2],a[3]},op[33]={};
                if(block.l){int code=soc_detail::evaluate(block.l,v.split,block.epsilon,a,z,a+1,4,4,op);if(code){fail(status,code);continue;}}
                double de[9]={},df[3],ds[9];
                nonlocal_detail::evaluate(v.soc?(block.l?2:1):0,v.split,diagonal,density,scalar,op,de,df);
                bool valid=isfinite(diagonal);
                for(int k=0;k<9;k++)valid=valid&&isfinite(de[k]);
                for(int axis=0;axis<3;axis++){
                    valid=valid&&isfinite(df[axis]);
                    for(int k=0;k<3;k++){ds[3*axis+k]=v.xyz[3*first+k]*df[axis]/v.volume;valid=valid&&isfinite(ds[3*axis+k]);}
                }
                if(!valid){fail(status,2);continue;}
                for(int k=0;k<9;k++){total_e[k]+=de[k];total_s[k]+=ds[k];}
                for(int k=0;k<3;k++)total_f[k]+=df[k];
            }
        }
        for(int k=0;k<9;k++){add(energy+k,total_e[k],status);add(stress+k,total_s[k],status);}
        for(int k=0;k<3;k++){add(force+3*size_t(left.atom-1)+k,-total_f[k],status);add(force+3*size_t(c.atom-1)+k,total_f[k],status);}
    }
}
int blocks(int n){return n?int((size_t(n)+127)/128>1024?1024:(size_t(n)+127)/128):1;}
}
cudaError_t add_nonlocal_compact_device(const NonlocalCompactView& v,const double* d,double* weights,double* e,double* f,double* s,int* status,cudaStream_t stream){
    if(v.neighbors<0||v.neighbors>INT_MAX/4||v.centers<0||v.blocks<0||v.queries<0||v.atoms<1||v.rows<1||v.columns<1||v.capacity<1||
       (v.components!=1&&v.components!=2&&v.components!=4&&v.components!=8)||(v.spinor!=1&&v.spinor!=2)||v.components<v.spinor||
       (v.soc!=0&&v.soc!=1)||(v.soc&&(v.components!=8||v.spinor!=2))||(v.split!=0&&v.split!=1)||!std::isfinite(v.volume)||v.volume<=0.||
       !v.center||!v.neighbor||!v.block||!v.num||!v.ptr||!v.list||!v.lookup||!v.integrals||!v.xyz||!d||!weights||!e||!f||!s||!status||!stream)return cudaErrorInvalidValue;
    gather<<<(v.rows>1024?1024:v.rows),128,0,stream>>>(v,d,weights,status);auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    contract<<<blocks(v.neighbors),128,0,stream>>>(v,d,weights,e,f,s,status);return cudaGetLastError();
}
}
