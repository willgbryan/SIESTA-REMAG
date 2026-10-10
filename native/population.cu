// SPDX-License-Identifier: GPL-3.0-or-later
#include "population.hpp"
#include <algorithm>
#include <climits>
#include <cmath>

namespace remag {
namespace {
constexpr int threads=256;
__global__ void initialize(int capacity,int rows,int atoms,int spins,
        const PopulationProblem* p,double* orbital,double* atomic,double* total,int* status) {
    const int components=min(spins,4);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*components;i+=gridDim.x*blockDim.x) orbital[i]=0.;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<atoms*components;i+=gridDim.x*blockDim.x) atomic[i]=0.;
    if(!blockIdx.x&&!threadIdx.x) {
        for(int s=0;s<components;s++)total[s]=0.;
        *status=(p->capacity!=capacity||p->rows!=rows||p->atoms!=atoms||p->spins!=spins)?1:0;
    }
}
__global__ void orbitals(int capacity,int rows,int spins,const int* num,const int* ptr,
        const double* overlap,const double* density,double* output,int* status) {
    const int components=min(spins,4);
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=gridDim.x*blockDim.x) {
        const int start=ptr[row],count=num[row];
        if(start<0||start>capacity||count<0||count>capacity-start) {atomicMax(status,1);continue;}
        if(row) {
            const int previous=ptr[row-1],length=num[row-1];
            if(previous<0||previous>capacity||length<0||length>capacity-previous||start<previous+length) {
                atomicMax(status,1);continue;
            }
        }
        double sums[4]{};
        bool finite=true;
        for(int i=start;i<start+count;i++) {
            const double s=overlap[i];
            finite=finite&&isfinite(s);
            for(int spin=0;spin<components;spin++) {
                double d=density[i+capacity*spin];
                if(spins==8&&spin>=2)d=.5*(d+density[i+capacity*(spin+4)]);
                finite=finite&&isfinite(d);
                sums[spin]+=d*s;
            }
        }
        for(int spin=0;spin<components;spin++)finite=finite&&isfinite(sums[spin]);
        if(!finite) {atomicMax(status,2);continue;}
        for(int spin=0;spin<components;spin++)output[row*components+spin]=sums[spin];
    }
}
__global__ void atoms(int rows,int atoms,int components,const int* lasto,
        const double* orbital,double* output,int* status) {
    for(int atom=blockIdx.x*blockDim.x+threadIdx.x;atom<atoms;atom+=gridDim.x*blockDim.x) {
        const int begin=lasto[atom],end=lasto[atom+1];
        if(begin<0||end<begin||end>rows||lasto[0]!=0||lasto[atoms]!=rows) {atomicMax(status,1);continue;}
        for(int spin=0;spin<components;spin++) {
            double value=0.;
            for(int row=begin;row<end;row++)value+=orbital[row*components+spin];
            if(!isfinite(value)) {atomicMax(status,2);continue;}
            output[atom*components+spin]=value;
        }
    }
}
__global__ void finish(int rows,int components,const double* orbital,double* total,int* status) {
    if(*status)return;
    double values[4]{};
    for(int row=0;row<rows;row++)for(int spin=0;spin<components;spin++)values[spin]+=orbital[row*components+spin];
    for(int spin=0;spin<components;spin++)if(!isfinite(values[spin])) {*status=2;return;}
    for(int spin=0;spin<components;spin++)total[spin]=values[spin];
}
}
cudaError_t population_device(int capacity,int rows,int count,int spins,const PopulationProblem* p,
        const int* num,const int* ptr,const int* lasto,const double* overlap,const double* density,
        double* orbital,double* atomic,double* total,int* status,cudaStream_t stream) {
    if(capacity<1||rows<1||count<1||(spins!=1&&spins!=2&&spins!=4&&spins!=8)||
       capacity>INT_MAX/spins||rows>INT_MAX/4||count>INT_MAX/4||
       !p||!num||!ptr||!lasto||!overlap||!density||!orbital||!atomic||!total||!status||
       orbital==atomic||orbital==total||atomic==total||orbital==density||atomic==density||total==density||
       orbital==overlap||atomic==overlap||total==overlap)return cudaErrorInvalidValue;
    const int blocks=std::min(1024,(rows-1)/threads+1);
    initialize<<<blocks,threads,0,stream>>>(capacity,rows,count,spins,p,orbital,atomic,total,status);
    orbitals<<<blocks,threads,0,stream>>>(capacity,rows,spins,num,ptr,overlap,density,orbital,status);
    atoms<<<std::min(1024,(count-1)/threads+1),threads,0,stream>>>(rows,count,std::min(spins,4),lasto,orbital,atomic,status);
    finish<<<1,1,0,stream>>>(rows,std::min(spins,4),orbital,total,status);
    return cudaGetLastError();
}
}
