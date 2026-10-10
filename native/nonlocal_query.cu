// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_query.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
namespace remag {
namespace {
__device__ bool span(int start,int count,int limit) {
    return start>=0&&start<=limit&&count>=0&&count<=limit-start;
}
__global__ void validate(NonlocalQueryView v,int* status) {
    for(size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
        index<size_t(v.centers)||index<size_t(v.neighbors)||index<size_t(v.projectors);
        index+=size_t(blockDim.x)*gridDim.x) {
        int error=0;
        if(index<size_t(v.centers)) {
            const int i=int(index);const auto c=v.center[i];
            const int begin=v.projector_ptr[i],end=v.projector_ptr[i+1];
            if(!span(c.neighbor,c.neighbors,v.neighbors)||c.projectors<1||c.query<0||
               size_t(c.query)+size_t(c.neighbors)*size_t(c.projectors)>size_t(v.queries)||
               begin<0||begin>end||end>v.projectors||end-begin!=c.projectors)error=1;
            if(i==0&&(c.neighbor!=0||c.query!=0||begin!=0))error=1;
            if(i+1==v.centers&&(size_t(c.neighbor)+size_t(c.neighbors)!=size_t(v.neighbors)||
               size_t(c.query)+size_t(c.neighbors)*size_t(c.projectors)!=size_t(v.queries)||end!=v.projectors))error=1;
            if(i) {
                const auto previous=v.center[i-1];
                if(size_t(c.neighbor)!=size_t(previous.neighbor)+size_t(previous.neighbors)||
                   size_t(c.query)!=size_t(previous.query)+size_t(previous.neighbors)*size_t(previous.projectors))error=1;
            }
        }
        if(index<size_t(v.neighbors)) {
            const auto n=v.neighbor[index];
            if(n.center<0||n.center>=v.centers||v.orbital_registry[index]<1)error=1;
            else {
                const auto c=v.center[n.center];
                if(!span(c.neighbor,c.neighbors,v.neighbors)||index<size_t(c.neighbor)||
                   index>=size_t(c.neighbor)+size_t(c.neighbors))error=1;
            }
            for(int axis=0;axis<3;axis++)if(!isfinite(v.displacement[3*index+axis]))error=2;
        }
        if(index<size_t(v.projectors)&&v.projector_registry[index]<1)error=1;
        if(error)atomicMax(status,error);
    }
}
__global__ void expand(NonlocalQueryView v,int* pairs,double* xyz,const int* status) {
    if(*status)return;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(v.neighbors);
        i+=size_t(blockDim.x)*gridDim.x) {
        const auto c=v.center[v.neighbor[i].center];
        const size_t first=size_t(c.query)+(i-size_t(c.neighbor))*size_t(c.projectors);
        const int offset=v.projector_ptr[v.neighbor[i].center];
        for(int p=0;p<c.projectors;p++) {
            const size_t q=first+size_t(p);
            pairs[2*q]=v.projector_registry[offset+p];pairs[2*q+1]=v.orbital_registry[i];
            for(int axis=0;axis<3;axis++)xyz[3*q+axis]=v.displacement[3*i+axis];
        }
    }
}
}
cudaError_t expand_nonlocal_queries_device(const NonlocalQueryView& v,int* pairs,double* xyz,
        int* status,cudaStream_t stream) {
    if(v.centers<0||v.neighbors<0||v.queries<0||v.projectors<0||v.queries>INT_MAX/4||
       (!v.centers&&(v.neighbors||v.queries||v.projectors))||
       (v.centers&&(!v.center||!v.projector_ptr))||
       (v.neighbors&&(!v.neighbor||!v.orbital_registry||!v.displacement))||
       (v.projectors&&!v.projector_registry)||(v.queries&&(!pairs||!xyz))||!status||!stream)
        return cudaErrorInvalidValue;
    if(!v.centers)return cudaSuccess;
    const size_t count=std::max({v.centers,v.neighbors,v.projectors});
    const int blocks=int(std::min(size_t(1024),(count+127)/128));
    validate<<<blocks,128,0,stream>>>(v,status);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    if(v.neighbors)expand<<<int(std::min(size_t(1024),(size_t(v.neighbors)+127)/128)),128,0,stream>>>(v,pairs,xyz,status);
    return cudaGetLastError();
}
}
