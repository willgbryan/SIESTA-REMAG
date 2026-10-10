// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_h0_detail.cuh"
#include <algorithm>
namespace remag {
namespace {
__global__ void validate(NonlocalH0View v,int* status) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<size_t(v.centers)||i<size_t(v.neighbors)||i<size_t(v.rows);i+=size_t(blockDim.x)*gridDim.x) {
        int error=0;
        if(i<size_t(v.centers))error=h0_detail::center_error(v,int(i));
        if(i<size_t(v.neighbors)){int code=h0_detail::neighbor_error(v,int(i));if(code>error)error=code;}
        if(i<size_t(v.rows)){int code=h0_detail::row_error(v,int(i));if(code>error)error=code;}
        if(error)atomicMax(status,error);
    }
}
__global__ void assemble(NonlocalH0View v,double* h,H0Complex* z,int* status) {
    if(*status)return;
    for(size_t row=blockIdx.x;row<size_t(v.rows);row+=gridDim.x) {
        const int start=v.ptr[row],count=v.num[row];
        for(size_t i=threadIdx.x;i<size_t(count);i+=blockDim.x) {
            const int error=h0_detail::add_entry(v,int(row),start+int(i),h,z);
            if(error)atomicMax(status,error);
        }
    }
}
}
cudaError_t add_nonlocal_h0_device(const NonlocalH0View& v,double* h,H0Complex* z,int* status,cudaStream_t stream) {
    if(v.neighbors<0||v.centers<0||v.blocks<0||v.queries<0||v.rows<1||v.columns<1||v.capacity<1||
       v.local_neighbors<0||v.local_neighbors>v.neighbors||(v.soc!=0&&v.soc!=1)||(v.split!=0&&v.split!=1)||
       (v.centers&&!v.center)||(v.neighbors&&!v.neighbor)||(v.blocks&&!v.block)||(v.queries&&!v.integrals)||
       !v.num||!v.ptr||!v.list||!v.row_ptr||(v.local_neighbors&&!v.row_neighbors)||!h||(v.soc&&!z)||!status||!stream)return cudaErrorInvalidValue;
    const int count=std::max({v.rows,v.neighbors,v.centers});
    const int blocks=int(std::min(size_t(1024),(size_t(count)+127)/128));
    validate<<<blocks,128,0,stream>>>(v,status);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    assemble<<<std::min(v.rows,65535),128,0,stream>>>(v,h,z,status);
    return cudaGetLastError();
}
}
