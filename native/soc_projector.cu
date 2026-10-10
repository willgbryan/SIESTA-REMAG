// SPDX-License-Identifier: GPL-3.0-or-later
// Angular/phase and scalar/SOC splitting conventions from pristine
// SIESTA m_nlefsm::calc_Vj_offsiteSO, pinned in upstream.lock.json.
#include "soc_projector.hpp"
#include "soc_projector_detail.cuh"
#include <cmath>
#include <algorithm>
namespace remag {
namespace {
__device__ void evaluate(const SocProjectorView& view,int query,double* output,int* status) {
    double* out=output+33*size_t(query);
    for(int k=0;k<33;k++)out[k]=0.;
    if(*status)return;
    const int width=2*view.l+1;
    const int code=soc_detail::evaluate(view.l,view.split,view.epsilon,
        view.left+2*size_t(width)*query,view.right+2*size_t(width)*query,
        view.left_gradient+6*size_t(width)*query,1,3,out);
    if(code)atomicCAS(status,0,code);
}
__global__ void projector(SocProjectorView view,double* output,int* status) {
    for(size_t query=size_t(blockIdx.x)*blockDim.x+threadIdx.x;query<size_t(view.queries);query+=size_t(blockDim.x)*gridDim.x)
        evaluate(view,int(query),output,status);
}
}
cudaError_t soc_projector_device(const SocProjectorView& view,double* output,int* status,cudaStream_t stream) {
    if(view.queries<0||view.l<1||view.l>5||(view.split!=0&&view.split!=1)||!status||
       (view.queries&&(!view.epsilon||!view.left||!view.right||!view.left_gradient||!output)))return cudaErrorInvalidValue;
    const int blocks=std::max(1,int(std::min(size_t(1024),(size_t(view.queries)+127)/128)));
    projector<<<blocks,128,0,stream>>>(view,output,status);
    return cudaGetLastError();
}
}
