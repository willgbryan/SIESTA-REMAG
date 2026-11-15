// CPU interface double for the actual C++ owner; no CUDA execution is claimed.
#pragma once
#include <cstdlib>
#include <cstring>
using cudaError_t=int;
using cudaStream_t=void*;using cudaGraph_t=void*;using cudaGraphExec_t=void*;using cudaGraphNode_t=void*;
using cudaGraphNodeType=int;
constexpr int cudaSuccess=0,cudaMemcpyDeviceToHost=1,cudaMemcpyHostToDevice=2;
constexpr int cudaStreamNonBlocking=1,cudaStreamCaptureModeGlobal=0,cudaGraphNodeTypeKernel=0;
inline bool copy_failure=false;
inline int copy_calls=0;
inline const char* cudaGetErrorString(int){return "injected copy failure";}
inline int cudaMemcpy(void* out,const void* in,size_t n,int direction){
    if(direction!=cudaMemcpyDeviceToHost)std::abort();++copy_calls;
    if(copy_failure)return 1;std::memcpy(out,in,n);return 0;
}
#define UNUSED_CUDA(name) template<class... A> int name(A...){std::abort();}
UNUSED_CUDA(cudaFree) UNUSED_CUDA(cudaMalloc) UNUSED_CUDA(cudaMemcpyAsync) UNUSED_CUDA(cudaMemsetAsync)
UNUSED_CUDA(cudaStreamSynchronize) UNUSED_CUDA(cudaStreamCreateWithFlags) UNUSED_CUDA(cudaStreamDestroy)
UNUSED_CUDA(cudaStreamBeginCapture) UNUSED_CUDA(cudaStreamEndCapture) UNUSED_CUDA(cudaGraphDestroy)
UNUSED_CUDA(cudaGraphExecDestroy) UNUSED_CUDA(cudaGraphGetNodes) UNUSED_CUDA(cudaGraphNodeGetType)
UNUSED_CUDA(cudaGraphInstantiate) UNUSED_CUDA(cudaGraphLaunch)
#undef UNUSED_CUDA
