// CPU-only transport double. It does not execute or simulate CUDA graphs.
#pragma once
using cudaError_t=int;
constexpr int cudaSuccess=0;
constexpr int cudaErrorNotSupported=801;
using cudaGraph_t=void*;
using cudaGraphExec_t=void*;
using cudaStream_t=void*;
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaDriverGetVersion(int*);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*,cudaGraph_t,unsigned long long);
cudaError_t cudaGraphUpload(cudaGraphExec_t,cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
