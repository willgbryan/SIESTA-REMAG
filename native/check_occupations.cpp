// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only host adapter. Staging and downloads here are explicitly outside the
// device routine; this adapter is never called from SIESTA.
#include "occupations.hpp"
#include <vector>

extern "C" int remag_check_fermi(const remag::FermiProblem* problem,const double* energies,
                                const double* weights,double* occupations,remag::FermiResult* result,int graph_mode) {
    std::vector<void*> buffers;
    cudaError_t error=cudaSuccess;
    auto allocate=[&](size_t bytes) {
        void* pointer=nullptr;
        auto status=cudaMalloc(&pointer,bytes);
        if(status!=cudaSuccess) error=status;
        if(pointer) buffers.push_back(pointer);
        return pointer;
    };
    if(problem->maxe<1||problem->maxspn<1||problem->nk<1) return -1;
    size_t bytes=size_t(problem->maxe)*problem->maxspn*problem->nk*sizeof(double);
    auto p=static_cast<remag::FermiProblem*>(allocate(sizeof(*problem)));
    auto e=static_cast<double*>(allocate(bytes));
    auto w=static_cast<double*>(allocate(size_t(problem->nk)*sizeof(double)));
    auto q=static_cast<double*>(allocate(bytes));
    auto r=static_cast<remag::FermiResult*>(allocate(sizeof(*result)));
    if(error==cudaSuccess) error=cudaMemcpy(p,problem,sizeof(*problem),cudaMemcpyHostToDevice);
    if(error==cudaSuccess) error=cudaMemcpy(e,energies,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess) error=cudaMemcpy(w,weights,size_t(problem->nk)*sizeof(double),cudaMemcpyHostToDevice);
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    if(error==cudaSuccess) error=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    if(error==cudaSuccess && graph_mode) {
        error=cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
        if(error==cudaSuccess) {
            error=remag::fermi_device(p,e,w,q,r,stream);
            auto ended=cudaStreamEndCapture(stream,&graph);
            if(error==cudaSuccess) error=ended;
        }
        if(error==cudaSuccess) error=cudaGraphInstantiate(&executable,graph,0);
        if(error==cudaSuccess) error=cudaGraphLaunch(executable,stream);
    } else if(error==cudaSuccess) error=remag::fermi_device(p,e,w,q,r,stream);
    if(error==cudaSuccess) error=cudaStreamSynchronize(stream);
    if(error==cudaSuccess) error=cudaMemcpy(result,r,sizeof(*result),cudaMemcpyDeviceToHost);
    if(error==cudaSuccess) error=cudaMemcpy(occupations,q,bytes,cudaMemcpyDeviceToHost);
    if(executable) cudaGraphExecDestroy(executable);
    if(graph) cudaGraphDestroy(graph);
    if(stream) cudaStreamDestroy(stream);
    for(void* pointer:buffers) cudaFree(pointer);
    return static_cast<int>(error);
}
