// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only staging and final exports for the Hamiltonian seed/SOC component.
#include "hamiltonian_seeds.hpp"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct HamiltonianSeedsCheck {size_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations;};
namespace {
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
struct Resources {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::unique_ptr<remag::HamiltonianSeedsPlan> plan;std::vector<void*> buffers;
    void* allocate(size_t bytes) {void* p=nullptr;check(cudaMalloc(&p,bytes));buffers.push_back(p);return p;}
    ~Resources() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        plan.reset();for(auto p:buffers) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
};
__global__ void prepare(size_t count,int matrix,const int* step,const double* input,double* current,double* h,
        const remag::HamiltonianSeedsProblem* all,remag::HamiltonianSeedsProblem* p,
        remag::HamiltonianSeedsResult* result) {
    const int cycle=*step;
    if(!blockIdx.x&&!threadIdx.x) {*p=all[cycle];*result=remag::HamiltonianSeedsResult{};}
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        current[i]=input[size_t(cycle)*count+i];
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<matrix;i+=gridDim.x*blockDim.x)
        h[i]=input[size_t(cycle)*count+count-matrix+i];
}
__global__ void pack(int matrix,int phase,const int* step,const double* h,
        const remag::HamiltonianSeedsResult* result,double* output,remag::HamiltonianSeedsResult* results) {
    const size_t slot=2*size_t(*step)+phase;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<matrix;i+=gridDim.x*blockDim.x) output[slot*matrix+i]=h[i];
    if(!blockIdx.x&&!threadIdx.x) results[slot]=*result;
}
__global__ void grid(int matrix,const double* contribution,double* h,const remag::HamiltonianSeedsResult* result) {
    if(result->status||result->stage!=1) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<matrix;i+=gridDim.x*blockDim.x) h[i]+=contribution[i];
}
__global__ void change_soc(remag::HamiltonianSeedsProblem* p) {p->soc=p->soc==1?2:1;}
__global__ void advance(int* step) {++*step;}
}

// Rows: kinetic, nonlocal, onsite(6), offsite(4 complex), density, grid, initial H.
// Modes exercise missing/duplicate preparation, changed SOC, and absent SOC data.
extern "C" int remag_check_hamiltonian_seeds(int n,int spins,int cycles,int mode,
        const remag::HamiltonianSeedsProblem* problems,const double* source,double* matrices,
        remag::HamiltonianSeedsResult* results,int graph_mode,HamiltonianSeedsCheck* info) {
    *info=HamiltonianSeedsCheck{};Resources r;
    try {
        if(cycles<1||cycles>1000||mode<0||mode>5) return -1;
        check(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
        r.plan=std::make_unique<remag::HamiltonianSeedsPlan>(n,spins,r.stream);
        info->workspace_bytes=r.plan->workspace_bytes();
        const int matrix=n*spins;const size_t count=16*size_t(n)+3*size_t(matrix);
        auto allocate=[&](size_t bytes,const void* input=nullptr) {
            void* p=r.allocate(bytes);
            if(input) check(cudaMemcpyAsync(p,input,bytes,cudaMemcpyHostToDevice,r.stream));
            return p;
        };
        auto all=static_cast<remag::HamiltonianSeedsProblem*>(allocate(sizeof(*problems)*cycles,problems));
        auto p=static_cast<remag::HamiltonianSeedsProblem*>(allocate(sizeof(*problems)));
        auto input=static_cast<double*>(allocate(count*sizeof(double)*cycles,source));
        auto current=static_cast<double*>(allocate(count*sizeof(double)));
        auto h=static_cast<double*>(allocate(size_t(matrix)*sizeof(double)));
        auto output=static_cast<double*>(allocate(2*size_t(matrix)*sizeof(double)*cycles));
        auto result=static_cast<remag::HamiltonianSeedsResult*>(allocate(sizeof(*results)));
        auto all_results=static_cast<remag::HamiltonianSeedsResult*>(allocate(2*sizeof(*results)*cycles));
        auto step=static_cast<int*>(allocate(sizeof(int)));check(cudaMemsetAsync(step,0,sizeof(int),r.stream));
        check(cudaStreamSynchronize(r.stream));
        const double* on=mode==4?nullptr:current+2*n;
        const cuDoubleComplex* off=mode==5?nullptr:reinterpret_cast<const cuDoubleComplex*>(current+8*n);
        auto apply=[&]() {
            prepare<<<32,256,0,r.stream>>>(count,matrix,step,input,current,h,all,p,result);
            if(mode!=1) check(static_cast<cudaError_t>(r.plan->prepare(p,current,current+n,on,off,current+16*n,h,result)));
            pack<<<32,256,0,r.stream>>>(matrix,0,step,h,result,output,all_results);
            grid<<<32,256,0,r.stream>>>(matrix,current+16*n+matrix,h,result);
            if(mode==3) change_soc<<<1,1,0,r.stream>>>(p);
            check(static_cast<cudaError_t>(r.plan->finish_grid(p,off,h,result)));
            if(mode==2) check(static_cast<cudaError_t>(r.plan->finish_grid(p,off,h,result)));
            pack<<<32,256,0,r.stream>>>(matrix,1,step,h,result,output,all_results);
            advance<<<1,1,0,r.stream>>>(step);check(cudaGetLastError());
        };
        if(graph_mode) {
            check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeGlobal));
            try {apply();} catch(...) {cudaStreamEndCapture(r.stream,&r.graph);throw;}
            check(cudaStreamEndCapture(r.stream,&r.graph));
            size_t size=0;check(cudaGraphGetNodes(r.graph,nullptr,&size));std::vector<cudaGraphNode_t> nodes(size);
            check(cudaGraphGetNodes(r.graph,nodes.data(),&size));
            for(auto node:nodes) {
                info->graph_nodes++;cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
                if(type!=cudaGraphNodeTypeKernel) info->forbidden_nodes++;
            }
            if(info->forbidden_nodes) throw std::runtime_error("Hamiltonian seed graph contains non-kernel nodes");
            check(cudaGraphInstantiate(&r.executable,r.graph,0));
        }
        for(int i=0;i<cycles;i++) {
            if(graph_mode) check(cudaGraphLaunch(r.executable,r.stream));else apply();
            info->iterations++;
        }
        check(cudaStreamSynchronize(r.stream));
        check(cudaMemcpy(matrices,output,2*size_t(matrix)*sizeof(double)*cycles,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(results,all_results,2*sizeof(*results)*cycles,cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& error) {std::fprintf(stderr,"REMAG Hamiltonian seed check: %s\n",error.what());return -2;}
}
