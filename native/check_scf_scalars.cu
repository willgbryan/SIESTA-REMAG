// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only staging/export for the device SCF scalar consumers.
#include "scf_scalars.hpp"
#include <climits>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct ScfScalarsCheck {size_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations;};
namespace {
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
struct Resources {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::unique_ptr<remag::ScfScalarsPlan> plan;std::vector<void*> buffers;
    void* allocate(size_t bytes) {void* p=nullptr;check(cudaMalloc(&p,bytes));buffers.push_back(p);return p;}
    ~Resources() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        plan.reset();for(auto p:buffers) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
};
__global__ void prepare(size_t count,const int* step,const double* source,double* state,
                        const remag::ScfEnergySeeds* all_seeds,remag::ScfEnergySeeds* seeds) {
    const int cycle=*step;
    if(!blockIdx.x&&!threadIdx.x) *seeds=all_seeds[cycle];
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        state[i]=source[size_t(cycle)*count+i];
}
__global__ void pack(int size,const int* step,const double* residual,const remag::ScfScalarsResult* result,
                     double* all_residual,remag::ScfScalarsResult* all_results) {
    const int cycle=*step;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x)
        all_residual[size_t(cycle)*size+i]=residual[i];
    if(!blockIdx.x&&!threadIdx.x) all_results[cycle]=*result;
}
__global__ void advance(int* step) {++*step;}
}
// Each source row concatenates H, Hold, D, Dold, E, Eold in Fortran layout.
extern "C" int remag_check_scf_scalars(const remag::ScfScalarsProblem* problem,int cycles,int corrupt_shape,
        const remag::ScfEnergySeeds* seeds,const int* num,const int* ptr,const double* overlap,const double* source,
        double* residual,remag::ScfScalarsResult* results,int graph_mode,ScfScalarsCheck* info) {
    *info=ScfScalarsCheck{};Resources r;
    try {
        if(cycles<1||cycles>1000) return -1;
        check(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
        const int n=problem->capacity,s=problem->spins,rows=problem->rows;
        r.plan=std::make_unique<remag::ScfScalarsPlan>(n,s,rows,r.stream);
        info->workspace_bytes=r.plan->workspace_bytes();
        const size_t matrix=size_t(n)*s,energy=size_t(n)*(s<4?s:4),count=4*matrix+2*energy;
        auto allocate=[&](size_t bytes,const void* source=nullptr) {
            void* p=r.allocate(bytes);
            if(source) check(cudaMemcpyAsync(p,source,bytes,cudaMemcpyHostToDevice,r.stream));
            return p;
        };
        auto p=static_cast<remag::ScfScalarsProblem*>(allocate(sizeof(*problem)));
        auto staged=*problem;if(corrupt_shape) staged.capacity++;
        check(cudaMemcpyAsync(p,&staged,sizeof(staged),cudaMemcpyHostToDevice,r.stream));
        auto all_seeds=static_cast<remag::ScfEnergySeeds*>(allocate(sizeof(*seeds)*cycles,seeds));
        auto current_seeds=static_cast<remag::ScfEnergySeeds*>(allocate(sizeof(*seeds)));
        auto counts=static_cast<int*>(allocate(size_t(rows)*sizeof(int),num));
        auto pointers=static_cast<int*>(allocate(size_t(rows)*sizeof(int),ptr));
        auto ov=static_cast<double*>(allocate(size_t(n)*sizeof(double),overlap));
        auto input=static_cast<double*>(allocate(count*sizeof(double)*cycles,source));
        auto current=static_cast<double*>(allocate(count*sizeof(double)));
        auto out=static_cast<double*>(allocate(matrix*sizeof(double),residual));
        auto all_out=static_cast<double*>(allocate(matrix*sizeof(double)*cycles));
        auto result=static_cast<remag::ScfScalarsResult*>(allocate(sizeof(*results)));
        auto all_results=static_cast<remag::ScfScalarsResult*>(allocate(sizeof(*results)*cycles));
        auto step=static_cast<int*>(allocate(sizeof(int)));check(cudaMemsetAsync(step,0,sizeof(int),r.stream));
        check(cudaStreamSynchronize(r.stream));
        auto apply=[&]() {
            prepare<<<32,256,0,r.stream>>>(count,step,input,current,all_seeds,current_seeds);
            check(static_cast<cudaError_t>(r.plan->apply(p,current_seeds,counts,pointers,ov,
                current,current+matrix,current+2*matrix,current+3*matrix,current+4*matrix,
                current+4*matrix+energy,out,result)));
            pack<<<32,256,0,r.stream>>>(int(matrix),step,out,result,all_out,all_results);
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
            if(info->forbidden_nodes) throw std::runtime_error("SCF scalar graph contains non-kernel nodes");
            check(cudaGraphInstantiate(&r.executable,r.graph,0));
        }
        for(int i=0;i<cycles;i++) {
            if(graph_mode) check(cudaGraphLaunch(r.executable,r.stream));else apply();
            info->iterations++;
        }
        check(cudaStreamSynchronize(r.stream));
        check(cudaMemcpy(residual,all_out,matrix*sizeof(double)*cycles,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(results,all_results,sizeof(*results)*cycles,cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& error) {std::fprintf(stderr,"REMAG SCF scalar check: %s\n",error.what());return -2;}
}
