// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only staging/export adapter; never called by SIESTA.
#include "pbe.hpp"
#include <climits>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct PbeCheck {uint64_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations;};
namespace {
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
struct Resources {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    std::unique_ptr<remag::PbePlan> plan;
    std::vector<void*> buffers;
    void* allocate(size_t size) {
        void* p=nullptr;check(cudaMalloc(&p,size));buffers.push_back(p);return p;
    }
    ~Resources() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        plan.reset();
        if(stream) cudaStreamDestroy(stream);
        for(auto p:buffers) cudaFree(p);
    }
};
__global__ void change_density(int size,double* density) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) density[i]*=.75;
}
void inventory(cudaGraph_t graph,PbeCheck& info) {
    size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes) {
        info.graph_nodes++;
        cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeGraph) {
            cudaGraph_t child=nullptr;check(cudaGraphChildGraphNodeGetGraph(node,&child));inventory(child,info);
        } else if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms params{};check(cudaGraphMemcpyNodeGetParams(node,&params));
            if(params.kind!=cudaMemcpyDeviceToDevice) info.forbidden_nodes++;
        } else if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty)
            info.forbidden_nodes++;
    }
}
}
extern "C" int remag_check_pbe(const remag::PbeProblem* problem,const double* density,
        double* potential,remag::PbeResult* result,int graph_mode,int corrupt_shape,PbeCheck* info) {
    *info=PbeCheck{};
    Resources resources;
    try {
        const auto& m=problem->mesh;
        if(m[0]<1||m[1]<1||m[2]<1||static_cast<long long>(m[0])*m[1]>INT_MAX/m[2]) return -1;
        int size=m[0]*m[1]*m[2];
        if((problem->spins!=1&&problem->spins!=2&&problem->spins!=4)||size>INT_MAX/problem->spins) return -1;
        size_t bytes=size_t(size)*problem->spins*sizeof(double);
        check(cudaStreamCreateWithFlags(&resources.stream,cudaStreamNonBlocking));
        resources.plan=std::make_unique<remag::PbePlan>(m[0],m[1],m[2],problem->spins,resources.stream);
        info->workspace_bytes=resources.plan->workspace_bytes();
        auto p=static_cast<remag::PbeProblem*>(resources.allocate(sizeof(*problem)));
        auto rho=static_cast<double*>(resources.allocate(bytes));
        auto v=static_cast<double*>(resources.allocate(bytes));
        auto r=static_cast<remag::PbeResult*>(resources.allocate(sizeof(*result)));
        auto staged=*problem;
        if(corrupt_shape==1) staged.mesh[0]=m[0]==1?2:m[0]-1;
        if(corrupt_shape==2) staged.spins=3;
        check(cudaMemcpyAsync(p,&staged,sizeof(staged),cudaMemcpyHostToDevice,resources.stream));
        check(cudaMemcpyAsync(rho,density,bytes,cudaMemcpyHostToDevice,resources.stream));
        check(cudaStreamSynchronize(resources.stream)); // explicit staging boundary
        auto apply=[&]() {
            int error=resources.plan->apply(p,rho,v,r);
            if(error) throw std::runtime_error("Pbe apply status "+std::to_string(error));
        };
        apply();check(cudaStreamSynchronize(resources.stream)); // initialization outside capture
        if(graph_mode) {
            check(cudaStreamBeginCapture(resources.stream,cudaStreamCaptureModeGlobal));
            try {apply();}
            catch(...) {cudaStreamEndCapture(resources.stream,&resources.graph);throw;}
            check(cudaStreamEndCapture(resources.stream,&resources.graph));
            inventory(resources.graph,*info);
            if(info->forbidden_nodes) throw std::runtime_error("Pbe graph contains host or non-device-copy nodes");
            check(cudaGraphInstantiate(&resources.executable,resources.graph,0));
        }
        for(int cycle=0;cycle<2;cycle++) {
            if(cycle) change_density<<<32,256,0,resources.stream>>>(size*problem->spins,rho);
            if(graph_mode) check(cudaGraphLaunch(resources.executable,resources.stream));
            else apply();
            info->iterations++;
        }
        check(cudaStreamSynchronize(resources.stream));
        check(cudaMemcpy(result,r,sizeof(*result),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(potential,v,bytes,cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"REMAG Pbe check: %s\n",error.what());return -2;
    }
}
