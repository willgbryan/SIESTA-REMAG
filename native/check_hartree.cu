// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only staging/export adapter; never called by SIESTA.
#include "hartree.hpp"
#include <climits>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct HartreeCheck {uint64_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations;};
namespace {
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
struct Resources {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    std::unique_ptr<remag::HartreePlan> plan;
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
void inventory(cudaGraph_t graph,HartreeCheck& info) {
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
extern "C" int remag_check_hartree(const remag::HartreeProblem* problem,const double* density,
        double* potential,remag::HartreeResult* result,int graph_mode,int corrupt_shape,HartreeCheck* info) {
    *info=HartreeCheck{};
    Resources resources;
    try {
        const auto& m=problem->mesh;
        if(m[0]<1||m[1]<1||m[2]<1||static_cast<long long>(m[0])*m[1]>INT_MAX/m[2]) return -1;
        int size=m[0]*m[1]*m[2];size_t bytes=size_t(size)*sizeof(double);
        check(cudaStreamCreateWithFlags(&resources.stream,cudaStreamNonBlocking));
        resources.plan=std::make_unique<remag::HartreePlan>(m[0],m[1],m[2],resources.stream);
        info->workspace_bytes=resources.plan->workspace_bytes();
        auto p=static_cast<remag::HartreeProblem*>(resources.allocate(sizeof(*problem)));
        auto rho=static_cast<double*>(resources.allocate(bytes));
        auto v=static_cast<double*>(resources.allocate(bytes));
        auto r=static_cast<remag::HartreeResult*>(resources.allocate(sizeof(*result)));
        auto staged=*problem;
        if(corrupt_shape) staged.mesh[0]=m[0]==1?2:m[0]-1;
        check(cudaMemcpyAsync(p,&staged,sizeof(staged),cudaMemcpyHostToDevice,resources.stream));
        check(cudaMemcpyAsync(rho,density,bytes,cudaMemcpyHostToDevice,resources.stream));
        check(cudaStreamSynchronize(resources.stream)); // explicit staging boundary
        auto apply=[&]() {
            int error=resources.plan->apply(p,rho,v,r);
            if(error) throw std::runtime_error("Hartree apply status "+std::to_string(error));
        };
        apply();check(cudaStreamSynchronize(resources.stream)); // initialization outside capture
        if(graph_mode) {
            check(cudaStreamBeginCapture(resources.stream,cudaStreamCaptureModeGlobal));
            try {apply();}
            catch(...) {cudaStreamEndCapture(resources.stream,&resources.graph);throw;}
            check(cudaStreamEndCapture(resources.stream,&resources.graph));
            inventory(resources.graph,*info);
            if(info->forbidden_nodes) throw std::runtime_error("Hartree graph contains host or non-device-copy nodes");
            check(cudaGraphInstantiate(&resources.executable,resources.graph,0));
        }
        for(int cycle=0;cycle<2;cycle++) {
            if(cycle) change_density<<<32,256,0,resources.stream>>>(size,rho);
            if(graph_mode) check(cudaGraphLaunch(resources.executable,resources.stream));
            else apply();
            info->iterations++;
        }
        check(cudaStreamSynchronize(resources.stream));
        check(cudaMemcpy(result,r,sizeof(*result),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(potential,v,bytes,cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"REMAG Hartree check: %s\n",error.what());return -2;
    }
}
