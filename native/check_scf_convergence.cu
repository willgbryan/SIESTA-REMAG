// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_convergence.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;
    Context() {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context() {
        if(executable) cudaGraphExecDestroy(executable);if(graph) cudaGraphDestroy(graph);
        for(void* p:memory) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t n,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);
        if(source) check(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice));return p;
    }
};
__global__ void select_inputs(const int* index,const remag::ScfConvergenceProblem* problems,const remag::ScfScalarsResult* inputs,
        remag::ScfConvergenceProblem* problem,remag::ScfScalarsResult* input) {*problem=problems[*index];*input=inputs[*index];}
__global__ void pack(int* index,const remag::ScfConvergenceResult* result,remag::ScfConvergenceResult* outputs) {
    outputs[*index]=*result;++*index;
}
}
struct Check {size_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations;};
extern "C" int remag_check_scf_convergence(int cycles,const remag::ScfConvergenceProblem* problems,
        const remag::ScfScalarsResult* inputs,remag::ScfConvergenceResult* outputs,int capture,Check* info) {
    try {
        if(cycles<1||!problems||!inputs||!outputs||!info) return -1;
        *info={};Context c;remag::ScfConvergencePlan plan(c.stream);
        info->workspace_bytes=plan.workspace_bytes();
        auto p=c.allocate<remag::ScfConvergenceProblem>(cycles,problems);
        auto s=c.allocate<remag::ScfScalarsResult>(cycles,inputs);
        auto output=c.allocate<remag::ScfConvergenceResult>(cycles);
        auto problem=c.allocate<remag::ScfConvergenceProblem>(1);
        auto input=c.allocate<remag::ScfScalarsResult>(1);
        auto result=c.allocate<remag::ScfConvergenceResult>(1);
        int zero=0;auto index=c.allocate<int>(1,&zero);
        check(cudaStreamSynchronize(c.stream));
        auto apply=[&] {
            select_inputs<<<1,1,0,c.stream>>>(index,p,s,problem,input);check(cudaGetLastError());
            check(static_cast<cudaError_t>(plan.apply(problem,input,result)));
            pack<<<1,1,0,c.stream>>>(index,result,output);check(cudaGetLastError());
        };
        if(capture) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();
            check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));
            std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));
            info->graph_nodes=static_cast<int>(n);
            for(auto node:nodes) {cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel) info->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,nullptr,nullptr,0));
        }
        for(int i=0;i<cycles;i++) {if(capture) check(cudaGraphLaunch(c.executable,c.stream));else apply();}
        check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(outputs,output,cycles*sizeof(*outputs),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&info->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"SCF convergence check: %s\n",e.what());return -1;}
}
