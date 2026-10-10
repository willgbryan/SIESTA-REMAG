// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_control.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" void remag_activity_phase(int,int,int) __attribute__((weak));
extern "C" void remag_activity_finish() __attribute__((weak));
extern "C" void remag_activity_graph(void*) __attribute__((weak));
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
        if(source) check(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice));
        else check(cudaMemsetAsync(p,0,n*sizeof(T),stream));return p;
    }
};
__global__ void select(int n,int count,const int* index,const remag::ScfControlProblem* problems,
        const remag::ScfScalarsResult* scalars,const double* inputs,const double* residuals,
        remag::ScfControlProblem* p,remag::ScfScalarsResult* scalar,double* input,double* residual,double* output) {
    const int k=*index;if(k>=count) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {input[i]=inputs[size_t(k)*n+i];residual[i]=residuals[size_t(k)*n+i];output[i]=123.;}
    if(!i) {*p=problems[k];*scalar=scalars[k];}
}
__global__ void save(int n,int count,const int* index,const double* value,const remag::ScfControlResult* result,
        const remag::ScfControlStep* history,double* values,remag::ScfControlResult* results,remag::ScfControlStep* records) {
    const int k=*index;if(k>=count) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) values[size_t(k)*n+i]=value[i];
    if(!i) {results[k]=*result;if(result->records) records[k]=history[result->records-1];}
}
__global__ void advance(int count,int* index,int* truncated,const remag::ScfControlResult* result,cudaGraphConditionalHandle handle) {
    ++*index;
    if(result->continue_loop&&*index>=count) *truncated=1;
    cudaGraphSetConditional(handle,result->continue_loop&&*index<count?1:0);
}
void audit(cudaGraph_t graph,int& count,int& forbidden) {
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);
    check(cudaGraphGetNodes(graph,nodes.data(),&n));
    for(auto node:nodes) {
        count++;cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeGraph) {cudaGraph_t child;check(cudaGraphChildGraphNodeGetGraph(node,&child));audit(child,count,forbidden);}
        else if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeEmpty) forbidden++;
    }
}
}
struct Check {size_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations,truncated;};
extern "C" int remag_check_scf_control(int n,int mixer_history,int capacity,int count,int launches,
        const remag::ScfControlProblem* problems,const remag::ScfScalarsResult* scalars,const double* inputs,const double* residuals,
        double* outputs,remag::ScfControlResult* results,remag::ScfControlStep* records,Check* info) {
    try {
        if(n<1||n>4096||capacity<1||capacity>200||count<1||count>600||launches<1||launches>count||
           !problems||!scalars||!inputs||!residuals||!outputs||!results||!records||!info) return -1;
        *info={};Context c;remag::ScfControlPlan plan(n,mixer_history,capacity,c.stream);
        info->workspace_bytes=plan.workspace_bytes();
        auto ps=c.allocate<remag::ScfControlProblem>(count,problems);auto ss=c.allocate<remag::ScfScalarsResult>(count,scalars);
        auto xs=c.allocate<double>(size_t(n)*count,inputs);auto fs=c.allocate<double>(size_t(n)*count,residuals);
        auto values=c.allocate<double>(size_t(n)*count);auto rs=c.allocate<remag::ScfControlResult>(count);
        auto logs=c.allocate<remag::ScfControlStep>(count);
        auto p=c.allocate<remag::ScfControlProblem>(1);auto scalar=c.allocate<remag::ScfScalarsResult>(1);
        auto x=c.allocate<double>(n);auto f=c.allocate<double>(n);auto output=c.allocate<double>(n);
        auto r=c.allocate<remag::ScfControlResult>(1);auto history=c.allocate<remag::ScfControlStep>(capacity);
        auto index=c.allocate<int>(1);auto truncated=c.allocate<int>(1);
        check(cudaStreamSynchronize(c.stream));check(cudaGraphCreate(&c.graph,0));
        cudaGraphConditionalHandle handle;check(cudaGraphConditionalHandleCreate(&handle,c.graph,1,cudaGraphCondAssignDefault));
        cudaGraphNodeParams condition{};condition.type=cudaGraphNodeTypeConditional;condition.conditional.handle=handle;
        condition.conditional.type=cudaGraphCondTypeWhile;condition.conditional.size=1;cudaGraphNode_t loop;
        check(cudaGraphAddNode(&loop,c.graph,nullptr,0,&condition));auto body=condition.conditional.phGraph_out[0];
        cudaGraph_t captured=nullptr;
        check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
        select<<<(n+255)/256,256,0,c.stream>>>(n,count,index,ps,ss,xs,fs,p,scalar,x,f,output);check(cudaGetLastError());
        check(cudaStreamEndCapture(c.stream,&captured));cudaGraphNode_t selected;
        audit(captured,info->graph_nodes,info->forbidden_nodes);
        check(cudaGraphAddChildGraphNode(&selected,body,nullptr,0,captured));check(cudaGraphDestroy(captured));
        auto controlled=plan.append(c.graph,body,selected,p,scalar,x,f,output,r,history);
        check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
        save<<<(n+255)/256,256,0,c.stream>>>(n,count,index,output,r,history,values,rs,logs);check(cudaGetLastError());
        advance<<<1,1,0,c.stream>>>(count,index,truncated,r,handle);check(cudaGetLastError());
        check(cudaStreamEndCapture(c.stream,&captured));cudaGraphNode_t saved;
        audit(captured,info->graph_nodes,info->forbidden_nodes);
        check(cudaGraphAddChildGraphNode(&saved,body,&controlled,1,captured));check(cudaGraphDestroy(captured));
        // CUDA 12.6 has no generic conditional-node getter. Every captured
        // body was audited before insertion; the three additional nodes here
        // are the explicitly constructed outer while and its two children.
        info->graph_nodes+=plan.graph_nodes()+3;
        info->forbidden_nodes+=plan.forbidden_nodes();
        if(info->forbidden_nodes) throw std::runtime_error("Forbidden SCF control graph nodes");
        check(cudaGraphInstantiate(&c.executable,c.graph,0));
        if(remag_activity_graph) remag_activity_graph(c.executable);
        check(cudaGraphUpload(c.executable,c.stream));check(cudaStreamSynchronize(c.stream));
        if(remag_activity_phase) remag_activity_phase(1,1,1);
        for(int k=0;k<launches;k++) check(cudaGraphLaunch(c.executable,c.stream));
        check(cudaStreamSynchronize(c.stream));
        if(remag_activity_phase) remag_activity_phase(2,1,0);
        check(cudaMemcpy(outputs,values,size_t(n)*count*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(results,rs,count*sizeof(*results),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(records,logs,count*sizeof(*records),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&info->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&info->truncated,truncated,sizeof(int),cudaMemcpyDeviceToHost));
        if(remag_activity_phase) remag_activity_phase(3,1,0);
        if(remag_activity_finish) remag_activity_finish();return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"SCF control check: %s\n",e.what());
        if(remag_activity_finish) remag_activity_finish();return -1;
    }
}
