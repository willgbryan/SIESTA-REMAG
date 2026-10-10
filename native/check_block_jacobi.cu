// SPDX-License-Identifier: GPL-3.0-or-later
#include "block_jacobi.hpp"
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

extern "C" void remag_activity_phase(int,int,int) __attribute__((weak));
extern "C" void remag_activity_finish() __attribute__((weak));
extern "C" void remag_activity_graph(void*) __attribute__((weak));
namespace {
void check(cudaError_t c) {if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> allocations;
    Context() {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context() {
        if(executable) cudaGraphExecDestroy(executable);if(graph) cudaGraphDestroy(graph);
        for(void* p:allocations) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t count,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));allocations.push_back(p);
        if(source) check(cudaMemcpy(p,source,count*sizeof(T),cudaMemcpyHostToDevice));return p;
    }
};
template<class T> __global__ void select(int size,const int* index,const T* all_h,const T* all_s,T* h,T* s) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) {h[i]=all_h[size_t(*index)*size+i];s[i]=all_s[size_t(*index)*size+i];}
}
template<class T> __global__ void identity(int n,T* h,T* s) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*n) return;
    double v=i%n==i/n?1.:0.;
    if constexpr(sizeof(T)==sizeof(double)) h[i]=s[i]=v;
    else h[i]=s[i]=make_cuDoubleComplex(v,0.);
}
template<class T> __global__ void pack(int n,const int* index,const T* z,const double* w,const remag::BlockJacobiResult* result,
        T* all_z,double* all_w,remag::BlockJacobiResult* all_results) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;size_t k=*index;
    if(i<n*n) all_z[k*n*n+i]=z[i];if(i<n) all_w[k*n+i]=w[i];if(i==0) all_results[k]=*result;
}
__global__ void advance(int* index,int cycles,int nested,cudaGraphConditionalHandle handle) {
    ++*index;
    // Each input is an independent generalized problem, so this diagnostic
    // deliberately continues after an error to verify next-problem reset.
    // A future SCF consumer must instead stop on a failed electronic solve.
    if(nested) cudaGraphSetConditional(handle,*index<cycles?1:0);
}
template<class T> cudaGraphNode_t select_node(cudaGraph_t graph,int n,int* index,const T* all_h,const T* all_s,T* h,T* s) {
    int size=n*n;void* args[]={&size,&index,&all_h,&all_s,&h,&s};cudaKernelNodeParams p{};
    p.func=reinterpret_cast<void*>(select<T>);p.gridDim=dim3((size+255)/256);p.blockDim=dim3(256);p.kernelParams=args;
    cudaGraphNode_t node;check(cudaGraphAddKernelNode(&node,graph,nullptr,0,&p));return node;
}
}
struct BlockCheck {
    size_t workspace_bytes,host_workspace_bytes;
    int graph_nodes,forbidden_nodes,iterations,stage;
};
template<class T> int run(int n,int cycles,int maximum,int nested,double tolerance,const void* host_h,const void* host_s,
                         void* host_z,double* host_w,remag::BlockJacobiResult* host_results,BlockCheck* report) {
    try {
        if(n<1||n>2048||cycles<1||cycles>8||!host_h||!host_s||!host_z||!host_w||!host_results||!report||
           nested<0||nested>3) return -1;
        *report={};report->stage=1;Context c;size_t size=size_t(n)*n;int zero=0;
        auto all_h=c.allocate<T>(size*cycles,static_cast<const T*>(host_h));
        auto all_s=c.allocate<T>(size*cycles,static_cast<const T*>(host_s));
        auto h=c.allocate<T>(size),s=c.allocate<T>(size),z=c.allocate<T>(size),all_z=c.allocate<T>(size*cycles);
        auto w=c.allocate<double>(n),all_w=c.allocate<double>(size_t(n)*cycles);
        auto result=c.allocate<remag::BlockJacobiResult>(1),results=c.allocate<remag::BlockJacobiResult>(cycles);
        auto index=c.allocate<int>(1,&zero);
        check(cudaMemsetAsync(w,0xff,n*sizeof(double),c.stream));
        remag::BlockJacobiPlan plan(n,sizeof(T)!=sizeof(double),maximum,tolerance,c.stream);
        report->workspace_bytes=plan.workspace_bytes();report->host_workspace_bytes=plan.host_workspace_bytes();
        report->stage=2;
        if(nested==3) {
            identity<<<(size+255)/256,256,0,c.stream>>>(n,h,s);check(cudaGetLastError());plan.warmup(h,s,result);
        } else plan.warmup(all_h,all_s,result);
        report->stage=3;check(cudaGraphCreate(&c.graph,0));cudaGraph_t body=c.graph;
        cudaGraphConditionalHandle handle=0;
        if(nested) {
            check(cudaGraphConditionalHandleCreate(&handle,c.graph,1,cudaGraphCondAssignDefault));
            cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=handle;
            p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;cudaGraphNode_t node;
            check(cudaGraphAddNode(&node,c.graph,nullptr,0,&p));body=p.conditional.phGraph_out[0];
        }
        cudaGraph_t outer_body=body;cudaGraphNode_t gate=nullptr;
        if(nested>=2) {
            cudaGraphConditionalHandle enabled;check(cudaGraphConditionalHandleCreate(&enabled,c.graph,1,cudaGraphCondAssignDefault));
            cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=enabled;
            p.conditional.type=cudaGraphCondTypeIf;p.conditional.size=1;
            check(cudaGraphAddNode(&gate,body,nullptr,0,&p));body=p.conditional.phGraph_out[0];
        }
        auto selected=select_node(body,n,index,all_h,all_s,h,s);
        cudaGraphNode_t solved;
        try {solved=plan.append(c.graph,body,selected,h,s,z,w,result);}
        catch(...) {
            report->graph_nodes=plan.graph_nodes();report->forbidden_nodes=plan.forbidden_nodes();throw;
        }
        void* pack_args[]={&n,&index,&z,&w,&result,&all_z,&all_w,&results};cudaKernelNodeParams packed{};
        packed.func=reinterpret_cast<void*>(pack<T>);packed.gridDim=dim3((size+255)/256);packed.blockDim=dim3(256);packed.kernelParams=pack_args;
        cudaGraphNode_t packed_node;check(cudaGraphAddKernelNode(&packed_node,body,&solved,1,&packed));
        void* advance_args[]={&index,&cycles,&nested,&handle};cudaKernelNodeParams advanced{};
        advanced.func=reinterpret_cast<void*>(advance);advanced.gridDim=dim3(1);advanced.blockDim=dim3(1);advanced.kernelParams=advance_args;
        cudaGraphNode_t advanced_node,dependency=gate?gate:packed_node;
        check(cudaGraphAddKernelNode(&advanced_node,outer_body,&dependency,1,&advanced));
        report->graph_nodes=plan.graph_nodes()+3+(nested!=0)+(nested>=2);report->forbidden_nodes=plan.forbidden_nodes();
        report->stage=4;check(cudaGraphInstantiate(&c.executable,c.graph,0));
        if(remag_activity_graph) remag_activity_graph(c.executable);
        check(cudaGraphUpload(c.executable,c.stream));check(cudaStreamSynchronize(c.stream));
        report->stage=5;
        // The measured region covers all three pre-staged problems, including
        // the first. Host replay and device-loop modes use the same boundary.
        if(remag_activity_phase) remag_activity_phase(1,1,1);
        if(nested) {
            check(cudaGraphLaunch(c.executable,c.stream));
        } else for(int i=0;i<cycles;i++) {
            check(cudaGraphLaunch(c.executable,c.stream));
        }
        check(cudaStreamSynchronize(c.stream));
        if(remag_activity_phase) remag_activity_phase(2,1,0);
        report->stage=6;
        check(cudaMemcpy(&report->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_z,all_z,size*cycles*sizeof(T),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_w,all_w,size_t(n)*cycles*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(host_results,results,cycles*sizeof(remag::BlockJacobiResult),cudaMemcpyDeviceToHost));
        if(std::getenv("REMAG_REPLAY_DIAGNOSTICS")) {
            remag::BlockJacobiLocalFailure failure{};
            check(cudaMemcpy(&failure,plan.local_failure(),sizeof(failure),cudaMemcpyDeviceToHost));
            if(FILE* f=std::fopen("local-failure.json","w")) {
                std::fprintf(f,"{\"info\":%d,\"pair\":%d,\"round\":%d,\"sweep\":%d,\"dimension\":%d}\n",failure.info,failure.pair,failure.round,failure.sweep,plan.local_dimension());std::fclose(f);
            }
            if(failure.info) {
                const size_t count=size_t(plan.local_dimension())*plan.local_dimension();
                std::vector<T> matrix(count);check(cudaMemcpy(matrix.data(),plan.failed_local_input(),count*sizeof(T),cudaMemcpyDeviceToHost));
                if(FILE* f=std::fopen("remag-failed-local.bin","wb")) {std::fwrite(matrix.data(),sizeof(T),count,f);std::fclose(f);}
            }
        }
        if(remag_activity_phase) remag_activity_phase(3,1,0);
        if(remag_activity_finish) remag_activity_finish();
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"Block Jacobi probe: %s\n",e.what());
        if(remag_activity_finish) remag_activity_finish();return -1;
    }
}
extern "C" int remag_check_block_jacobi(int n,int cycles,int complex_values,int maximum,int nested,double tolerance,
        const void* h,const void* s,void* z,double* w,remag::BlockJacobiResult* results,BlockCheck* report) {
    if(complex_values) return run<cuDoubleComplex>(n,cycles,maximum,nested,tolerance,h,s,z,w,results,report);
    return run<double>(n,cycles,maximum,nested,tolerance,h,s,z,w,results,report);
}
