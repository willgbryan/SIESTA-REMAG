// SPDX-License-Identifier: GPL-3.0-or-later
// Isolated capability probe for the small standard solves in blocked Jacobi.
// This is not a generalized eigensolver or an SCF integration.
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
void check(cusolverStatus_t code) {
    if(code!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER status "+std::to_string(int(code)));
}
struct Context {
    cudaStream_t stream=nullptr;cusolverDnHandle_t solver=nullptr;cusolverDnParams_t params=nullptr;
    cudaGraph_t captured=nullptr,outer=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;
    Context() {
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cusolverDnCreate(&solver));check(cusolverDnSetStream(solver,stream));
        check(cusolverDnCreateParams(&params));
    }
    ~Context() {
        if(executable) cudaGraphExecDestroy(executable);
        if(outer) cudaGraphDestroy(outer);if(captured) cudaGraphDestroy(captured);
        if(params) cusolverDnDestroyParams(params);if(solver) cusolverDnDestroy(solver);
        for(void* p:memory) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t n,const T* source=nullptr) {
        if(!n) return nullptr;
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);
        if(source) check(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice));return p;
    }
};
template<class T> __global__ void select_matrix(size_t size,const int* index,const T* inputs,T* a) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size) a[i]=inputs[size_t(*index)*size+i];
}
template<class T> __global__ void pack(size_t matrix_size,int values,int batch,const int* index,
        const T* a,const double* w,const int* info,T* vectors,double* eigenvalues,int* statuses) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,k=*index;
    if(i<matrix_size) vectors[k*matrix_size+i]=a[i];
    if(i<size_t(values)) eigenvalues[k*values+i]=w[i];
    if(i<size_t(batch)) statuses[k*batch+i]=info[i];
}
__global__ void advance(int* index,const int* info,int batch,int cycles,
        cudaGraphConditionalHandle handle,int conditional) {
    bool valid=true;for(int i=0;i<batch;i++) valid=valid&&info[i]==0;
    ++*index;
    if(conditional) cudaGraphSetConditional(handle,valid&&*index<cycles?1:0);
}
void inspect(cudaGraph_t graph,int& count,int& forbidden) {
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);
    check(cudaGraphGetNodes(graph,nodes.data(),&n));
    for(auto node:nodes) {
        cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));count++;
        if(type==cudaGraphNodeTypeGraph) {cudaGraph_t child;check(cudaGraphChildGraphNodeGetGraph(node,&child));inspect(child,count,forbidden);}
        else if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms p{};check(cudaGraphMemcpyNodeGetParams(node,&p));
            if(p.kind!=cudaMemcpyDeviceToDevice) forbidden++;
        } else if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset) forbidden++;
    }
}
}
struct SmallEigenCheck {size_t device_workspace,host_workspace;int graph_nodes,forbidden_nodes,iterations,stage;};
template<class T> int run(int n,int batch,int cycles,int capture,int workspace_fill,
        const void* matrices,void* vectors,double* values,int* statuses,SmallEigenCheck* report) {
    try {
        if(n<1||n>64||batch<1||batch>32||cycles<2||cycles>8||workspace_fill < -1||workspace_fill>255||
           !matrices||!vectors||!values||!statuses||!report) return -1;
        *report={};Context c;report->stage=1;
        size_t size=size_t(n)*n*batch;int count=n*batch,zero=0;
        auto original=c.allocate<T>(size*cycles,static_cast<const T*>(matrices));
        auto a=c.allocate<T>(size);auto w=c.allocate<double>(count);auto info=c.allocate<int>(batch);
        auto index=c.allocate<int>(1,&zero);auto output=c.allocate<T>(size*cycles);
        auto eigenvalues=c.allocate<double>(size_t(count)*cycles);auto results=c.allocate<int>(size_t(batch)*cycles);
        constexpr auto dtype=(std::is_same_v<T,double>) ? CUDA_R_64F : CUDA_C_64F;
        check(cusolverDnXsyevBatched_bufferSize(c.solver,c.params,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
            n,dtype,a,n,CUDA_R_64F,w,dtype,&report->device_workspace,&report->host_workspace,batch));
        auto work=c.allocate<unsigned char>(report->device_workspace);
        std::vector<unsigned char> host_work(report->host_workspace);
        auto solve=[&] {
            select_matrix<<<(size+255)/256,256,0,c.stream>>>(size,index,original,a);check(cudaGetLastError());
            // Compare the default workspace contract with explicit initialization.
            // The latter is a device memset and remains visible in captured graphs.
            if(workspace_fill>=0&&report->device_workspace)
                check(cudaMemsetAsync(work,workspace_fill,report->device_workspace,c.stream));
            check(cusolverDnXsyevBatched(c.solver,c.params,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
                n,dtype,a,n,CUDA_R_64F,w,dtype,work,report->device_workspace,
                host_work.empty()?nullptr:host_work.data(),report->host_workspace,info,batch));
        };
        report->stage=2;solve();check(cudaStreamSynchronize(c.stream));
        std::vector<int> warm(batch);check(cudaMemcpy(warm.data(),info,batch*sizeof(int),cudaMemcpyDeviceToHost));
        for(int status:warm) if(status) throw std::runtime_error("Warmup solver info "+std::to_string(status));
        auto body=[&] {
            solve();pack<<<(size+255)/256,256,0,c.stream>>>(size,count,batch,index,a,w,info,output,eigenvalues,results);
            check(cudaGetLastError());
        };
        if(capture) {
            report->stage=3;check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
            try {body();} catch(...) {cudaStreamEndCapture(c.stream,&c.captured);throw;}
            check(cudaStreamEndCapture(c.stream,&c.captured));inspect(c.captured,report->graph_nodes,report->forbidden_nodes);
            report->stage=4;check(cudaGraphCreate(&c.outer,0));cudaGraphConditionalHandle handle;
            check(cudaGraphConditionalHandleCreate(&handle,c.outer,1,cudaGraphCondAssignDefault));
            cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
            params.conditional.handle=handle;params.conditional.type=cudaGraphCondTypeWhile;params.conditional.size=1;
            cudaGraphNode_t condition,child,finish;check(cudaGraphAddNode(&condition,c.outer,nullptr,0,&params));
            cudaGraph_t loop=params.conditional.phGraph_out[0];check(cudaGraphAddChildGraphNode(&child,loop,nullptr,0,c.captured));
            int enabled=1;void* arguments[]={&index,&info,&batch,&cycles,&handle,&enabled};
            cudaKernelNodeParams kernel{};kernel.func=reinterpret_cast<void*>(advance);
            kernel.gridDim=dim3(1);kernel.blockDim=dim3(1);kernel.kernelParams=arguments;
            check(cudaGraphAddKernelNode(&finish,loop,&child,1,&kernel));check(cudaGraphInstantiate(&c.executable,c.outer,0));
            report->stage=5;check(cudaGraphLaunch(c.executable,c.stream));
        } else {
            report->stage=5;
            for(int k=0;k<cycles;k++) {body();advance<<<1,1,0,c.stream>>>(index,info,batch,cycles,0,0);check(cudaGetLastError());}
        }
        check(cudaStreamSynchronize(c.stream));report->stage=6;
        check(cudaMemcpy(&report->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));
        if(report->iterations!=cycles) throw std::runtime_error("Device loop stopped at a failed eigensolve");
        check(cudaMemcpy(vectors,output,size*cycles*sizeof(T),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(values,eigenvalues,size_t(count)*cycles*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(statuses,results,size_t(batch)*cycles*sizeof(int),cudaMemcpyDeviceToHost));return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Small eigensolver probe: %s\n",e.what());return -1;}
}
extern "C" int remag_check_small_eigensolver(int n,int batch,int cycles,int complex_values,int capture,int workspace_fill,
        const void* matrices,void* vectors,double* values,int* statuses,SmallEigenCheck* report) {
    if(complex_values) return run<cuDoubleComplex>(n,batch,cycles,capture,workspace_fill,matrices,vectors,values,statuses,report);
    return run<double>(n,batch,cycles,capture,workspace_fill,matrices,vectors,values,statuses,report);
}
