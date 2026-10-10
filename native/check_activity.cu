// SPDX-License-Identifier: GPL-3.0-or-later
// Known array/scalar traffic validates the optional CUPTI observer independently
// of SIESTA and custom bridge counters. This is not an SCF algorithm.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" void remag_activity_phase(int,int,int) __attribute__((weak));
extern "C" void remag_activity_finish() __attribute__((weak));
extern "C" void remag_activity_graph(void*) __attribute__((weak));
void check(cudaError_t code) {
    if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__global__ void increment(double* a,int n,int* status,int iteration) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) a[i]+=1.;
    if(i==0) *status=iteration;
}
__global__ void reset_loop(int* index) {*index=0;}
__global__ void advance_loop(int* index,cudaGraphConditionalHandle handle) {
    ++*index;cudaGraphSetConditional(handle,*index<2?1:0);
}
int main(int argc,char** argv) {
    if(!remag_activity_phase||!remag_activity_finish) return 2;
    const bool unique_scope=argc==2&&std::strcmp(argv[1],"--conditional-scope")==0;
    const bool conditional=unique_scope||(argc==2&&std::strcmp(argv[1],"--conditional")==0);
    const bool graph_mode=conditional||(argc==2&&std::strcmp(argv[1],"--graph")==0);
    if(argc!=1&&!graph_mode) return 2;
    try {
        constexpr int n=4096;
        std::vector<double> h(n);
        for(int i=0;i<n;i++) h[i]=i;
        double *a=nullptr,*b=nullptr;
        int* status=nullptr;
        check(cudaMalloc(&a,n*sizeof(double)));check(cudaMalloc(&b,n*sizeof(double)));
        check(cudaMalloc(&status,sizeof(int)));
        check(cudaMemcpy(a,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        int observed=0;
        int* host_status=&observed;
        cudaStream_t stream=nullptr;
        cudaGraph_t graph=nullptr;
        int* loop_index=nullptr;
        cudaGraphExec_t executable=nullptr;
        if(graph_mode) {
            check(cudaMallocHost(&host_status,sizeof(int)));
            *host_status=0;
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            cudaGraphConditionalHandle handle=0;
            cudaGraph_t captured=nullptr,body=nullptr;
            if(conditional) {
                check(cudaMalloc(&loop_index,sizeof(int)));check(cudaGraphCreate(&graph,0));
                check(cudaGraphConditionalHandleCreate(&handle,graph,1,cudaGraphCondAssignDefault));
                void* args[]={&loop_index};cudaKernelNodeParams reset{};
                reset.func=reinterpret_cast<void*>(reset_loop);reset.gridDim=dim3(1);reset.blockDim=dim3(1);reset.kernelParams=args;
                cudaGraphNode_t start,condition;check(cudaGraphAddKernelNode(&start,graph,nullptr,0,&reset));
                cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=handle;
                p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;
                check(cudaGraphAddNode(&condition,graph,&start,1,&p));body=p.conditional.phGraph_out[0];
            }
            // Capture belongs to initialization; both executions belong to
            // later SCF regions. A capture-origin mapping must fail this test.
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            increment<<<n/256,256,0,stream>>>(a,n,status,1);check(cudaGetLastError());
            check(cudaMemcpyAsync(b,a,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
            if(conditional) {advance_loop<<<1,1,0,stream>>>(loop_index,handle);check(cudaGetLastError());}
            else check(cudaMemcpyAsync(host_status,status,sizeof(int),cudaMemcpyDeviceToHost,stream));
            check(cudaStreamEndCapture(stream,&captured));
            if(conditional) {
                cudaGraphNode_t child;check(cudaGraphAddChildGraphNode(&child,body,nullptr,0,captured));
                check(cudaGraphDestroy(captured));
            } else graph=captured;
            check(cudaGraphInstantiate(&executable,graph,0));
            if(remag_activity_graph) remag_activity_graph(executable);
            check(cudaGraphUpload(executable,stream));
        }
        check(cudaDeviceSynchronize());
        remag_activity_phase(1,1,1);
        if(graph_mode) {
            check(cudaGraphLaunch(executable,stream));
            if(conditional) check(cudaMemcpyAsync(host_status,status,sizeof(int),cudaMemcpyDeviceToHost,stream));
        }
        else {
            increment<<<n/256,256>>>(a,n,status,1);check(cudaGetLastError());
            check(cudaMemcpy(b,a,n*sizeof(double),cudaMemcpyDeviceToDevice));
            check(cudaMemcpy(host_status,status,sizeof(int),cudaMemcpyDeviceToHost));
        }
        check(cudaDeviceSynchronize());
        if(*host_status!=1) throw std::runtime_error("first scalar mismatch");
        if(!unique_scope) remag_activity_phase(1,1,2);
        if(graph_mode) {
            check(cudaGraphLaunch(executable,stream));
            if(conditional) check(cudaMemcpyAsync(host_status,status,sizeof(int),cudaMemcpyDeviceToHost,stream));
        }
        else {
            increment<<<n/256,256>>>(b,n,status,2);check(cudaGetLastError());
            check(cudaMemcpy(host_status,status,sizeof(int),cudaMemcpyDeviceToHost));
        }
        check(cudaDeviceSynchronize());
        if(*host_status!=(graph_mode?1:2)) throw std::runtime_error("second scalar mismatch");
        remag_activity_phase(2,1,0);
        check(cudaMemcpy(h.data(),b,n*sizeof(double),cudaMemcpyDeviceToHost));
        for(int i=0;i<n;i++) if(h[i]!=i+(conditional?4.:2.)) throw std::runtime_error("array mismatch");
        check(cudaFree(a));check(cudaFree(b));check(cudaFree(status));
        if(graph_mode) {
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
            check(cudaStreamDestroy(stream));check(cudaFreeHost(host_status));
            if(loop_index) check(cudaFree(loop_index));
        }
        remag_activity_phase(3,1,0);
        remag_activity_finish();
        std::printf("Known array/scalar activity completed (graph=%d, conditional=%d); trace must be checked separately.\n",int(graph_mode),int(conditional));
        return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"Activity check: %s\n",error.what());
        return 1;
    }
}
