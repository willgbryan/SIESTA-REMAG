// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only staging and export. Every sequence input is staged before execution.
#include "pulay.hpp"
#include "pulay_math.hpp"
#include <climits>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

struct PulayCheck {uint64_t workspace_bytes;int graph_nodes,forbidden_nodes,iterations;};
namespace {
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
struct Resources {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::unique_ptr<remag::PulayPlan> plan;std::vector<void*> buffers;
    void* allocate(size_t n) {void* p=nullptr;check(cudaMalloc(&p,n));buffers.push_back(p);return p;}
    ~Resources() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);if(graph) cudaGraphDestroy(graph);
        plan.reset();for(auto p:buffers) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
};
__global__ void prepare(int size,int reset_before,const int* step,remag::PulayProblem* problem,
                        const double* source_x,const double* source_f,double* x,double* f) {
    const int cycle=*step;
    if(blockIdx.x==0&&threadIdx.x==0) problem->epoch=1+int(reset_before>0&&cycle+1>=reset_before);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        x[i]=source_x[size_t(cycle)*size+i];f[i]=source_f[size_t(cycle)*size+i];
    }
}
__global__ void pack(int size,const int* step,const double* next,const remag::PulayResult* result,
                     double* sequence,remag::PulayResult* records) {
    const int cycle=*step;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) sequence[size_t(cycle)*size+i]=next[i];
    if(!blockIdx.x&&!threadIdx.x) records[cycle]=*result;
}
__global__ void advance(int* step) {++*step;}
__global__ void coefficient_probe(const double* gram,const double* projection,double* output) {
    const double value=*gram;
    output[0]=value;output[1]=1./value;output[2]=__ddiv_rn(1.,value);
    double coefficient=123.;
    auto global=remag::pulay_coefficients(1,gram,projection,1.e-8,&coefficient);
    output[3]=global.status;output[4]=global.inverse_first;output[5]=coefficient;
    double local_gram[225]{},local_projection[15]{};local_gram[0]=value;local_projection[0]=*projection;
    coefficient=123.;auto local=remag::pulay_coefficients(1,local_gram,local_projection,1.e-8,&coefficient);
    output[6]=local.status;output[7]=local.inverse_first;output[8]=coefficient;
}
void inventory(cudaGraph_t graph,PulayCheck& info) {
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);
    check(cudaGraphGetNodes(graph,nodes.data(),&n));
    for(auto node:nodes) {
        info.graph_nodes++;cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel) info.forbidden_nodes++;
    }
}
}
extern "C" int remag_check_pulay_coefficient_probe(double gram,double projection,double* output) {
    Resources r;
    try {
        check(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
        auto g=static_cast<double*>(r.allocate(sizeof(double)));
        auto p=static_cast<double*>(r.allocate(sizeof(double)));
        auto out=static_cast<double*>(r.allocate(9*sizeof(double)));
        check(cudaMemcpyAsync(g,&gram,sizeof(double),cudaMemcpyHostToDevice,r.stream));
        check(cudaMemcpyAsync(p,&projection,sizeof(double),cudaMemcpyHostToDevice,r.stream));
        coefficient_probe<<<1,1,0,r.stream>>>(g,p,out);check(cudaGetLastError());
        check(cudaStreamSynchronize(r.stream));check(cudaMemcpy(output,out,9*sizeof(double),cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"REMAG Pulay coefficient probe: %s\n",e.what());return -2;}
}
extern "C" int remag_check_pulay(const remag::PulayProblem* problem,int cycles,int reset_before,
        const double* xin,const double* residual,double* next,remag::PulayResult* results,int graph_mode,PulayCheck* info) {
    *info=PulayCheck{};Resources r;
    try {
        if(problem->size<1||cycles<1||cycles>10000||reset_before<0||reset_before>cycles) return -1;
        check(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
        r.plan=std::make_unique<remag::PulayPlan>(problem->size,problem->history,r.stream);
        info->workspace_bytes=r.plan->workspace_bytes();
        const size_t bytes=size_t(problem->size)*sizeof(double);
        auto p=static_cast<remag::PulayProblem*>(r.allocate(sizeof(remag::PulayProblem)));
        auto all_x=static_cast<double*>(r.allocate(bytes*cycles));auto all_f=static_cast<double*>(r.allocate(bytes*cycles));
        auto x=static_cast<double*>(r.allocate(bytes));auto f=static_cast<double*>(r.allocate(bytes));
        auto output=static_cast<double*>(r.allocate(bytes));auto sequence=static_cast<double*>(r.allocate(bytes*cycles));
        auto result=static_cast<remag::PulayResult*>(r.allocate(sizeof(remag::PulayResult)));
        auto records=static_cast<remag::PulayResult*>(r.allocate(sizeof(*result)*cycles));
        auto step=static_cast<int*>(r.allocate(sizeof(int)));check(cudaMemsetAsync(step,0,sizeof(int),r.stream));
        check(cudaMemcpyAsync(p,problem,sizeof(*problem),cudaMemcpyHostToDevice,r.stream));
        check(cudaMemcpyAsync(all_x,xin,bytes*cycles,cudaMemcpyHostToDevice,r.stream));
        check(cudaMemcpyAsync(all_f,residual,bytes*cycles,cudaMemcpyHostToDevice,r.stream));
        check(cudaMemcpyAsync(output,next,bytes,cudaMemcpyHostToDevice,r.stream));
        check(cudaStreamSynchronize(r.stream));
        auto apply=[&]() {
            prepare<<<32,256,0,r.stream>>>(problem->size,reset_before,step,p,all_x,all_f,x,f);
            check(static_cast<cudaError_t>(r.plan->apply(p,x,f,output,result)));
            pack<<<32,256,0,r.stream>>>(problem->size,step,output,result,sequence,records);
            advance<<<1,1,0,r.stream>>>(step);check(cudaGetLastError());
        };
        if(graph_mode) {
            check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeGlobal));
            try {apply();} catch(...) {cudaStreamEndCapture(r.stream,&r.graph);throw;}
            check(cudaStreamEndCapture(r.stream,&r.graph));inventory(r.graph,*info);
            if(info->forbidden_nodes) throw std::runtime_error("Pulay graph contains a non-kernel node");
            check(cudaGraphInstantiate(&r.executable,r.graph,0));
        }
        for(int i=0;i<cycles;i++) {
            if(graph_mode) check(cudaGraphLaunch(r.executable,r.stream));else apply();
            info->iterations++;
        }
        check(cudaStreamSynchronize(r.stream));
        check(cudaMemcpy(next,sequence,bytes*cycles,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(results,records,sizeof(*results)*cycles,cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"REMAG Pulay check: %s\n",e.what());return -2;}
}
