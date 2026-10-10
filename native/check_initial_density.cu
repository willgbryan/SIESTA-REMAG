// SPDX-License-Identifier: GPL-3.0-or-later
#include "initial_density.hpp"
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;std::vector<void*> buffers;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
        for(auto p:buffers)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* buffer(size_t n,const T* host=nullptr){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));buffers.push_back(p);
        if(host)check(cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream));return p;}
};
}
extern "C" int remag_check_initial_density(int capacity,int spins,int captured,int inplace,int dependency,
        const remag::InitialDensityProblem* p,const double* overlap,const double* density,
        double* output,double* energy,remag::InitialDensityResult* result,int* graph_info) {
    try {
        if(!p||!overlap||!density||!output||!energy||!result||!graph_info||capacity<1||capacity>1000000||
           (spins!=1&&spins!=2&&spins!=4&&spins!=8)||(captured!=0&&captured!=1)||(inplace!=0&&inplace!=1))return -1;
        Context c;auto problem=c.buffer(1,p);auto status=c.buffer(1,&dependency);auto s=c.buffer(capacity,overlap);
        auto d=c.buffer(size_t(capacity)*spins,density);auto out=inplace?d:c.buffer<double>(size_t(capacity)*spins);
        auto e=c.buffer<double>(size_t(capacity)*std::min(spins,4));auto r=c.buffer<remag::InitialDensityResult>(1);
        if(!inplace)check(cudaMemsetAsync(out,0xff,size_t(capacity)*spins*8,c.stream));
        check(cudaMemsetAsync(e,0xff,size_t(capacity)*std::min(spins,4)*8,c.stream));check(cudaStreamSynchronize(c.stream));
        auto apply=[&]{check(remag::initial_density_device(capacity,spins,problem,status,s,d,out,e,r,c.stream));};
        graph_info[0]=graph_info[1]=0;
        if(captured) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));
            graph_info[0]=n;for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)graph_info[1]++;}
            check(cudaGraphInstantiate(&c.exec,c.graph,0));check(cudaGraphLaunch(c.exec,c.stream));
        }else apply();
        check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(output,out,size_t(capacity)*spins*8,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(energy,e,size_t(capacity)*std::min(spins,4)*8,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(result,r,sizeof(*result),cudaMemcpyDeviceToHost));return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Initial density probe: %s\n",e.what());return -1;}
}
