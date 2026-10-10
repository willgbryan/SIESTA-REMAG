// SPDX-License-Identifier: GPL-3.0-or-later
#include "soc_projector.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct SocProjectorCheck {size_t allocated,staged,exported,poisoned;int graph_nodes,forbidden_nodes,stage;};
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t staged=0,allocated=0;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* source){T* p=nullptr;size_t bytes=(n?n:1)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;
        if(n&&source){check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);}return p;}
};
}
// config = l, split, query count. Right derivatives deliberately absent.
extern "C" int remag_check_soc_projector(const int* config,int captured,int cycles,int initial,
        const double* epsilon,const double* left,const double* right,const double* gradient,
        double* result,int* status,SocProjectorCheck* report) {
    try {
        if(!config||!epsilon||!left||!right||!gradient||!result||!status||!report)return -1;
        *report={};const int l=config[0],split=config[1],n=config[2];
        if(n<0||n>1000000||l<1||l>5||(split!=0&&split!=1)||cycles<1||cycles>3||(captured!=0&&captured!=1))return -1;
        report->stage=1;Context c;const size_t width=2*l+1,amplitudes=2*width*n,nout=33*size_t(n);
        remag::SocProjectorView view{n,l,split,c.upload(2,epsilon),c.upload(amplitudes,left),c.upload(amplitudes,right),c.upload(3*amplitudes,gradient)};
        auto out=c.upload<double>(nout,nullptr);auto code=c.upload(1,&initial);check(cudaStreamSynchronize(c.stream));
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::soc_projector_device(view,out,code,c.stream));};
        if(captured){check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(c.graph,nodes.data(),&count));report->graph_nodes=count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));}
        report->stage=3;
        for(int i=0;i<cycles;i++) {
            check(cudaMemsetAsync(out,0xff,nout*sizeof(double),c.stream));report->poisoned+=nout*sizeof(double);
            if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();
        }
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        if(nout)check(cudaMemcpy(result,out,nout*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(status,code,4,cudaMemcpyDeviceToHost));report->exported=nout*sizeof(double)+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"SOC projector probe: %s\n",e.what());return -1;}
}
