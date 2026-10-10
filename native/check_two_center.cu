// SPDX-License-Identifier: GPL-3.0-or-later
#include "two_center_state.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct TwoCenterCheck {size_t allocated,staged,exported,poisoned;int graph_nodes,forbidden_nodes,stage;};
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t allocated=0,staged=0;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* source){T* p=nullptr;size_t bytes=(n?n:1)*sizeof(T);
        check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;
        if(n&&source){check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);}return p;}
};
}
extern "C" int remag_check_two_center_comparison(int count,int captured,int initial,const double* actual,
        const double* reference,remag::TwoCenterComparison* result,int* status,TwoCenterCheck* report) {
    try {
        if(count<1||count>1000000||(captured!=0&&captured!=1)||!actual||!reference||!result||!status||!report)return -1;
        *report={};report->stage=1;Context c;remag::TwoCenterComparison zero{};
        auto a=c.upload(4*size_t(count),actual);auto b=c.upload(4*size_t(count),reference);
        auto out=c.upload(1,&zero);auto code=c.upload(1,&initial);check(cudaStreamSynchronize(c.stream));
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::compare_two_center_device(count,a,b,out,code,c.stream));};
        if(captured) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));report->graph_nodes=n;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphLaunch(c.executable,c.stream));
        }else apply();
        report->stage=3;check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(result,out,sizeof(*result),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,4,cudaMemcpyDeviceToHost));
        report->exported=sizeof(*result)+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Two-center comparison probe: %s\n",e.what());return -1;}
}
// config: registry, radials, terms, rows, cols, bases[2], registry bounds[4].
extern "C" int remag_check_two_center(const int* config,double delta,double pi,int count,int captured,int cycles,int initial,
        const int* ind,const int* offsets,const int* ilm,const int* ri,const double* cutoffs,const double* radial,const double* coefficients,
        const int* pairs,const double* xyz,double* result,int* status,TwoCenterCheck* report) {
    try {
        if(!config||!ind||!offsets||!ilm||!ri||!cutoffs||!radial||!coefficients||!pairs||!xyz||!result||!status||!report)return -1;
        *report={};const int nf=config[0],nr=config[1],ny=config[2],rows=config[3],cols=config[4];
        if(nf<1||nf>1000||nr<0||nr>10000||ny<0||ny>1000000||rows<1||cols<1||size_t(rows)*cols>1000000||
           count<1||count>1000000||(captured!=0&&captured!=1)||cycles<1||cycles>3)return -1;
        report->stage=1;Context c;
        remag::TwoCenterView t{nf,nr,ny,rows,cols,config[5],config[6],config[7],config[8],config[9],config[10],delta,pi,
            c.upload(nf+1,ind),c.upload(size_t(rows)*cols+1,offsets),c.upload(ny,ilm),c.upload(ny,ri),
            c.upload(nf,cutoffs),c.upload(size_t(nr)*2050,radial),c.upload(ny,coefficients)};
        auto dpairs=c.upload(2*size_t(count),pairs);auto dxyz=c.upload(3*size_t(count),xyz);auto out=c.upload<double>(4*size_t(count),nullptr);auto code=c.upload(1,&initial);
        check(cudaStreamSynchronize(c.stream));report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::evaluate_two_center_device(t,count,dpairs,dxyz,out,code,c.stream));};
        if(captured){
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));report->graph_nodes=n;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));
        }
        report->stage=3;
        for(int i=0;i<cycles;i++){
            check(cudaMemsetAsync(out,255,32*size_t(count),c.stream));report->poisoned+=32*size_t(count);
            if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();
        }
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(result,out,32*size_t(count),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,4,cudaMemcpyDeviceToHost));
        report->exported=32*size_t(count)+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Two-center probe: %s\n",e.what());return -1;}
}
