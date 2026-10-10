// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_compact.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct NonlocalCompactCheck {size_t allocated,staged,exported,poisoned;int graph_nodes,forbidden_nodes,stage;};
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
// config: neighbors,centers,blocks,queries,atoms,rows,columns,capacity,
// components,spinor,soc,split,cycles. Host descriptor arrays use packed ints.
extern "C" int remag_check_nonlocal_compact(const int* config,double volume,int captured,int initial,
        const int* centers,const int* neighbors,const int* indices,const double* epsilon,
        const int* num,const int* ptr,const int* list,const int* lookup,
        const double* integrals,const double* xyz,const double* density,const double* initial_values,
        double* result,int* status,NonlocalCompactCheck* report) {
    try {
        if(!config||!centers||!neighbors||!indices||!epsilon||!num||!ptr||!list||!lookup||!integrals||!xyz||!density||!initial_values||!result||!status||!report)return -1;
        *report={};const int nn=config[0],nc=config[1],nb=config[2],nq=config[3],na=config[4],nr=config[5],no=config[6],cap=config[7],components=config[8],cycles=config[12];
        if(nn<0||nn>1000000||nc<0||nc>1000000||nb<0||nb>1000000||nq<0||nq>10000000||na<1||na>10000||nr<1||nr>10000||no<1||no>10000||cap<1||cap>1000000||
           (components!=1&&components!=2&&components!=4&&components!=8)||cycles<1||cycles>3||(captured!=0&&captured!=1))return -1;
        std::vector<remag::NonlocalCenter> hc(nc);std::vector<remag::NonlocalNeighbor> hn(nn);std::vector<remag::NonlocalBlock> hb(nb);
        for(int i=0;i<nc;i++){auto a=centers+7*i;hc[i]={a[0],a[1],a[2],a[3],a[4],a[5],a[6]};}
        for(int i=0;i<nn;i++){auto a=neighbors+4*i;hn[i]={a[0],a[1],a[2],a[3]};}
        for(int i=0;i<nb;i++)hb[i]={indices[2*i],indices[2*i+1],{epsilon[2*i],epsilon[2*i+1]}};
        report->stage=1;Context context;
        remag::NonlocalCompactView v{nn,nc,nb,nq,na,nr,no,cap,components,config[9],config[10],config[11],volume,
            context.upload(nc,hc.data()),context.upload(nn,hn.data()),context.upload(nb,hb.data()),
            context.upload(nr,num),context.upload(nr,ptr),context.upload(cap,list),context.upload(size_t(nr)*no,lookup),
            context.upload(4*size_t(nq),integrals),context.upload(3*size_t(nq),xyz)};
        auto d=context.upload(size_t(cap)*components,density);auto out=context.upload(3*na+18,initial_values);
        auto weights=context.upload<double>(cap,nullptr);auto code=context.upload(1,&initial);check(cudaStreamSynchronize(context.stream));
        report->allocated=context.allocated;report->staged=context.staged;report->stage=2;
        auto apply=[&]{check(remag::add_nonlocal_compact_device(v,d,weights,out,out+9,out+9+3*na,code,context.stream));};
        if(captured){check(cudaStreamBeginCapture(context.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(context.stream,&context.graph));
            size_t count=0;check(cudaGraphGetNodes(context.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(context.graph,nodes.data(),&count));report->graph_nodes=count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&context.executable,context.graph,0));}
        report->stage=3;for(int i=0;i<cycles;i++){
            // Poison scratch before each replay, outside capture. Unwritten
            // or wrong-row lookup entries must never masquerade as zeros.
            check(cudaMemsetAsync(weights,0xff,8*size_t(cap),context.stream));report->poisoned+=8*size_t(cap);
            if(captured)check(cudaGraphLaunch(context.executable,context.stream));else apply();}
        check(cudaStreamSynchronize(context.stream));report->stage=4;
        check(cudaMemcpy(result,out,8*(3*na+18),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,4,cudaMemcpyDeviceToHost));
        report->exported=8*(3*na+18)+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Compact nonlocal probe: %s\n",e.what());return -1;}
}
