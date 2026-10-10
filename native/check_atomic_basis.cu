// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_basis.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct AtomicBasisCheck {size_t allocated,staged,exported,poisoned;int graph_nodes,forbidden_nodes,stage;};
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
template<class F,class P>void execute(Context& c,int captured,int cycles,AtomicBasisCheck* report,F apply,P poison){
    check(cudaStreamSynchronize(c.stream));report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
    if(captured){
        check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
        size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));
        report->graph_nodes=n;
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
        check(cudaGraphInstantiate(&c.executable,c.graph,0));
    }
    // Poison every output before each replay, outside the captured kernel.
    // Incomplete writes (including capped-grid tails) cannot inherit old data.
    report->stage=3;for(int i=0;i<cycles;i++){poison();if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();}
    check(cudaStreamSynchronize(c.stream));report->stage=4;
}
}
extern "C" int remag_check_atomic_queries(int ns,int samples,int kind,int filter,int count,int captured,int cycles,int initial,
        const int* z,const int* index,const double* parameters,const double* values,const double* second,const int* species,
        const double* xyz,double* result,int* active,int* status,AtomicBasisCheck* report){
    try {
        if(ns<1||ns>100||samples<0||samples>1000000||count<1||count>1000000||(captured!=0&&captured!=1)||cycles<1||cycles>3||
            !z||!index||!parameters||!values||!second||!species||!xyz||!result||!active||!status||!report)return -1;
        *report={};report->stage=1;Context c;
        remag::AtomicBasisView b{ns,samples,c.upload(ns,z),c.upload(4*ns,index),c.upload(4*ns,parameters),c.upload(samples,values),c.upload(samples,second)};
        auto qs=c.upload(count,species);auto r=c.upload(3*count,xyz);auto out=c.upload<double>(4*count,nullptr);
        auto mask=c.upload<int>(count,nullptr);auto code=c.upload(1,&initial);
        execute(c,captured,cycles,report,[&]{check(remag::evaluate_atomic_queries_device(b,kind,filter,count,qs,r,out,mask,code,c.stream));},[&]{
            check(cudaMemsetAsync(out,255,32*size_t(count),c.stream));check(cudaMemsetAsync(mask,255,4*size_t(count),c.stream));report->poisoned+=36*size_t(count);});
        check(cudaMemcpy(result,out,32*size_t(count),cudaMemcpyDeviceToHost));check(cudaMemcpy(active,mask,4*size_t(count),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));report->exported=36*size_t(count)+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Atomic basis query: %s\n",e.what());return -1;}
}
extern "C" int remag_check_atomic_geometry(int ns,int samples,int kind,const int* shape,int captured,int cycles,int initial,
        const int* z,const int* index,const double* parameters,const double* values,const double* second,
        const int* pa,const int* po,const int* pc,const int* as,const int* ua,const double* offset,const double* sub,const double* atom,
        double* result,int* indices,int* status,AtomicBasisCheck* report){
    try {
        if(ns<1||ns>100||samples<0||samples>1000000||!shape||(captured!=0&&captured!=1)||cycles<1||cycles>3||
            !z||!index||!parameters||!values||!second||!pa||!po||!pc||!as||!ua||!offset||!sub||!atom||!result||!indices||!status||!report)return -1;
        const int pairs=shape[0],nsp=shape[1],na=shape[2],noff=shape[3];
        if(pairs<1||nsp<1||size_t(pairs)*nsp>1000000||na<1||na>1000000||noff<1||noff>1000000)return -1;
        *report={};report->stage=1;Context c;const size_t n=size_t(pairs)*nsp;
        remag::AtomicBasisView b{ns,samples,c.upload(ns,z),c.upload(4*ns,index),c.upload(4*ns,parameters),c.upload(samples,values),c.upload(samples,second)};
        remag::AtomicGeometryView g{pairs,nsp,na,noff,shape[4],shape[5],shape[6],{shape[7],shape[8],shape[9]},
            c.upload(pairs,pa),c.upload(pairs,po),c.upload(pairs,pc),c.upload(na,as),c.upload(na,ua),
            c.upload(3*noff,offset),c.upload(3*nsp,sub),c.upload(3*na,atom)};
        auto v=c.upload<double>(n,nullptr);auto gr=c.upload<double>(3*n,nullptr);auto dx=c.upload<double>(3*n,nullptr);
        auto active=c.upload<int>(n,nullptr);auto point=c.upload<int>(n,nullptr);auto unit=c.upload<int>(n,nullptr);auto code=c.upload(1,&initial);
        execute(c,captured,cycles,report,[&]{check(remag::evaluate_atomic_geometry_device(b,g,kind,v,gr,dx,active,point,unit,code,c.stream));},[&]{
            check(cudaMemsetAsync(v,255,8*n,c.stream));check(cudaMemsetAsync(gr,255,24*n,c.stream));check(cudaMemsetAsync(dx,255,24*n,c.stream));
            check(cudaMemsetAsync(active,255,4*n,c.stream));check(cudaMemsetAsync(point,255,4*n,c.stream));check(cudaMemsetAsync(unit,255,4*n,c.stream));report->poisoned+=68*n;});
        check(cudaMemcpy(result,v,8*n,cudaMemcpyDeviceToHost));check(cudaMemcpy(result+n,gr,24*n,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(result+4*n,dx,24*n,cudaMemcpyDeviceToHost));check(cudaMemcpy(indices,active,4*n,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(indices+n,point,4*n,cudaMemcpyDeviceToHost));check(cudaMemcpy(indices+2*n,unit,4*n,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));report->exported=68*n+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Atomic basis geometry: %s\n",e.what());return -1;}
}
extern "C" int remag_check_atomic_comparison(int n,int captured,int initial,const double* av,const int* ai,const double* bv,const int* bi,
        remag::AtomicBasisComparison* result,int* status,AtomicBasisCheck* report){
    try {
        if(n<1||n>1000000||(captured!=0&&captured!=1)||!av||!ai||!bv||!bi||!result||!status||!report)return -1;
        *report={};report->stage=1;Context c;
        auto make=[&](const double* v,const int* i){auto dv=c.upload(7*n,v);auto di=c.upload(3*n,i);return remag::AtomicBasisValues{dv,dv+n,dv+4*n,di,di+n,di+2*n};};
        auto a=make(av,ai),b=make(bv,bi);remag::AtomicBasisComparison zero{};auto out=c.upload(1,&zero);auto code=c.upload(1,&initial);
        execute(c,captured,1,report,[&]{check(remag::compare_atomic_geometry_device(n,a,b,out,code,c.stream));},[]{});
        check(cudaMemcpy(result,out,sizeof(*result),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));
        report->exported=sizeof(*result)+sizeof(int);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Atomic basis comparison: %s\n",e.what());return -1;}
}
