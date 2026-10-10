// SPDX-License-Identifier: GPL-3.0-or-later
#include "grid_force.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t bytes=0,staged=0;
    Context() {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);if(graph) cudaGraphDestroy(graph);
        for(void* p:memory) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* upload(size_t n,const T* source) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);bytes+=n*sizeof(T);
        check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);return p;
    }
};
}
struct GridForceCheck {size_t allocated,staged,exported;int graph_nodes,forbidden_nodes,stage;};
extern "C" int remag_check_grid_force(int nc,int nsp,int nh,int atoms,int captured,int cycles,
        const remag::GridForceProblem* problem,const int* orbital_atoms,const double* density,const double* occupation,
        const double* phi,const double* gradient,const double* displacement,const double* potential,const double* auxiliary,
        const double* initial_force,const double* initial_stress,double* output,int* status,GridForceCheck* report) {
    try {
        if(nc<1||nc>1024||nsp<1||nsp>64||atoms<1||atoms>nc||cycles<1||cycles>8||
           (nh!=1&&nh!=2&&nh!=4&&nh!=8)||!problem||!orbital_atoms||!density||!occupation||!phi||!gradient||!displacement||
           !potential||!auxiliary||!initial_force||!initial_stress||!output||!status||!report||(captured!=0&&captured!=1)) return -1;
        *report={};report->stage=1;Context c;const int ng=nh<4?nh:4;
        std::vector<int> orbs(nc),lookup(size_t(nc)*nc);int ends[]={0,nc},zero=0;
        for(int i=0;i<nc;i++) {orbs[i]=i+1;for(int j=0;j<nc;j++) lookup[size_t(i)*nc+j]=j*nc+i;}
        remag::GridContractionView grid{{nc,1,nsp,nc,nc*nc,1},c.upload(2,ends),c.upload(nc,orbs.data()),
            c.upload(1,&zero),c.upload(lookup.size(),lookup.data()),nullptr};
        remag::GridForceView v{grid,atoms,nc,c.upload(nc,orbital_atoms),c.upload(size_t(nc)*nsp,phi),
            c.upload(size_t(nc)*nsp*3,gradient),c.upload(size_t(nc)*nsp*3,displacement),c.upload(nc,occupation)};
        auto p=c.upload(1,problem);auto d=c.upload(size_t(nc)*nc*nh,density);auto pot=c.upload(size_t(nsp)*ng,potential);
        auto aux=c.upload(nsp,auxiliary);auto f=c.upload(3*atoms,initial_force);auto s=c.upload(9,initial_stress);auto code=c.upload(1,&zero);
        report->allocated=c.bytes;report->staged=c.staged;check(cudaStreamSynchronize(c.stream));report->stage=2;
        auto apply=[&] {check(remag::add_grid_force_device(v,nh,p,d,pot,aux,f,s,code,c.stream));};
        if(captured) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
            check(cudaGraphGetNodes(c.graph,nodes.data(),&count));report->graph_nodes=int(count);
            for(auto node:nodes) {cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel) report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphUpload(c.executable,c.stream));check(cudaStreamSynchronize(c.stream));
        }
        report->stage=3;
        for(int i=0;i<cycles;i++) {if(captured) check(cudaGraphLaunch(c.executable,c.stream));else apply();}
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(output,f,3*atoms*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output+3*atoms,s,9*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));
        report->exported=(3*atoms+9)*sizeof(double)+sizeof(int);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Grid force probe: %s\n",e.what());return -1;}
}
