// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_grid_force.hpp"
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
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),(n?n:1)*sizeof(T)));memory.push_back(p);bytes+=(n?n:1)*sizeof(T);
        if(n) check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);return p;
    }
};
}
struct AtomicGridForceCheck {size_t allocated,staged,exported;int graph_nodes,forbidden_nodes,stage;};
extern "C" int remag_check_atomic_masked_force(int points,int entries,int atoms,int nsd,int captured,int cycles,
        const remag::AtomicGridForceProblem* problem,const int* point_index,const int* atom_index,
        const double* gradient,const double* displacement,const double* potential,const double* auxiliary,
        const double* deformation,const double* initial_force,const double* initial_stress,
        double* output,int* status,AtomicGridForceCheck* report,const int* active,int initial_status) {
    try {
        if(points<1||points>1000000||entries<0||entries>1000000||atoms<1||atoms>10000||cycles<1||cycles>8||
           (nsd!=1&&nsd!=2)||!problem||!point_index||!atom_index||!gradient||!displacement||
           !potential||!auxiliary||!deformation||!initial_force||!initial_stress||!output||!status||!report||
           (captured!=0&&captured!=1)) return -1;
        *report={};report->stage=1;Context c;int zero=initial_status;
        remag::AtomicGridForceView v{points,entries,atoms,c.upload(entries,point_index),c.upload(entries,atom_index),
            c.upload(size_t(entries)*3,gradient),c.upload(size_t(entries)*3,displacement),active?c.upload(entries,active):nullptr};
        auto p=c.upload(1,problem);auto pot=c.upload(size_t(points)*nsd,potential);
        auto aux=c.upload(points,auxiliary);auto rho=c.upload(points,deformation);
        auto f=c.upload(3*atoms,initial_force);auto s=c.upload(9,initial_stress);auto code=c.upload(1,&zero);
        report->allocated=c.bytes;report->staged=c.staged;check(cudaStreamSynchronize(c.stream));report->stage=2;
        auto apply=[&] {check(remag::add_atomic_grid_force_device(v,p,pot,aux,rho,f,s,code,c.stream));};
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
    } catch(const std::exception& e) {std::fprintf(stderr,"Atomic grid force probe: %s\n",e.what());return -1;}
}

extern "C" int remag_check_atomic_grid_force(int points,int entries,int atoms,int nsd,int captured,int cycles,
        const remag::AtomicGridForceProblem* problem,const int* point_index,const int* atom_index,
        const double* gradient,const double* displacement,const double* potential,const double* auxiliary,
        const double* deformation,const double* initial_force,const double* initial_stress,
        double* output,int* status,AtomicGridForceCheck* report) {
    return remag_check_atomic_masked_force(points,entries,atoms,nsd,captured,cycles,problem,point_index,atom_index,
        gradient,displacement,potential,auxiliary,deformation,initial_force,initial_stress,output,status,report,nullptr,0);
}
