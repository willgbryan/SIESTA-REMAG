// SPDX-License-Identifier: GPL-3.0-or-later
#include "orbital_basis.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t allocated=0,staged=0;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* source) {
        T* p=nullptr;size_t bytes=(n?n:1)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;
        if(n&&source){check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);}
        return p;
    }
};
}
struct OrbitalBasisCheck {size_t allocated,staged,exported;int graph_nodes,forbidden_nodes,stage;};
extern "C" int remag_check_orbital_basis(int species_count,int orbitals,int samples,int count,int captured,int cycles,int initial_status,
        double pi,const int* species,const int* orbital,const double* radial,const double* values,const double* second,
        const int* query_species,const int* query_orbital,const double* xyz,double* result,int* status,OrbitalBasisCheck* report) {
    try {
        if(!report||!result||!status||species_count<1||species_count>16||orbitals<1||orbitals>10000||samples<0||samples>10000000||
           count<1||count>1000000||cycles<1||cycles>3||(captured!=0&&captured!=1)||!species||!orbital||!radial||!values||!second||
           !query_species||!query_orbital||!xyz)return -1;
        *report={};report->stage=1;Context c;
        remag::OrbitalBasisView b{species_count,orbitals,samples,pi,c.upload(2*species_count,species),c.upload(4*orbitals,orbital),
            c.upload(2*orbitals,radial),c.upload(samples,values),c.upload(samples,second)};
        auto qs=c.upload(count,query_species);auto qo=c.upload(count,query_orbital);auto x=c.upload(3*count,xyz);
        auto out=c.upload<double>(4*count,nullptr);auto code=c.upload(1,&initial_status);check(cudaStreamSynchronize(c.stream));
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&] {check(remag::evaluate_orbital_queries_device(b,count,qs,qo,x,out,code,c.stream));};
        if(captured) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));
            report->graph_nodes=n;for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));
        }
        report->stage=3;for(int i=0;i<cycles;i++){if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();}
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(result,out,size_t(4)*count*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));
        report->exported=size_t(4)*count*sizeof(double)+sizeof(int);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Orbital basis probe: %s\n",e.what());return -1;}
}
// Geometry checker composes the same evaluation with compact SIESTA mappings.
extern "C" int remag_check_orbital_geometry(int species_count,int orbitals,int samples,int entries,int nsp,int no,int na,int offsets,
        int captured,double pi,const int* species,const int* orbital,const double* radial,const double* values,const double* second,
        const int* entry_orbital,const int* entry_offset,const int* orbital_atom,const int* orbital_local,const int* atom_species,
        const double* offset,const double* subpoint,const double* atom_offset,double* output,int* status,OrbitalBasisCheck* report) {
    try {
        if(!report||!output||!status||species_count<1||species_count>16||orbitals<1||orbitals>10000||samples<0||samples>10000000||
            entries<1||entries>1000000||nsp<1||nsp>64||no<1||no>1000000||na<1||na>1000000||offsets<1||offsets>1000000||
            (captured!=0&&captured!=1)||!species||!orbital||!radial||!values||!second||!entry_orbital||!entry_offset||
            !orbital_atom||!orbital_local||!atom_species||!offset||!subpoint||!atom_offset)return -1;
        *report={};report->stage=1;Context c;int zero=0;
        remag::OrbitalBasisView b{species_count,orbitals,samples,pi,c.upload(2*species_count,species),c.upload(4*orbitals,orbital),
            c.upload(2*orbitals,radial),c.upload(samples,values),c.upload(samples,second)};
        remag::OrbitalGeometryView g{entries,nsp,no,na,offsets,c.upload(entries,entry_orbital),c.upload(entries,entry_offset),c.upload(no,orbital_atom),
            c.upload(no,orbital_local),c.upload(na,atom_species),c.upload(3*offsets,offset),c.upload(3*nsp,subpoint),c.upload(3*na,atom_offset)};
        const size_t n=size_t(entries)*nsp;
        auto phi=c.upload<double>(n,nullptr);auto gradient=c.upload<double>(3*n,nullptr);auto dx=c.upload<double>(3*n,nullptr);auto code=c.upload(1,&zero);
        check(cudaStreamSynchronize(c.stream));report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::evaluate_orbital_geometry_device(b,g,phi,gradient,dx,code,c.stream));};
        if(captured) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));
            report->graph_nodes=n;for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphLaunch(c.executable,c.stream));
        } else apply();
        report->stage=3;check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(output,phi,n*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(output+n,gradient,3*n*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output+4*n,dx,3*n*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));
        report->exported=7*n*sizeof(double)+sizeof(int);return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"Orbital geometry probe: %s\n",e.what());return -1;}
}
extern "C" int remag_check_orbital_comparison(int n,int captured,int initial_status,const double* actual,const double* reference,
        remag::OrbitalComparison* result,int* status,OrbitalBasisCheck* report) {
    try {
        if(n<1||n>1000000||(captured!=0&&captured!=1)||!actual||!reference||!result||!status||!report)return -1;
        *report={};report->stage=1;Context c;remag::OrbitalComparison zero{};
        auto a=c.upload(size_t(7)*n,actual);auto b=c.upload(size_t(7)*n,reference);
        auto out=c.upload(1,&zero);auto code=c.upload(1,&initial_status);check(cudaStreamSynchronize(c.stream));
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::compare_orbital_geometry_device(n,a,a+n,a+4*n,b,b+n,b+4*n,out,code,c.stream));};
        if(captured) {
            check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t nodes_count=0;check(cudaGraphGetNodes(c.graph,nullptr,&nodes_count));std::vector<cudaGraphNode_t> nodes(nodes_count);
            check(cudaGraphGetNodes(c.graph,nodes.data(),&nodes_count));report->graph_nodes=nodes_count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphLaunch(c.executable,c.stream));
        } else apply();
        report->stage=3;check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(result,out,sizeof(*result),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));
        report->exported=sizeof(*result)+sizeof(int);report->stage=4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Orbital comparison probe: %s\n",e.what());return -1;}
}
