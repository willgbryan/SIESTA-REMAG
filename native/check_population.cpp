// SPDX-License-Identifier: GPL-3.0-or-later
#include "population.hpp"
#include "population_observables.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct PopulationCheck {size_t allocated,staged,exported,poisoned;int graph_nodes,forbidden_nodes,stage;};
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t staged=0,allocated=0;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:memory)cudaFree(p);
        if(stream)cudaStreamDestroy(stream);}
    template<class T>T* buffer(size_t n,const T* source=nullptr){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));
        memory.push_back(p);allocated+=n*sizeof(T);if(source){check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);}return p;}
};
}
// Config follows the pristine population oracle: atoms, rows, capacity, spins.
extern "C" int remag_check_population(const int* config,int captured,int replays,const remag::PopulationProblem* problem,
        const int* num,const int* ptr,const int* lasto,const double* overlap,const double* density,
        double* output,int* status,PopulationCheck* report) {
    try {
        if(!report)return -1;
        *report={};
        if(!config||!problem||!num||!ptr||!lasto||!overlap||!density||!output||!status)return -1;
        const int atoms=config[0],rows=config[1],capacity=config[2],spins=config[3];
        if(atoms<1||atoms>1000000||rows<1||rows>1000000||capacity<1||capacity>10000000||
           (spins!=1&&spins!=2&&spins!=4&&spins!=8)||(captured!=0&&captured!=1)||replays<1||replays>3)return -1;
        report->stage=1;Context c;const int components=spins<4?spins:4;
        auto p=c.buffer(1,problem);auto n=c.buffer(rows,num);auto starts=c.buffer(rows,ptr);auto end=c.buffer(atoms+1,lasto);
        auto s=c.buffer(capacity,overlap);auto d=c.buffer(size_t(capacity)*spins,density);
        const size_t size=size_t(components)*(rows+atoms+1);auto out=c.buffer<double>(size);auto code=c.buffer<int>(1);
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::population_device(capacity,rows,atoms,spins,p,n,starts,end,s,d,
            out,out+size_t(rows)*components,out+size_t(rows+atoms)*components,code,c.stream));};
        check(cudaStreamSynchronize(c.stream));
        if(captured){check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
            check(cudaGraphGetNodes(c.graph,nodes.data(),&count));report->graph_nodes=count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));}
        report->stage=3;
        for(int i=0;i<replays;i++){
            // Fixture poisoning is deliberately outside the captured operation.
            check(cudaMemsetAsync(out,0xff,size*sizeof(double),c.stream));check(cudaMemsetAsync(code,0x7f,sizeof(int),c.stream));
            report->poisoned+=size*sizeof(double)+sizeof(int);
            if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();
        }
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(output,out,size*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,sizeof(int),cudaMemcpyDeviceToHost));
        report->exported=size*sizeof(double)+sizeof(int);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Population probe: %s\n",e.what());return -1;}
}

// The derived operation preserves the population status from its caller.
extern "C" int remag_check_population_observables(const int* config,int captured,int initial_status,
        const remag::PopulationObservableProblem* problem,const int* ids,const double* valence,
        const double* orbital,const double* atomic,const double* total,
        double* output,int* status,PopulationCheck* report) {
    try {
        if(!report)return -1;
        *report={};
        if(!config||!problem||!ids||!valence||!orbital||!atomic||!total||!output||!status)return -1;
        const int rows=config[0],atoms=config[1],species=config[2],components=config[3];
        if(rows<1||rows>1000000||atoms<1||atoms>1000000||species<1||species>1000000||
           (components!=1&&components!=2&&components!=4)||(captured!=0&&captured!=1))return -1;
        Context c;report->stage=1;
        auto p=c.buffer(1,problem);auto id=c.buffer(atoms,ids);auto z=c.buffer(species,valence);
        auto orb=c.buffer(size_t(rows)*components,orbital),atom=c.buffer(size_t(atoms)*components,atomic),tot=c.buffer(components,total);
        const size_t size=size_t(5)*rows+size_t(6)*atoms+size_t(5)*species+10;
        auto out=c.buffer<double>(size);auto code=c.buffer(1,&initial_status);
        report->allocated=c.allocated;report->staged=c.staged;
        auto apply=[&]{check(remag::population_observables_device(rows,atoms,species,components,p,id,z,orb,atom,tot,
            out,out+size_t(5)*rows,out+size_t(5)*rows+size_t(6)*atoms,
            out+size_t(5)*rows+size_t(6)*atoms+size_t(5)*species,code,c.stream));};
        check(cudaStreamSynchronize(c.stream));report->stage=2;
        if(captured){check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
            check(cudaGraphGetNodes(c.graph,nodes.data(),&count));report->graph_nodes=count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));}
        report->stage=3;
        // Repeated execution must overwrite every output, but never clear an
        // earlier failure. Poisoning belongs to the probe, outside capture.
        for(int replay=0;replay<3;replay++){
            check(cudaMemsetAsync(out,0xff,size*8,c.stream));report->poisoned+=size*8;
            if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();
        }
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(output,out,size*8,cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,4,cudaMemcpyDeviceToHost));
        report->exported=size*8+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Population observable probe: %s\n",e.what());return -1;}
}
