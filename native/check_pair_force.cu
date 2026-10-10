// SPDX-License-Identifier: GPL-3.0-or-later
#include "pair_force.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct PairForceCheck {size_t allocated,staged,exported;int graph_nodes,forbidden_nodes,stage;};
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;std::vector<void*> memory;size_t staged=0,allocated=0;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* source){T* p=nullptr;size_t bytes=(n?n:1)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;
        if(n){check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);}return p;}
};
}
// config: entries,atoms,rows,columns,capacity,components,spinor,cycles.
extern "C" int remag_check_pair_force(const int* config,int captured,int initial,const remag::PairForceProblem* problem,
        const int* qr,const int* qc,const int* atoms,const int* num,const int* ptr,const int* list,
        const double* integral,const double* xyz,const double* density,const double* initial_values,
        double* result,int* status,PairForceCheck* report) {
    try {
        if(!config||!problem||!qr||!qc||!atoms||!num||!ptr||!list||!integral||!xyz||!density||!initial_values||!result||!status||!report)return -1;
        *report={};const int n=config[0],na=config[1],nr=config[2],nc=config[3],capacity=config[4],components=config[5],spinor=config[6],cycles=config[7];
        if(n<0||n>1000000||na<1||na>10000||nr<1||nr>10000||nc<1||capacity<1||capacity>1000000||
           (components!=1&&components!=2&&components!=4&&components!=8)||components<spinor||cycles<1||cycles>3||(captured!=0&&captured!=1))return -1;
        report->stage=1;Context c;
        remag::PairForceView v{n,na,nr,nc,capacity,spinor,c.upload(n,qr),c.upload(n,qc),c.upload(2*size_t(n),atoms),
            c.upload(nr,num),c.upload(nr,ptr),c.upload(capacity,list),c.upload(4*size_t(n),integral),c.upload(3*size_t(n),xyz)};
        auto p=c.upload(1,problem);auto d=c.upload(size_t(capacity)*components,density);
        auto out=c.upload(3*na+10,initial_values);auto code=c.upload(1,&initial);check(cudaStreamSynchronize(c.stream));
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{check(remag::add_pair_force_device(v,p,d,out,out+1,out+1+3*na,code,c.stream));};
        if(captured){check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t n=0;check(cudaGraphGetNodes(c.graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(c.graph,nodes.data(),&n));report->graph_nodes=n;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));}
        report->stage=3;for(int i=0;i<cycles;i++){if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();}
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(result,out,8*(3*na+10),cudaMemcpyDeviceToHost));check(cudaMemcpy(status,code,4,cudaMemcpyDeviceToHost));report->exported=8*(3*na+10)+4;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Pair-force probe: %s\n",e.what());return -1;}
}
