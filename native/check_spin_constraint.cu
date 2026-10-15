// SPDX-License-Identifier: GPL-3.0-or-later
#include "spin_constraint.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
struct SpinConstraintCheck {size_t allocated,staged,exported;int graph_nodes,forbidden_nodes,stage;};
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
// config = kind, rows, entries, spins, pairs, sites. pairs holds (row,col,begin,end)
// per pair in hamiltonian.cu order; H is updated in place cycles times, then
// row moments of the density are computed once. With penalty > 0 the field
// is rebuilt from the density before every addition (field is then ignored).
extern "C" int remag_check_spin_constraint(const int* config,int captured,int cycles,const int* pairs,const int* indices,
        const double* overlap,const double* field,const int* num,const int* ptr,const double* density,
        double penalty,const int* orbital_site,const double* site_fields,const double* targets,
        double* h,double* moments,double* penalized_field,SpinConstraintCheck* report) {
    try {
        if(!config||!pairs||!indices||!overlap||!field||!num||!ptr||!density||!h||!moments||!report)return -1;
        *report={};const int kind=config[0],n=config[1],nnz=config[2],spins=config[3],np=config[4],sites=config[5];
        if(n<1||nnz<1||np<1||cycles<1||cycles>3||(captured!=0&&captured!=1)||penalty<0.||
           (penalty>0.&&(sites<1||!orbital_site||!site_fields||!targets||!penalized_field)))return -1;
        report->stage=1;Context c;
        remag::HamiltonianAssemblyView view{};
        view.shape={kind,n,nnz,spins,n,1,nnz,np};
        view.pairs=c.upload(size_t(np),reinterpret_cast<const remag::HamiltonianPair*>(pairs));
        view.indices=c.upload(size_t(nnz),indices);view.overlap=c.upload(size_t(nnz),overlap);
        auto f=c.upload(3*size_t(n),field);auto hd=c.upload(size_t(nnz)*spins,h);
        auto nd=c.upload(size_t(n),num);auto pd=c.upload(size_t(n),ptr);auto dd=c.upload(size_t(nnz)*spins,density);
        auto md=c.upload<double>(3*size_t(n),nullptr);
        remag::SpinConstraintDevice constraint{};
        if(penalty>0.) constraint={sites,penalty,c.upload(size_t(n),orbital_site),c.upload(3*size_t(sites),site_fields),
            c.upload(3*size_t(sites),targets),c.upload<double>(3*size_t(n),nullptr),c.upload<double>(3*size_t(n),nullptr)};
        check(cudaStreamSynchronize(c.stream));
        report->allocated=c.allocated;report->staged=c.staged;report->stage=2;
        auto apply=[&]{
            if(penalty>0.) {
                check(remag::spin_constraint_penalty_field_device(n,spins,size_t(nnz),nd,pd,view.overlap,dd,constraint,c.stream));
                check(remag::spin_constraint_field_device(view,spins,constraint.orbital_field,hd,c.stream));
            } else check(remag::spin_constraint_field_device(view,spins,f,hd,c.stream));
        };
        if(captured){check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(c.stream,&c.graph));
            size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(c.graph,nodes.data(),&count));report->graph_nodes=count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report->forbidden_nodes++;}
            check(cudaGraphInstantiate(&c.executable,c.graph,0));}
        report->stage=3;
        for(int i=0;i<cycles;i++){if(captured)check(cudaGraphLaunch(c.executable,c.stream));else apply();}
        check(remag::spin_constraint_moments_device(n,spins,size_t(nnz),nd,pd,view.overlap,dd,md,c.stream));
        check(cudaStreamSynchronize(c.stream));report->stage=4;
        check(cudaMemcpy(h,hd,size_t(nnz)*spins*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(moments,md,3*size_t(n)*sizeof(double),cudaMemcpyDeviceToHost));
        if(penalty>0.) check(cudaMemcpy(penalized_field,constraint.orbital_field,3*size_t(n)*sizeof(double),cudaMemcpyDeviceToHost));
        report->exported=(size_t(nnz)*spins+3*size_t(n))*sizeof(double);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Spin-constraint probe: %s\n",e.what());return -1;}
}
