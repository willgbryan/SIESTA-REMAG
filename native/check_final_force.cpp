// SPDX-License-Identifier: GPL-3.0-or-later
#include "final_grid_state.hpp"
#include "atomic_grid_force.hpp"
#include "grid_force.hpp"
#include "execution.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* m) {if(!v) throw std::runtime_error(m);}
struct Context {
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* h){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);check(cudaMemcpyAsync(p,h,n*sizeof(T),cudaMemcpyHostToDevice,stream));return p;}
};
template<class F> void captured(cudaStream_t stream,F apply) {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    try {
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));apply();check(cudaStreamEndCapture(stream,&graph));
        size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));require(n==1,"Expected one captured force kernel");
        cudaGraphNode_t node;check(cudaGraphGetNodes(graph,&node,&n));cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        require(type==cudaGraphNodeTypeKernel,"Forbidden force graph node");
        check(cudaGraphInstantiate(&exec,graph,0));check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
    } catch(...) {if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);throw;}
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
}
}
struct FinalForceCheck {size_t staged_bytes;int status,graphs,rejections,consumer_mask;};
extern "C" int remag_check_final_force(int nh,int partial_core,int mode,int forces,int stress,double* output,FinalForceCheck* report) {
    try {
        require((nh==1||nh==2||nh==4||nh==8)&&(partial_core==0||partial_core==1)&&mode>=0&&mode<=9&&
            (forces==0||forces==1)&&(stress==0||stress==1)&&output&&report,"Invalid final-force fixture");
        setenv("REMAG_GRID_FORCE","1",1);setenv("REMAG_ATOMIC_GRID_FORCE","1",1);unsetenv("REMAG_ATOMIC_FORCE_SHADOW");
        *report={};Context c;const auto epoch=remag::geometry_epoch();const int ng=nh<4?nh:4,nsd=nh==1?1:2;
        const double density[]={.4,.3,.2,-.1,.07,.08,.12,.14},energy[4]={},potential[]={1.2,1.8,.2,.3},auxiliary=.6;
        const double deformation=mode==9?std::numeric_limits<double>::quiet_NaN():.8;
        auto d=c.upload(nh,density);auto e=c.upload(ng,energy);auto v=c.upload(ng,potential);auto a=c.upload(1,&auxiliary);auto rho=c.upload(1,&deformation);
        const double core_grad[]={.2,-.3,.4},neutral_grad[]={-.2,.5,-.1},df_grad[]={.3,.2,-.4},dx[]={.4,.2,-.3};
        int one=1,zero=0,ends[]={0,1};double phi=.7,occupation=.3;
        remag::AtomicGridForceView av{1,1,1,c.upload(1,&one),c.upload(1,&one),c.upload(3,core_grad),c.upload(3,dx)};
        auto neutral_gradient=c.upload(3,neutral_grad);
        remag::GridContractionView grid{{1,1,1,1,1,1},c.upload(2,ends),c.upload(1,&one),c.upload(1,&zero),c.upload(1,&zero),nullptr};
        remag::GridForceView gv{grid,1,1,c.upload(1,&one),c.upload(1,&phi),c.upload(3,df_grad),av.displacement,c.upload(1,&occupation)};
        remag::AtomicGridForceProblem cp{.1,2.,1,nsd,forces,stress},np{.1,2.,2,1,forces,stress};
        remag::GridForceProblem dp{.1,2.,forces,stress};auto core_problem=c.upload(1,&cp);auto neutral_problem=c.upload(1,&np);auto df_problem=c.upload(1,&dp);
        check(cudaStreamSynchronize(c.stream));
        remag::retain_final_grid_state({grid.shape,nh,ng,epoch,.1,2.,d,e,v,a,rho,0,{1,1,1},1,partial_core},c.stream);
        auto rejected=[&](auto fn){bool threw=false;try{fn();}catch(const std::exception&){threw=true;}require(threw,"Required ownership rejection did not occur");report->rejections++;};
        if(mode==1) rejected([&]{remag::borrow_final_grid_state(epoch,partial_core?4:1);});
        if(mode==2) rejected([&]{remag::borrow_final_grid_state(epoch+1,partial_core?1:2);});
        if(mode==6) rejected([&]{remag::reset_final_grid_state();});
        double force[]={.1,-.2,.3},tensor[]={.01,.02,.03,.04,.05,.06,.07,.08,.09};
        remag::FinalForceAccumulatorView previous{};
        for(int kind=partial_core?1:2;kind<=2;kind++) {
            const auto fields=remag::borrow_final_grid_state(epoch,kind);
            auto pool=remag::final_force_accumulator(epoch,1,forces,stress,force,tensor);
            if(previous.stream)require(pool.stream==previous.stream&&pool.force==previous.force&&pool.stress==previous.stress&&pool.status==previous.status,"Accumulator pointers changed");
            previous=pool;
            if(kind==(partial_core?1:2)) {
                if(mode==3)rejected([&]{remag::borrow_final_grid_state(epoch,4);});
                if(mode==4){force[0]+=1.;rejected([&]{remag::final_force_accumulator(epoch,1,forces,stress,force,tensor);});force[0]=.1;}
                if(mode==7)rejected([&]{remag::complete_final_grid_force(epoch,kind);});
            }
            auto current=av;if(kind==2)current.gradient=neutral_gradient;
            captured(pool.stream,[&]{check(remag::add_atomic_grid_force_device(current,kind==1?core_problem:neutral_problem,fields.potential,fields.auxiliary,fields.deformation,pool.force,pool.stress,pool.status,pool.stream));});
            report->graphs++;
            remag::AtomicForceRecord record{kind,1,1,kind==1?nsd:1,1,88,56,56};remag::record_final_atomic_force(epoch,record);
            if(mode==8&&kind==(partial_core?1:2))rejected([&]{remag::record_final_atomic_force(epoch,record);});
            remag::complete_final_grid_force(epoch,kind);report->consumer_mask|=kind;
            if(mode==5&&kind==(partial_core?1:2))rejected([&]{remag::borrow_final_grid_state(epoch,kind);});
        }
        const auto fields=remag::borrow_final_grid_state(epoch,4);auto pool=remag::final_force_accumulator(epoch,1,forces,stress,force,tensor);
        require(pool.force==previous.force&&pool.stress==previous.stress&&pool.status==previous.status&&pool.stream==previous.stream,"Final accumulator changed");
        require(remag::final_atomic_force_records(epoch).size()==size_t(partial_core?2:1),"Missing terminal atomic records");
        captured(pool.stream,[&]{check(remag::add_grid_force_device(gv,nh,df_problem,fields.density,fields.potential,fields.auxiliary,pool.force,pool.stress,pool.status,pool.stream));});report->graphs++;
        // Only terminal readbacks. A failed result is retained as a negative
        // fixture, never published as valid; the completed owner can close.
        check(cudaMemcpy(&report->status,pool.status,sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output,pool.force,3*sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(output+3,pool.stress,9*sizeof(double),cudaMemcpyDeviceToHost));
        report->staged_bytes=pool.staged_bytes;remag::complete_final_grid_force(epoch,4);report->consumer_mask|=4;
        remag::reset_final_grid_state();return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"Final force probe: %s\n",e.what());return -1;}
}
