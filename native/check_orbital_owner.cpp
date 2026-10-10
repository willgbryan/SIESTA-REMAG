// SPDX-License-Identifier: GPL-3.0-or-later
#include "orbital_basis_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr,other=nullptr;int* status=nullptr;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));check(cudaMalloc(reinterpret_cast<void**>(&status),sizeof(int)));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(other)cudaStreamSynchronize(other);if(status)cudaFree(status);if(stream)cudaStreamDestroy(stream);if(other)cudaStreamDestroy(other);}
};
}
int main(int argc,char** argv){
    try {
        require(argc==4,"Expected scenario, shadow flag, report path");const std::string mode=argv[1];const bool shadow=std::stoi(argv[2])==1;
        const std::vector<std::string> accepted={"normal","reuse","borrow_without_stage","stage_in_scf","duplicate_stage","unconsumed_epoch","stale_borrow","wrong_shape","borrow_in_scf","finish_before_borrow","finish_before_generation","generate_before_borrow","duplicate_borrow","duplicate_generation","wrong_stream","finish_in_scf","duplicate_finish","mode_change","device_error","both_modes","missing_comparison"};
        require(std::find(accepted.begin(),accepted.end(),mode)!=accepted.end(),"Unknown owner scenario");require(shadow||mode!="missing_comparison","Missing comparison is a shadow-only case");
        setenv("REMAG_GRID_FORCE","1",1);
        auto setmode=[&](bool s){setenv("REMAG_ORBITAL_BASIS",s?"0":"1",1);setenv("REMAG_ORBITAL_BASIS_SHADOW",s?"1":"0",1);};setmode(shadow);
        Context c;std::vector<std::string> rejections;int graphs=0,code=-1;double first=0.,second=0.;
        auto reject=[&](const char* name,auto fn){bool threw=false;try{fn();}catch(const std::exception&){threw=true;}require(threw,"Expected owner rejection missing");rejections.emplace_back(name);};
        const int config[]={1,1,2,1,1,1,1,1,1},sp[]={0,1},orb[]={2,0,0,0},one=1;
        const double radial[]={1.,2.},d2[]={0.,0.},sub[]={0.,0.,0.},atom[]={0.,0.,0.};
        double values[]={.3,.7},offset[]={.2,0.,0.};if(mode=="device_error")values[0]=std::numeric_limits<double>::quiet_NaN();
        auto stage=[&](){return remag_orbital_basis_stage(config,std::acos(-1.),sp,orb,radial,values,d2,&one,&one,&one,&one,&one,offset,sub,atom);};
        check(cudaMemset(c.status,0,sizeof(int)));require(remag_phase_enter(0,0,0)==0,"Initial phase");remag_geometry_changed();auto epoch=remag::geometry_epoch();
        auto borrow=[&](){return remag::begin_orbital_basis(epoch,1,1,1,1,c.stream);};
        if(mode=="borrow_without_stage")reject("borrow_without_stage",borrow);
        if(mode=="stage_in_scf") {require(remag_phase_enter(1,0,1)==0,"SCF phase");require(stage()!=0,"SCF staging accepted");rejections.emplace_back("stage_in_scf");require(remag_phase_enter(0,0,0)==0,"Initialization phase");}
        if(mode=="both_modes") {setenv("REMAG_ORBITAL_BASIS","1",1);setenv("REMAG_ORBITAL_BASIS_SHADOW","1",1);require(stage()!=0,"Conflicting modes accepted");rejections.emplace_back("both_modes");setmode(shadow);}
        require(stage()==0,"Initial compact stage failed");
        bool terminal=false;
        if(mode=="duplicate_stage") {require(stage()!=0,"Duplicate stage accepted");rejections.emplace_back("duplicate_stage");}
        if(mode=="unconsumed_epoch") {
            remag_geometry_changed();require(stage()!=0,"Unconsumed replacement accepted");rejections.emplace_back("unconsumed_epoch");
            reject("stale_after_epoch_change",borrow);terminal=true;
        }
        if(!terminal) {
            if(mode=="stale_borrow")reject("stale_borrow",[&]{remag::begin_orbital_basis(epoch+1,1,1,1,1,c.stream);});
            if(mode=="wrong_shape")for(int i=0;i<4;i++)reject("wrong_shape",[&]{remag::begin_orbital_basis(epoch,1+(i==0),1+(i==1),1+(i==2),1+(i==3),c.stream);});
            if(mode=="borrow_in_scf") {require(remag_phase_enter(1,0,1)==0,"SCF phase");reject("borrow_in_scf",borrow);}
            require(remag_phase_enter(2,0,0)==0,"Post phase");
            if(mode=="mode_change") {setmode(!shadow);reject("mode_change",borrow);setmode(shadow);}
            if(mode=="finish_before_borrow")reject("finish_before_borrow",[&]{remag::finish_orbital_basis(epoch,0);});
            if(mode=="generate_before_borrow")reject("generate_before_borrow",[&]{remag::generate_orbital_basis(epoch,c.status,c.stream);});
            auto execute=[&](bool inject){
                auto view=borrow();require(bool(view.comparison)==shadow,"Unexpected comparison allocation");
                if(inject&&mode=="duplicate_borrow")reject("duplicate_borrow",borrow);
                if(inject&&mode=="finish_before_generation")reject("finish_before_generation",[&]{remag::finish_orbital_basis(epoch,0);});
                if(inject&&mode=="wrong_stream")reject("wrong_stream",[&]{remag::generate_orbital_basis(epoch,c.status,c.other);});
                cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
                check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));remag::generate_orbital_basis(epoch,c.status,c.stream);
                if(inject&&mode=="duplicate_generation")reject("duplicate_generation",[&]{remag::generate_orbital_basis(epoch,c.status,c.stream);});
                const bool compare=shadow&&mode!="missing_comparison";
                if(compare)check(remag::compare_orbital_geometry_device(1,view.phi,view.gradient,view.displacement,view.phi,view.gradient,view.displacement,view.comparison,c.status,c.stream));
                check(cudaStreamEndCapture(c.stream,&graph));size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));require(n==size_t(compare?2:1),"Owner graph count differs");
                std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden owner graph node");}
                check(cudaGraphInstantiate(&executable,graph,0));check(cudaGraphLaunch(executable,c.stream));check(cudaStreamSynchronize(c.stream));graphs++;
                double phi;check(cudaMemcpy(&phi,view.phi,sizeof(double),cudaMemcpyDeviceToHost));check(cudaMemcpy(&code,c.status,sizeof(int),cudaMemcpyDeviceToHost));
                cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
                if(inject&&mode=="finish_in_scf") {require(remag_phase_enter(1,0,1)==0,"SCF phase");reject("finish_in_scf",[&]{remag::finish_orbital_basis(epoch,code);});require(remag_phase_enter(2,0,0)==0,"Post phase");}
                if(mode=="device_error"||mode=="missing_comparison")reject(mode.c_str(),[&]{remag::finish_orbital_basis(epoch,code);});
                else remag::finish_orbital_basis(epoch,code);
                return phi;
            };
            first=execute(true);
            if(mode=="duplicate_finish")reject("duplicate_finish",[&]{remag::finish_orbital_basis(epoch,0);});
            if(mode=="reuse") {
                require(remag_phase_enter(0,1,0)==0,"Next initialization");remag_geometry_changed();epoch=remag::geometry_epoch();offset[0]=.4;
                require(stage()==0,"Next epoch stage failed");require(remag_phase_enter(2,1,0)==0,"Next post phase");second=execute(false);
                require(first!=second,"Reused stale derivatives");
            }
            require(code==(mode=="device_error"?2:0),"Unexpected generated device status");
        }
        FILE* f=std::fopen(argv[3],"w");require(f,"Cannot retain owner report");
        std::fprintf(f,"{\"scenario\":\"%s\",\"shadow\":%s,\"passed\":true,\"graphs\":%d,\"status\":%d,\"first_phi\":%.17g,\"second_phi\":%.17g,\"rejections\":[",mode.c_str(),shadow?"true":"false",graphs,code,first,second);
        for(size_t i=0;i<rejections.size();i++)std::fprintf(f,"%s\"%s\"",i?",":"",rejections[i].c_str());std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Orbital owner probe: %s\n",e.what());return 1;}
}
