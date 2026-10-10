// SPDX-License-Identifier: GPL-3.0-or-later
#include "final_grid_state.hpp"
#include "grid.hpp"
#include "execution.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int remag_grid(int,int,int,int,int,int,int,int,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const double*,double*,double);
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    Device(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Device(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    double* upload(int n,const double* a){double* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*8));memory.push_back(p);check(cudaMemcpy(p,a,n*8,cudaMemcpyHostToDevice));return p;}
};
}
int main(int argc,char** argv){
    try {
        require(argc==4,"Expected scenario, spin count, report");std::string label=argv[1],mode=label;bool nonlocal=mode.rfind("nl_",0)==0;if(nonlocal)mode=mode.substr(3);setenv("REMAG_NONLOCAL_FORCE_SHADOW",nonlocal?"1":"0",1);int nh=std::stoi(argv[2]);require(nh==1||nh==2||nh==4||nh==8,"Invalid spins");
        setenv("REMAG_PAIR_FORCE_SHADOW","1",1);setenv("REMAG_GRID_FORCE","1",1);setenv("REMAG_ATOMIC_GRID_FORCE","0",1);setenv("REMAG_ATOMIC_FORCE_SHADOW","0",1);setenv("REMAG_GRID_VERIFY_EPOCH","0",1);
        int rejected=0;auto reject=[&](auto fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}require(failed,"Missing expected rejection");rejected++;};
        Device d;std::vector<double> outputs;const int ng=nh<4?nh:4,geometries=mode=="reuse"?2:1;
        for(int geometry=0;geometry<geometries;geometry++){
            remag_geometry_changed();const auto epoch=remag::geometry_epoch();remag_phase_enter(0,geometry,0);
            double density[]={.4,.3,.2,.1,.05,.06,.07,.08},energy[]={1.2,1.3,1.4,1.5},potential[4]={},zero=0.;density[0]+=geometry;energy[0]+=geometry;
            int nsc[]={1,1,1},ends[]={0,1},one=1,num=1,ptr=0,list=1;float phi=.5f;double rho[4]={};
            require(remag_grid(0,1,1,1,1,1,ng,nh,nsc,ends,&one,&num,&ptr,&list,&phi,density,rho,1.)==0,"Grid owner stage");
            auto D=d.upload(nh,density),E=d.upload(ng,energy),V=d.upload(ng,potential),A=d.upload(1,&zero),R=d.upload(1,&zero);
            remag_phase_enter(2,geometry,0);
            if(mode=="missing")reject([&]{remag::borrow_final_pair_state(epoch,1);});
            remag::retain_final_grid_state({{1,1,1,1,1,1},nh,ng,epoch,1.,1.,D,E,V,A,R,0,{1,1,1},1,0},d.stream);
            // Mutate the original device arrays after D2D retention. The pair
            // borrower must observe the immutable snapshot, not these sources.
            check(cudaMemset(D,0,nh*8));check(cudaMemset(E,0,ng*8));
            remag::validate_grid_sparse_topology(1,1,1,&num,&ptr,&list);
            if(mode=="num"){num=0;reject([&]{remag::validate_grid_sparse_topology(1,1,1,&num,&ptr,&list);});num=1;}
            if(mode=="ptr"){ptr=1;reject([&]{remag::validate_grid_sparse_topology(1,1,1,&num,&ptr,&list);});ptr=0;}
            if(mode=="list"){list=0;reject([&]{remag::validate_grid_sparse_topology(1,1,1,&num,&ptr,&list);});list=1;}
            if(mode=="shape")reject([&]{remag::validate_grid_sparse_topology(2,1,1,&num,&ptr,&list);});
            if(mode=="wrong_epoch")reject([&]{remag::borrow_final_pair_state(epoch+1,1);});
            if(mode=="wrong_consumer")reject([&]{remag::borrow_final_pair_state(epoch,2);});
            if(mode=="overlap_first")reject([&]{remag::borrow_final_pair_state(epoch,4);});
            if(mode=="grid_first")reject([&]{remag::borrow_final_grid_state(epoch,4);});
            if(mode=="mode"){setenv("REMAG_PAIR_FORCE_SHADOW","0",1);reject([&]{remag::borrow_final_pair_state(epoch,1);});setenv("REMAG_PAIR_FORCE_SHADOW","1",1);}
            if(mode=="scf"){remag_phase_enter(1,geometry,1);reject([&]{remag::borrow_final_pair_state(epoch,1);});remag_phase_enter(2,geometry,0);}
            auto kinetic=remag::borrow_final_pair_state(epoch,1);
            if(mode=="duplicate")reject([&]{remag::borrow_final_pair_state(epoch,1);});
            if(mode=="grid_during_pair")reject([&]{remag::borrow_final_grid_state(epoch,4);});
            if(mode=="reset_active")reject([&]{remag::reset_final_grid_state();});
            if(mode=="wrong_complete")reject([&]{remag::complete_final_pair_shadow(epoch,4);});
            double gotD[8]={},gotE[4]={};check(cudaMemcpy(gotD,kinetic.density,nh*8,cudaMemcpyDeviceToHost));
            for(int i=0;i<nh;i++){require(gotD[i]==density[i],"Retained D changed");outputs.push_back(gotD[i]);}
            remag::complete_final_pair_shadow(epoch,1);
            if(mode=="duplicate_complete")reject([&]{remag::complete_final_pair_shadow(epoch,1);});
            if(mode=="overlap_early")reject([&]{remag::borrow_final_pair_state(epoch,4);});
            if(nonlocal){
                const int consumer=nh==8?3:2;
                require(!remag::final_grid_forces_complete(epoch),"Analysis allowed before nonlocal");
                if(mode=="skip_nonlocal")reject([&]{remag::borrow_final_grid_state(epoch,4);});
                if(mode=="nonlocal_spin")reject([&]{remag::borrow_final_pair_state(epoch,nh==8?2:3);});
                if(mode=="nonlocal_reset")reject([&]{remag::reset_final_grid_state();});
                auto nl=remag::borrow_final_pair_state(epoch,consumer);require(nl.density==kinetic.density,"Nonlocal D identity differs");
                if(mode=="nonlocal_duplicate")reject([&]{remag::borrow_final_pair_state(epoch,consumer);});
                if(mode=="nonlocal_grid")reject([&]{remag::borrow_final_grid_state(epoch,4);});
                if(mode=="nonlocal_complete")reject([&]{remag::complete_final_pair_shadow(epoch,nh==8?2:3);});
                remag::complete_final_pair_shadow(epoch,consumer);
                if(mode=="nonlocal_reborrow")reject([&]{remag::borrow_final_pair_state(epoch,consumer);});
            }
            auto grid=remag::borrow_final_grid_state(epoch,4);require(grid.density==kinetic.density,"Grid and pair owners disagree");
            if(mode=="pair_during_grid")reject([&]{remag::borrow_final_pair_state(epoch,4);});
            remag::complete_final_grid_force(epoch,4);require(!remag::final_grid_forces_complete(epoch),"Analysis allowed before overlap");
            if(mode=="reset_incomplete")reject([&]{remag::reset_final_grid_state();});
            auto overlap=remag::borrow_final_pair_state(epoch,4);require(overlap.energy_density==kinetic.energy_density,"Retained E identity changed");
            check(cudaMemcpy(gotE,overlap.energy_density,ng*8,cudaMemcpyDeviceToHost));
            for(int i=0;i<ng;i++){require(gotE[i]==energy[i],"Retained E changed");outputs.push_back(gotE[i]);}
            remag::complete_final_pair_shadow(epoch,4);require(remag::final_grid_forces_complete(epoch),"Completed pair/grid owner not publishable");
            if(mode=="reborrow")reject([&]{remag::borrow_final_pair_state(epoch,4);});
            remag::reset_final_grid_state();
        }
        FILE* f=std::fopen(argv[3],"w");require(f,"Report creation");std::fprintf(f,"{\"scenario\":\"%s\",\"spins\":%d,\"geometries\":%d,\"rejections\":%d,\"retained_values\":[",label.c_str(),nh,geometries,rejected);
        for(size_t i=0;i<outputs.size();i++)std::fprintf(f,"%s%.17g",i?",":"",outputs[i]);std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Pair handoff probe: %s\n",e.what());return 1;}
}
