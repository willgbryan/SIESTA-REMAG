// SPDX-License-Identifier: GPL-3.0-or-later
#include "grid.hpp"
#include "execution.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int remag_grid(int,int,int,int,int,int,int,int,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const double*,double*,double);
extern "C" int remag_force_shadow_begin(int,int,int,int,int,int,int,int,const int*,const double*,const double*,const double*,const double*,double,double,int,int,const double*,const double*);
namespace {void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}}
int main(int argc,char** argv) {
    try {
        require(argc==3,"Expected scenario and report path");const std::string scenario=argv[1];
        setenv("REMAG_FORCE_SHADOW","1",1);setenv("REMAG_GRID_FORCE","0",1);setenv("REMAG_GRID_VERIFY_EPOCH","1",1);
        int nsc[]={1,1,1},ends[]={0,1},orbs[]={1,1},num=1,ptr=0,col=1,atom=1;
        float phi[]={.5f,.25f};double density=1.,rho=0.,occupation=1.,potential=2.,auxiliary=0.,force[3]={},stress[9]={};
        int stages=0;std::vector<int> rejected;
        const bool topology=scenario=="topology"||scenario=="topology-lease";
        auto stage=[&]{require(remag_phase_enter(0,stages,0)==0,"Init phase");require(remag_grid(topology?2:0,1,1,1,1,1,1,1,nsc,ends,orbs,&num,&ptr,&col,phi,topology?nullptr:&density,topology?nullptr:&rho,1.)==0,"Grid stage failed");stages++;require(remag_phase_enter(2,stages,0)==0,"Post phase");};
        auto begin=[&](int entries,int no=1){return remag_force_shadow_begin(no,1,1,1,1,1,1,entries,&atom,&density,&occupation,&potential,&auxiliary,1.,1.,1,1,force,stress);};
        auto reject=[&](int code){require(code==-230,"Expected force-owner rejection");rejected.push_back(code);};
        remag_geometry_changed();
        if(scenario=="missing")reject(begin(1));
        stage();
        if(scenario=="count")reject(begin(2));
        if(scenario=="shape")reject(begin(1,2));
        if(scenario=="stale"||scenario=="restage") {
            remag_geometry_changed();reject(begin(1));
            if(scenario=="restage")ends[1]=2;
            stage();
        }
        if(scenario=="ends"){ends[0]=1;reject(begin(1));ends[0]=0;}
        if(scenario=="orbs"){orbs[0]=2;reject(begin(1));orbs[0]=1;}
        if(scenario=="phi"){phi[0]=.75f;reject(begin(1));phi[0]=.5f;}
        if(scenario=="nsc"){nsc[0]=2;reject(begin(1));nsc[0]=1;}
        if(scenario=="lease"||scenario=="topology-lease") {
            const auto v=remag::lease_grid_contraction();require(v.entries==1,"Lease lost entries");reject(begin(1));
            require(remag_grid(0,1,1,1,1,1,1,1,nsc,ends,orbs,&num,&ptr,&col,phi,&density,&rho,1.)==-117,"Leased grid accepted staging");
            require(remag_grid(2,1,1,1,1,1,1,1,nsc,ends,orbs,&num,&ptr,&col,phi,nullptr,nullptr,1.)==-117,"Leased grid accepted topology staging");
            remag::release_grid_contraction();
        }
        auto view=remag::grid_contraction_view();require(view.entries==ends[1],"Retained count mismatch");
        require(begin(view.entries)==0,"Valid force begin failed");
        FILE* f=std::fopen(argv[2],"w");require(f,"Report failed");
        std::fprintf(f,"{\"scenario\":\"%s\",\"passed\":true,\"stages\":%d,\"epoch\":%llu,\"entries\":%d,\"rho\":%.17g,\"rejections\":[",scenario.c_str(),stages,static_cast<unsigned long long>(remag::geometry_epoch()),view.entries,rho);
        for(size_t i=0;i<rejected.size();i++)std::fprintf(f,"%s%d",i?",":"",rejected[i]);
        std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Grid owner probe: %s\n",e.what());return 1;}
}
