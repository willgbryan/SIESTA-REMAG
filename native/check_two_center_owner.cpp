// SPDX-License-Identifier: GPL-3.0-or-later
#include "two_center_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}}
int main(int argc,char** argv) {
    try {
        require(argc==3,"Expected scenario and report");const std::string scenario=argv[1];
        setenv("REMAG_TWO_CENTER_SHADOW","1",1);remag_geometry_changed();require(remag_phase_enter(0,0,0)==0,"Initialization phase");
        int config[]={1,1,1,1,1,0,0,1,1,1,1},ind[]={1,2},offsets[]={1,2},ilm=1,ri=1;
        double cutoff=1.,coefficient=1.,radial[2050]={},xyz[]={.5,0.,0.},gradient[3]={};
        std::fill(radial,radial+1025,1.);const double pi=std::acos(-1.),value=std::sqrt(1./(4*pi));
        std::vector<int> rejected;
        auto reject=[&](int code){require(code==-250,"Expected two-center rejection");rejected.push_back(code);};
        auto stage=[&](int kind){return remag_two_center_stage(kind,config,.1,pi,ind,offsets,&ilm,&ri,&cutoff,radial,&coefficient);};
        auto observe=[&](int table,double reference){return remag_two_center_observe(table,1,1,xyz,reference,gradient);};
        if(scenario=="missing")reject(remag_two_center_begin(1));
        if(scenario=="table_order")reject(stage(2));
        if(scenario=="bad_map"){ind[1]=3;reject(stage(1));ind[1]=2;}
        if(scenario=="bad_dimensions"){config[3]=100000000;config[4]=100000000;reject(stage(1));config[3]=config[4]=1;}
        if(scenario=="nonfinite_table"){radial[1024]=std::numeric_limits<double>::quiet_NaN();reject(stage(1));radial[1024]=1.;}
        if(scenario=="mode_stage"){setenv("REMAG_TWO_CENTER_SHADOW","0",1);reject(stage(1));setenv("REMAG_TWO_CENTER_SHADOW","1",1);}
        if(scenario=="scf_stage"){remag_phase_enter(1,0,1);reject(stage(1));remag_phase_enter(0,0,0);}
        require(stage(1)==0,"S stage");
        if(scenario=="duplicate")reject(stage(1));
        if(scenario=="missing_t")reject(remag_two_center_begin(1));
        require(stage(2)==0,"T stage");
        if(scenario=="snapshot") {std::fill(radial,radial+2050,77.);coefficient=99.;cutoff=0.;ind[1]=99;offsets[1]=99;ilm=99;ri=99;}
        require(remag_phase_enter(2,0,0)==0,"Post phase");
        // Non-force setup integral calls are deliberately ignored outside a batch.
        require(observe(2,value)==0,"Unexpected setup observation rejection");
        if(scenario=="wrong_begin")reject(remag_two_center_begin(4));
        const int geometries=scenario=="reuse"?2:1;
        for(int geometry=0;geometry<geometries;geometry++) {
            if(geometry){remag_geometry_changed();require(remag_phase_enter(2,geometry,0)==0,"Changed geometry");xyz[0]=3.;}
            const int consumers[]={1,scenario=="soc"?3:2,4};
            for(int consumer:consumers) {
                const int table=consumer==1?2:1;require(remag_two_center_begin(consumer)==0,"Consumer begin");
                if(consumer==1&&geometry==0) {
                    if(scenario=="duplicate_begin")reject(remag_two_center_begin(1));
                    if(scenario=="wrong_table")reject(observe(1,value));
                    if(scenario=="index")reject(remag_two_center_observe(2,2,1,xyz,value,gradient));
                    if(scenario=="null_xyz")reject(remag_two_center_observe(2,1,1,nullptr,value,gradient));
                    if(scenario=="wrong_finish")reject(remag_two_center_finish(4));
                    if(scenario=="mode_consume"){setenv("REMAG_TWO_CENTER_SHADOW","0",1);reject(observe(2,value));setenv("REMAG_TWO_CENTER_SHADOW","1",1);}
                    if(scenario=="scf_consume"){remag_phase_enter(1,0,1);reject(observe(2,value));remag_phase_enter(2,0,0);}
                }
                double reference=geometry?0.:value;
                if(scenario=="mismatch")reference+=1.;
                if(scenario=="nonfinite")reference=std::numeric_limits<double>::quiet_NaN();
                if(scenario!="empty")require(observe(table,reference)==0,"Observe");
                if(scenario=="snapshot")xyz[0]=3.;
                if(scenario=="stale_active") {
                    remag_geometry_changed();reject(remag_two_center_finish(consumer));reject(observe(table,reference));reject(remag_two_center_begin(1));break;
                }
                if(scenario=="mismatch"||scenario=="nonfinite") {
                    reject(remag_two_center_finish(consumer));reject(remag_two_center_begin(2));break;
                }
                require(remag_two_center_finish(consumer)==0,"Consumer finish");
                if(scenario=="snapshot")xyz[0]=.5;
                if(scenario=="duplicate_finish"&&consumer==1)reject(remag_two_center_finish(consumer));
            }
        }
        FILE* f=std::fopen(argv[2],"w");require(f,"Report file");
        std::fprintf(f,"{\"scenario\":\"%s\",\"passed\":true,\"epoch\":%llu,\"rejections\":[",scenario.c_str(),static_cast<unsigned long long>(remag::geometry_epoch()));
        for(size_t i=0;i<rejected.size();i++)std::fprintf(f,"%s%d",i?",":"",rejected[i]);
        std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Two-center owner probe: %s\n",e.what());return 1;}
}
