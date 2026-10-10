// SPDX-License-Identifier: GPL-3.0-or-later
#include "pair_geometry_state.hpp"
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
void ok(int v){require(v==0,"Geometry operation failed");}
void check(cudaError_t v){require(v==cudaSuccess,"Geometry CUDA operation failed");}
template<class T>void append(std::vector<double>& out,const T* device,size_t n){
    std::vector<T> host(n);if(n)check(cudaMemcpy(host.data(),device,n*sizeof(T),cudaMemcpyDeviceToHost));
    out.insert(out.end(),host.begin(),host.end());
}
}
int main(int argc,char** argv){
    try{
        require(argc==3,"Expected scenario and report");const std::string mode=argv[1];
        require(mode=="normal"||mode=="empty"||mode=="recovery"||mode=="stale_begin"||mode=="stale_active"||mode=="stale_staged","Unknown geometry scenario");
        const bool recovery=mode=="recovery",empty=mode=="empty",stale=mode.rfind("stale_",0)==0;
        setenv("REMAG_PAIR_FORCE_SHADOW","1",1);setenv("REMAG_TWO_CENTER_SHADOW","1",1);
        std::vector<std::string> rejections;std::vector<std::vector<double>> snapshots;
        auto reject=[&](const char* label,auto operation){bool caught=false;try{operation();}catch(const std::exception&){caught=true;}require(caught,label);rejections.emplace_back(label);};
        const int geometries=recovery?2:1;
        for(int g=0;g<geometries;g++){
            ok(remag_phase_enter(0,g,0));int num[2]={3,2},ptr[2]={1,5},list[8]={0,1,1,2,0,1,2,0};
            int pairs[4]={1,2,2,1};double xyz[6]={.1,-.2,.3,-.1,.2,-.3};for(double& v:xyz)v+=.125*g;
            auto prepare=[&]{ok(remag_pair_geometry_prepare(2,2,2,8,2.3,num,ptr,list));};
            auto add=[&](int i){ok(remag_pair_geometry_add(i+1,2-i,i+1,2-i,pairs[2*i],pairs[2*i+1],xyz+3*i));};
            auto validate=[&](int c){remag::validate_pair_geometry(c,2,2,2,8,2.3,num,ptr,list);};
            auto stage=[&](int c){return remag::stage_pair_geometry(c,empty?0:2,empty?nullptr:pairs,empty?nullptr:xyz);};
            if(recovery){
                reject("missing_bind",[&]{ok(remag_pair_geometry_bind());});reject("missing_finish",[&]{ok(remag_pair_geometry_finish());});
                reject("bad_volume",[&]{ok(remag_pair_geometry_prepare(2,2,2,8,0.,num,ptr,list));});
                ptr[1]=-1;reject("bad_span",prepare);ptr[1]=5;list[1]=3;reject("bad_column",prepare);list[1]=1;
                ok(remag_phase_enter(1,g,1));reject("prepare_in_scf",prepare);ok(remag_phase_enter(0,g,0));
            }
            prepare();
            if(recovery){
                reject("duplicate_prepare",prepare);reject("early_bind",[&]{ok(remag_pair_geometry_bind());});reject("early_begin",[&]{remag::begin_pair_geometry(1);});
                reject("query_row",[&]{ok(remag_pair_geometry_add(3,2,1,2,1,2,xyz));});
                reject("query_column",[&]{ok(remag_pair_geometry_add(1,3,1,2,1,2,xyz));});
                reject("query_atom",[&]{ok(remag_pair_geometry_add(1,2,1,3,1,2,xyz));});
                reject("query_registry",[&]{ok(remag_pair_geometry_add(1,2,1,2,0,2,xyz));});
                double bad[3]={.1,.2,std::numeric_limits<double>::quiet_NaN()};reject("query_xyz",[&]{ok(remag_pair_geometry_add(1,2,1,2,1,2,bad));});
                ok(remag_phase_enter(1,g,1));reject("query_in_scf",[&]{add(0);});ok(remag_phase_enter(0,g,0));
            }
            if(!empty){add(0);add(1);}ok(remag_pair_geometry_finish());
            if(recovery){
                reject("duplicate_finish",[&]{ok(remag_pair_geometry_finish());});reject("query_after_finish",[&]{add(0);});
                reject("bind_without_epoch",[&]{ok(remag_pair_geometry_bind());});
            }
            remag_geometry_changed();ok(remag_pair_geometry_bind());
            if(recovery){reject("duplicate_bind",[&]{ok(remag_pair_geometry_bind());});reject("overlap_first",[&]{validate(4);});}
            ok(remag_phase_enter(2,g,0));
            if(mode=="stale_begin"){
                remag_geometry_changed();reject("stale_validate",[&]{validate(1);});reject("held_prepare",prepare);continue;
            }
            for(int c:{1,4}){
                if(recovery){
                    reject("changed_shape",[&]{remag::validate_pair_geometry(c,2,2,3,8,2.3,num,ptr,list);});
                    num[0]=2;reject("changed_num",[&]{validate(c);});num[0]=3;
                    ptr[0]=0;reject("changed_ptr",[&]{validate(c);});ptr[0]=1;
                    list[1]=2;reject("changed_list",[&]{validate(c);});list[1]=1;
                    reject("changed_volume",[&]{remag::validate_pair_geometry(c,2,2,2,8,3.3,num,ptr,list);});
                }
                validate(c);remag::begin_pair_geometry(c);
                if(recovery){
                    reject("duplicate_begin",[&]{remag::begin_pair_geometry(c);});reject("incomplete_stage",[&]{stage(c);});
                    reject("changed_query",[&]{remag::observe_pair_geometry(c,2,2,1,2);});
                }
                if(!empty){remag::observe_pair_geometry(c,1,2,1,2);remag::observe_pair_geometry(c,2,1,2,1);}
                if(mode=="stale_active"){
                    remag_geometry_changed();reject("stale_stage",[&]{stage(c);});reject("stale_complete",[&]{remag::complete_pair_geometry(c);});reject("held_prepare",prepare);break;
                }
                if(recovery){
                    reject("extra_query",[&]{remag::observe_pair_geometry(c,2,1,2,1);});
                    pairs[0]=2;reject("changed_pairs",[&]{stage(c);});pairs[0]=1;
                    double saved=xyz[0];xyz[0]+=.5;reject("changed_xyz",[&]{stage(c);});xyz[0]=saved;
                    reject("changed_count",[&]{remag::stage_pair_geometry(c,1,pairs,xyz);});
                }
                const auto view=stage(c);std::vector<double> values;
                // Mutating caller storage must not change the prepared device snapshot.
                const double saved=xyz[0];xyz[0]=99.;list[1]=99;
                const size_t n=empty?0:2;
                require(view.queries==int(n),"Staged query count differs");
                append(values,view.row,n);append(values,view.column,n);append(values,view.atoms,2*n);
                append(values,view.pairs,2*n);append(values,view.xyz,3*n);
                append(values,view.num,2);append(values,view.ptr,2);append(values,view.list,8);
                xyz[0]=saved;list[1]=1;snapshots.push_back(values);
                if(mode=="stale_staged"){
                    remag_geometry_changed();reject("stale_complete",[&]{remag::complete_pair_geometry(c);});reject("held_prepare",prepare);break;
                }
                if(recovery){reject("duplicate_stage",[&]{stage(c);});reject("wrong_complete",[&]{remag::complete_pair_geometry(c==1?4:1);});}
                remag::complete_pair_geometry(c);
                if(recovery)reject("duplicate_complete",[&]{remag::complete_pair_geometry(c);});
            }
        }
        FILE* f=std::fopen(argv[2],"w");require(f,"Cannot write geometry report");
        std::fprintf(f,"{\"scenario\":\"%s\",\"geometries\":%d,\"held\":%s,\"rejections\":[",mode.c_str(),geometries,stale?"true":"false");
        for(size_t i=0;i<rejections.size();i++)std::fprintf(f,"%s\"%s\"",i?",":"",rejections[i].c_str());
        std::fputs("],\"snapshots\":[",f);
        for(size_t i=0;i<snapshots.size();i++){std::fprintf(f,"%s[",i?",":"");for(size_t j=0;j<snapshots[i].size();j++)std::fprintf(f,"%s%.17g",j?",":"",snapshots[i][j]);std::fputc(']',f);}
        std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Pair geometry probe: %s\n",e.what());return 1;}
}
