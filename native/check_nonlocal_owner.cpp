// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_force_state.hpp"
#include "final_grid_state.hpp"
#include "grid.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int remag_grid(int,int,int,int,int,int,int,int,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const double*,double*,double);
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void ok(int code){require(code==0,"Owner call failed");}
struct Device {
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    Device(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Device(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(size_t n,const T* a){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)));memory.push_back(p);
        if(n)check(cudaMemcpy(p,a,n*sizeof(T),cudaMemcpyHostToDevice));
        return p;}
};
struct Fixture {
    int num[2]={3,2},ptr[2]={1,5},list[8]={0,1,1,2,0,1,2,0};
    int row[2]={1,2},column[2]={1,2},atom[2]={1,2},offset[2]={0,1},angular[2]={0,0};
    int pairs[8]={1,1,1,2,2,1,2,2};
    double epsilon[4]={.7,.7,-.4,-.4},xyz[12]={.4,-.3,.2,.4,-.3,.2,-.1,.6,-.4,-.1,.6,-.4};
    double integrals[16]={.3,.1,-.2,.05,.1,.05,.07,-.03,.2,-.15,.1,.2,-.15,-.08,.12,-.04};
    double initial[24]={},volume=2.3;std::vector<double> density;
    Fixture(int spins,int geometry):density(8*spins){
        for(int s=0;s<spins;s++)for(int i=0;i<8;i++)density[8*s+i]=.025*(s+1)+.01*i+.125*geometry;
        for(int i=9;i<24;i++)initial[i]=.01*(i-8);
        for(int i=0;i<12;i++)xyz[i]+=.03*geometry;
        for(int i=0;i<16;i++)integrals[i]+=.02*geometry;
    }
    std::vector<double> reference(int spins,int split,bool empty) const {
        std::vector<double> out(initial,initial+24);if(empty)return out;
        for(int i=0;i<2;i++)for(int j=0;j<2;j++){
            double diagonal=0.;int last=-1;
            for(int k=ptr[i];k<ptr[i]+num[i];k++)if(list[k]==column[j]){
                last=k;if(spins==8)diagonal=0.;
                for(int s=0;s<std::min(2,spins);s++)diagonal+=density[8*s+k];
            }
            if(last<0)continue;
            for(int b=0;b<2;b++){
                const double* left=integrals+4*(2*i+b);const double* right=integrals+4*(2*j+b);
                const double potential=epsilon[2*b]*left[0]*right[0];
                if(spins!=8||split)out[0]+=diagonal*potential;
                else{out[1]+=density[last]*potential;out[2]+=density[8+last]*potential;out[5]+=density[32+last]*potential;out[6]+=density[40+last]*potential;}
                for(int axis=0;axis<3;axis++){
                    const double force=2.*diagonal*epsilon[2*b]*right[0]*left[1+axis];
                    out[9+3*i+axis]-=force;out[12+axis]+=force;
                    for(int k=0;k<3;k++)out[15+3*axis+k]+=xyz[6*i+k]*force/volume;
                }
            }
        }return out;
    }
};
struct Cycle {Fixture input;std::vector<double> expected,retained;int status=-1,nodes=0;bool executed=false,completed=false;
    Cycle(const Fixture& f,int spins,int split,bool empty):input(f),expected(f.reference(spins,split,empty)){}
};
template<class T>void array(FILE* f,const char* name,const T* a,size_t n){std::fprintf(f,",\"%s\":[",name);for(size_t i=0;i<n;i++)std::fprintf(f,"%s%.17g",i?",":"",double(a[i]));std::fputc(']',f);}
}
int main(int argc,char** argv){
    try{
        require(argc==5,"Expected scenario, spins, split, report");std::string mode=argv[1];const int spins=std::stoi(argv[2]),split=std::stoi(argv[3]);
        require((spins==1||spins==2||spins==4||spins==8)&&(split==0||(spins==8&&split==1)),"Unsupported fixture spin/split");
        require(mode=="normal"||mode=="captured"||mode=="recovery"||mode=="empty"||mode=="failed_reference"||mode=="failed_status"||mode=="stale_setup"||mode=="stale_staged","Unknown scenario");
        for(const char* name:{"REMAG_NONLOCAL_FORCE_SHADOW","REMAG_PAIR_FORCE_SHADOW","REMAG_TWO_CENTER_SHADOW","REMAG_GRID_FORCE"})setenv(name,"1",1);
        for(const char* name:{"REMAG_ATOMIC_GRID_FORCE","REMAG_ATOMIC_FORCE_SHADOW","REMAG_GRID_VERIFY_EPOCH"})setenv(name,"0",1);
        const bool recovery=mode=="recovery",empty=mode=="empty",failed=mode=="failed_reference"||mode=="failed_status",stale=mode=="stale_setup"||mode=="stale_staged";
        const int ng=std::min(4,spins),consumer=spins==8?3:2,geometries=recovery?2:1;
        auto device=std::make_unique<Device>();auto& d=*device;std::vector<std::string> rejected;std::vector<Cycle> cycles;
        auto reject=[&](const char* label,auto fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}require(caught,label);rejected.emplace_back(label);};
        for(int g=0;g<geometries;g++){
            ok(remag_phase_enter(0,g,0));Fixture f(spins,g);cycles.emplace_back(f,spins,split,empty);auto& cycle=cycles.back();
            auto begin=[&](int stage,double volume){return remag_nonlocal_begin(stage,consumer,2,2,2,8,split,volume,f.num,f.ptr,f.list,f.initial+9,f.initial+15);};
            auto center=[&]{return remag_nonlocal_center(2,2,2,2,f.row,f.column,f.atom,f.offset,f.angular,f.epsilon);};
            auto observe=[&](int q){remag::observe_nonlocal_geometry(1,f.pairs[2*q],f.pairs[2*q+1],f.xyz+3*q);};
            auto bad_centers=[&](const std::string& prefix){
                auto attempt=[&](const char* label,auto change,auto restore){change();reject((prefix+label).c_str(),[&]{ok(center());});restore();};
                attempt("row",[&]{f.row[1]=3;},[&]{f.row[1]=2;});
                attempt("column",[&]{f.column[1]=3;},[&]{f.column[1]=2;});
                attempt("atom",[&]{f.atom[1]=3;},[&]{f.atom[1]=2;});
                attempt("offset",[&]{f.offset[1]=2;},[&]{f.offset[1]=1;});
                attempt("angular_low",[&]{f.angular[1]=std::numeric_limits<int>::min();},[&]{f.angular[1]=0;});
                attempt("angular_high",[&]{f.angular[1]=std::numeric_limits<int>::max();},[&]{f.angular[1]=0;});
                attempt("epsilon",[&]{f.epsilon[3]=std::numeric_limits<double>::quiet_NaN();},[&]{f.epsilon[3]=-.4;});
                reject((prefix+"coverage").c_str(),[&]{ok(remag_nonlocal_center(1,2,2,2,f.row,f.column,f.atom,f.offset,f.angular,f.epsilon));});
            };
            if(recovery){
                reject("missing_end",[&]{ok(remag_nonlocal_end(0,nullptr,nullptr,nullptr));});
                reject("missing_bind",[&]{ok(remag_nonlocal_bind_geometry());});
                reject("bad_mode",[&]{ok(begin(2,f.volume));});
                reject("bad_volume",[&]{ok(begin(0,0.));});
                for(const char* flag:{"REMAG_NONLOCAL_FORCE_SHADOW","REMAG_PAIR_FORCE_SHADOW","REMAG_TWO_CENTER_SHADOW"}){
                    setenv(flag,"0",1);reject(flag,[&]{ok(begin(0,f.volume));});setenv(flag,"1",1);
                }
                f.ptr[1]=-1;reject("bad_span",[&]{ok(begin(0,f.volume));});f.ptr[1]=5;
                f.ptr[1]=2;reject("overlapping_rows",[&]{ok(begin(0,f.volume));});f.ptr[1]=5;
                f.list[1]=3;reject("bad_sparse_column",[&]{ok(begin(0,f.volume));});f.list[1]=1;
                ok(remag_phase_enter(1,g,1));reject("prepare_in_scf",[&]{ok(begin(0,f.volume));});ok(remag_phase_enter(0,g,0));
            }
            ok(begin(0,f.volume));
            if(recovery){
                reject("duplicate_prepare",[&]{ok(begin(0,f.volume));});
                reject("early_bind",[&]{ok(remag_nonlocal_bind_geometry());});
                reject("early_center",[&]{ok(center());});
                reject("table",[&]{remag::observe_nonlocal_geometry(2,1,1,f.xyz);});
                reject("registry_a",[&]{remag::observe_nonlocal_geometry(1,0,1,f.xyz);});
                reject("registry_b",[&]{remag::observe_nonlocal_geometry(1,1,0,f.xyz);});
                reject("null_xyz",[&]{remag::observe_nonlocal_geometry(1,1,1,nullptr);});
                double bad[3]={.4,-.3,std::numeric_limits<double>::quiet_NaN()};
                reject("nonfinite_xyz",[&]{remag::observe_nonlocal_geometry(1,1,1,bad);});
                ok(remag_phase_enter(1,g,1));reject("observe_in_scf",[&]{observe(0);});ok(remag_phase_enter(0,g,0));
            }
            if(!empty){for(int q=0;q<4;q++)observe(q);
                if(recovery){bad_centers("initial_");reject("unfinished_center",[&]{ok(remag_nonlocal_end(0,nullptr,nullptr,nullptr));});}
                ok(center());}
            ok(remag_nonlocal_end(0,nullptr,nullptr,nullptr));
            if(recovery){
                reject("bind_without_epoch_advance",[&]{ok(remag_nonlocal_bind_geometry());});
                reject("duplicate_initial_end",[&]{ok(remag_nonlocal_end(0,nullptr,nullptr,nullptr));});
                reject("center_after_staging",[&]{ok(center());});
            }
            remag_geometry_changed();const auto epoch=remag::geometry_epoch();ok(remag_nonlocal_bind_geometry());
            if(recovery)reject("duplicate_bind",[&]{ok(remag_nonlocal_bind_geometry());});
            int nsc[3]={1,1,1},ends[2]={0,2},orbs[2]={1,2};float phi[2]={.5f,.25f};double rho[4]={};
            auto grid=[&](const int* num){ok(remag_grid(0,2,1,1,2,8,ng,spins,nsc,ends,orbs,num,f.ptr,f.list,phi,f.density.data(),rho,1.));};
            grid(f.num);
            if(recovery)reject("missing_retained",[&]{ok(begin(1,f.volume));});
            std::vector<double> energy(8*ng,.2),potential(ng,.1);double zero=0.;
            auto D=d.upload(f.density.size(),f.density.data()),E=d.upload(energy.size(),energy.data()),V=d.upload(potential.size(),potential.data()),A=d.upload(1,&zero),R=d.upload(1,&zero);
            auto integrals=d.upload(empty?0:16,f.integrals);int initial_status=mode=="failed_status"?7:0;auto status=d.upload(1,&initial_status);
            ok(remag_phase_enter(2,g,0));remag::retain_final_grid_state({{2,1,1,2,8,1},spins,ng,epoch,1.,f.volume,D,E,V,A,R,0,{1,1,1},1,0},d.stream);
            check(cudaMemset(D,0,f.density.size()*8));
            if(recovery)reject("missing_kinetic",[&]{ok(begin(1,f.volume));});
            auto kinetic=remag::borrow_final_pair_state(epoch,1);cycle.retained.resize(f.density.size());
            check(cudaMemcpy(cycle.retained.data(),kinetic.density,f.density.size()*8,cudaMemcpyDeviceToHost));
            require(cycle.retained==f.density,"Snapshot borrowed mutable source D");remag::complete_final_pair_shadow(epoch,1);
            if(mode=="stale_setup"){
                remag_geometry_changed();reject("stale_begin",[&]{ok(begin(1,f.volume));});
            }else{
                if(recovery){
                    reject("changed_volume",[&]{ok(begin(1,f.volume+1.));});
                    reject("changed_consumer",[&]{ok(remag_nonlocal_begin(1,consumer==2?3:2,2,2,2,8,split,f.volume,f.num,f.ptr,f.list,f.initial+9,f.initial+15));});
                    reject("changed_split",[&]{ok(remag_nonlocal_begin(1,consumer,2,2,2,8,1-split,f.volume,f.num,f.ptr,f.list,f.initial+9,f.initial+15));});
                    reject("changed_shape",[&]{ok(remag_nonlocal_begin(1,consumer,2,2,3,8,split,f.volume,f.num,f.ptr,f.list,f.initial+9,f.initial+15));});
                    f.num[0]=2;reject("changed_num",[&]{ok(begin(1,f.volume));});f.num[0]=3;
                    f.ptr[0]=0;reject("changed_ptr",[&]{ok(begin(1,f.volume));});f.ptr[0]=1;
                    f.list[1]=2;reject("changed_list",[&]{ok(begin(1,f.volume));});f.list[1]=1;
                    int other_num[2]={2,2};grid(other_num);reject("grid_fingerprint",[&]{ok(begin(1,f.volume));});grid(f.num);
                    f.initial[9]=std::numeric_limits<double>::quiet_NaN();reject("nonfinite_initial",[&]{ok(begin(1,f.volume));});f.initial[9]=.01;
                    ok(remag_phase_enter(1,g,1));reject("final_begin_in_scf",[&]{ok(begin(1,f.volume));});ok(remag_phase_enter(2,g,0));
                }
                ok(begin(1,f.volume));
                if(recovery){
                    reject("duplicate_final_begin",[&]{ok(begin(1,f.volume));});
                    reject("early_final_end",[&]{ok(remag_nonlocal_end(1,f.initial,f.initial+9,f.initial+15));});
                    reject("early_pending",[&]{remag::nonlocal_force_shadow_pending(consumer);});
                    f.pairs[0]=3;reject("changed_registry",[&]{observe(0);});f.pairs[0]=1;
                    double saved=f.xyz[2];f.xyz[2]+=.5;reject("changed_xyz",[&]{observe(0);});f.xyz[2]=saved;
                }
                if(!empty){for(int q=0;q<4;q++)observe(q);
                    if(recovery){
                        reject("extra_query",[&]{observe(0);});bad_centers("final_");
                        f.row[1]=0;reject("changed_local_row",[&]{ok(center());});f.row[1]=2;
                        reject("changed_center",[&]{ok(remag_nonlocal_center(2,2,1,2,f.row,f.column,f.atom,f.offset,f.angular,f.epsilon));});
                    }
                    ok(center());
                    if(recovery)reject("extra_center",[&]{ok(remag_nonlocal_center(0,2,2,2,f.row,f.column,f.atom,f.offset,f.angular,f.epsilon));});
                }
                auto reference=cycle.expected;if(mode=="failed_reference")reference[0]+=1.;
                if(recovery){
                    auto bad=reference;bad[23]=std::numeric_limits<double>::quiet_NaN();
                    reject("nonfinite_reference",[&]{ok(remag_nonlocal_end(1,bad.data(),bad.data()+9,bad.data()+15));});
                    reject("stage_without_reference",[&]{remag::stage_nonlocal_force_shadow(consumer,4,f.pairs,f.xyz,integrals,d.stream);});
                }
                ok(remag_nonlocal_end(1,reference.data(),reference.data()+9,reference.data()+15));
                require(remag::nonlocal_force_shadow_pending(consumer),"Completed reference not pending");
                auto stage=[&]{return remag::stage_nonlocal_force_shadow(consumer,empty?0:4,empty?nullptr:f.pairs,empty?nullptr:f.xyz,integrals,d.stream);};
                if(recovery){
                    reject("duplicate_reference",[&]{ok(remag_nonlocal_end(1,reference.data(),reference.data()+9,reference.data()+15));});
                    reject("center_after_reference",[&]{ok(center());});
                    reject("wrong_pending",[&]{remag::nonlocal_force_shadow_pending(consumer==2?3:2);});
                    f.pairs[0]=3;reject("stage_pairs",[&]{stage();});f.pairs[0]=1;
                    double saved=f.xyz[11];f.xyz[11]+=.5;reject("stage_xyz",[&]{stage();});f.xyz[11]=saved;
                    reject("stage_count",[&]{remag::stage_nonlocal_force_shadow(consumer,3,f.pairs,f.xyz,integrals,d.stream);});
                    reject("stage_consumer",[&]{remag::stage_nonlocal_force_shadow(consumer==2?3:2,4,f.pairs,f.xyz,integrals,d.stream);});
                    reject("apply_before_stage",[&]{remag::apply_nonlocal_force_shadow(status,d.stream);});
                    reject("finish_before_stage",[&]{remag::finish_nonlocal_force_shadow(0);});
                }
                stage();
                if(mode=="stale_staged"){
                    remag_geometry_changed();reject("stale_apply",[&]{remag::apply_nonlocal_force_shadow(status,d.stream);});
                    reject("stale_finish",[&]{remag::finish_nonlocal_force_shadow(0);});
                }else{
                    if(recovery){
                        reject("duplicate_stage",[&]{stage();});
                        reject("finish_before_apply",[&]{remag::finish_nonlocal_force_shadow(0);});
                        reject("wrong_stream",[&]{remag::apply_nonlocal_force_shadow(status,nullptr);});
                        reject("missing_status",[&]{remag::apply_nonlocal_force_shadow(nullptr,d.stream);});
                        ok(remag_phase_enter(1,g,1));reject("apply_in_scf",[&]{remag::apply_nonlocal_force_shadow(status,d.stream);});ok(remag_phase_enter(2,g,0));
                    }
                    if(mode=="captured"){
                        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
                        check(cudaStreamBeginCapture(d.stream,cudaStreamCaptureModeGlobal));remag::apply_nonlocal_force_shadow(status,d.stream);check(cudaStreamEndCapture(d.stream,&graph));
                        size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));
                        cycle.nodes=int(n);for(auto node:nodes){cudaGraphNodeType kind;check(cudaGraphNodeGetType(node,&kind));require(kind==cudaGraphNodeTypeKernel,"Forbidden owner graph node");}
                        check(cudaGraphInstantiate(&executable,graph,0));check(cudaGraphLaunch(executable,d.stream));check(cudaStreamSynchronize(d.stream));
                        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
                    }else remag::apply_nonlocal_force_shadow(status,d.stream);
                    cycle.executed=true;
                    if(recovery)reject("duplicate_apply",[&]{remag::apply_nonlocal_force_shadow(status,d.stream);});
                    check(cudaStreamSynchronize(d.stream));check(cudaMemcpy(&cycle.status,status,4,cudaMemcpyDeviceToHost));
                    if(failed){
                        reject("failed_completion",[&]{remag::finish_nonlocal_force_shadow(cycle.status);});
                        reject("retry_completion",[&]{remag::finish_nonlocal_force_shadow(0);});
                    }else{remag::finish_nonlocal_force_shadow(cycle.status);cycle.completed=true;}
                }
            }
            if(failed||stale){
                require(!remag::final_grid_forces_complete(epoch),"Failed owner allowed analysis");
                reject("held_grid",[&]{remag::borrow_final_grid_state(epoch,4);});
                reject("held_overlap",[&]{remag::borrow_final_pair_state(epoch,4);});
                reject("held_reset",[&]{remag::reset_final_grid_state();});
                reject("held_prepare",[&]{ok(begin(0,f.volume));});
            }else{
                if(recovery){
                    reject("duplicate_finish",[&]{remag::finish_nonlocal_force_shadow(0);});
                    reject("missing_final_begin",[&]{ok(begin(1,f.volume));});
                }
                remag::borrow_final_grid_state(epoch,4);remag::complete_final_grid_force(epoch,4);
                remag::borrow_final_pair_state(epoch,4);remag::complete_final_pair_shadow(epoch,4);
                require(remag::final_grid_forces_complete(epoch),"Completed owner did not allow analysis");remag::reset_final_grid_state();
            }
        }
        FILE* out=std::fopen(argv[4],"w");require(out,"Cannot create probe report");
        std::fprintf(out,"{\"scenario\":\"%s\",\"spins\":%d,\"split\":%d,\"held\":%s,\"rejections\":[",mode.c_str(),spins,split,(failed||stale)?"true":"false");
        for(size_t i=0;i<rejected.size();i++)std::fprintf(out,"%s\"%s\"",i?",":"",rejected[i].c_str());
        std::fputs("],\"cycles\":[",out);
        for(size_t i=0;i<cycles.size();i++){const auto& c=cycles[i];const auto& f=c.input;
            std::fprintf(out,"%s{\"geometry\":%zu,\"volume\":%.17g,\"queries\":%d,\"executed\":%s,\"completed\":%s,\"status\":%d,\"graph_nodes\":%d",i?",":"",i,f.volume,empty?0:4,c.executed?"true":"false",c.completed?"true":"false",c.status,c.nodes);
            array(out,"num",f.num,2);array(out,"ptr",f.ptr,2);array(out,"list",f.list,8);array(out,"pairs",f.pairs,8);array(out,"xyz",f.xyz,12);array(out,"integrals",f.integrals,16);array(out,"epsilon",f.epsilon,4);
            array(out,"initial",f.initial,24);array(out,"density",f.density.data(),f.density.size());array(out,"expected",c.expected.data(),c.expected.size());array(out,"retained",c.retained.data(),c.retained.size());std::fputc('}',out);
        }std::fputs("]}\n",out);std::fclose(out);
        // A terminally failed production owner deliberately retains its lease.
        // Keep its borrowed stream alive through process teardown as well.
        if(failed||stale)device.release();
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Nonlocal owner probe: %s\n",e.what());return 1;}
}
