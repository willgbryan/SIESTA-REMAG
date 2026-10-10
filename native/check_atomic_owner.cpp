// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_basis_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
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
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));check(cudaMalloc(reinterpret_cast<void**>(&status),4));check(cudaMemset(status,0,4));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(other)cudaStreamSynchronize(other);if(status)cudaFree(status);if(stream)cudaStreamDestroy(stream);if(other)cudaStreamDestroy(other);}
};
}
int main(int argc,char** argv){
    try {
        require(argc==5,"Expected scenario, core flag, shadow flag, report path");std::string mode=argv[1];int core=std::stoi(argv[2]);require(core==0||core==1,"Invalid core mode");
        const bool shadow=std::stoi(argv[3])==1;
        auto setmode=[&](bool mode){setenv("REMAG_ATOMIC_BASIS_SHADOW",mode?"1":"0",1);setenv("REMAG_ATOMIC_BASIS",mode?"0":"1",1);};
        setenv("REMAG_GRID_FORCE","1",1);setenv("REMAG_ATOMIC_GRID_FORCE","1",1);setmode(shadow);
        Context ctx;int graphs=0,status=-1;std::vector<std::array<double,19>> probes;std::vector<std::string> rejections;
        auto reject=[&](const char* label,auto fn){bool threw=false;try{fn();}catch(const std::exception&){threw=true;}require(threw,"Expected rejection missing");rejections.emplace_back(label);};
        auto rejectc=[&](const char* label,int code){require(code!=0,"Expected C rejection missing");rejections.emplace_back(label);};
        int config[]={1,4,2,1,1,2,1,2,1,2,1,1,core},z[]={26},index[]={2,0,2,2},pa[]={1,1},po[]={1,2},pc[]={1,2},one=1,mesh[]={2,1,1};
        double params[]={1.,2.,1.,2.},values[]={.3,.7,.3,.7},second[]={0.,0.,0.,0.},offset[]={.2,0.,0.,2.,0.,0.},zero[]={0.,0.,0.};
        auto stage=[&]{return remag_atomic_basis_stage(config,z,index,params,values,second,pa,po,pc,&one,&one,offset,zero,zero);};
        require(remag_phase_enter(0,0,0)==0,"Initialization phase");remag_geometry_changed();auto epoch=remag::geometry_epoch();int first=core?1:2;
        auto begin=[&](int k){remag::begin_atomic_basis(epoch,k,2,1,1,mesh,1,core);};
        auto finish=[&]{remag::finish_atomic_basis(epoch,status);};
        if(mode=="begin_without_stage")reject(mode.c_str(),[&]{begin(first);});
        if(mode=="stage_in_scf"){require(remag_phase_enter(1,0,1)==0,"SCF phase");rejectc(mode.c_str(),stage());require(remag_phase_enter(0,0,0)==0,"Init phase");}
        if(mode=="wrong_mode"){setenv("REMAG_ATOMIC_GRID_FORCE","0",1);rejectc(mode.c_str(),stage());setenv("REMAG_ATOMIC_GRID_FORCE","1",1);}
        if(mode=="bad_shape"){config[3]=2;rejectc(mode.c_str(),stage());config[3]=1;}
        if(mode=="duplicate_pair"){po[1]=1;rejectc(mode.c_str(),stage());po[1]=2;}
        if(mode=="device_error")values[0]=values[2]=std::numeric_limits<double>::quiet_NaN();
        if(mode=="both_modes"){setenv("REMAG_ATOMIC_BASIS_SHADOW","1",1);setenv("REMAG_ATOMIC_BASIS","1",1);rejectc(mode.c_str(),stage());setmode(shadow);}
        require(stage()==0,"Initial stage failed");
        if(mode=="duplicate_stage")rejectc(mode.c_str(),stage());
        if(mode=="unconsumed_epoch"){
            remag_geometry_changed();rejectc(mode.c_str(),stage());reject("stale_after_epoch_change",[&]{begin(first);});
        }else{
            if(mode=="snapshot")offset[0]=.9;
            if(mode=="begin_in_scf"){require(remag_phase_enter(1,0,1)==0,"SCF phase");reject(mode.c_str(),[&]{begin(first);});}
            require(remag_phase_enter(2,0,0)==0,"Post phase");
            if(mode=="stale_epoch")reject(mode.c_str(),[&]{remag::begin_atomic_basis(epoch+1,first,2,1,1,mesh,1,core);});
            if(mode=="wrong_shape")reject(mode.c_str(),[&]{remag::begin_atomic_basis(epoch,first,3,1,1,mesh,1,core);});
            if(mode=="wrong_kind")reject(mode.c_str(),[&]{begin(3);});
            if(mode=="out_of_order")reject(mode.c_str(),[&]{begin(core?2:1);});
            if(mode=="mode_change"){setmode(!shadow);reject(mode.c_str(),[&]{begin(first);});setmode(shadow);}
            if(mode=="finish_before_begin")reject(mode.c_str(),finish);
            if(mode=="prepare_before_begin")reject(mode.c_str(),[&]{remag::prepare_atomic_basis(epoch,first,0,ctx.stream);});
            auto execute=[&](bool inject,double x){
                for(int k=first;k<=2;k++){
                    begin(k);bool fault=inject&&k==first;double grad[]={.2,0.,0.},dx[]={x,0.,0.};
                    auto point=[&]{return remag_atomic_basis_point(k,1,1,1,1,1,.3+.2*x,grad,dx);};
                    if(fault&&mode=="duplicate_begin")reject(mode.c_str(),[&]{begin(k);});
                    if(fault&&mode=="invalid_point")rejectc(mode.c_str(),remag_atomic_basis_point(k,1,1,1,2,1,.34,grad,dx));
                    if(fault&&mode=="unstaged_pair")rejectc(mode.c_str(),remag_atomic_basis_point(k,1,3,1,1,1,.34,grad,dx));
                    if(shadow&&mode!="missing_point")require(point()==0,"Reference point failed");
                    if(fault&&mode=="cpu_callback")rejectc(mode.c_str(),point());
                    if(fault&&mode=="duplicate_point")rejectc(mode.c_str(),point());
                    if(fault&&mode=="finish_before_generation")reject(mode.c_str(),finish);
                    if(fault&&mode=="generate_before_prepare")reject(mode.c_str(),[&]{remag::enqueue_atomic_basis(epoch,k,ctx.status,ctx.stream);});
                    if(fault&&mode=="selected_count")reject(mode.c_str(),[&]{remag::prepare_atomic_basis(epoch,k,2,ctx.stream);});
                    auto view=remag::prepare_atomic_basis(epoch,k,!shadow||mode=="missing_point"?0:1,ctx.stream);
                    if(fault&&mode=="duplicate_prepare")reject(mode.c_str(),[&]{remag::prepare_atomic_basis(epoch,k,1,ctx.stream);});
                    if(fault&&mode=="point_after_prepare")rejectc(mode.c_str(),point());
                    if(fault&&mode=="wrong_stream")reject(mode.c_str(),[&]{remag::enqueue_atomic_basis(epoch,k,ctx.status,ctx.other);});
                    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
                    check(cudaStreamBeginCapture(ctx.stream,cudaStreamCaptureModeGlobal));remag::enqueue_atomic_basis(epoch,k,ctx.status,ctx.stream);
                    if(fault&&mode=="duplicate_generation")reject(mode.c_str(),[&]{remag::enqueue_atomic_basis(epoch,k,ctx.status,ctx.stream);});
                    check(cudaStreamEndCapture(ctx.stream,&graph));size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));require(n==size_t(shadow?2:1),"Graph count differs");
                    std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden graph node");}
                    check(cudaGraphInstantiate(&executable,graph,0));if(mode!="unlaunched")check(cudaGraphLaunch(executable,ctx.stream));check(cudaStreamSynchronize(ctx.stream));graphs++;
                    if(!shadow){
                        require(view.points==2&&view.entries==2&&view.atoms==1&&view.active,"Invalid production view");
                        int masks[2],points[2],atoms[2];double gradients[6],displacements[6];
                        check(cudaMemcpy(masks,view.active,sizeof(masks),cudaMemcpyDeviceToHost));check(cudaMemcpy(points,view.point_index,sizeof(points),cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(atoms,view.atom_index,sizeof(atoms),cudaMemcpyDeviceToHost));check(cudaMemcpy(gradients,view.gradient,sizeof(gradients),cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(displacements,view.displacement,sizeof(displacements),cudaMemcpyDeviceToHost));
                        std::array<double,19> probe{};probe[0]=x;
                        for(int j=0;j<2;j++){probe[1+j]=masks[j];probe[3+j]=points[j];probe[5+j]=atoms[j];}
                        for(int j=0;j<6;j++){probe[7+j]=gradients[j];probe[13+j]=displacements[j];}
                        require(masks[0]==1&&masks[1]==0&&points[0]==1&&points[1]==2&&atoms[0]==1&&atoms[1]==1,"Generated selection/map mismatch");
                        for(int j=0;j<6;j++){
                            require(std::isfinite(gradients[j])&&std::abs(gradients[j]-(j==0&&mode!="device_error"?.2:0.))<1.e-14,"Generated gradient mismatch");
                            require(displacements[j]==(j==0?x:j==3?2.:0.),"Generated displacement mismatch");
                        }
                        probes.push_back(probe);
                    }
                    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
                    if(fault&&mode=="missing_term"&&core)reject(mode.c_str(),finish);
                }
                check(cudaMemcpy(&status,ctx.status,4,cudaMemcpyDeviceToHost));
                if(inject&&mode=="finish_in_scf"){require(remag_phase_enter(1,0,1)==0,"SCF phase");reject(mode.c_str(),finish);require(remag_phase_enter(2,0,0)==0,"Post phase");}
                if(mode=="sticky"){status=7;reject(mode.c_str(),finish);}
                else if(mode=="device_error"||mode=="missing_point"||mode=="unlaunched")reject(mode.c_str(),finish);
                else finish();
            };
            execute(true,.2);
            if(mode=="duplicate_finish")reject(mode.c_str(),finish);
            if(mode=="reuse"){
                require(remag_phase_enter(0,1,0)==0,"Next init phase");remag_geometry_changed();epoch=remag::geometry_epoch();offset[0]=.4;
                require(stage()==0,"Next stage failed");require(remag_phase_enter(2,1,0)==0,"Next post phase");execute(false,.4);
            }
        }
        FILE* f=std::fopen(argv[4],"w");require(f,"Report failed");std::fprintf(f,"{\"scenario\":\"%s\",\"core\":%d,\"shadow\":%s,\"passed\":true,\"graphs\":%d,\"status\":%d,\"rejections\":[",mode.c_str(),core,shadow?"true":"false",graphs,status);
        for(size_t i=0;i<rejections.size();i++)std::fprintf(f,"%s\"%s\"",i?",":"",rejections[i].c_str());std::fputs("],\"probes\":[",f);
        for(size_t i=0;i<probes.size();i++){std::fputs(i?",[":"[",f);for(size_t j=0;j<probes[i].size();j++)std::fprintf(f,"%s%.17g",j?",":"",probes[i][j]);std::fputs("]",f);}
        std::fputs("]}\n",f);std::fclose(f);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Atomic owner probe: %s\n",e.what());return 1;}
}
