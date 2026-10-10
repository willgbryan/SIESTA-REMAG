// SPDX-License-Identifier: GPL-3.0-or-later
#include "density_history_state.hpp"
#include "density_history.hpp"
#include "initial_state.hpp"
#include "final_grid_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace remag {
namespace {
static_assert(sizeof(DensityHistoryCommand)==48&&sizeof(DensityHistoryResult)==40,
    "Update the production history boundary audit for this ABI");
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void kernel(int e){check(static_cast<cudaError_t>(e));}
enum class Stage {empty,selected,geometry,prepared,bound,finished,failed};
struct Record {
    uint64_t token=0,epoch=0;
    int atoms=0,rows=0,capacity=0,spins=0,depth=0,mode=0,previous_count=0,count=0;
    bool reused=false,bound=false,finished=false,saved=false,forces_accepted=false;
    int status=-1,rank=0,fallback=0;
    size_t setup_bytes=0,initial_status_bytes=0,post_status_bytes=0,coefficient_bytes=0,workspace_bytes=0;
    int setup_copies=0,initial_status_copies=0,post_status_copies=0;
};
struct Evidence {
    std::vector<Record> rows;
    ~Evidence(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-density-history.json","w");if(!f)return;
        std::fputs("{\"schema\":1,\"scope\":\"Production device density history; geometry staging and explicit status/coefficient publications outside SCF\",\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++) {
            const auto& r=rows[i];
            std::fprintf(f,"%s{\"token\":%llu,\"epoch\":%llu,\"atoms\":%d,\"rows\":%d,\"capacity\":%d,\"spins\":%d,\"depth\":%d,\"mode\":%d,\"previous_count\":%d,\"count\":%d,\"reused\":%s,\"bound\":%s,\"finished\":%s,\"saved\":%s,\"forces_accepted\":%s,\"status\":%d,\"rank\":%d,\"fallback\":%d,\"setup_bytes\":%zu,\"setup_copies\":%d,\"initial_status_bytes\":%zu,\"initial_status_copies\":%d,\"post_status_bytes\":%zu,\"post_status_copies\":%d,\"coefficient_export_bytes\":%zu,\"workspace_bytes\":%zu}",
                i?",":"",static_cast<unsigned long long>(r.token),static_cast<unsigned long long>(r.epoch),
                r.atoms,r.rows,r.capacity,r.spins,r.depth,r.mode,r.previous_count,r.count,
                r.reused?"true":"false",r.bound?"true":"false",r.finished?"true":"false",r.saved?"true":"false",r.forces_accepted?"true":"false",
                r.status,r.rank,r.fallback,r.setup_bytes,r.setup_copies,r.initial_status_bytes,r.initial_status_copies,
                r.post_status_bytes,r.post_status_copies,r.coefficient_bytes,r.workspace_bytes);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
} evidence;
struct State {
    Stage stage=Stage::empty;
    bool configured=false,reuse=false;
    int depth=0,mode=0,count=0,atoms=0,rows=0,spins=0,capacity=0,column_capacity=0;
    uint64_t epoch=0;
    size_t record=0;
    const double* initial_density=nullptr;
    cudaStream_t stream=nullptr;
    std::unique_ptr<DensityHistoryPlan> plan;
    DensityHistoryCommand command{};
    DensityHistoryCommand* device_command=nullptr;
    DensityHistoryResult* result=nullptr;
    double *xyz=nullptr,*coefficients=nullptr;
    int *num=nullptr,*ptr=nullptr,*columns=nullptr;
    std::vector<void*> storage;
    std::vector<double> printed_coefficients;
    ~State(){
        if(stream)cudaStreamSynchronize(stream);
        plan.reset();for(auto p:storage)cudaFree(p);if(columns)cudaFree(columns);
        if(stream)cudaStreamDestroy(stream);
    }
    template<class T>T* allocate(size_t n){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n*sizeof(T))));storage.push_back(p);return p;}
    template<class T>void upload(T* dst,const T* src,size_t n){
        check(cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyHostToDevice,stream));
        auto& r=evidence.rows[record];r.setup_bytes+=n*sizeof(T);r.setup_copies++;
        transfer(Transfer::density_history_setup,true,n*sizeof(T));
    }
    DensityHistoryResult collect(uint64_t token,int expected_count,bool post=false){
        DensityHistoryResult value{};
        check(cudaMemcpyAsync(&value,result,sizeof(value),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));transfer(Transfer::density_history_output,false,sizeof(value));
        auto& r=evidence.rows[record];
        (post?r.post_status_bytes:r.initial_status_bytes)+=sizeof(value);
        (post?r.post_status_copies:r.initial_status_copies)++;
        r.status=value.status;
        if(value.status||value.generation!=1||value.epoch!=token||value.count!=expected_count){stage=Stage::failed;throw std::runtime_error("Device history operation failed its status/epoch/count contract");}
        return value;
    }
} state;
void enabled(){require(density_history_enabled()&&initial_state_enabled()&&integral_force_enabled()&&!in_scf(),
    "Device history requires GPU initialization/populations/integral forces outside SCF");}
void configure(int depth,int mode){
    enabled();auto& s=state;
    require((s.stage==Stage::empty||s.stage==Stage::finished)&&depth>=0&&depth<=32&&mode>=0&&mode<=3,
        "History configuration changed during a live geometry or exceeds supported bounds");
    require(!s.configured||(depth==s.depth&&mode==s.mode),"History policy changed within this serial calculation");
    s.configured=true;s.depth=depth;s.mode=mode;
}
void select(int reuse){
    enabled();auto& s=state;
    require(s.configured&&(s.stage==Stage::empty||s.stage==Stage::finished)&&(reuse==0||reuse==1)&&(!reuse||s.count>0),
        "Invalid history source selection");
    s.reuse=reuse;s.stage=Stage::selected;
}
void geometry(int atoms,int rows,int capacity,int spins,const int* num,const int* ptr,const int* cols,const int* nsc,const double* xyz){
    enabled();auto& s=state;
    require(s.stage==Stage::selected&&atoms>0&&rows>0&&rows<=1000000&&atoms<=1000000&&capacity>0&&capacity<=INT_MAX/8&&
        (spins==1||spins==2||spins==4||spins==8)&&num&&ptr&&cols&&nsc&&xyz,"Invalid production history geometry");
    require(!s.plan||(atoms==s.atoms&&rows==s.rows&&spins==s.spins),"History atom/orbital/spin distribution changed");
    Record r;r.token=evidence.rows.size()+1;r.atoms=atoms;r.rows=rows;r.capacity=capacity;r.spins=spins;r.depth=s.depth;r.mode=s.mode;
    r.previous_count=s.count;r.count=s.count;r.reused=s.reuse;s.record=evidence.rows.size();evidence.rows.push_back(r);
    if(!s.plan){
        s.atoms=atoms;s.rows=rows;s.spins=spins;
        check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
        s.plan=std::make_unique<DensityHistoryPlan>(DensityHistoryConfig{rows,rows,atoms,spins,capacity,s.depth},s.stream);
        s.device_command=s.allocate<DensityHistoryCommand>(1);s.result=s.allocate<DensityHistoryResult>(1);
        s.num=s.allocate<int>(rows);s.ptr=s.allocate<int>(rows);s.xyz=s.allocate<double>(size_t(3)*atoms);
        s.coefficients=s.allocate<double>(std::max(1,s.depth));
        s.command={1,1,0,capacity,{nsc[0],nsc[1],nsc[2]},s.mode};
        s.upload(s.device_command,&s.command,1);kernel(s.plan->reset(s.device_command,s.result));s.collect(0,0);
    }
    kernel(s.plan->reserve(capacity));
    if(capacity>s.column_capacity){
        int* next=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&next),size_t(capacity)*sizeof(int)));
        if(s.columns)check(cudaFree(s.columns));
        s.columns=next;s.column_capacity=capacity;
    }
    s.capacity=capacity;s.command={1,1,r.token,capacity,{nsc[0],nsc[1],nsc[2]},s.mode};
    s.upload(s.device_command,&s.command,1);s.upload(s.num,num,rows);s.upload(s.ptr,ptr,rows);
    s.upload(s.columns,cols,capacity);s.upload(s.xyz,xyz,size_t(3)*atoms);
    kernel(s.plan->advance(s.device_command,s.num,s.ptr,s.columns,s.result));s.collect(r.token,s.count);
    s.printed_coefficients.clear();s.initial_density=nullptr;s.epoch=0;s.stage=Stage::geometry;
    evidence.rows[s.record].workspace_bytes=s.plan->workspace_bytes()+sizeof(DensityHistoryCommand)+sizeof(DensityHistoryResult)+
        size_t(2*rows+s.column_capacity)*sizeof(int)+size_t(3*atoms+std::max(1,s.depth))*sizeof(double);
}
void coefficients(int count,double* output){
    enabled();auto& s=state;
    require(s.stage==Stage::prepared&&s.reuse&&count==s.count&&count>0&&output,"History coefficient output has no matching request");
    if(s.printed_coefficients.empty()){
        s.printed_coefficients.resize(count);
        check(cudaMemcpyAsync(s.printed_coefficients.data(),s.coefficients,size_t(count)*sizeof(double),cudaMemcpyDeviceToHost,s.stream));
        check(cudaStreamSynchronize(s.stream));transfer(Transfer::density_history_output,false,size_t(count)*sizeof(double));
        evidence.rows[s.record].coefficient_bytes=size_t(count)*sizeof(double);
    }
    std::copy(s.printed_coefficients.begin(),s.printed_coefficients.end(),output);
}
void finish(int save){
    enabled();auto& s=state;
    require(s.stage==Stage::bound&&(save==0||save==1)&&(!save||s.depth>0)&&final_grid_forces_complete(s.epoch),
        "History retention precedes accepted forces or duplicates a geometry");
    auto& r=evidence.rows[s.record];r.forces_accepted=true;
    if(save){
        auto view=borrow_final_history_state(s.epoch);
        require(view.shape.nuo==s.rows&&view.shape.nnz==s.capacity&&view.h_spins==s.spins&&view.convergence,
            "Final history density has a different shape or lacks device convergence");
        kernel(s.plan->push(s.device_command,view.density,s.xyz,view.convergence,s.result));
        auto result=s.collect(s.command.epoch,std::min(s.depth,s.count+1),true);
        require(result.last_push==s.command.epoch,"History push did not commit this geometry");
        complete_final_history_state(s.epoch);s.count=result.count;r.saved=true;
    }
    r.count=s.count;r.finished=true;s.stage=Stage::finished;
}
}
bool density_history_enabled(){const char* p=std::getenv("REMAG_GPU_HISTORY");return p&&std::strcmp(p,"1")==0;}
bool density_history_reuses(){enabled();require(state.stage==Stage::geometry,"History density requested outside initialization");return state.reuse;}
void prepare_history_density(double* density){
    enabled();auto& s=state;require(s.stage==Stage::geometry&&density,"Missing initial history density allocation");
    if(s.reuse){
        kernel(s.plan->extrapolate(s.device_command,s.xyz,density,s.coefficients,s.result));
        auto value=s.collect(s.command.epoch,s.count);auto& r=evidence.rows[s.record];r.rank=value.rank;r.fallback=value.fallback;
    }
    s.initial_density=density;s.stage=Stage::prepared;
}
void bind_history_density(uint64_t epoch,const double* density){
    enabled();auto& s=state;
    require(s.stage==Stage::prepared&&epoch&&epoch==geometry_epoch()&&density==s.initial_density,
        "History density lease does not match the initial/SCF owner");
    s.epoch=epoch;s.stage=Stage::bound;auto& r=evidence.rows[s.record];r.epoch=epoch;r.bound=true;
}
}
namespace {
template<class F>int bridge(F f){try{f();return 0;}catch(const std::exception& e){std::fprintf(stderr,"Device density history: %s\n",e.what());return -190;}}
}
extern "C" int remag_history_enabled(){return remag::density_history_enabled();}
extern "C" int remag_history_configure(int d,int m){return bridge([&]{remag::configure(d,m);});}
extern "C" int remag_history_count(){try{remag::enabled();remag::require(remag::state.configured&&remag::state.stage!=remag::Stage::failed,"History policy is unavailable");return remag::state.count;}catch(const std::exception& e){std::fprintf(stderr,"Device history count: %s\n",e.what());return -190;}}
extern "C" int remag_history_select(int r){return bridge([&]{remag::select(r);});}
extern "C" int remag_history_geometry(int a,int n,int c,int s,const int* num,const int* ptr,const int* col,const int* nsc,const double* xyz){return bridge([&]{remag::geometry(a,n,c,s,num,ptr,col,nsc,xyz);});}
extern "C" int remag_history_coefficients(int n,double* values){return bridge([&]{remag::coefficients(n,values);});}
extern "C" int remag_history_finish(int save){return bridge([&]{remag::finish(save);});}
