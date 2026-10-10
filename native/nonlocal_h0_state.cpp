#include "spin_rotation.hpp"
// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_h0_state.hpp"
#include "two_center_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int remag_initialization_enabled();
namespace remag {
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Record {
    uint64_t preparation_epoch=0,epoch=0;
    int rows=0,columns=0,capacity=0,consumer=0,split=0,queries=0,local_neighbors=0,nodes=0,status=-1;
    size_t staged=0,allocated=0,exported=0,raw_bytes=0;
    bool compared=false,passed=false,bound=false,force_borrowed=false,complete=false;
    bool production=false,scf_borrowed=false,scf_released=false,poison_requested=false,host_poisoned=false;
    int query_expansion_nodes=0;bool compact_queries=false;
    double scalar_error=0.,soc_error=0.;std::string file;
    double geometry_prepare=-1.,owner_prepare=-1.,graph_execution=-1.,validation_export=-1.;
};
struct Evidence {
    std::vector<Record> rows;
    ~Evidence(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-nonlocal-h0.json","w");if(!f)return;
        std::fputs("{\"schema\":1,\"scope\":\"Nonlocal H0 device owner; optional upstream matrix comparison; integrals retained through authoritative forces\",\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++){
            const auto& r=rows[i];char scalar[64],soc[64];
            if(std::isfinite(r.scalar_error))std::snprintf(scalar,sizeof(scalar),"%.17g",r.scalar_error);else std::strcpy(scalar,"null");
            if(std::isfinite(r.soc_error))std::snprintf(soc,sizeof(soc),"%.17g",r.soc_error);else std::strcpy(soc,"null");
            if(!r.compared){std::strcpy(scalar,"null");std::strcpy(soc,"null");}
            std::fprintf(f,"%s{\"preparation_epoch\":%llu,\"epoch\":%llu,\"rows\":%d,\"columns\":%d,\"capacity\":%d,\"consumer\":%d,\"split\":%d,\"queries\":%d,\"local_neighbors\":%d,\"graph_nodes\":%d,\"status\":%d,\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"export_bytes\":%zu,\"raw_bytes\":%zu,\"raw_file\":\"%s\",\"compared\":%s,\"passed\":%s,\"bound\":%s,\"force_borrowed\":%s,\"complete\":%s,\"scalar_max_error\":%s,\"soc_max_error\":%s,\"derivative_upload_bytes\":0,\"integral_download_bytes\":0",
                i?",":"",static_cast<unsigned long long>(r.preparation_epoch),static_cast<unsigned long long>(r.epoch),
                r.rows,r.columns,r.capacity,r.consumer,r.split,r.queries,r.local_neighbors,r.nodes,r.status,
                r.staged,r.allocated,r.exported,r.raw_bytes,r.file.c_str(),r.compared?"true":"false",r.passed?"true":"false",
                r.bound?"true":"false",r.force_borrowed?"true":"false",r.complete?"true":"false",scalar,soc);
            std::fprintf(f,",\"compact_queries\":%s,\"query_expansion_nodes\":%d",r.compact_queries?"true":"false",r.query_expansion_nodes);
            std::fprintf(f,",\"production\":%s,\"scf_borrowed\":%s,\"scf_released\":%s",
                r.production?"true":"false",r.scf_borrowed?"true":"false",r.scf_released?"true":"false");
            std::fprintf(f,",\"poison_requested\":%s,\"host_poisoned\":%s,\"setup_timing_seconds\":{",
                r.poison_requested?"true":"false",r.host_poisoned?"true":"false");
            const char* timing_names[]={"geometry_preparation_and_staging","owner_preparation","graph_launch_and_wait","validation_and_export"};
            const double timings[]={r.geometry_prepare,r.owner_prepare,r.graph_execution,r.validation_export};
            for(int j=0;j<4;j++){
                std::fprintf(f,"%s\"%s\":",j?",":"",timing_names[j]);
                if(timings[j]>=0.&&std::isfinite(timings[j]))std::fprintf(f,"%.17g",timings[j]);else std::fputs("null",f);
            }
            std::fputs("}}",f);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
};
Evidence& evidence(){static Evidence records;return records;}
struct State {
    std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
    size_t record=0;std::vector<double> initial;std::vector<H0Complex> initial_soc;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;double* integrals=nullptr;double* h=nullptr;H0Complex* z=nullptr;bool capturing=false;
    ~State(){
        if(capturing)cudaStreamEndCapture(stream,&graph);
        if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);
        if(graph)cudaGraphDestroy(graph);
        for(auto p:memory)cudaFree(p);
        if(stream)cudaStreamDestroy(stream);
    }
    template<class T>T* allocate(size_t count){
        T* p=nullptr;const size_t bytes=std::max(count,size_t(1))*sizeof(T);
        check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);evidence().rows[record].allocated+=bytes;return p;
    }
    template<class T>T* upload(size_t count,const T* input){
        auto p=allocate<T>(count);
        if(count){check(cudaMemcpyAsync(p,input,count*sizeof(T),cudaMemcpyHostToDevice,stream));
            evidence().rows[record].staged+=count*sizeof(T);transfer(evidence().rows[record].production?Transfer::nonlocal_h0:Transfer::nonlocal_h0_shadow,true,count*sizeof(T));}
        return p;
    }
    template<class T>void download(size_t count,T* output,const T* input){
        if(count){check(cudaMemcpy(output,input,count*sizeof(T),cudaMemcpyDeviceToHost));
            evidence().rows[record].exported+=count*sizeof(T);transfer(evidence().rows[record].production?Transfer::nonlocal_h0:Transfer::nonlocal_h0_shadow,false,count*sizeof(T));}
    }
};
std::unique_ptr<State> state;
Record& current(){
    require(nonlocal_h0_active()&&!in_scf()&&state,"Missing/out-of-phase H0 owner");
    return evidence().rows[state->record];
}
template<class T>void save(FILE* file,const T* values,size_t count){
    require(std::fwrite(values,sizeof(T),count,file)==count,"Cannot retain H0 matrix comparison");
}
}
bool nonlocal_h0_shadow_enabled(){const char* v=std::getenv("REMAG_NONLOCAL_H0_SHADOW");return v&&std::strcmp(v,"1")==0;}
bool nonlocal_h0_enabled(){const char* v=std::getenv("REMAG_NONLOCAL_H0");return v&&std::strcmp(v,"1")==0;}
bool nonlocal_h0_active(){return nonlocal_h0_enabled()||nonlocal_h0_shadow_enabled();}
bool nonlocal_h0_poison_enabled(){const char* v=std::getenv("REMAG_NONLOCAL_H0_POISON_HOST");return v&&std::strcmp(v,"1")==0;}
void begin_nonlocal_h0_shadow(int rows,int columns,int capacity,int consumer,int split,const double* initial,const H0Complex* initial_soc){
    require(nonlocal_h0_active()&&!(nonlocal_h0_enabled()&&nonlocal_h0_shadow_enabled())&&!in_scf()&&!state&&rows>0&&columns>=rows&&capacity>0&&
        (consumer==2||consumer==3)&&(split==0||split==1)&&initial&&(consumer!=3||initial_soc),"Invalid H0 shadow start");
    require(size_t(capacity)*(consumer==3?72:8)<(size_t(4)<<30),"H0 shadow allocation limit");
    auto s=std::make_unique<State>();s->record=evidence().rows.size();
    Record r;r.preparation_epoch=geometry_epoch();r.rows=rows;r.columns=columns;r.capacity=capacity;r.consumer=consumer;r.split=split;
    r.production=nonlocal_h0_enabled();r.poison_requested=nonlocal_h0_poison_enabled();
    require(!r.poison_requested||r.production,"H0 host poisoning requires production device matrices");
    require(!r.production||remag_initialization_enabled(),"Production H0 requires GPU initialization");
    if(!r.production)s->initial.assign(initial,initial+capacity);
    require(std::all_of(s->initial.begin(),s->initial.end(),[](double x){return std::isfinite(x);}),"Nonfinite initial H0");
    if(consumer==3&&!r.production){
        s->initial_soc.assign(initial_soc,initial_soc+4*size_t(capacity));
        require(std::all_of(s->initial_soc.begin(),s->initial_soc.end(),[](auto x){return std::isfinite(x.real)&&std::isfinite(x.imag);}),"Nonfinite initial SOC H0");
    }
    evidence().rows.push_back(r);state=std::move(s);
}
void evaluate_nonlocal_h0_shadow(NonlocalH0View v,const NonlocalNeighbor* host_neighbors,const int* pairs,const double* xyz,
        const double* reference,const H0Complex* reference_soc,const NonlocalQueryView* compact){
    auto& r=current();auto& s=*state;
    require(!r.passed&&!r.compared&&!r.bound&&r.preparation_epoch==geometry_epoch()&&reference&&
        v.rows==r.rows&&v.columns==r.columns&&v.capacity==r.capacity&&v.soc==(r.consumer==3)&&v.split==r.split&&
        v.neighbors>=0&&v.queries>=0&&(v.neighbors==0||host_neighbors)&&(v.queries==0||(pairs&&xyz))&&(!v.soc||reference_soc),
        "H0 shadow geometry/reference mismatch");
    require(bool(compact)==r.production,"Production H0 requires compact query metadata");
    if(compact){
        require(compact->centers==v.centers&&compact->neighbors==v.neighbors&&compact->queries==v.queries&&
            compact->center==v.center&&compact->neighbor==v.neighbor,"H0 compact query geometry differs");
        r.compact_queries=true;r.query_expansion_nodes=v.centers?1+bool(v.neighbors):0;
    }
    const auto geometry_ready=std::chrono::steady_clock::now();
    r.geometry_prepare=std::chrono::duration<double>(geometry_ready-s.started).count();
    r.queries=v.queries;
    std::vector<int> counts(v.rows),offsets(size_t(v.rows)+1),indices;
    for(int i=0;i<v.neighbors;i++){
        const int row=host_neighbors[i].row;require(row>=0&&row<=v.rows,"Invalid H0 neighbor row");if(row)counts[row-1]++;
    }
    for(int row=0;row<v.rows;row++)offsets[row+1]=offsets[row]+counts[row];
    indices.resize(offsets.back());auto positions=offsets;
    for(int i=0;i<v.neighbors;i++)if(host_neighbors[i].row)indices[positions[host_neighbors[i].row-1]++]=i;
    r.local_neighbors=int(indices.size());v.local_neighbors=r.local_neighbors;
    require(size_t(32)*v.queries+size_t(4)*(v.rows+1+indices.size())+size_t(v.capacity)*(v.soc?72:8)+4<(size_t(4)<<30),"H0 integral owner allocation limit");
    check(cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking));
    v.row_ptr=s.upload(offsets.size(),offsets.data());v.row_neighbors=s.upload(indices.size(),indices.data());
    if(r.production){
        s.h=s.allocate<double>(v.capacity);check(cudaMemsetAsync(s.h,0,size_t(v.capacity)*sizeof(double),s.stream));
        if(v.soc){s.z=s.allocate<H0Complex>(4*size_t(v.capacity));check(cudaMemsetAsync(s.z,0,4*size_t(v.capacity)*sizeof(H0Complex),s.stream));}
    }else{
        s.h=s.upload(s.initial.size(),s.initial.data());s.z=v.soc?s.upload(s.initial_soc.size(),s.initial_soc.data()):nullptr;
    }
    auto h=s.h;auto z=s.z;
    int zero=0;auto status=s.upload(1,&zero);s.integrals=s.allocate<double>(4*size_t(v.queries));v.integrals=s.integrals;
    const auto table=borrow_integral_two_center(r.consumer);
    check(cudaStreamSynchronize(s.stream));
    check(cudaStreamBeginCapture(s.stream,cudaStreamCaptureModeGlobal));s.capturing=true;
    if(compact)check(expand_nonlocal_queries_device(*compact,const_cast<int*>(pairs),const_cast<double*>(xyz),status,s.stream));
    if(v.queries)check(evaluate_two_center_device(table,v.queries,pairs,xyz,s.integrals,status,s.stream));
    check(add_nonlocal_h0_device(v,h,z,status,s.stream));
    check(cudaStreamEndCapture(s.stream,&s.graph));s.capturing=false;
    size_t count=0;check(cudaGraphGetNodes(s.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(s.graph,nodes.data(),&count));r.nodes=int(count);
    require(r.nodes==(v.queries?3:2)+r.query_expansion_nodes,"Unexpected H0 setup graph size");
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden H0 setup graph node");}
    check(cudaGraphInstantiate(&s.executable,s.graph,0));
    const auto graph_ready=std::chrono::steady_clock::now();
    r.owner_prepare=std::chrono::duration<double>(graph_ready-geometry_ready).count();
    check(cudaGraphLaunch(s.executable,s.stream));check(cudaStreamSynchronize(s.stream));
    const auto graph_done=std::chrono::steady_clock::now();
    r.graph_execution=std::chrono::duration<double>(graph_done-graph_ready).count();
    if(r.production){
        s.download(1,&r.status,status);r.passed=r.status==0;
        r.validation_export=std::chrono::duration<double>(std::chrono::steady_clock::now()-graph_done).count();
        require(r.passed,"GPU H0 assembly failed; scientific outputs invalid");return;
    }
    std::vector<double> actual(v.capacity);std::vector<H0Complex> actual_soc(s.initial_soc.size());
    s.download(actual.size(),actual.data(),h);s.download(actual_soc.size(),actual_soc.data(),z);s.download(1,&r.status,status);
    r.compared=true;r.passed=r.status==0;
    for(size_t i=0;i<actual.size();i++){
        const double error=std::abs(actual[i]-reference[i]);r.scalar_error=std::max(r.scalar_error,error);
        r.passed&=std::isfinite(reference[i])&&std::isfinite(actual[i])&&error<=1e-12+1e-12*std::abs(reference[i]);
    }
    for(size_t i=0;i<actual_soc.size();i++){
        const auto a=actual_soc[i],b=reference_soc[i];const double error=std::hypot(a.real-b.real,a.imag-b.imag);
        r.soc_error=std::max(r.soc_error,error);
        r.passed&=std::isfinite(b.real)&&std::isfinite(b.imag)&&std::isfinite(a.real)&&std::isfinite(a.imag)&&error<=1e-12+1e-12*std::hypot(b.real,b.imag);
    }
    char name[96];std::snprintf(name,sizeof(name),"remag-nonlocal-h0-%04zu.bin",s.record+1);r.file=name;
    FILE* file=std::fopen(name,"wb");require(file,"Cannot open H0 matrix evidence");
    try{
        save(file,s.initial.data(),s.initial.size());save(file,reference,actual.size());save(file,actual.data(),actual.size());
        if(v.soc){save(file,s.initial_soc.data(),s.initial_soc.size());save(file,reference_soc,actual_soc.size());save(file,actual_soc.data(),actual_soc.size());}
    }catch(...){std::fclose(file);throw;}
    require(std::fclose(file)==0,"Cannot close H0 matrix evidence");
    r.raw_bytes=3*size_t(v.capacity)*(v.soc?72:8);
    r.validation_export=std::chrono::duration<double>(std::chrono::steady_clock::now()-graph_done).count();
    require(r.passed,"GPU H0 matrix differs from upstream; retained failing comparison");
}
void record_nonlocal_h0_host_poison(int capacity,const double* h,const H0Complex* z){
    auto& r=current();require(r.production&&r.poison_requested&&r.passed&&!r.bound&&!r.host_poisoned&&
        capacity==r.capacity&&h&&(r.consumer!=3||z),"Invalid H0 host poisoning control");
    require(std::all_of(h,h+capacity,[](double x){return std::isnan(x);}),"Host scalar H0 was not poisoned");
    if(r.consumer==3)require(std::all_of(z,z+4*size_t(capacity),[](auto x){return std::isnan(x.real)&&std::isnan(x.imag);}),
        "Host SOC H0 was not poisoned");
    r.host_poisoned=true;
}
void bind_nonlocal_h0_shadow(uint64_t epoch){
    if(!nonlocal_h0_active())return;
    auto& r=current();require((r.production||r.compared)&&r.passed&&(!r.poison_requested||r.host_poisoned)&&!r.bound&&epoch==geometry_epoch()&&epoch>r.preparation_epoch,"H0 geometry bind order");
    r.epoch=epoch;r.bound=true;
}
NonlocalH0Matrices borrow_nonlocal_h0_matrices(uint64_t epoch,int rows,int columns,int capacity,int soc){
    auto& r=current();require(r.production&&r.bound&&r.passed&&!r.scf_borrowed&&!r.complete&&
        epoch==r.epoch&&epoch==geometry_epoch()&&rows==r.rows&&columns==r.columns&&capacity==r.capacity&&
        (soc==0||soc==2)&&(soc==2)==(r.consumer==3)&&state->h&&((soc==2)==bool(state->z)),
        "Invalid retained H0 SCF lease");
    r.scf_borrowed=true;return {state->h,state->z};
}
void release_nonlocal_h0_matrices(uint64_t epoch){
    auto& r=current();require(r.production&&r.scf_borrowed&&!r.scf_released&&!r.complete&&
        epoch==r.epoch&&epoch==geometry_epoch(),"Invalid retained H0 SCF release");r.scf_released=true;
}
const double* borrow_nonlocal_h0_integrals(uint64_t epoch,int consumer,int queries){
    auto& r=current();require(r.bound&&r.passed&&(!r.production||(r.scf_borrowed&&r.scf_released))&&!r.force_borrowed&&!r.complete&&epoch==r.epoch&&epoch==geometry_epoch()&&consumer==r.consumer&&queries==r.queries&&state->integrals,
        "Invalid retained H0 integral lease");r.force_borrowed=true;return state->integrals;
}
void finish_nonlocal_h0_shadow(uint64_t epoch){
    if(!nonlocal_h0_active())return;
    auto& r=current();require(r.bound&&(!r.production||(r.scf_borrowed&&r.scf_released))&&r.force_borrowed&&!r.complete&&r.epoch==epoch&&epoch==geometry_epoch(),"H0 integral release before final force completion");
    r.complete=true;if(!remag_orientation_retain_geometry())state.reset();
}
}
extern "C" int remag_nonlocal_h0_shadow_enabled(){return remag::nonlocal_h0_shadow_enabled();}
extern "C" int remag_nonlocal_h0_enabled(){return remag::nonlocal_h0_enabled();}
extern "C" int remag_nonlocal_h0_active(){return remag::nonlocal_h0_active();}
extern "C" int remag_nonlocal_h0_poison_enabled(){return remag::nonlocal_h0_poison_enabled();}

extern "C" int remag_reuse_nonlocal_h0() {
    try {
        auto& old=remag::current();auto& s=*remag::state;
        remag::require(remag_orientation_retain_geometry()&&old.production&&old.bound&&old.passed&&old.complete&&
            old.scf_borrowed&&old.scf_released&&old.force_borrowed&&old.status==0&&
            remag::geometry_epoch()==old.epoch+1,"H0 reuse precedes complete electronic/force leases");
        auto next=old;next.preparation_epoch=old.epoch;next.epoch=remag::geometry_epoch();
        next.scf_borrowed=false;next.scf_released=false;next.force_borrowed=false;next.complete=false;
        next.staged=0;next.allocated=0;next.exported=0;
        next.geometry_prepare=-1.;next.owner_prepare=-1.;next.graph_execution=-1.;next.validation_export=-1.;
        s.record=remag::evidence().rows.size();remag::evidence().rows.push_back(next);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG H0 geometry reuse: %s\n",e.what());return -280;}
}
