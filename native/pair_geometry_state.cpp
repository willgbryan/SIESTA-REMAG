#include "spin_rotation.hpp"
// SPDX-License-Identifier: GPL-3.0-or-later
#include "pair_geometry_state.hpp"
#include "final_grid_state.hpp"
#include "two_center_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Record {
    uint64_t epoch=0;int atoms=0,rows=0,columns=0,capacity=0,queries=0,consumed=0;
    double volume=0.;size_t staged=0,allocated=0;bool complete=false;
};
struct Records {
    std::vector<Record> rows;
    ~Records(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-pair-geometry.json","w");if(!f)return;
        const bool complete=std::all_of(rows.begin(),rows.end(),[](const auto& r){return r.complete;});
        std::fprintf(f,"{\"scope\":\"pre-SCF orbital-pair geometry shared by kinetic and overlap retained-D/E consumers\",\"complete\":%s,\"cycles\":[",complete?"true":"false");
        for(size_t i=0;i<rows.size();i++){const auto& r=rows[i];
            std::fprintf(f,"%s{\"epoch\":%llu,\"atoms\":%d,\"rows\":%d,\"columns\":%d,\"capacity\":%d,\"queries\":%d,\"volume\":%.17g,\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"consumer_mask\":%d,\"complete\":%s}",
                i?",":"",static_cast<unsigned long long>(r.epoch),r.atoms,r.rows,r.columns,r.capacity,r.queries,r.volume,r.staged,r.allocated,r.consumed,r.complete?"true":"false");}
        std::fputs("]}\n",f);std::fclose(f);
    }
};
Records& records(){static Records r;return r;}
struct Geometry {
    size_t record=0,seen=0;uint64_t preparation_epoch=0;
    int active=0;bool ready=false,bound=false,staged=false,integral=false;
    std::vector<int> num,ptr,list,row,column,atoms,pairs;
    std::vector<double> xyz;std::vector<void*> memory;
    cudaStream_t stream=nullptr;remag::PairGeometryView view{};
    ~Geometry(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* upload(const std::vector<T>& a){
        T* p=nullptr;size_t bytes=std::max(size_t(1),a.size())*sizeof(T);
        check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);auto& r=records().rows[record];r.allocated+=bytes;
        if(!a.empty()){check(cudaMemcpyAsync(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
            r.staged+=a.size()*sizeof(T);remag::transfer(remag::Transfer::pair_geometry,true,a.size()*sizeof(T));}return p;
    }
};
std::unique_ptr<Geometry> geometry;
void valid(){require(remag::pair_force_active()&&remag_two_center_active()&&!remag::in_scf(),"Pair geometry mode/phase mismatch");}
Record& current(){valid();require(bool(geometry),"Missing pair geometry");auto& r=records().rows[geometry->record];
    require(geometry->integral==remag::integral_force_enabled(),"Pair geometry mode changed");
    require(!geometry->bound||r.epoch==remag::geometry_epoch(),"Stale pair geometry");return r;}
int failed(const std::exception& e){std::fprintf(stderr,"REMAG pair geometry: %s\n",e.what());return -280;}
}
extern "C" int remag_pair_force_active(){return remag::pair_force_active();}
extern "C" int remag_pair_geometry_prepare(int atoms,int rows,int columns,int capacity,double volume,const int* num,const int* ptr,const int* list){
    try{
        valid();require(!geometry&&atoms>0&&rows>=atoms&&columns>=rows&&capacity>0&&capacity<=2147483647/8&&
            std::isfinite(volume)&&volume>0.&&num&&ptr&&list,"Invalid pair geometry preparation");
        auto g=std::make_unique<Geometry>();g->record=records().rows.size();g->preparation_epoch=remag::geometry_epoch();
        g->integral=remag::integral_force_enabled();
        g->num.assign(num,num+rows);g->ptr.assign(ptr,ptr+rows);g->list.assign(list,list+capacity);
        for(int i=0;i<rows;i++){
            require(ptr[i]>=0&&ptr[i]<=capacity&&num[i]>=0&&num[i]<=capacity-ptr[i],"Invalid pair geometry sparse span");
            for(int j=0;j<num[i];j++)require(list[ptr[i]+j]>=1&&list[ptr[i]+j]<=columns,"Invalid pair geometry sparse column");}
        Record r;r.atoms=atoms;r.rows=rows;r.columns=columns;r.capacity=capacity;r.volume=volume;
        records().rows.push_back(r);geometry=std::move(g);return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_pair_geometry_add(int row,int column,int a,int b,int registry_a,int registry_b,const double* xyz){
    try{
        auto& r=current();auto& g=*geometry;
        require(!g.ready&&!g.bound&&row>=1&&row<=r.rows&&column>=1&&column<=r.columns&&a>=1&&a<=r.atoms&&b>=1&&b<=r.atoms&&
            registry_a>0&&registry_b>0&&xyz,"Invalid pair geometry query");
        require(std::all_of(xyz,xyz+3,[](double v){return std::isfinite(v);}),"Nonfinite pair geometry displacement");
        require(48*(g.row.size()+1)+8*size_t(r.rows)+4*size_t(r.capacity)<=(size_t(4)<<30),"Pair geometry staging limit");
        g.row.push_back(row);g.column.push_back(column);g.atoms.push_back(a);g.atoms.push_back(b);
        g.pairs.push_back(registry_a);g.pairs.push_back(registry_b);g.xyz.insert(g.xyz.end(),xyz,xyz+3);return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_pair_geometry_finish(){
    try{
        auto& r=current();auto& g=*geometry;require(!g.ready&&!g.bound,"Pair geometry already staged");
        check(cudaStreamCreateWithFlags(&g.stream,cudaStreamNonBlocking));r.queries=int(g.row.size());auto& v=g.view;v.queries=r.queries;
        v.row=g.upload(g.row);v.column=g.upload(g.column);v.atoms=g.upload(g.atoms);v.pairs=g.upload(g.pairs);v.xyz=g.upload(g.xyz);
        v.num=g.upload(g.num);v.ptr=g.upload(g.ptr);v.list=g.upload(g.list);check(cudaStreamSynchronize(g.stream));g.ready=true;return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_pair_geometry_bind(){
    try{
        if(!remag::pair_force_active())return 0;
        auto& r=current();auto& g=*geometry;require(g.ready&&!g.bound&&remag::geometry_epoch()>g.preparation_epoch,"Pair geometry bind order");
        r.epoch=remag::geometry_epoch();g.bound=true;return 0;
    }catch(const std::exception& e){return failed(e);}
}
namespace remag {
void validate_pair_geometry(int consumer,int atoms,int rows,int columns,int capacity,double volume,const int* num,const int* ptr,const int* list){
    auto& r=current();auto& g=*geometry;
    require(g.ready&&g.bound&&!g.active&&((consumer==1&&r.consumed==0)||(consumer==4&&r.consumed==1))&&
        atoms==r.atoms&&rows==r.rows&&columns==r.columns&&capacity==r.capacity&&volume==r.volume&&num&&ptr&&list&&
        std::equal(g.num.begin(),g.num.end(),num)&&std::equal(g.ptr.begin(),g.ptr.end(),ptr)&&std::equal(g.list.begin(),g.list.end(),list),
        "Pair geometry changed or consumed out of order");
}
void begin_pair_geometry(int consumer){
    auto& r=current();auto& g=*geometry;require(g.ready&&g.bound&&!g.active&&((consumer==1&&r.consumed==0)||(consumer==4&&r.consumed==1)),"Pair geometry consumer order");
    g.active=consumer;g.seen=0;g.staged=false;
}
void observe_pair_geometry(int consumer,int row,int column,int a,int b){
    current();auto& g=*geometry;const size_t i=g.seen;
    require(g.active==consumer&&!g.staged&&i<g.row.size()&&g.row[i]==row&&g.column[i]==column&&g.atoms[2*i]==a&&g.atoms[2*i+1]==b,"Final pair query differs from pre-SCF geometry");
    g.seen++;
}
PairGeometryView stage_pair_geometry(int consumer,size_t n,const int* pairs,const double* xyz){
    auto& r=current();auto& g=*geometry;
    require(g.active==consumer&&!g.staged&&n==size_t(r.queries)&&g.seen==n&&(!n||(pairs&&xyz&&
        std::equal(g.pairs.begin(),g.pairs.end(),pairs)&&std::equal(g.xyz.begin(),g.xyz.end(),xyz))),"Final pair integral geometry differs from pre-SCF geometry");
    g.staged=true;return g.view;
}
void complete_pair_geometry(int consumer){
    auto& r=current();auto& g=*geometry;require(g.active==consumer&&g.staged&&(consumer==1||consumer==4),"Pair geometry completion without staged consumer");
    r.consumed|=consumer;g.active=0;g.staged=false;
    if(consumer==4&&!g.integral){r.complete=true;geometry.reset();}
}
PairGeometryView borrow_integral_pair_geometry(int consumer,int atoms,const FinalGridStateView& fields){
    auto& r=current();auto& g=*geometry;
    require(g.integral&&fields.epoch==r.epoch&&atoms==r.atoms&&fields.shape.nuo==r.rows&&fields.shape.no==r.columns&&
        fields.shape.nnz==r.capacity&&std::abs(fields.volume-r.volume)<=1e-12*std::max(1.,std::abs(r.volume)),
        "Production pair geometry differs from retained electronic state");
    validate_grid_sparse_topology(r.rows,r.columns,r.capacity,g.num.data(),g.ptr.data(),g.list.data());
    begin_pair_geometry(consumer);g.seen=r.queries;g.staged=true;return g.view;
}
void finish_integral_pair_geometry(uint64_t epoch,int status){
    auto& r=current();auto& g=*geometry;require(g.integral&&r.epoch==epoch&&!g.active&&r.consumed==5&&!r.complete&&status==0,
        "Production pair geometry publication before successful completion");
    r.complete=true;if(!remag_orientation_retain_geometry())geometry.reset();
}
}

extern "C" int remag_reuse_pair_geometry() {
    try {
        valid();require(remag_orientation_retain_geometry()&&geometry,"Missing retained pair geometry");
        auto& g=*geometry;const auto& old=records().rows[g.record];
        require(g.integral&&g.ready&&g.bound&&!g.active&&old.complete&&old.consumed==5&&
            remag::geometry_epoch()==old.epoch+1,"Pair geometry reuse precedes complete forces");
        Record next=old;next.epoch=remag::geometry_epoch();next.consumed=0;next.complete=false;
        next.staged=0;next.allocated=0;g.preparation_epoch=old.epoch;
        g.record=records().rows.size();records().rows.push_back(next);
        g.seen=0;g.staged=false;return 0;
    }catch(const std::exception& e){return failed(e);}
}
