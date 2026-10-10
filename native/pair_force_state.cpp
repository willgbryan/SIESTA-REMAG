// SPDX-License-Identifier: GPL-3.0-or-later
#include "pair_force_state.hpp"
#include "pair_force.hpp"
#include "two_center_state.hpp"
#include "final_grid_state.hpp"
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
    uint64_t epoch=0;int consumer=0,atoms=0,rows=0,columns=0,capacity=0,spinor=0,components=0,queries=0,status=-1;
    bool complete=false,passed=false;size_t staged=0,allocated=0,exported=0;
    double volume=0.;std::vector<double> initial,reference,actual;
};
struct Records {
    std::vector<Record> rows;
    ~Records(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-pair-force.json","w");if(!f)return;
        bool complete=rows.back().consumer==4&&std::all_of(rows.begin(),rows.end(),[](const auto& r){return r.complete&&r.passed;});
        std::fprintf(f,"{\"scope\":\"kinetic/overlap shadow borrowing final device D/E; CPU outputs authoritative\",\"geometry_staged_before_scf\":true,\"complete\":%s,\"cycles\":[",complete?"true":"false");
        for(size_t i=0;i<rows.size();i++){
            const auto& r=rows[i];
            std::fprintf(f,"%s{\"epoch\":%llu,\"consumer\":%d,\"atoms\":%d,\"rows\":%d,\"columns\":%d,\"capacity\":%d,\"spinor\":%d,\"components\":%d,\"queries\":%d,\"volume\":%.17g,\"status\":%d,\"complete\":%s,\"passed\":%s,\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"export_bytes\":%zu,\"electronic_inputs_resident\":true,\"electronic_upload_bytes\":0,\"contraction_graph_nodes\":%d",
                i?",":"",static_cast<unsigned long long>(r.epoch),r.consumer,r.atoms,r.rows,r.columns,r.capacity,r.spinor,r.components,r.queries,r.volume,r.status,r.complete?"true":"false",r.passed?"true":"false",r.staged,r.allocated,r.exported,r.complete?1:0);
            const char* names[]={"initial","reference","actual"};const std::vector<double>* arrays[]={&r.initial,&r.reference,&r.actual};
            for(int k=0;k<3;k++){std::fprintf(f,",\"%s\":[",names[k]);for(size_t j=0;j<arrays[k]->size();j++){double v=(*arrays[k])[j];std::fprintf(f,"%s",j?",":"");if(std::isfinite(v))std::fprintf(f,"%.17g",v);else std::fputs("null",f);}std::fputc(']',f);}
            std::fputc('}',f);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
};
Records& records(){static Records r;return r;}
struct Batch {
    size_t record=0;remag::FinalGridStateView fields{};remag::PairForceView view{};
    std::vector<void*> memory;
    cudaStream_t stream=nullptr;remag::PairForceProblem* problem=nullptr;double* output=nullptr;
    bool referenced=false,staged=false,applied=false;
    ~Batch(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);}
    template<class T>T* upload(size_t n,const T* host){
        T* p=nullptr;size_t bytes=std::max(size_t(1),n)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);
        auto& r=records().rows[record];r.allocated+=bytes;
        if(n){check(cudaMemcpyAsync(p,host,n*sizeof(T),cudaMemcpyHostToDevice,stream));r.staged+=n*sizeof(T);remag::transfer(remag::Transfer::pair_force_shadow,true,n*sizeof(T));}return p;
    }
};
std::unique_ptr<Batch> batch;
void valid(){require(remag::pair_force_shadow_enabled()&&remag_two_center_shadow_enabled()&&!remag::in_scf(),"Pair shadow phase/mode mismatch");}
Record& current(){valid();require(batch&&records().rows[batch->record].epoch==remag::geometry_epoch(),"Missing/stale pair shadow");return records().rows[batch->record];}
std::vector<double> values(int n,double e,const double* f,const double* s){
    require(f&&s,"Missing pair force reference");std::vector<double> v(1,e);v.insert(v.end(),f,f+3*n);v.insert(v.end(),s,s+9);
    require(std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);}),"Nonfinite pair reference");return v;
}
int failed(const std::exception& e){std::fprintf(stderr,"REMAG pair force: %s\n",e.what());return -260;}
}
extern "C" int remag_pair_force_shadow_enabled(){return remag::pair_force_shadow_enabled()?1:0;}
extern "C" int remag_pair_force_begin(int consumer,int atoms,int rows,int columns,int capacity,int spinor,double volume,
        const int* num,const int* ptr,const int* list,const double* force,const double* stress){
    try {
        valid();require(!batch&&(consumer==1||consumer==4)&&atoms>0&&rows>=atoms&&columns>=rows&&capacity>0&&capacity<=2147483647/8&&
            (spinor==1||spinor==2)&&std::isfinite(volume)&&volume>0&&num&&ptr&&list,"Invalid pair shadow geometry");
        require(remag::two_center_query_count(consumer)==0,"Pair shadow must begin before integral queries");
        // Check dimensions before using them to read caller metadata.
        remag::validate_grid_sparse_topology(rows,columns,capacity,num,ptr,list);
        remag::validate_pair_geometry(consumer,atoms,rows,columns,capacity,volume,num,ptr,list);
        auto s=std::make_unique<Batch>();
        auto initial=values(atoms,0.,force,stress);
        auto fields=remag::borrow_final_pair_state(remag::geometry_epoch(),consumer);
        require(fields.shape.nuo==rows&&fields.shape.no==columns&&fields.shape.nnz==capacity&&
            spinor==std::min(2,fields.h_spins)&&std::abs(volume-fields.volume)<=1.e-12*std::max(1.,std::abs(volume)),"Pair/final electronic shape or volume mismatch");
        Record r;r.epoch=fields.epoch;r.consumer=consumer;r.atoms=atoms;r.rows=rows;r.columns=columns;r.capacity=capacity;r.spinor=spinor;
        r.components=consumer==1?fields.h_spins:fields.energy_spins;r.volume=volume;r.initial=std::move(initial);
        remag::begin_pair_geometry(consumer);
        s->fields=fields;s->record=records().rows.size();records().rows.push_back(std::move(r));batch=std::move(s);return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_pair_force_observe(int row,int col,int a,int b){
    try {
        auto& r=current();auto& s=*batch;require(!s.referenced&&!s.staged&&row>=1&&row<=r.rows&&col>=1&&col<=r.columns&&
            a>=1&&a<=r.atoms&&b>=1&&b<=r.atoms&&remag::two_center_query_count(r.consumer)==size_t(r.queries)+1,"Pair query/integral alignment mismatch");
        remag::observe_pair_geometry(r.consumer,row,col,a,b);r.queries++;return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_pair_force_reference(int consumer,double energy,const double* force,const double* stress){
    try {
        auto& r=current();require(r.consumer==consumer&&!batch->referenced&&!batch->staged&&remag::two_center_query_count(consumer)==size_t(r.queries),"Invalid pair reference order");
        require(consumer==1||energy==0.,"Overlap must not add energy");r.reference=values(r.atoms,energy,force,stress);batch->referenced=true;return 0;
    }catch(const std::exception& e){return failed(e);}
}
namespace remag {
bool pair_force_shadow_pending(int consumer){
    if(!pair_force_shadow_enabled()||(consumer!=1&&consumer!=4))return false;
    auto& r=current();require(r.consumer==consumer&&batch->referenced,"Missing pair reference before execution");return true;
}
PairGeometryView stage_pair_force_shadow(uint64_t epoch,int consumer,int n,const int* pairs,const double* xyz,const double* integrals,cudaStream_t stream){
    auto& r=current();auto& s=*batch;require(epoch==r.epoch&&consumer==r.consumer&&n==r.queries&&s.referenced&&!s.staged&&integrals&&stream,"Invalid pair staging lifetime");
    const auto geometry=stage_pair_geometry(consumer,size_t(n),pairs,xyz);
    s.stream=stream;s.view={n,r.atoms,r.rows,r.columns,r.capacity,r.spinor,geometry.row,geometry.column,geometry.atoms,
        geometry.num,geometry.ptr,geometry.list,integrals,geometry.xyz};
    PairForceProblem problem{r.volume,r.consumer,0};s.problem=s.upload(1,&problem);s.output=s.upload(r.initial.size(),r.initial.data());
    check(cudaStreamSynchronize(stream));s.staged=true;return geometry;
}
void apply_pair_force_shadow(int* status,cudaStream_t stream){
    auto& r=current();auto& s=*batch;require(s.staged&&!s.applied&&status&&stream==s.stream,"Invalid pair execution lifetime");
    check(add_pair_force_device(s.view,s.problem,r.consumer==1?s.fields.density:s.fields.energy_density,s.output,s.output+1,s.output+1+3*r.atoms,status,stream));
    s.applied=true;
}
void finish_pair_force_shadow(int status){
    auto& r=current();auto& s=*batch;require(s.staged&&s.applied&&!r.complete,"Pair completion before execution or after terminal result");
    r.actual.resize(r.initial.size());check(cudaMemcpy(r.actual.data(),s.output,r.actual.size()*sizeof(double),cudaMemcpyDeviceToHost));
    r.exported=r.actual.size()*sizeof(double);transfer(Transfer::pair_force_shadow,false,r.exported);r.status=status;r.complete=true;r.passed=status==0;
    for(size_t i=0;i<r.actual.size();i++){
        const double tolerance=i==0?1.e-10+1.e-10*std::abs(r.reference[i]):i<=size_t(3*r.atoms)?1.e-10:1.e-12;
        r.passed&=std::isfinite(r.actual[i])&&std::abs(r.actual[i]-r.reference[i])<=tolerance;
    }
    require(r.passed,"Pair force full contribution comparison failed");
    complete_pair_geometry(r.consumer);complete_final_pair_shadow(r.epoch,r.consumer);
    // The parent owns the stream and destroys it only after this handoff.
    batch.reset();
}
}
