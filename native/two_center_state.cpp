// SPDX-License-Identifier: GPL-3.0-or-later
#include "two_center_state.hpp"
#include "execution.hpp"
#include "pair_force_state.hpp"
#include "nonlocal_force_state.hpp"
#include "final_grid_state.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" void remag_radial_batch_stats(unsigned long long*,double*);
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct TableRecord {int kind=0;std::array<int,11> shape{};double delta=0.,pi=0.;size_t staged=0,allocated=0,uses=0;};
struct Record {uint64_t epoch=0;int consumer=0,table=0,status=-1,nodes=0;size_t queries=0,staged=0,allocated=0,exported=0;bool complete=false,passed=false;remag::TwoCenterComparison comparison{};};
struct Records {
    std::vector<TableRecord> tables;std::vector<Record> rows;
    ~Records(){
        if(tables.empty()&&rows.empty())return;
        const bool production=remag::integral_force_enabled();
        FILE* f=std::fopen(production?"remag-two-center-tables.json":"remag-two-center.json","w");if(!f)return;
        const bool complete=!rows.empty()&&rows.back().consumer==4&&std::all_of(rows.begin(),rows.end(),[](const Record& r){return r.complete&&r.passed;});
        if(production)std::fputs("{\"scope\":\"immutable two-center tables; execution/status in shared integral-force publication\",\"comparison_performed\":false,\"tables\":[",f);
        else std::fprintf(f,"{\"scope\":\"material two-center derivative shadow; CPU force integrals remain authoritative\",\"complete\":%s,\"atol\":1e-10,\"rtol\":1e-10,\"tables\":[",complete?"true":"false");
        for(size_t i=0;i<tables.size();i++){
            const auto& t=tables[i];std::fprintf(f,"%s{\"kind\":%d,\"shape\":[",i?",":"",t.kind);
            for(int j=0;j<11;j++)std::fprintf(f,"%s%d",j?",":"",t.shape[j]);
            std::fprintf(f,"],\"delta\":%.17g,\"pi\":%.17g,\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"uses\":%zu}",t.delta,t.pi,t.staged,t.allocated,t.uses);
        }
        std::fputs("],\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++){
            const auto& r=rows[i];const auto& c=r.comparison;
            std::fprintf(f,"%s{\"epoch\":%llu,\"consumer\":%d,\"table\":%d,\"queries\":%zu,\"status\":%d,\"complete\":%s,\"passed\":%s,\"graph_nodes\":%d,\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"export_bytes\":%zu,\"host_reference_bytes\":%zu,",
                i?",":"",static_cast<unsigned long long>(r.epoch),r.consumer,r.table,r.queries,r.status,r.complete?"true":"false",r.passed?"true":"false",r.nodes,r.staged,r.allocated,r.exported,32*r.queries);
            std::fprintf(f,"\"compared\":%llu,\"mismatches\":%llu,\"nonfinite\":%llu,\"max_absolute_error\":[%.17g,%.17g],\"max_scaled_error\":[%.17g,%.17g]}",c.compared,c.mismatches,c.nonfinite,c.absolute[0],c.absolute[1],c.scaled[0],c.scaled[1]);
        }
        unsigned long long radial[4]={};double seconds=0.;remag_radial_batch_stats(radial,&seconds);
        std::fprintf(f,"],\"radial_setup\":{\"calls\":%llu,\"compact_calls\":%llu,\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,\"seconds\":%.9f}}\n",radial[0],radial[3],radial[1],radial[2],seconds);std::fclose(f);
    }
};
Records& records(){static Records value;return value;}
struct Device {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;std::vector<void*> memory;size_t staged=0,allocated=0;
    remag::Transfer boundary;
    explicit Device(remag::Transfer tag):boundary(tag){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Device(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* allocate(size_t n){T* p=nullptr;size_t bytes=std::max(size_t(1),n)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;return p;}
    template<class T>T* upload(size_t n,const T* p){auto out=allocate<T>(n);if(n){check(cudaMemcpyAsync(out,p,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);remag::transfer(boundary,true,n*sizeof(T));}return out;}
};
struct Table {Device device{remag::Transfer::two_center_setup};remag::TwoCenterView view{};size_t record=0;};
struct Batch {uint64_t epoch=0;int consumer=0,table=0;size_t record=0;std::vector<int> pairs;std::vector<double> xyz,reference;};
std::array<std::unique_ptr<Table>,2> tables;std::unique_ptr<Batch> batch;
void valid(){require(remag_two_center_shadow_enabled()&&!remag::in_scf(),"Two-center shadow phase/mode mismatch");}
int failed(const std::exception& e){std::fprintf(stderr,"REMAG two-center: %s\n",e.what());return -250;}
}
namespace remag {
size_t two_center_query_count(int consumer){valid();require(batch&&batch->consumer==consumer&&batch->epoch==geometry_epoch(),"Pair query outside its integral lifetime");return batch->reference.size()/4;}
TwoCenterView borrow_integral_two_center(int consumer){
    require(integral_force_enabled()&&!remag_two_center_shadow_enabled()&&!in_scf()&&!batch&&
        consumer>=1&&consumer<=4&&tables[0]&&tables[1],"Invalid production integral table request");
    auto& t=*tables[consumer==1?1:0];records().tables[t.record].uses++;return t.view;
}
}
extern "C" int remag_two_center_shadow_enabled(){const char* v=std::getenv("REMAG_TWO_CENTER_SHADOW");return v&&std::strcmp(v,"1")==0;}
extern "C" int remag_two_center_active(){return remag_two_center_shadow_enabled()||remag::integral_force_enabled();}
extern "C" int remag_two_center_stage(int kind,const int* c,double delta,double pi,const int* ind,const int* offsets,
        const int* ilm,const int* radial_index,const double* cutoff,const double* radial,const double* coefficients){
    try {
        require(remag_two_center_active()&&!remag::in_scf(),"Two-center table mode/phase mismatch");
        require((kind==1||kind==2)&&!batch&&records().rows.empty()&&!tables[kind-1],"Invalid two-center table lifetime");
        require(kind==1?!tables[1]:bool(tables[0]),"Two-center table staging order");
        require(c&&ind&&offsets&&ilm&&radial_index&&cutoff&&radial&&coefficients,"Missing two-center table");
        const int nf=c[0],nr=c[1],ny=c[2],rows=c[3],cols=c[4];
        require(nf>0&&nf<=1000000&&nr>=0&&nr<=1000000&&ny>=0&&ny<=100000000&&rows>0&&cols>0&&size_t(rows)*cols<=100000000&&c[5]>=0&&c[6]>=0,"Invalid two-center dimensions");
        require(c[7]>0&&c[8]>=c[7]&&c[8]<=nf&&c[9]>0&&c[10]>=c[9]&&c[10]<=nf&&std::isfinite(delta)&&delta>0.&&std::isfinite(pi)&&pi>3.14&&pi<3.15,"Invalid two-center bounds/constants");
        const size_t bytes=4*(size_t(nf)+1+size_t(rows)*cols+1+2*size_t(ny))+8*(size_t(nf)+2050*size_t(nr)+size_t(ny));
        require(bytes<=(size_t(4)<<30),"Two-center table exceeds staging limit");
        require(ind[0]>=1,"Invalid initial harmonic index");
        for(int i=0;i<nf;i++)require(ind[i]>=1&&ind[i+1]>=ind[i]&&std::isfinite(cutoff[i])&&cutoff[i]>=0.,"Invalid harmonic/cutoff map");
        require(ind[c[7]-1]>c[5]&&static_cast<long long>(ind[c[8]])-1<=static_cast<long long>(c[5])+rows&&ind[c[9]-1]>c[6]&&static_cast<long long>(ind[c[10]])-1<=static_cast<long long>(c[6])+cols,"Table harmonic bounds differ");
        require(offsets[0]==1&&offsets[size_t(rows)*cols]==ny+1,"Invalid angular endpoints");
        for(size_t i=0;i<size_t(rows)*cols;i++)require(offsets[i]>=1&&offsets[i+1]>=offsets[i]&&offsets[i+1]<=ny+1,"Invalid angular map");
        for(int i=0;i<ny;i++)require(ilm[i]>=1&&ilm[i]<=121&&radial_index[i]>=1&&radial_index[i]<=nr&&std::isfinite(coefficients[i]),"Invalid angular/radial expansion");
        for(size_t i=0;i<2050*size_t(nr);i++)require(std::isfinite(radial[i]),"Nonfinite radial table");
        auto t=std::make_unique<Table>();auto& d=t->device;
        t->view={nf,nr,ny,rows,cols,c[5],c[6],c[7],c[8],c[9],c[10],delta,pi,d.upload(nf+1,ind),d.upload(size_t(rows)*cols+1,offsets),d.upload(ny,ilm),d.upload(ny,radial_index),d.upload(nf,cutoff),d.upload(2050*size_t(nr),radial),d.upload(ny,coefficients)};
        check(cudaStreamSynchronize(d.stream));require(d.staged==bytes,"Two-center table staging count differs");
        TableRecord r;r.kind=kind;std::copy(c,c+11,r.shape.begin());r.delta=delta;r.pi=pi;r.staged=d.staged;r.allocated=d.allocated;t->record=records().tables.size();records().tables.push_back(r);tables[kind-1]=std::move(t);return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_two_center_begin(int consumer){
    try {
        valid();const auto epoch=remag::geometry_epoch();require(epoch>0&&consumer>=1&&consumer<=4&&tables[0]&&tables[1]&&!batch,"Invalid two-center consumer begin");
        const auto& rows=records().rows;
        if(rows.empty())require(consumer==1,"Two-center force sequence must start with kinetic");
        else {const auto& previous=rows.back();require(previous.complete&&previous.passed&&epoch>=previous.epoch,"Previous two-center consumer incomplete");
            if(epoch>previous.epoch)require(previous.consumer==4&&consumer==1,"Incomplete previous two-center geometry");
            else require((previous.consumer==1&&(consumer==2||consumer==3))||((previous.consumer==2||previous.consumer==3)&&consumer==4),"Two-center consumer order differs");}
        auto b=std::make_unique<Batch>();b->epoch=epoch;b->consumer=consumer;b->table=consumer==1?2:1;b->record=records().rows.size();
        Record r;r.epoch=epoch;r.consumer=consumer;r.table=b->table;records().rows.push_back(r);batch=std::move(b);return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_two_center_observe(int table,int a,int b,const double* xyz,double value,const double* gradient){
    try {
        remag::observe_nonlocal_geometry(table,a,b,xyz);
        if(!batch)return 0;
        valid();auto& s=*batch;require(s.epoch==remag::geometry_epoch()&&table==s.table&&xyz&&gradient,"Two-center observation owner mismatch");
        const auto& t=tables[table-1]->view;require(a>=t.first1&&a<=t.last1&&b>=t.first2&&b<=t.last2,"Two-center query registry mismatch");
        require(s.reference.size()/4<(size_t(4)<<30)/96,"Two-center shadow exceeds 4 GiB limit");
        s.pairs.push_back(a);s.pairs.push_back(b);s.xyz.insert(s.xyz.end(),xyz,xyz+3);s.reference.push_back(value);s.reference.insert(s.reference.end(),gradient,gradient+3);
        records().rows[s.record].queries=s.reference.size()/4;return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_two_center_finish(int consumer){
    try {
        valid();require(batch&&batch->consumer==consumer&&batch->epoch==remag::geometry_epoch(),"Two-center completion owner mismatch");auto& s=*batch;auto& r=records().rows[s.record];const size_t n=r.queries;
        const bool pair=remag::pair_force_shadow_pending(consumer);
        const bool nonlocal=remag::nonlocal_force_shadow_pending(consumer);
        if(n||pair||nonlocal) {
            Device d(remag::Transfer::two_center_shadow);auto reference=d.upload(4*n,s.reference.data());auto actual=d.allocate<double>(4*n);
            const int* pairs=nullptr;const double* xyz=nullptr;
            if(nonlocal){auto geometry=remag::stage_nonlocal_force_shadow(consumer,n,s.pairs.data(),s.xyz.data(),actual,d.stream);pairs=geometry.pairs;xyz=geometry.xyz;}
            else if(pair){auto geometry=remag::stage_pair_force_shadow(s.epoch,consumer,int(n),s.pairs.data(),s.xyz.data(),actual,d.stream);pairs=geometry.pairs;xyz=geometry.xyz;}
            else {pairs=d.upload(2*n,s.pairs.data());xyz=d.upload(3*n,s.xyz.data());}
            remag::TwoCenterComparison zero{};int initial=0;auto comparison=d.upload(1,&zero);auto status=d.upload(1,&initial);check(cudaStreamSynchronize(d.stream));r.staged=d.staged;r.allocated=d.allocated;
            check(cudaStreamBeginCapture(d.stream,cudaStreamCaptureModeGlobal));
            if(n){check(remag::evaluate_two_center_device(tables[s.table-1]->view,int(n),pairs,xyz,actual,status,d.stream));
                check(remag::compare_two_center_device(n,actual,reference,comparison,status,d.stream));}
            if(pair)remag::apply_pair_force_shadow(status,d.stream);
            if(nonlocal)remag::apply_nonlocal_force_shadow(status,d.stream);
            check(cudaStreamEndCapture(d.stream,&d.graph));size_t count=0;check(cudaGraphGetNodes(d.graph,nullptr,&count));r.nodes=int(count);require(count==(n?size_t(2):size_t(0))+(pair?1:0)+(nonlocal?2:0),"Two-center graph node count differs");
            std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(d.graph,nodes.data(),&count));for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden two-center graph node");}
            check(cudaGraphInstantiate(&d.executable,d.graph,0));check(cudaGraphLaunch(d.executable,d.stream));check(cudaStreamSynchronize(d.stream));
            check(cudaMemcpy(&r.comparison,comparison,sizeof(r.comparison),cudaMemcpyDeviceToHost));remag::transfer(remag::Transfer::two_center_shadow,false,sizeof(r.comparison));
            check(cudaMemcpy(&r.status,status,4,cudaMemcpyDeviceToHost));remag::transfer(remag::Transfer::two_center_shadow,false,4);
            r.exported=60;
            if(pair)remag::finish_pair_force_shadow(r.status);
            if(nonlocal)remag::finish_nonlocal_force_shadow(r.status);
        }else r.status=0;
        r.complete=true;r.passed=r.status==0&&r.comparison.compared==4*n&&r.comparison.mismatches==0&&r.comparison.nonfinite==0;
        records().tables[tables[s.table-1]->record].uses++;batch.reset();require(r.passed,"Two-center full derivative comparison failed");return 0;
    }catch(const std::exception& e){return failed(e);}
}
