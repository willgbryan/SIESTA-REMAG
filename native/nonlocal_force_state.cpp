#include "spin_rotation.hpp"
// SPDX-License-Identifier: GPL-3.0-or-later
#include "nonlocal_force_state.hpp"
#include "nonlocal_compact.hpp"
#include "nonlocal_h0_state.hpp"
#include "nonlocal_query.hpp"
#include "final_grid_state.hpp"
#include "two_center_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Record {
    unsigned long long epoch=0;int consumer=0,atoms=0,rows=0,columns=0,capacity=0,components=0,queries=0,centers=0,neighbors=0,blocks=0,status=-1;
    size_t geometry_bytes=0,staged=0,allocated=0,exported=0;bool complete=false,passed=false,integral=false;
    double volume=0.;std::vector<double> initial,reference,actual;
    bool compact_queries=false;int projector_entries=0;
    size_t query_metadata_bytes=0,query_output_bytes=0;
};
struct Records {std::vector<Record> rows;~Records(){
    if(rows.empty())return;
    const bool production=rows.front().integral;
    FILE* f=std::fopen(production?"remag-nonlocal-geometry.json":"remag-nonlocal-force.json","w");if(!f)return;
    bool complete=std::all_of(rows.begin(),rows.end(),[](auto& r){return r.complete&&r.passed;});
    std::fprintf(f,"{\"scope\":\"%s\",\"complete\":%s,\"cycles\":[",production?"prepared nonlocal geometry held until authoritative shared force publication":"compact pre-SCF geometry and retained-D nonlocal shadow; CPU outputs authoritative",complete?"true":"false");
    for(size_t i=0;i<rows.size();i++){auto& r=rows[i];
        std::fprintf(f,"%s{\"epoch\":%llu,\"consumer\":%d,\"atoms\":%d,\"rows\":%d,\"columns\":%d,\"capacity\":%d,\"components\":%d,\"queries\":%d,\"centers\":%d,\"neighbors\":%d,\"blocks\":%d,\"volume\":%.17g,\"geometry_bytes\":%zu,\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"export_bytes\":%zu,\"electronic_upload_bytes\":0,\"graph_nodes\":2,\"status\":%d,\"complete\":%s,\"passed\":%s",
            i?",":"",r.epoch,r.consumer,r.atoms,r.rows,r.columns,r.capacity,r.components,r.queries,r.centers,r.neighbors,r.blocks,r.volume,r.geometry_bytes,r.staged,r.allocated,r.exported,r.status,r.complete?"true":"false",r.passed?"true":"false");
        std::fprintf(f,",\"compact_queries\":%s,\"projector_entries\":%d,\"query_metadata_bytes\":%zu,\"query_output_bytes\":%zu",
            r.compact_queries?"true":"false",r.projector_entries,r.query_metadata_bytes,r.query_output_bytes);
        const char* names[]={"initial","reference","actual"};const std::vector<double>* arrays[]={&r.initial,&r.reference,&r.actual};
        for(int k=0;k<3;k++){std::fprintf(f,",\"%s\":[",names[k]);for(size_t j=0;j<arrays[k]->size();j++){double x=(*arrays[k])[j];std::fprintf(f,"%s",j?",":"");if(std::isfinite(x))std::fprintf(f,"%.17g",x);else std::fputs("null",f);}std::fputc(']',f);}std::fputc('}',f);
    }std::fputs("]}\n",f);std::fclose(f);
}};
Records& records(){static Records r;return r;}
struct Geometry {
    size_t record=0;int consumer=0,split=0,mode=0;bool ready=false,bound=false,referenced=false,staged=false,applied=false;
    uint64_t preparation_epoch=0;
    size_t seen_queries=0,seen_centers=0,covered_queries=0;cudaStream_t own_stream=nullptr,stream=nullptr;
    std::vector<int> num,ptr,list,pairs;std::vector<double> xyz;
    std::vector<int> projector_ptr{0},projector_registry,orbital_registry;
    std::vector<double> neighbor_xyz;remag::NonlocalQueryView query_view{};
    std::vector<remag::NonlocalCenter> centers;std::vector<remag::NonlocalNeighbor> neighbors;std::vector<remag::NonlocalBlock> blocks;
    const int* device_pairs=nullptr;
    std::vector<void*> memory;remag::NonlocalCompactView view{};remag::FinalGridStateView fields{};double *output=nullptr,*weights=nullptr;
    ~Geometry(){if(stream)cudaStreamSynchronize(stream);if(own_stream)cudaStreamSynchronize(own_stream);for(auto p:memory)cudaFree(p);if(own_stream)cudaStreamDestroy(own_stream);}
    template<class T>T* allocate(size_t n){T* p=nullptr;size_t bytes=std::max(size_t(1),n)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);records().rows[record].allocated+=bytes;return p;}
    template<class T>T* upload(size_t n,const T* a,bool geometry){auto p=allocate<T>(n);if(n){check(cudaMemcpyAsync(p,a,n*sizeof(T),cudaMemcpyHostToDevice,geometry?own_stream:stream));
        auto& r=records().rows[record];if(geometry)r.geometry_bytes+=n*sizeof(T);else r.staged+=n*sizeof(T);
        remag::transfer(geometry?remag::Transfer::nonlocal_geometry:remag::Transfer::nonlocal_force_shadow,true,n*sizeof(T));}return p;}
};
std::unique_ptr<Geometry> geometry;
void valid(){require(remag_nonlocal_force_active()&&remag_two_center_active()&&remag::pair_force_active()&&!remag::in_scf(),"Nonlocal mode/phase mismatch");}
Record& current(){valid();require(bool(geometry),"Missing nonlocal geometry");auto& r=records().rows[geometry->record];
    require(r.integral==remag::integral_force_enabled(),"Nonlocal geometry mode changed");
    require(!geometry->bound||r.epoch==remag::geometry_epoch(),"Stale nonlocal geometry");return r;}
std::vector<double> values(int atoms,const double* energy,const double* force,const double* stress){
    require(force&&stress,"Missing nonlocal output");std::vector<double> v(9,0.);if(energy)std::copy(energy,energy+9,v.begin());
    v.insert(v.end(),force,force+3*atoms);v.insert(v.end(),stress,stress+9);
    require(std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);}),"Nonfinite nonlocal output");return v;
}
int failed(const std::exception& e){std::fprintf(stderr,"REMAG nonlocal force: %s\n",e.what());return -270;}
}
extern "C" int remag_nonlocal_force_shadow_enabled(){const char* v=std::getenv("REMAG_NONLOCAL_FORCE_SHADOW");return v&&std::strcmp(v,"1")==0;}
extern "C" int remag_nonlocal_force_active(){return remag_nonlocal_force_shadow_enabled()||remag::integral_force_enabled();}
extern "C" int remag_nonlocal_begin(int mode,int consumer,int atoms,int rows,int columns,int capacity,int split,double volume,
        const int* num,const int* ptr,const int* list,const double* force,const double* stress){
    try{valid();require((mode==0||(mode==1&&remag_nonlocal_force_shadow_enabled()))&&(consumer==2||consumer==3)&&atoms>0&&rows>=atoms&&columns>=rows&&capacity>0&&capacity<2147483647/8&&
        (split==0||split==1)&&std::isfinite(volume)&&volume>0&&num&&ptr&&list,"Invalid nonlocal geometry dimensions");
        if(mode==0){
            require(!geometry,"Previous nonlocal geometry unfinished");auto g=std::make_unique<Geometry>();Record r;r.consumer=consumer;r.atoms=atoms;r.rows=rows;r.columns=columns;r.capacity=capacity;r.volume=volume;
            r.integral=remag::integral_force_enabled();
            r.compact_queries=remag::nonlocal_h0_enabled();
            g->record=records().rows.size();g->consumer=consumer;g->split=split;g->preparation_epoch=remag::geometry_epoch();g->num.assign(num,num+rows);g->ptr.assign(ptr,ptr+rows);g->list.assign(list,list+capacity);
            std::vector<std::pair<int,int>> spans;
            for(int row=0;row<rows;row++){require(ptr[row]>=0&&ptr[row]<=capacity&&num[row]>=0&&num[row]<=capacity-ptr[row],"Invalid nonlocal sparse span");
                if(num[row])spans.emplace_back(ptr[row],ptr[row]+num[row]);
                for(int j=0;j<num[row];j++)require(list[ptr[row]+j]>=1&&list[ptr[row]+j]<=columns,"Invalid nonlocal sparse column");}
            std::sort(spans.begin(),spans.end());
            for(size_t i=1;i<spans.size();i++)require(spans[i].first>=spans[i-1].second,"Overlapping nonlocal sparse rows");
            records().rows.push_back(r);geometry=std::move(g);
        }else{
            auto& r=current();auto& g=*geometry;
            require(g.ready&&g.bound&&!g.mode&&r.epoch==remag::geometry_epoch()&&consumer==g.consumer&&split==g.split&&atoms==r.atoms&&rows==r.rows&&columns==r.columns&&capacity==r.capacity&&volume==r.volume&&
                std::equal(g.num.begin(),g.num.end(),num)&&std::equal(g.ptr.begin(),g.ptr.end(),ptr)&&std::equal(g.list.begin(),g.list.end(),list),"Nonlocal geometry changed after staging");
            remag::validate_grid_sparse_topology(rows,columns,capacity,num,ptr,list);
            r.initial=values(atoms,nullptr,force,stress);g.fields=remag::borrow_final_pair_state(r.epoch,consumer);
            require(g.fields.shape.nuo==rows&&g.fields.shape.no==columns&&g.fields.shape.nnz==capacity&&std::abs(g.fields.volume-volume)<=1e-12*std::max(1.,std::abs(volume)),"Nonlocal retained D shape/volume differs");
            r.components=g.fields.h_spins;g.mode=1;g.seen_queries=0;g.seen_centers=0;g.covered_queries=0;
        }return 0;
    }catch(const std::exception& e){return failed(e);}
}
namespace remag {
void observe_nonlocal_geometry(int table,int a,int b,const double* xyz){
    if(!remag_nonlocal_force_active()||!geometry||geometry->referenced||(geometry->ready&&!geometry->mode))return;
    auto& r=current();auto& g=*geometry;require(!r.compact_queries&&table==1&&a>0&&b>0&&xyz,"Nonlocal integral table differs");
    for(int k=0;k<3;k++)require(std::isfinite(xyz[k]),"Nonfinite nonlocal displacement");
    require(g.seen_queries<size_t(2147483647/4),"Nonlocal query limit");
    if(!g.mode){g.pairs.push_back(a);g.pairs.push_back(b);g.xyz.insert(g.xyz.end(),xyz,xyz+3);}
    else{size_t n=g.seen_queries;require(n<size_t(r.queries)&&g.pairs[2*n]==a&&g.pairs[2*n+1]==b&&std::equal(g.xyz.begin()+3*n,g.xyz.begin()+3*n+3,xyz),"Final nonlocal integral geometry differs from pre-SCF staging");}
    g.seen_queries++;
}
}
extern "C" int remag_nonlocal_center(int count,int projectors,int atom,int blocks,const int* row,const int* column,const int* atoms,const int* offset,const int* angular,const double* epsilon){
    try{auto& r=current();auto& g=*geometry;require(!g.referenced&&(!g.ready||g.mode)&&count>=0&&projectors>0&&atom>=1&&atom<=r.atoms&&blocks>0&&blocks<=projectors&&row&&column&&atoms&&offset&&angular&&epsilon&&
            g.seen_queries-g.covered_queries==size_t(count)*projectors,"Nonlocal overlap/center coverage differs");
        if(r.compact_queries)require(g.projector_ptr.size()==g.centers.size()+2&&
            g.projector_ptr.back()-g.projector_ptr[g.centers.size()]==projectors&&
            g.orbital_registry.size()==g.neighbors.size()+size_t(count),"Compact center metadata coverage differs");
        remag::NonlocalCenter c{int(g.mode?g.centers.at(g.seen_centers).neighbor:g.neighbors.size()),count,int(g.mode?g.centers.at(g.seen_centers).block:g.blocks.size()),blocks,projectors,int(g.covered_queries),atom};
        if(g.mode){require(g.seen_centers<g.centers.size()&&std::memcmp(&c,&g.centers[g.seen_centers],sizeof(c))==0,"Nonlocal center changed");}
        std::vector<remag::NonlocalNeighbor> next_neighbors;std::vector<remag::NonlocalBlock> next_blocks;
        for(int i=0;i<count;i++){
            require(row[i]>=0&&row[i]<=r.rows&&column[i]>=1&&column[i]<=r.columns&&atoms[i]>0&&(!row[i]||atoms[i]<=r.atoms),"Invalid nonlocal neighbor");
            remag::NonlocalNeighbor n{row[i],column[i],atoms[i],int(g.seen_centers)};
            if(g.mode)require(std::memcmp(&n,&g.neighbors[c.neighbor+i],sizeof(n))==0,"Nonlocal neighbor changed");else next_neighbors.push_back(n);
        }
        for(int i=0;i<blocks;i++){
            const int l=angular[i];require(l>=0&&l<=5&&(g.consumer==3||l==0),"Invalid nonlocal angular order");
            const int width=l?2*(2*l+1):1;
            require(offset[i]>=0&&offset[i]<=projectors-width&&std::isfinite(epsilon[2*i])&&std::isfinite(epsilon[2*i+1]),"Invalid nonlocal projector block");
            remag::NonlocalBlock b{offset[i],l,{epsilon[2*i],epsilon[2*i+1]}};
            if(g.mode)require(std::memcmp(&b,&g.blocks[c.block+i],sizeof(b))==0,"Nonlocal projector block changed");else next_blocks.push_back(b);
        }
        if(!g.mode){g.centers.push_back(c);g.neighbors.insert(g.neighbors.end(),next_neighbors.begin(),next_neighbors.end());g.blocks.insert(g.blocks.end(),next_blocks.begin(),next_blocks.end());}
        g.seen_centers++;g.covered_queries=g.seen_queries;return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_nonlocal_end(int mode,const double* energy,const double* force,const double* stress){
    try{auto& r=current();auto& g=*geometry;require(mode==g.mode&&!g.referenced&&g.covered_queries==g.seen_queries,"Nonlocal geometry completion order");
        if(!mode){
            require(!g.ready&&!g.bound,"Nonlocal geometry already staged");r.queries=int(g.seen_queries);r.centers=int(g.centers.size());r.neighbors=int(g.neighbors.size());r.blocks=int(g.blocks.size());
            require(8*size_t(r.rows)+4*size_t(r.capacity)+28*g.centers.size()+16*g.neighbors.size()+24*g.blocks.size()+32*size_t(r.queries)<(size_t(4)<<30),"Nonlocal geometry staging limit");
            if(r.compact_queries){
                require(g.pairs.empty()&&g.xyz.empty()&&g.projector_ptr.size()==g.centers.size()+1&&
                    g.orbital_registry.size()==g.neighbors.size()&&g.neighbor_xyz.size()==3*g.neighbors.size(),"Incomplete compact query metadata");
                r.projector_entries=int(g.projector_registry.size());
                r.query_metadata_bytes=4*(g.projector_ptr.size()+g.projector_registry.size()+g.orbital_registry.size())+8*g.neighbor_xyz.size();
                r.query_output_bytes=32*size_t(r.queries);
                require(8*size_t(r.rows)+4*size_t(r.capacity)+28*g.centers.size()+16*g.neighbors.size()+24*g.blocks.size()+
                    r.query_metadata_bytes+r.query_output_bytes<(size_t(4)<<30),"Compact nonlocal allocation limit");
            }
            check(cudaStreamCreateWithFlags(&g.own_stream,cudaStreamNonBlocking));
            g.view.center=g.upload(g.centers.size(),g.centers.data(),true);g.view.neighbor=g.upload(g.neighbors.size(),g.neighbors.data(),true);g.view.block=g.upload(g.blocks.size(),g.blocks.data(),true);
            g.view.num=g.upload(g.num.size(),g.num.data(),true);g.view.ptr=g.upload(g.ptr.size(),g.ptr.data(),true);g.view.list=g.upload(g.list.size(),g.list.data(),true);
            if(r.compact_queries){
                auto& q=g.query_view;q.centers=r.centers;q.neighbors=r.neighbors;q.queries=r.queries;q.projectors=r.projector_entries;
                q.center=g.view.center;q.neighbor=g.view.neighbor;
                q.projector_ptr=g.upload(g.projector_ptr.size(),g.projector_ptr.data(),true);
                q.projector_registry=g.upload(g.projector_registry.size(),g.projector_registry.data(),true);
                q.orbital_registry=g.upload(g.orbital_registry.size(),g.orbital_registry.data(),true);
                q.displacement=g.upload(g.neighbor_xyz.size(),g.neighbor_xyz.data(),true);
                auto pairs=g.allocate<int>(2*size_t(r.queries));g.device_pairs=pairs;
                g.view.xyz=g.allocate<double>(3*size_t(r.queries));
                // Invalid expansion leaves outputs untouched. Zero registry IDs
                // make the downstream evaluator reject before reading xyz.
                if(r.queries)check(cudaMemsetAsync(pairs,0,2*size_t(r.queries)*sizeof(int),g.own_stream));
            }else{
                g.device_pairs=g.upload(g.pairs.size(),g.pairs.data(),true);g.view.xyz=g.upload(g.xyz.size(),g.xyz.data(),true);
            }
            check(cudaStreamSynchronize(g.own_stream));g.ready=true;
        }else{require(g.ready&&g.bound&&!g.staged&&g.seen_queries==size_t(r.queries)&&g.seen_centers==g.centers.size()&&energy,"Incomplete final nonlocal geometry");r.reference=values(r.atoms,energy,force,stress);g.referenced=true;}
        return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_nonlocal_h0_matrix(int phase,int capacity,const double* h,const remag::H0Complex* z){
    try{
        auto& r=current();auto& g=*geometry;
        require(remag::nonlocal_h0_active()&&remag::integral_force_enabled()&&!remag_nonlocal_force_shadow_enabled()&&
            !g.mode&&!g.bound&&capacity==r.capacity&&(phase==0||phase==1||phase==2)&&h&&(r.consumer!=3||z),"Invalid H0 capture mode/lifetime");
        if(phase==0){
            require(!g.ready,"H0 initial capture after geometry completion");
            remag::begin_nonlocal_h0_shadow(r.rows,r.columns,r.capacity,r.consumer,g.split,h,z);
        }else if(phase==2){
            require(g.ready,"H0 host poisoning before geometry completion");
            remag::record_nonlocal_h0_host_poison(capacity,h,z);
        }else{
            require(g.ready,"H0 comparison before geometry completion");
            remag::NonlocalH0View view{r.neighbors,r.centers,r.blocks,r.queries,r.rows,r.columns,r.capacity,r.consumer==3,g.split,0,
                g.view.center,g.view.neighbor,g.view.block,g.view.num,g.view.ptr,g.view.list,nullptr,nullptr,nullptr};
            remag::evaluate_nonlocal_h0_shadow(view,g.neighbors.data(),g.device_pairs,g.view.xyz,h,z,
                r.compact_queries?&g.query_view:nullptr);
        }
        return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_nonlocal_h0_neighbor(int orbital,const double* xyz){
    try{current();auto& g=*geometry;
        require(remag::nonlocal_h0_enabled()&&!g.ready&&!g.mode&&!g.bound&&orbital>0&&xyz&&
            g.projector_ptr.size()==g.centers.size()+1&&g.orbital_registry.size()<size_t(2147483647/4),"Invalid H0 neighbor query");
        for(int k=0;k<3;k++)require(std::isfinite(xyz[k]),"Nonfinite compact displacement");
        g.orbital_registry.push_back(orbital);g.neighbor_xyz.insert(g.neighbor_xyz.end(),xyz,xyz+3);return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_nonlocal_h0_projectors(int count,int projectors,const int* registry){
    try{auto& r=current();auto& g=*geometry;
        require(r.compact_queries&&remag::nonlocal_h0_enabled()&&!g.ready&&!g.mode&&!g.bound&&count>=0&&projectors>0&&registry&&
            g.projector_ptr.size()==g.centers.size()+1&&g.orbital_registry.size()==g.neighbors.size()+size_t(count)&&
            g.seen_queries==g.covered_queries,"Invalid H0 center query metadata");
        const size_t queries=size_t(count)*size_t(projectors);
        require(queries<=size_t(2147483647/4)-g.covered_queries&&
            size_t(projectors)<=size_t(2147483647)-g.projector_registry.size(),"Compact query limit");
        for(int p=0;p<projectors;p++)require(registry[p]>0,"Invalid compact projector registry");
        g.projector_registry.insert(g.projector_registry.end(),registry,registry+projectors);
        g.projector_ptr.push_back(int(g.projector_registry.size()));g.seen_queries+=queries;return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_nonlocal_h0_split_warning(){
    try{auto& r=current();auto& g=*geometry;
        require(remag::nonlocal_h0_enabled()&&g.ready&&!g.mode&&!g.bound&&r.consumer==3,"Invalid SOC H0 warning consumer");
        // Preserve the upstream diagnostic only where a selected sparse pair
        // visits a block with opposite-sign relativistic projector energies.
        for(const auto& center:g.centers){
            bool opposite=false;
            for(int k=0;k<center.blocks;k++){const auto& b=g.blocks[center.block+k];
                opposite|=b.l>0&&b.epsilon[0]*b.epsilon[1]<0.;}
            if(!opposite)continue;
            for(int l=0;l<center.neighbors;l++){
                const int row=g.neighbors[center.neighbor+l].row;if(!row)continue;
                for(int j=0;j<g.num[row-1];j++){
                    const int column=g.list[g.ptr[row-1]+j];
                    for(int k=0;k<center.neighbors;k++)if(g.neighbors[center.neighbor+k].column==column)return 1;
                }
            }
        }return 0;
    }catch(const std::exception& e){return failed(e);}
}
extern "C" int remag_nonlocal_bind_geometry(){
    try{if(!remag_nonlocal_force_active())return 0;auto& r=current();auto& g=*geometry;require(g.ready&&!g.bound&&!g.mode&&remag::geometry_epoch()>g.preparation_epoch,"Nonlocal geometry bind order");r.epoch=remag::geometry_epoch();remag::bind_nonlocal_h0_shadow(r.epoch);g.bound=true;return 0;}
    catch(const std::exception& e){return failed(e);}
}
namespace remag {
void validate_nonlocal_h0_topology(int rows,int columns,int capacity,const int* num,const int* ptr,const int* list){
    auto& r=current();auto& g=*geometry;
    require(nonlocal_h0_enabled()&&g.ready&&g.bound&&!g.mode&&rows==r.rows&&columns==r.columns&&capacity==r.capacity&&
        num&&ptr&&list&&std::equal(g.num.begin(),g.num.end(),num)&&std::equal(g.ptr.begin(),g.ptr.end(),ptr)&&
        std::equal(g.list.begin(),g.list.end(),list),"H0 SCF topology differs from prepared geometry");
}
PreparedNonlocalForceView borrow_integral_nonlocal(int consumer,int atoms,const FinalGridStateView& fields,cudaStream_t stream){
    auto& r=current();auto& g=*geometry;
    require(integral_force_enabled()&&!remag_nonlocal_force_shadow_enabled()&&g.ready&&g.bound&&!g.mode&&!g.referenced&&!g.staged&&
        consumer==g.consumer&&atoms==r.atoms&&stream&&fields.epoch==r.epoch&&fields.shape.nuo==r.rows&&fields.shape.no==r.columns&&
        fields.shape.nnz==r.capacity&&((consumer==3)==(fields.h_spins==8))&&
        std::abs(fields.volume-r.volume)<=1e-12*std::max(1.,std::abs(r.volume)),"Invalid prepared production nonlocal geometry");
    validate_grid_sparse_topology(r.rows,r.columns,r.capacity,g.num.data(),g.ptr.data(),g.list.data());
    g.mode=2;g.stream=stream;g.fields=fields;r.components=fields.h_spins;auto& v=g.view;
    v.neighbors=r.neighbors;v.centers=r.centers;v.blocks=r.blocks;v.queries=r.queries;v.atoms=r.atoms;v.rows=r.rows;v.columns=r.columns;v.capacity=r.capacity;
    v.components=r.components;v.spinor=std::min(2,r.components);v.soc=consumer==3;v.split=g.split;v.volume=r.volume;v.lookup=grid_contraction_view().lookup;
    g.staged=true;return {v,g.device_pairs};
}
void apply_integral_nonlocal(const double* integrals,double* weights,const FinalForceAccumulatorView& pool){
    current();auto& g=*geometry;require(integral_force_enabled()&&g.mode==2&&g.staged&&!g.applied&&integrals&&weights&&
        pool.stream==g.stream&&pool.energy&&pool.force&&pool.stress&&pool.status,"Invalid production nonlocal execution");
    g.view.integrals=integrals;check(add_nonlocal_compact_device(g.view,g.fields.density,weights,pool.energy+1,pool.force,pool.stress,pool.status,pool.stream));g.applied=true;
}
void finish_integral_nonlocal(int status){
    auto& r=current();auto& g=*geometry;require(integral_force_enabled()&&g.mode==2&&g.staged&&g.applied&&!r.complete&&status==0,
        "Invalid production nonlocal publication");r.status=status;r.complete=true;r.passed=true;remag::finish_nonlocal_h0_shadow(r.epoch);if(remag_orientation_retain_geometry())g.stream=nullptr;else geometry.reset();
}
bool nonlocal_force_shadow_pending(int consumer){if(!remag_nonlocal_force_shadow_enabled()||(consumer!=2&&consumer!=3))return false;
    auto& r=current();require(r.consumer==consumer&&geometry->referenced,"Missing nonlocal reference");return true;}
NonlocalIntegralGeometry stage_nonlocal_force_shadow(int consumer,size_t n,const int* pairs,const double* xyz,const double* integrals,cudaStream_t stream){
    auto& r=current();auto& g=*geometry;require(consumer==g.consumer&&n==size_t(r.queries)&&(n==0||(pairs&&xyz))&&integrals&&stream&&g.referenced&&!g.staged&&
        (!n||(std::equal(g.pairs.begin(),g.pairs.end(),pairs)&&std::equal(g.xyz.begin(),g.xyz.end(),xyz))),"Nonlocal execution geometry differs");
    g.stream=stream;auto& v=g.view;v.neighbors=r.neighbors;v.centers=r.centers;v.blocks=r.blocks;v.queries=r.queries;v.atoms=r.atoms;v.rows=r.rows;v.columns=r.columns;v.capacity=r.capacity;
    v.components=r.components;v.spinor=std::min(2,r.components);v.soc=consumer==3;v.split=g.split;v.volume=r.volume;v.lookup=grid_contraction_view().lookup;v.integrals=integrals;
    g.output=g.upload(r.initial.size(),r.initial.data(),false);g.weights=g.allocate<double>(r.capacity);check(cudaStreamSynchronize(stream));g.staged=true;return {g.device_pairs,v.xyz};
}
void apply_nonlocal_force_shadow(int* status,cudaStream_t stream){auto& r=current();auto& g=*geometry;require(g.staged&&!g.applied&&status&&stream==g.stream,"Nonlocal execution lifetime");
    check(add_nonlocal_compact_device(g.view,g.fields.density,g.weights,g.output,g.output+9,g.output+9+3*r.atoms,status,stream));g.applied=true;}
void finish_nonlocal_force_shadow(int status){auto& r=current();auto& g=*geometry;require(g.staged&&g.applied&&!r.complete,"Nonlocal completion before execution or after terminal result");
    r.actual.resize(r.initial.size());check(cudaMemcpy(r.actual.data(),g.output,r.actual.size()*8,cudaMemcpyDeviceToHost));r.exported=r.actual.size()*8;transfer(Transfer::nonlocal_force_shadow,false,r.exported);
    r.status=status;r.complete=true;r.passed=status==0;
    for(size_t i=0;i<r.actual.size();i++){double tolerance=i<9?1e-10+1e-10*std::abs(r.reference[i]):i<size_t(9+3*r.atoms)?1e-10:1e-12;r.passed&=std::isfinite(r.actual[i])&&std::abs(r.actual[i]-r.reference[i])<=tolerance;}
    require(r.passed,"Nonlocal full contribution comparison failed");complete_final_pair_shadow(r.epoch,r.consumer);geometry.reset();
}
}

extern "C" int remag_reuse_nonlocal_geometry() {
    try {
        valid();require(remag_orientation_retain_geometry()&&geometry,"Missing retained nonlocal geometry");
        auto& g=*geometry;const auto& old=records().rows[g.record];
        require(old.integral&&g.ready&&g.bound&&g.mode==2&&g.staged&&g.applied&&old.complete&&old.passed&&old.status==0&&
            remag::geometry_epoch()==old.epoch+1,"Nonlocal geometry reuse precedes complete forces");
        Record next=old;next.epoch=remag::geometry_epoch();next.complete=false;next.passed=false;next.status=-1;
        next.geometry_bytes=0;next.staged=0;next.allocated=0;next.exported=0;
        next.initial.clear();next.reference.clear();next.actual.clear();g.preparation_epoch=old.epoch;
        g.record=records().rows.size();records().rows.push_back(next);
        g.mode=0;g.referenced=false;g.staged=false;g.applied=false;
        g.seen_queries=0;g.seen_centers=0;g.covered_queries=0;
        g.fields={};g.stream=nullptr;g.view.integrals=nullptr;g.view.lookup=nullptr;return 0;
    }catch(const std::exception& e){return failed(e);}
}
