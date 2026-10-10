// SPDX-License-Identifier: GPL-3.0-or-later
#include "atomic_basis_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct TermRecord {
    size_t selected=0,staged=0,allocated=0,exported=0;
    bool begun=false,generated=false,consumed=false,passed=false;int status=-1;
    remag::AtomicBasisComparison comparison{};
};
struct Record {
    uint64_t epoch=0;std::array<int,13> shape{};size_t staged=0,allocated=0;
    bool complete=false,shadow=false;TermRecord term[2];
};
struct Records {
    std::vector<Record> rows;
    ~Records(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-atomic-basis.json","w");if(!f)return;
        std::fputs("{\"scope\":\"geometry-owned core/neutral derivatives; explicit production or shadow mode\",\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++){
            const auto& r=rows[i];std::fprintf(f,"%s{\"epoch\":%llu,\"shape\":[",i?",":"",static_cast<unsigned long long>(r.epoch));
            for(int j=0;j<13;j++)std::fprintf(f,"%s%d",j?",":"",r.shape[j]);
            std::fprintf(f,"],\"compact_setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"complete\":%s,\"shadow\":%s,\"host_derivatives_authoritative\":%s,\"terms\":[",r.staged,r.allocated,r.complete?"true":"false",r.shadow?"true":"false",r.shadow?"true":"false");
            const int first=r.shape[12]?0:1;
            for(int k=first;k<2;k++){
                const auto& t=r.term[k];const auto& c=t.comparison;
                std::fprintf(f,"%s{\"kind\":%d,\"reference_selected\":%s,\"shadow_setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"shadow_export_bytes\":%zu,\"begun\":%s,\"generated\":%s,\"consumed\":%s,\"passed\":%s,\"status\":%d,\"comparison\":",
                    k==first?"":",",k+1,r.shadow?std::to_string(t.selected).c_str():"null",t.staged,t.allocated,t.exported,t.begun?"true":"false",t.generated?"true":"false",t.consumed?"true":"false",t.passed?"true":"false",t.status);
                if(!t.consumed||!r.shadow)std::fputs("null",f);
                else std::fprintf(f,"{\"points\":%llu,\"selected\":%llu,\"reference_selected\":%llu,\"compared\":%llu,\"index_compared\":%llu,\"mismatches\":%llu,\"nonfinite\":%llu,\"mask_mismatches\":%llu,\"index_mismatches\":%llu,\"absolute\":[%.17g,%.17g,%.17g],\"scaled\":[%.17g,%.17g,%.17g]}",
                    c.points,c.selected,c.reference_selected,c.compared,c.index_compared,c.mismatches,c.nonfinite,c.mask_mismatches,c.index_mismatches,
                    c.absolute[0],c.absolute[1],c.absolute[2],c.scaled[0],c.scaled[1],c.scaled[2]);
                std::fputs("}",f);
            }
            std::fputs("]}",f);
        }
        std::fputs("],\"atol\":1e-10,\"rtol\":1e-10,\"displacements_exact\":true}\n",f);std::fclose(f);
    }
};
Records& records(){static Records value;return value;}
struct Arrays {
    double *value=nullptr,*gradient=nullptr,*displacement=nullptr;int *active=nullptr,*point=nullptr,*atom=nullptr;
    remag::AtomicBasisValues view()const{return {value,gradient,displacement,active,point,atom};}
};
struct Term {
    bool begun=false,prepared=false,generated=false;size_t selected=0;cudaStream_t stream=nullptr;
    std::vector<double> value,gradient,displacement;std::vector<int> active,point,atom;
    Arrays actual,reference;remag::AtomicBasisComparison* comparison=nullptr;
};
struct Owner {
    uint64_t epoch=0;size_t record=0,staged=0,allocated=0;int core=0;bool complete=false,shadow=false;
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    remag::AtomicBasisView basis{};remag::AtomicGeometryView geometry{};
    std::vector<int> starts,pair_offset,pair_cluster,unit_atom;Term term[2];
    Owner(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Owner(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* allocate(size_t n){T* p=nullptr;const size_t bytes=(n?n:1)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;return p;}
    template<class T>T* upload(size_t n,const T* data,remag::Transfer boundary=remag::Transfer::atomic_basis_setup,cudaStream_t borrowed=nullptr){
        auto p=allocate<T>(n);if(n){check(cudaMemcpyAsync(p,data,n*sizeof(T),cudaMemcpyHostToDevice,borrowed?borrowed:stream));
            staged+=n*sizeof(T);remag::transfer(boundary,true,n*sizeof(T));}return p;
    }
};
std::unique_ptr<Owner> owner;
bool enabled(const char* n){const char* v=std::getenv(n);return v&&std::strcmp(v,"1")==0;}
void mode(){require(remag::atomic_basis_active()&&!(remag::atomic_basis_shadow_enabled()&&remag::atomic_basis_device_enabled())&&enabled("REMAG_ATOMIC_GRID_FORCE")&&enabled("REMAG_GRID_FORCE"),"Choose atomic device or shadow derivatives with authoritative GPU forces");}
void valid(uint64_t epoch){mode();require(!remag::in_scf()&&owner&&owner->epoch==epoch&&epoch==remag::geometry_epoch()&&!owner->complete&&owner->shadow==remag::atomic_basis_shadow_enabled(),"Invalid atomic basis phase/epoch/lifetime");}
Term& term(uint64_t epoch,int kind){valid(epoch);require((kind==1&&owner->core)||kind==2,"Unrequested atomic basis term");return owner->term[kind-1];}
}
bool remag::atomic_basis_shadow_enabled(){return enabled("REMAG_ATOMIC_BASIS_SHADOW");}
bool remag::atomic_basis_device_enabled(){return enabled("REMAG_ATOMIC_BASIS");}
bool remag::atomic_basis_active(){return atomic_basis_shadow_enabled()||atomic_basis_device_enabled();}
extern "C" int remag_atomic_basis_device_enabled(){return remag::atomic_basis_device_enabled();}
extern "C" int remag_atomic_basis_active(){return remag::atomic_basis_active();}
extern "C" int remag_atomic_basis_shadow_enabled(){return remag::atomic_basis_shadow_enabled();}
extern "C" int remag_atomic_basis_stage(const int* c,const int* z,const int* index,const double* parameters,const double* values,const double* second,
        const int* pa,const int* po,const int* pc,const int* species,const int* ua,const double* offset,const double* sub,const double* atom){
    try {
        mode();require(!remag::in_scf()&&c,"Invalid atomic basis staging phase");
        auto epoch=remag::geometry_epoch();require(epoch&&(!owner||(owner->complete&&owner->epoch<epoch)),"Duplicate or unconsumed atomic basis epoch");
        require(c[0]>0&&c[0]<=10000&&c[1]>=0&&c[2]>0&&c[3]>0&&c[3]<=4096&&c[4]>0&&c[5]>0&&c[6]>0&&c[7]>0&&c[8]>0&&c[8]<=16&&
            c[3]==c[8]*c[8]*c[8]&&c[9]>0&&c[10]>0&&c[11]>0&&c[9]%c[8]==0&&c[10]%c[8]==0&&c[11]%c[8]==0&&
            size_t(c[9])*c[10]<=size_t(c[7])&&size_t(c[9])*c[10]*c[11]==size_t(c[7])&&(c[12]==0||c[12]==1)&&
            size_t(c[2])*c[3]<=(size_t(4)<<30)/136,"Invalid/oversized atomic basis shape");
        require(z&&index&&parameters&&values&&second&&pa&&po&&pc&&species&&ua&&offset&&sub&&atom,"Missing atomic basis input");
        auto s=std::make_unique<Owner>();s->epoch=epoch;s->core=c[12];s->shadow=remag::atomic_basis_shadow_enabled();if(s->shadow)s->starts.assign(size_t(c[4])+1,0);
        for(int i=0;i<c[4];i++)require(species[i]>0&&species[i]<=c[0]&&ua[i]>0&&ua[i]<=c[6],"Invalid atomic species/unit mapping");
        for(int i=0;i<c[2];i++){
            require(pa[i]>0&&pa[i]<=c[4]&&po[i]>0&&po[i]<=c[5]&&pc[i]>0&&pc[i]<=c[7]/c[3],"Invalid compact atomic pair");
            require(!i||pa[i]>pa[i-1]||(pa[i]==pa[i-1]&&po[i]>po[i-1]),"Unordered or duplicate compact atomic pair");
            if(s->shadow)s->starts[pa[i]]++;
        }
        if(s->shadow){
        for(int i=1;i<=c[4];i++)s->starts[i]+=s->starts[i-1];
        s->pair_offset.assign(po,po+c[2]);s->pair_cluster.assign(pc,pc+c[2]);s->unit_atom.assign(ua,ua+c[4]);
        }
        s->basis={c[0],c[1],s->upload(c[0],z),s->upload(size_t(4)*c[0],index),s->upload(size_t(4)*c[0],parameters),s->upload(c[1],values),s->upload(c[1],second)};
        s->geometry={c[2],c[3],c[4],c[5],c[6],c[7],c[8],{c[9],c[10],c[11]},s->upload(c[2],pa),s->upload(c[2],po),s->upload(c[2],pc),
            s->upload(c[4],species),s->upload(c[4],ua),s->upload(size_t(3)*c[5],offset),s->upload(size_t(3)*c[3],sub),s->upload(size_t(3)*c[4],atom)};
        check(cudaStreamSynchronize(s->stream));Record r;r.epoch=epoch;r.shadow=s->shadow;std::copy(c,c+13,r.shape.begin());r.staged=s->staged;r.allocated=s->allocated;
        s->record=records().rows.size();records().rows.push_back(r);owner=std::move(s);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Atomic basis preparation: %s\n",e.what());return -261;}
}
void remag::begin_atomic_basis(uint64_t epoch,int kind,int points,int nsp,int atoms,const int* mesh,int nsm,int core){
    auto& t=term(epoch,kind);auto& s=*owner;const auto& g=s.geometry;
    require(!t.begun&&points==g.points&&nsp==g.nsp&&atoms==g.unit_atoms&&mesh&&std::equal(mesh,mesh+3,g.mesh)&&nsm==g.nsm&&core==s.core,"Atomic basis consumer shape or lifetime differs");
    require(kind!=2||!s.core||s.term[0].generated,"Atomic basis consumers out of order");
    const size_t n=size_t(g.pairs)*g.nsp;
    if(s.shadow){t.value.assign(n,0.);t.gradient.assign(3*n,0.);t.displacement.assign(3*n,0.);
    t.active.assign(n,0);t.point.assign(n,0);t.atom.assign(n,0);}
    t.begun=true;records().rows[s.record].term[kind-1].begun=true;
}
extern "C" int remag_atomic_basis_point(int kind,int atom,int offset,int sub,int point,int ua,double value,const double* gradient,const double* dx){
    try {
        auto& t=term(remag::geometry_epoch(),kind);auto& s=*owner;const auto& g=s.geometry;
        require(s.shadow&&t.begun&&!t.prepared&&atom>0&&atom<=g.atoms&&offset>0&&offset<=g.offsets&&sub>0&&sub<=g.nsp&&gradient&&dx,"Invalid atomic basis reference point");
        auto begin=s.pair_offset.begin()+s.starts[atom-1],end=s.pair_offset.begin()+s.starts[atom];auto it=std::lower_bound(begin,end,offset);
        require(it!=end&&*it==offset,"Unstaged atomic reference pair");const size_t pair=it-s.pair_offset.begin(),i=pair*g.nsp+sub-1;
        require(!t.active[i]&&point==(s.pair_cluster[pair]-1)*g.nsp+sub&&ua==s.unit_atom[atom-1],"Duplicate or changed atomic reference mapping");
        require(std::isfinite(value)&&std::all_of(gradient,gradient+3,[](double v){return std::isfinite(v);})&&std::all_of(dx,dx+3,[](double v){return std::isfinite(v);}),"Nonfinite CPU atomic derivative");
        if(kind==2){
            const int p=point-1,sp=p%g.nsp,cl=p/g.nsp,n=g.nsm,nx=g.mesh[0]/n,ny=g.mesh[1]/n;
            point=1+(cl%nx)*n+sp%n+g.mesh[0]*(((cl/nx)%ny)*n+(sp/n)%n+g.mesh[1]*((cl/(nx*ny))*n+sp/(n*n)));
        }
        t.active[i]=1;t.point[i]=point;t.atom[i]=ua;t.value[i]=value;std::copy(gradient,gradient+3,t.gradient.data()+3*i);std::copy(dx,dx+3,t.displacement.data()+3*i);t.selected++;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Atomic basis reference: %s\n",e.what());return -262;}
}
remag::AtomicGridForceView remag::prepare_atomic_basis(uint64_t epoch,int kind,size_t selected,cudaStream_t stream){
    auto& t=term(epoch,kind);auto& s=*owner;require(t.begun&&!t.prepared&&selected==t.selected&&stream,"Incomplete or duplicate atomic reference");
    const size_t n=size_t(s.geometry.pairs)*s.geometry.nsp,before=s.staged,allocated=s.allocated;
    t.actual={s.allocate<double>(n),s.allocate<double>(3*n),s.allocate<double>(3*n),s.allocate<int>(n),s.allocate<int>(n),s.allocate<int>(n)};
    if(s.shadow){
    auto b=Transfer::atomic_basis_shadow;
    t.reference={s.upload(n,t.value.data(),b,stream),s.upload(3*n,t.gradient.data(),b,stream),s.upload(3*n,t.displacement.data(),b,stream),
        s.upload(n,t.active.data(),b,stream),s.upload(n,t.point.data(),b,stream),s.upload(n,t.atom.data(),b,stream)};
    AtomicBasisComparison zero{};t.comparison=s.upload(1,&zero,b,stream);check(cudaStreamSynchronize(stream));}
    t.prepared=true;t.stream=stream;
    auto& r=records().rows[s.record];r.allocated=s.allocated;auto& v=r.term[kind-1];v.selected=t.selected;v.staged=s.staged-before;v.allocated=s.allocated-allocated;
    return {s.geometry.points,int(n),s.geometry.unit_atoms,t.actual.point,t.actual.atom,t.actual.gradient,t.actual.displacement,t.actual.active};
}
void remag::enqueue_atomic_basis(uint64_t epoch,int kind,int* status,cudaStream_t stream){
    auto& t=term(epoch,kind);auto& s=*owner;require(t.prepared&&!t.generated&&status&&stream==t.stream,"Invalid atomic basis generation");
    const auto& a=t.actual;
    check(evaluate_atomic_geometry_device(s.basis,s.geometry,kind,a.value,a.gradient,a.displacement,a.active,a.point,a.atom,status,stream));
    if(s.shadow)check(compare_atomic_geometry_device(t.active.size(),a.view(),t.reference.view(),t.comparison,status,stream));
    t.generated=true;records().rows[s.record].term[kind-1].generated=true;
}
void remag::finish_atomic_basis(uint64_t epoch,int force_status){
    valid(epoch);auto& s=*owner;auto& r=records().rows[s.record];const size_t n=size_t(s.geometry.pairs)*s.geometry.nsp;bool passed=force_status==0;
    for(int k=s.core?0:1;k<2;k++)require(s.term[k].generated,"Atomic basis final publication before every consumer");
    for(int k=s.core?0:1;k<2;k++){
        auto& t=s.term[k];auto& v=r.term[k];v.consumed=true;v.status=force_status;
        if(!s.shadow){v.passed=force_status==0;passed=passed&&v.passed;continue;}
        check(cudaMemcpy(&v.comparison,t.comparison,sizeof(v.comparison),cudaMemcpyDeviceToHost));
        transfer(Transfer::atomic_basis_shadow,false,sizeof(v.comparison));v.exported=sizeof(v.comparison);v.consumed=true;v.status=force_status;
        const auto& c=v.comparison;v.passed=force_status==0&&c.points==n&&c.selected==t.selected&&c.reference_selected==t.selected&&
            c.compared==4*n+3*t.selected&&c.index_compared==2*t.selected&&!c.mismatches&&!c.nonfinite&&!c.mask_mismatches&&!c.index_mismatches;
        for(int j=0;j<3;j++)v.passed=v.passed&&std::isfinite(c.absolute[j])&&std::isfinite(c.scaled[j])&&c.absolute[j]>=0.&&c.scaled[j]>=0.&&c.scaled[j]<=1.;
        v.passed=v.passed&&c.absolute[2]==0.&&c.scaled[2]==0.;passed=passed&&v.passed;
    }
    require(passed,"Full atomic derivative/selection shadow failed");s.complete=r.complete=true;
}
