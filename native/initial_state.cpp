// SPDX-License-Identifier: GPL-3.0-or-later
#include "initial_state.hpp"
#include "spin_rotation.hpp"
#include "population.hpp"
#include "population_observables.hpp"
#include "population_state.hpp"
#include "execution.hpp"
#include "density_history_state.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" int remag_initialization_enabled();
extern "C" int remag_cycle_enabled();
namespace remag {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
enum class Stage {open,ready,borrowed,released,failed};
struct Record {
    uint64_t token=0,epoch=0;
    int atoms=0,rows=0,capacity=0,spins=0,species=0,requests=0,status=-1;
    size_t density_upload=0,population_setup=0,status_export=0,exports[7]{};
    int reads[7]{};
    bool evaluated=false,closed=false,borrowed=false,released=false,from_history=false;
};
struct Evidence {
    std::vector<Record> records;
    ~Evidence(){
        if(records.empty())return;
        FILE* f=std::fopen("remag-initial-state.json","w");if(!f)return;
        std::fputs("{\"schema\":1,\"scope\":\"Initial unnormalized device D from explicit host staging or retained GPU history, requested population output, and one SCF density lease\",\"cycles\":[",f);
        for(size_t i=0;i<records.size();i++) {
            const auto& r=records[i];
            std::fprintf(f,"%s{\"token\":%llu,\"epoch\":%llu,\"atoms\":%d,\"rows\":%d,\"capacity\":%d,\"spins\":%d,\"species\":%d,\"requests\":%d,\"status\":%d,\"density_upload_bytes\":%zu,\"population_setup_bytes\":%zu,\"status_export_bytes\":%zu,\"evaluated\":%s,\"closed\":%s,\"borrowed\":%s,\"released\":%s,\"export_bytes\":[",
                i?",":"",static_cast<unsigned long long>(r.token),static_cast<unsigned long long>(r.epoch),
                r.atoms,r.rows,r.capacity,r.spins,r.species,r.requests,r.status,r.density_upload,r.population_setup,r.status_export,
                r.evaluated?"true":"false",r.closed?"true":"false",r.borrowed?"true":"false",r.released?"true":"false");
            for(int k=0;k<7;k++)std::fprintf(f,"%s%zu",k?",":"",r.exports[k]);
            std::fputs("],\"reads\":[",f);
            for(int k=0;k<7;k++)std::fprintf(f,"%s%d",k?",":"",r.reads[k]);
            std::fprintf(f,"],\"density_from_history\":%s}",r.from_history?"true":"false");
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
} evidence;
struct State {
    Stage stage=Stage::open;size_t record=0;
    int atoms,rows,capacity,spins,species,components;
    uint64_t epoch=0;
    const double* host_identity=nullptr;
    cudaStream_t stream=nullptr;
    double *density=nullptr,*raw=nullptr,*views=nullptr;
    int* status=nullptr;
    std::vector<void*> storage;
    std::vector<int> num,ptr;
    std::vector<double> overlap,host[7];
    State(int a,int n,int c,int s,int ns):atoms(a),rows(n),capacity(c),spins(s),species(ns),components(std::min(s,4)){}
    ~State(){if(stream)cudaStreamSynchronize(stream);for(auto p:storage)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* allocate(size_t n){T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));storage.push_back(p);return p;}
    template<class T>T* upload(size_t n,const T* src,bool is_density=false){
        T* p=allocate<T>(n);check(cudaMemcpyAsync(p,src,n*sizeof(T),cudaMemcpyHostToDevice,stream));
        auto& r=evidence.records[record];(is_density?r.density_upload:r.population_setup)+=n*sizeof(T);
        transfer(is_density?Transfer::initial_density_setup:Transfer::initial_population_setup,true,n*sizeof(T));return p;
    }
};
std::unique_ptr<State> state;
void current(Stage stage){require(initial_state_enabled()&&!in_scf()&&state&&state->stage==stage,"Initial density owner is absent, stale or out of order");}
void begin(int atoms,int rows,int capacity,int spins,int species,int requests,const int* num,const int* ptr,
        const double* overlap,const double* density,const int* lasto,const int* ids,const double* valence){
    require(initial_state_enabled()&&remag_cycle_enabled()&&integral_force_enabled()&&!in_scf()&&
        (!state||(state->stage==Stage::released&&final_grid_forces_complete(state->epoch))),
        "Previous initial density/force ownership is incomplete");
    require(atoms>0&&atoms<=1000000&&rows>0&&rows<=1000000&&capacity>0&&species>0&&species<=1000000&&
        (spins==1||spins==2||spins==4||spins==8)&&capacity<=INT_MAX/spins&&requests>=0&&requests<=3&&
        num&&ptr&&overlap&&density&&lasto&&ids&&valence,"Invalid initial density configuration");
    require(lasto[0]==0&&lasto[atoms]==rows&&std::is_sorted(lasto,lasto+atoms+1),"Invalid initial atom partition");
    require(std::all_of(ids,ids+atoms,[&](int i){return i>=1&&i<=species;})&&
        std::all_of(valence,valence+species,[](double x){return std::isfinite(x);}),"Invalid initial species metadata");
    auto s=std::make_unique<State>(atoms,rows,capacity,spins,species);s->record=evidence.records.size();
    Record r;r.token=s->record+1;r.atoms=atoms;r.rows=rows;r.capacity=capacity;r.spins=spins;r.species=species;r.requests=requests;
    r.from_history=density_history_enabled()&&density_history_reuses();
    evidence.records.push_back(r);s->host_identity=density;s->num.assign(num,num+rows);s->ptr.assign(ptr,ptr+rows);s->overlap.assign(overlap,overlap+capacity);
    check(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking));
    s->density=r.from_history?s->allocate<double>(size_t(capacity)*spins):s->upload(size_t(capacity)*spins,density,true);
    if(density_history_enabled())prepare_history_density(s->density);
    require(orientation_density(capacity,spins,s->density,s->stream)==0,"GPU spin rotation failed");
    if(requests) {
        PopulationProblem p{capacity,rows,atoms,spins};PopulationObservableProblem v{rows,atoms,species,s->components};
        auto problem=s->upload(1,&p);auto view_problem=s->upload(1,&v);
        auto dnum=s->upload(rows,num);auto dptr=s->upload(rows,ptr);auto ends=s->upload(atoms+1,lasto);
        auto dids=s->upload(atoms,ids);auto z=s->upload(species,valence);auto ov=s->upload(capacity,overlap);
        s->raw=s->allocate<double>(size_t(s->components)*(rows+atoms+1));
        s->views=s->allocate<double>(size_t(5)*rows+size_t(6)*atoms+size_t(5)*species+10);s->status=s->allocate<int>(1);
        check(population_device(capacity,rows,atoms,spins,problem,dnum,dptr,ends,ov,s->density,
            s->raw,s->raw+size_t(rows)*s->components,s->raw+size_t(rows+atoms)*s->components,s->status,s->stream));
        check(population_observables_device(rows,atoms,species,s->components,view_problem,dids,z,
            s->raw,s->raw+size_t(rows)*s->components,s->raw+size_t(rows+atoms)*s->components,
            s->views,s->views+size_t(5)*rows,s->views+size_t(5)*rows+size_t(6)*atoms,
            s->views+size_t(5)*rows+size_t(6)*atoms+size_t(5)*species,s->status,s->stream));
        evidence.records[s->record].evaluated=true;
    }
    check(cudaStreamSynchronize(s->stream));state=std::move(s);
}
}
bool initial_state_enabled(){return remag_initialization_enabled()&&population_enabled();}
bool initial_population_window(){return state&&state->stage==Stage::open;}
void read_initial_population(int kind,int atoms,int rows,int spins,int count,double* output){
    current(Stage::open);auto& s=*state;auto& r=evidence.records[s.record];
    require(kind>=1&&kind<=7&&atoms==s.atoms&&rows==s.rows&&spins==s.spins&&output&&r.requests&&r.evaluated,
        "Initial population export has no matching request or shape");
    require((r.requests&2)||kind==7,"Initial spin request does not authorize orbital/atomic tables");
    const size_t sizes[]={size_t(s.components)*rows,size_t(s.components)*atoms,size_t(s.components),size_t(5)*rows,size_t(6)*atoms,size_t(5)*s.species,10};
    const size_t offsets[]={0,size_t(s.components)*rows,size_t(s.components)*(rows+atoms),0,size_t(5)*rows,size_t(5)*rows+size_t(6)*atoms,size_t(5)*rows+size_t(6)*atoms+size_t(5)*s.species};
    const size_t n=sizes[kind-1];require(count>=0&&size_t(count)==n,"Initial population output size differs");
    if(!r.status_export) {
        check(cudaMemcpy(&r.status,s.status,sizeof(int),cudaMemcpyDeviceToHost));r.status_export=sizeof(int);
        transfer(Transfer::initial_population_output,false,sizeof(int));
        if(r.status){s.stage=Stage::failed;throw std::runtime_error("Initial population computation failed");}
    }
    auto& host=s.host[kind-1];
    if(host.empty()) {
        std::vector<double> values(n);
        check(cudaMemcpy(values.data(),(kind<=3?s.raw:s.views)+offsets[kind-1],n*sizeof(double),cudaMemcpyDeviceToHost));
        r.exports[kind-1]=n*sizeof(double);transfer(Transfer::initial_population_output,false,n*sizeof(double));
        require(std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}),"Invalid initial population output");host=std::move(values);
    }
    std::copy(host.begin(),host.end(),output);r.reads[kind-1]++;
}
double* borrow_initial_density(uint64_t epoch,int rows,int capacity,int spins,const int* num,const int* ptr,
        const double* overlap,const double* host_identity){
    current(Stage::ready);auto& s=*state;
    require(epoch&&epoch==geometry_epoch()&&rows==s.rows&&capacity==s.capacity&&spins==s.spins&&host_identity==s.host_identity&&
        num&&ptr&&overlap&&std::equal(s.num.begin(),s.num.end(),num)&&std::equal(s.ptr.begin(),s.ptr.end(),ptr)&&
        std::memcmp(s.overlap.data(),overlap,size_t(capacity)*sizeof(double))==0,"Initial density lease does not match SCF geometry or host identity");
    if(density_history_enabled())bind_history_density(epoch,s.density);
    s.stage=Stage::borrowed;s.epoch=epoch;auto& r=evidence.records[s.record];r.borrowed=true;r.epoch=epoch;return s.density;
}
void release_initial_density(uint64_t epoch){
    current(Stage::borrowed);require(epoch==state->epoch,"Initial density release has a stale epoch");
    state->stage=Stage::released;evidence.records[state->record].released=true;
}
}
extern "C" int remag_initial_state_enabled(){return remag::initial_state_enabled();}
extern "C" int remag_initial_state_begin(int a,int n,int c,int s,int ns,int req,const int* num,const int* ptr,
        const double* ov,const double* d,const int* lasto,const int* ids,const double* valence){
    try{remag::begin(a,n,c,s,ns,req,num,ptr,ov,d,lasto,ids,valence);return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"Initial density preparation: %s\n",e.what());return -181;}
}
extern "C" int remag_initial_state_end(){
    try{
        remag::current(remag::Stage::open);auto& s=*remag::state;auto& r=remag::evidence.records[s.record];
        remag::require(!r.requests||r.status_export==sizeof(int),"Requested initial populations were not published");
        s.stage=remag::Stage::ready;r.closed=true;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Initial population close: %s\n",e.what());return -182;}
}
