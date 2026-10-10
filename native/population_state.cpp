// SPDX-License-Identifier: GPL-3.0-or-later
#include "population_state.hpp"
#include "initial_state.hpp"
#include "population.hpp"
#include "population_observables.hpp"
#include "execution.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace remag {
namespace {
void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
enum class Stage {prepared,bound,evaluating,published,failed};
struct Record {
    uint64_t epoch=0;
    int atoms=0,rows=0,capacity=0,spins=0,species=0;
    size_t setup=0,retained=0,workspace=0,exports[7]{};
    int reads[7]{};
    bool evaluated=false,published=false;
};
struct Evidence {
    std::vector<Record> records;
    ~Evidence(){
        if(records.empty())return;
        FILE* f=std::fopen("remag-population.json","w");if(!f)return;
        std::fputs("{\"schema\":2,\"scope\":\"GPU population and derived observables with shared final status and requested terminal exports\",\"cycles\":[",f);
        for(size_t i=0;i<records.size();i++){
            const auto& r=records[i];
            std::fprintf(f,"%s{\"epoch\":%llu,\"atoms\":%d,\"rows\":%d,\"capacity\":%d,\"spins\":%d,\"species\":%d,\"setup_bytes\":%zu,\"retained_topology_bytes\":%zu,\"workspace_bytes\":%zu,\"evaluated\":%s,\"published\":%s,\"export_bytes\":[",
                i?",":"",static_cast<unsigned long long>(r.epoch),r.atoms,r.rows,r.capacity,r.spins,r.species,r.setup,r.retained,r.workspace,
                r.evaluated?"true":"false",r.published?"true":"false");
            for(int k=0;k<7;k++)std::fprintf(f,"%s%zu",k?",":"",r.exports[k]);
            std::fputs("],\"reads\":[",f);
            for(int k=0;k<7;k++)std::fprintf(f,"%s%d",k?",":"",r.reads[k]);
            std::fputs("]}",f);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
} evidence;
struct State {
    Stage stage=Stage::prepared;
    uint64_t epoch=0;
    int atoms,rows,capacity,spins,components,species;
    size_t record=0;
    cudaStream_t stream=nullptr;
    PopulationProblem* problem=nullptr;
    PopulationObservableProblem* observable_problem=nullptr;
    int *lasto=nullptr,*num=nullptr,*ptr=nullptr,*status=nullptr,*atom_species=nullptr;
    double *overlap=nullptr,*output=nullptr,*valence=nullptr,*views=nullptr;
    std::vector<void*> memory;
    std::vector<double> host[7];
    State(int a,int n,int c,int s,int ns):atoms(a),rows(n),capacity(c),spins(s),components(std::min(s,4)),species(ns){}
    ~State(){cudaDeviceSynchronize();if(stream)cudaStreamSynchronize(stream);for(void* p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* allocate(size_t n){
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);
        evidence.records[record].workspace+=n*sizeof(T);return p;
    }
    template<class T>T* upload(size_t n,const T* input){
        T* p=allocate<T>(n);check(cudaMemcpyAsync(p,input,n*sizeof(T),cudaMemcpyHostToDevice,stream));
        evidence.records[record].setup+=n*sizeof(T);transfer(Transfer::population_setup,true,n*sizeof(T));return p;
    }
};
std::unique_ptr<State> state;
void current(uint64_t epoch,Stage stage){
    require(population_enabled()&&integral_force_enabled()&&!in_scf()&&state&&state->stage==stage&&
            state->epoch==epoch&&epoch==geometry_epoch(),"Population state is stale, incomplete or out of order");
}
}
bool population_enabled(){const char* p=std::getenv("REMAG_POPULATION");return p&&std::strcmp(p,"1")==0;}
void prepare_population_state(uint64_t epoch,int atoms,int rows,int capacity,int spins,int species,const int* lasto,const int* atom_species,const double* valence){
    require(population_enabled()&&integral_force_enabled()&&!in_scf()&&!state&&epoch&&epoch==geometry_epoch()&&
        atoms>0&&atoms<=1000000&&rows>0&&rows<=1000000&&capacity>0&&(spins==1||spins==2||spins==4||spins==8)&&
        capacity<=INT_MAX/spins&&species>0&&species<=1000000&&lasto&&atom_species&&valence,"Invalid population preparation");
    require(lasto[0]==0&&lasto[atoms]==rows&&std::is_sorted(lasto,lasto+atoms+1),"Invalid population atom partition");
    require(std::all_of(atom_species,atom_species+atoms,[&](int id){return id>=1&&id<=species;})&&
        std::all_of(valence,valence+species,[](double v){return std::isfinite(v);}),"Invalid population species");
    auto s=std::make_unique<State>(atoms,rows,capacity,spins,species);s->epoch=epoch;s->record=evidence.records.size();
    evidence.records.push_back({epoch,atoms,rows,capacity,spins,species});
    check(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking));
    PopulationProblem problem{capacity,rows,atoms,spins};
    s->problem=s->upload(1,&problem);s->lasto=s->upload(atoms+1,lasto);
    PopulationObservableProblem observable_problem{rows,atoms,species,s->components};
    s->observable_problem=s->upload(1,&observable_problem);
    s->atom_species=s->upload(atoms,atom_species);s->valence=s->upload(species,valence);
    s->views=s->allocate<double>(size_t(5)*rows+size_t(6)*atoms+size_t(5)*species+10);
    s->output=s->allocate<double>(size_t(s->components)*(rows+atoms+1));s->status=s->allocate<int>(1);
    check(cudaStreamSynchronize(s->stream));state=std::move(s);
}
void bind_population_topology(uint64_t epoch,const int* num,const int* ptr,const double* overlap,cudaStream_t stream){
    if(!population_enabled())return;
    current(epoch,Stage::prepared);require(num&&ptr&&overlap&&stream,"Missing population topology");auto& s=*state;
    s.num=s.allocate<int>(s.rows);s.ptr=s.allocate<int>(s.rows);s.overlap=s.allocate<double>(s.capacity);
    check(cudaMemcpyAsync(s.num,num,size_t(s.rows)*4,cudaMemcpyDeviceToDevice,stream));
    check(cudaMemcpyAsync(s.ptr,ptr,size_t(s.rows)*4,cudaMemcpyDeviceToDevice,stream));
    check(cudaMemcpyAsync(s.overlap,overlap,size_t(s.capacity)*8,cudaMemcpyDeviceToDevice,stream));
    check(cudaStreamSynchronize(stream));evidence.records[s.record].retained=size_t(s.rows)*8+size_t(s.capacity)*8;
    s.stage=Stage::bound;
}
void evaluate_final_population(const FinalGridStateView& view,const FinalForceAccumulatorView& pool){
    if(!population_enabled())return;
    current(view.epoch,Stage::bound);auto& s=*state;
    require(final_population_window(view.epoch,view.density),"Population computation outside final collection");
    require(view.shape.nuo==s.rows&&view.shape.nnz==s.capacity&&view.h_spins==s.spins&&view.density&&pool.stream&&pool.status,
        "Population does not match the final electronic snapshot");
    // Mark attempted before launching: failed or partially launched work cannot
    // be retried to clear status or discard the unpublished epoch.
    s.stage=Stage::evaluating;
    check(population_device(s.capacity,s.rows,s.atoms,s.spins,s.problem,s.num,s.ptr,s.lasto,s.overlap,view.density,
        s.output,s.output+size_t(s.rows)*s.components,s.output+size_t(s.rows+s.atoms)*s.components,s.status,pool.stream));
    check(population_observables_device(s.rows,s.atoms,s.species,s.components,s.observable_problem,s.atom_species,s.valence,
        s.output,s.output+size_t(s.rows)*s.components,s.output+size_t(s.rows+s.atoms)*s.components,
        s.views,s.views+size_t(5)*s.rows,s.views+size_t(5)*s.rows+size_t(6)*s.atoms,
        s.views+size_t(5)*s.rows+size_t(6)*s.atoms+size_t(5)*s.species,s.status,pool.stream));
    check(merge_population_status_device(s.status,pool.status,pool.stream));
    evidence.records[s.record].evaluated=true;
}
void complete_population_publication(uint64_t epoch){
    if(!population_enabled())return;
    current(epoch,Stage::evaluating);
    require(final_grid_forces_complete(epoch),"Population publication before force acceptance");
    state->stage=Stage::published;evidence.records[state->record].published=true;
}
void reset_population_state(){
    require(!in_scf()&&(!state||state->stage==Stage::published),"Unpublished population cannot be discarded");state.reset();
}
void read_final_population(uint64_t epoch,int kind,int atoms,int rows,int spins,int output_count,double* output){
    current(epoch,Stage::published);auto& s=*state;
    require(final_grid_forces_complete(epoch)&&kind>=1&&kind<=7&&atoms==s.atoms&&rows==s.rows&&spins==s.spins&&output,
        "Population export requested before publication or with a different shape");
    const size_t sizes[]={size_t(s.components)*rows,size_t(s.components)*atoms,size_t(s.components),
        size_t(5)*rows,size_t(6)*atoms,size_t(5)*s.species,10};
    const size_t offsets[]={0,size_t(s.components)*rows,size_t(s.components)*(rows+atoms),
        0,size_t(5)*rows,size_t(5)*rows+size_t(6)*atoms,size_t(5)*rows+size_t(6)*atoms+size_t(5)*s.species};
    const size_t size=sizes[kind-1];
    require(output_count>=0&&size_t(output_count)==size,"Population export output size differs");
    auto& host=s.host[kind-1];auto& r=evidence.records[s.record];
    if(host.empty()){
        const double* source=(kind<=3?s.output:s.views)+offsets[kind-1];
        std::vector<double> values(size);
        try {
            check(cudaMemcpy(values.data(),source,size*8,cudaMemcpyDeviceToHost));
            transfer(Transfer::population_output,false,size*8);r.exports[kind-1]+=size*8;
            require(std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}),"Nonfinite population export");
            host=std::move(values);
        }catch(...){s.stage=Stage::failed;throw;}
    }
    std::copy(host.begin(),host.end(),output);r.reads[kind-1]++;
}
}
extern "C" int remag_population_enabled(){return remag::population_enabled()?1:0;}
extern "C" int remag_population_begin(int atoms,int rows,int capacity,int spins,int species,const int* lasto,const int* atom_species,const double* valence){
    try{remag::prepare_population_state(remag::geometry_epoch(),atoms,rows,capacity,spins,species,lasto,atom_species,valence);return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"Population preparation: %s\n",e.what());return -171;}
}
extern "C" int remag_population_read(int kind,int atoms,int rows,int spins,int output_count,double* output){
    try{
        if(remag::initial_population_window())remag::read_initial_population(kind,atoms,rows,spins,output_count,output);
        else remag::read_final_population(remag::geometry_epoch(),kind,atoms,rows,spins,output_count,output);
        return 0;
    }
    catch(const std::exception& e){std::fprintf(stderr,"Population export: %s\n",e.what());return -172;}
}
