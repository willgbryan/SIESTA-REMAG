// SPDX-License-Identifier: GPL-3.0-or-later
#include "final_grid_state.hpp"
#include "atomic_basis_state.hpp"
#include "nonlocal_force_state.hpp"
#include "execution.hpp"
#include "initial_mesh_state.hpp"
#include "population_state.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace remag {
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
struct Snapshot {
    FinalGridStateView view{};
    int required=4,borrowed=0,consumed=0,active=0;
    int pair_required=0;
    int pair_consumed=0,pair_active=0;
    int atoms=0,forces=0,stress=0;
    bool integral=false,hartree_added=false,publication_attempted=false,collected=false,published=false;
    bool history_active=false,history_consumed=false;
    FinalForceAccumulatorView accumulator{};
    std::vector<double> initial_force,initial_stress;
    std::vector<AtomicForceRecord> atomic_records;
    std::vector<IntegralForceRecord> integral_records;
    std::vector<void*> memory;
    ~Snapshot() {
        if(accumulator.stream) cudaStreamSynchronize(accumulator.stream);
        for(void* p:memory) cudaFree(p);
        if(accumulator.stream) cudaStreamDestroy(accumulator.stream);
    }
    template<class T> T* stage(size_t n,const T* source) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);
        check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,accumulator.stream));
        accumulator.staged_bytes+=n*sizeof(T);transfer(Transfer::grid_flow_geometry,true,n*sizeof(T));return p;
    }
    template<class T>const T* copy(const T* source,size_t n,cudaStream_t stream) {
        require(source&&n&&n<=SIZE_MAX/sizeof(T),"Missing final electronic field");
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);
        check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        view.bytes+=n*sizeof(T);return p;
    }
};
std::unique_ptr<Snapshot> snapshot;
}
bool final_grid_state_enabled() {
    const char* v=std::getenv("REMAG_FORCE_SHADOW_RESIDENT");const char* force=std::getenv("REMAG_GRID_FORCE");
    return (v&&std::strcmp(v,"1")==0)||(force&&std::strcmp(force,"1")==0);
}
bool atomic_grid_force_enabled() {
    const char* value=std::getenv("REMAG_ATOMIC_GRID_FORCE");return value&&std::strcmp(value,"1")==0;
}
bool integral_force_enabled() {
    const char* value=std::getenv("REMAG_INTEGRAL_FORCE");return value&&std::strcmp(value,"1")==0;
}
bool pair_force_active() {return pair_force_shadow_enabled()||integral_force_enabled();}
void reset_final_grid_state() {
    require(!in_scf()&&(!snapshot||(snapshot->consumed==snapshot->required&&!snapshot->pair_active&&!snapshot->history_active&&
        (!snapshot->pair_required||snapshot->pair_consumed==snapshot->pair_required)&&
        (!snapshot->integral||snapshot->published))),"Final force consumer incomplete before new geometry");reset_population_state();snapshot.reset();
}
bool final_grid_forces_complete(uint64_t epoch) {
    return !in_scf()&&snapshot&&!snapshot->active&&!snapshot->history_active&&snapshot->consumed==snapshot->required&&
        !snapshot->pair_active&&(!snapshot->pair_required||snapshot->pair_consumed==snapshot->pair_required)&&
        (!snapshot->integral||snapshot->published)&&
        snapshot->view.epoch==epoch&&epoch==geometry_epoch();
}
FinalGridStateView borrow_final_history_state(uint64_t epoch) {
    require(final_grid_forces_complete(epoch)&&!snapshot->history_consumed&&snapshot->view.convergence,
        "History requested stale, unfinished or already retained final electronic state");
    snapshot->history_active=true;return snapshot->view;
}
void complete_final_history_state(uint64_t epoch) {
    require(!in_scf()&&snapshot&&snapshot->history_active&&!snapshot->history_consumed&&
        snapshot->view.epoch==epoch&&epoch==geometry_epoch(),"History completion outside its snapshot lifetime");
    snapshot->history_active=false;snapshot->history_consumed=true;
}
bool final_population_window(uint64_t epoch,const double* density) {
    return !in_scf()&&snapshot&&snapshot->integral&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        snapshot->view.density==density&&snapshot->publication_attempted&&!snapshot->collected&&!snapshot->published&&
        !snapshot->active&&!snapshot->pair_active&&snapshot->consumed==snapshot->required&&snapshot->pair_consumed==7&&
        snapshot->hartree_added&&snapshot->integral_records.size()==3;
}
bool pair_force_shadow_enabled() {
    const char* v=std::getenv("REMAG_PAIR_FORCE_SHADOW");return v&&std::strcmp(v,"1")==0;
}
FinalGridStateView borrow_final_pair_state(uint64_t epoch,int consumer) {
    require(pair_force_active()&&!in_scf()&&snapshot&&snapshot->integral==integral_force_enabled()&&snapshot->pair_required&&
        !snapshot->active&&!snapshot->pair_active&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        ((consumer==1&&snapshot->pair_consumed==0&&snapshot->consumed==0)||
         ((consumer==2||consumer==3)&&snapshot->pair_required==7&&snapshot->pair_consumed==1&&snapshot->consumed==0&&
          ((consumer==3)==(snapshot->view.h_spins==8)))||
         (consumer==4&&snapshot->pair_consumed==(snapshot->pair_required&3)&&snapshot->consumed==snapshot->required)),
        "Pair force requested stale, overlapping or unordered electronic state");
    snapshot->pair_active=consumer;return snapshot->view;
}
void complete_final_pair_shadow(uint64_t epoch,int consumer) {
    require(pair_force_active()&&!in_scf()&&snapshot&&snapshot->integral==integral_force_enabled()&&snapshot->pair_required&&
        snapshot->pair_active==consumer&&(consumer==1||consumer==2||consumer==3||consumer==4)&&
        snapshot->view.epoch==epoch&&epoch==geometry_epoch(),"Pair force completion outside its electronic lifetime");
    if(snapshot->integral)require(!snapshot->integral_records.empty()&&snapshot->integral_records.back().consumer==consumer,
        "Integral force consumer did not record its computation");
    snapshot->pair_consumed|=consumer==3?2:consumer;snapshot->pair_active=0;
}
void retain_final_grid_state(const FinalGridStateView& v,cudaStream_t stream) {
    require(final_grid_state_enabled()&&!in_scf()&&!snapshot&&stream&&v.epoch==geometry_epoch()&&
            v.shape.nnz>0&&v.shape.np>0&&v.shape.nsp>0&&(v.h_spins==1||v.h_spins==2||v.h_spins==4||v.h_spins==8)&&
            v.energy_spins==(v.h_spins<4?v.h_spins:4),"Invalid final electronic handoff");
    require(v.nsm>0&&v.nsm<=64&&(v.partial_core==0||v.partial_core==1)&&
            v.shape.nsp==v.nsm*v.nsm*v.nsm&&v.mesh[0]>0&&v.mesh[1]>0&&v.mesh[2]>0&&
            v.mesh[0]%v.nsm==0&&v.mesh[1]%v.nsm==0&&v.mesh[2]%v.nsm==0&&
            size_t(v.mesh[0])*v.mesh[1]*v.mesh[2]==size_t(v.shape.np)*v.shape.nsp,"Invalid final mesh metadata");
    auto s=std::make_unique<Snapshot>();s->view=v;s->view.bytes=0;s->integral=integral_force_enabled();
    require(!s->integral||(atomic_grid_force_enabled()&&!pair_force_shadow_enabled()&&!remag_nonlocal_force_shadow_enabled()&&v.hartree_stress),
        "Integral forces require authoritative atomic/grid forces, retained Hartree stress and no pair/nonlocal shadows");
    require(!remag_nonlocal_force_shadow_enabled()||pair_force_shadow_enabled(),"Nonlocal shadow requires kinetic/overlap leases");
    s->pair_required=s->integral?7:pair_force_shadow_enabled()?(remag_nonlocal_force_shadow_enabled()?7:5):0;
    const char* atomic=std::getenv("REMAG_ATOMIC_FORCE_SHADOW");
    if((atomic&&std::strcmp(atomic,"1")==0)||atomic_grid_force_enabled()) s->required|=2|(v.partial_core?1:0);
    try {
        const size_t points=size_t(v.shape.np)*v.shape.nsp;
        s->view.density=s->copy(v.density,size_t(v.shape.nnz)*v.h_spins,stream);
        s->view.energy_density=s->copy(v.energy_density,size_t(v.shape.nnz)*v.energy_spins,stream);
        s->view.potential=s->copy(v.potential,points*v.energy_spins,stream);
        s->view.auxiliary=s->copy(v.auxiliary,points,stream);
        s->view.deformation=s->copy(v.deformation,points,stream);
        if(s->integral)s->view.hartree_stress=s->copy(v.hartree_stress,9,stream);
        if(v.convergence)s->view.convergence=s->copy(v.convergence,1,stream);
        check(cudaStreamSynchronize(stream)); // post-SCF ownership boundary
    } catch(...) {cudaStreamSynchronize(stream);throw;}
    snapshot=std::move(s);
}
FinalGridStateView borrow_final_grid_state(uint64_t epoch,int consumer) {
    require(!in_scf()&&snapshot&&snapshot->integral==integral_force_enabled()&&(consumer==1||consumer==2||consumer==4)&&
            (!snapshot->integral||snapshot->hartree_added)&&
            !snapshot->pair_active&&(!snapshot->pair_required||snapshot->pair_consumed==(snapshot->pair_required&3))&&
            (snapshot->required&consumer)&&!(snapshot->borrowed&consumer)&&!snapshot->active&&
            (snapshot->consumed&(consumer-1))==(snapshot->required&(consumer-1))&&
            snapshot->view.epoch==epoch&&epoch==geometry_epoch(),
            "Final force requested stale or already consumed electronic state");
    snapshot->borrowed|=consumer;snapshot->active=consumer;return snapshot->view;
}
void complete_final_grid_force(uint64_t epoch,int consumer) {
    require(!in_scf()&&snapshot&&snapshot->active==consumer&&(snapshot->borrowed&consumer)&&
            !(snapshot->consumed&consumer)&&snapshot->view.epoch==epoch&&epoch==geometry_epoch(),
            "Final force completed outside its electronic lifetime");
    if(atomic_grid_force_enabled()) require(snapshot->accumulator.stream&&
        (consumer==4||(!snapshot->atomic_records.empty()&&snapshot->atomic_records.back().kind==consumer)),
        "Shared final force consumer did not record its computation");
    snapshot->consumed|=consumer;snapshot->active=0;
}
FinalForceAccumulatorView final_force_accumulator(uint64_t epoch,int atoms,int forces,int stress,
        const double* initial_force,const double* initial_stress) {
    require(atomic_grid_force_enabled()&&!in_scf()&&snapshot&&snapshot->integral==integral_force_enabled()&&
        (snapshot->active||(snapshot->integral&&snapshot->pair_active))&&!snapshot->publication_attempted&&
        snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&atoms>0&&atoms<=snapshot->view.shape.nuo&&
        (forces==0||forces==1)&&(stress==0||stress==1)&&initial_force&&initial_stress,"Invalid shared final force request");
    auto& s=*snapshot;
    if(!s.accumulator.stream) {
        require(!s.consumed&&(s.integral?(s.pair_active==1&&s.pair_consumed==0):s.active==(s.required&-s.required)),"Shared force accumulator started after its first consumer");
        require(std::all_of(initial_force,initial_force+3*atoms,[](double x){return std::isfinite(x);})&&
                std::all_of(initial_stress,initial_stress+9,[](double x){return std::isfinite(x);}),"Nonfinite initial force contribution");
        s.atoms=atoms;s.forces=forces;s.stress=stress;s.initial_force.assign(initial_force,initial_force+3*atoms);
        s.initial_stress.assign(initial_stress,initial_stress+9);
        check(cudaStreamCreateWithFlags(&s.accumulator.stream,cudaStreamNonBlocking));
        if(s.integral){
            std::vector<double> values(10,0.);values.insert(values.end(),s.initial_force.begin(),s.initial_force.end());values.insert(values.end(),s.initial_stress.begin(),s.initial_stress.end());
            s.accumulator.energy=s.stage(values.size(),values.data());s.accumulator.force=s.accumulator.energy+10;s.accumulator.stress=s.accumulator.force+3*atoms;
            check(cudaStreamSynchronize(s.accumulator.stream));
        }else{s.accumulator.force=s.stage(s.initial_force.size(),s.initial_force.data());s.accumulator.stress=s.stage(9,s.initial_stress.data());}
        int zero=0;s.accumulator.status=s.stage(1,&zero);check(cudaStreamSynchronize(s.accumulator.stream));
    } else require(s.atoms==atoms&&s.forces==forces&&s.stress==stress&&
        std::equal(s.initial_force.begin(),s.initial_force.end(),initial_force)&&
        std::equal(s.initial_stress.begin(),s.initial_stress.end(),initial_stress),"Host force contribution changed between device consumers");
    return s.accumulator;
}
void record_final_integral_force(uint64_t epoch,const IntegralForceRecord& r){
    require(integral_force_enabled()&&!in_scf()&&snapshot&&snapshot->integral&&snapshot->accumulator.energy&&
        snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&snapshot->pair_active==r.consumer&&r.queries>=0&&
        ((r.consumer==1||r.consumer==4)?!r.retained_integrals&&r.graph_nodes==(r.queries?2:1):
         (r.consumer==2||r.consumer==3)&&r.graph_nodes==(r.retained_integrals?2:(r.queries?3:2)))&&
        std::none_of(snapshot->integral_records.begin(),snapshot->integral_records.end(),[&](const auto& v){return v.consumer==r.consumer;}),
        "Invalid final integral-force computation record");
    snapshot->integral_records.push_back(r);
}
void add_final_hartree_stress(uint64_t epoch){
    require(integral_force_enabled()&&!in_scf()&&snapshot&&snapshot->integral&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        snapshot->pair_consumed==3&&!snapshot->pair_active&&!snapshot->consumed&&!snapshot->active&&!snapshot->hartree_added&&
        snapshot->accumulator.stream&&snapshot->view.hartree_stress,"Invalid retained Hartree-stress handoff");
    check(add_final_stress_device(snapshot->view.hartree_stress,snapshot->accumulator.stress,snapshot->accumulator.status,snapshot->accumulator.stream));
    snapshot->hartree_added=true;
}
const std::vector<IntegralForceRecord>& final_integral_force_records(uint64_t epoch){
    require(integral_force_enabled()&&!in_scf()&&snapshot&&snapshot->integral&&snapshot->view.epoch==epoch&&epoch==geometry_epoch(),"Invalid integral-force records request");
    return snapshot->integral_records;
}
FinalForcePublication collect_final_forces(uint64_t epoch,int atoms,const double* initial_force,const double* initial_stress){
    require(integral_force_enabled()&&!in_scf()&&snapshot&&snapshot->integral&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        !snapshot->active&&!snapshot->pair_active&&snapshot->consumed==snapshot->required&&snapshot->pair_consumed==7&&snapshot->hartree_added&&
        snapshot->integral_records.size()==3&&!snapshot->publication_attempted&&atoms==snapshot->atoms&&initial_force&&initial_stress&&
        std::equal(snapshot->initial_force.begin(),snapshot->initial_force.end(),initial_force)&&
        std::equal(snapshot->initial_stress.begin(),snapshot->initial_stress.end(),initial_stress),"Final force publication before completion or after host modification");
    auto& s=*snapshot;s.publication_attempted=true;FinalForcePublication result;
    evaluate_final_population(s.view,s.accumulator);
    check(cudaStreamSynchronize(s.accumulator.stream));
    check(cudaMemcpy(&result.status,s.accumulator.status,sizeof(int),cudaMemcpyDeviceToHost));
    result.exported_bytes=sizeof(int);transfer(Transfer::final_force_output,false,sizeof(int));
    require(result.status==0,"Final force device failure; publication rejected");
    std::vector<double> values(10+3*atoms+9);
    check(cudaMemcpy(values.data(),s.accumulator.energy,values.size()*8,cudaMemcpyDeviceToHost));
    result.exported_bytes+=values.size()*8;transfer(Transfer::final_force_output,false,values.size()*8);
    require(std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}),"Nonfinite final force publication");
    result.energy.assign(values.begin(),values.begin()+10);result.force.assign(values.begin()+10,values.begin()+10+3*atoms);
    result.stress.assign(values.begin()+10+3*atoms,values.end());s.collected=true;return result;
}
void complete_final_force_publication(uint64_t epoch){
    require(integral_force_enabled()&&!in_scf()&&snapshot&&snapshot->integral&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        snapshot->collected&&!snapshot->published&&!snapshot->active&&!snapshot->pair_active&&snapshot->pair_consumed==7&&
        snapshot->consumed==snapshot->required,"Invalid final force publication completion");snapshot->published=true;
    complete_population_publication(epoch);
    if(initial_mesh_enabled())complete_initial_mesh_forces(epoch);
}
void record_final_atomic_force(uint64_t epoch,const AtomicForceRecord& r) {
    require(atomic_grid_force_enabled()&&!in_scf()&&snapshot&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        snapshot->accumulator.stream&&(r.kind==1||r.kind==2)&&snapshot->active==r.kind&&r.entries>=0&&
        r.points==snapshot->view.shape.np*snapshot->view.shape.nsp&&
        r.diagonal_spins==(r.kind==1?std::min(2,snapshot->view.energy_spins):1)&&r.graph_nodes==(atomic_basis_shadow_enabled()?3:atomic_basis_device_enabled()?2:1)&&
        r.setup_bytes==(atomic_basis_device_enabled()?0:size_t(56)*r.entries)+32&&
        r.derivative_upload_bytes==r.setup_bytes-32&&r.host_derivative_storage_bytes==r.derivative_upload_bytes&&
        std::none_of(snapshot->atomic_records.begin(),snapshot->atomic_records.end(),[&](auto v){return v.kind==r.kind;}),
        "Invalid shared atomic-force record");
    snapshot->atomic_records.push_back(r);snapshot->accumulator.staged_bytes+=r.setup_bytes;
}
const std::vector<AtomicForceRecord>& final_atomic_force_records(uint64_t epoch) {
    require(atomic_grid_force_enabled()&&!in_scf()&&snapshot&&snapshot->view.epoch==epoch&&epoch==geometry_epoch()&&
        snapshot->active==4&&snapshot->atomic_records.size()==size_t(snapshot->view.partial_core?2:1),
        "Incomplete atomic terms before final force publication");
    return snapshot->atomic_records;
}
}
