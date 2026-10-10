// SPDX-License-Identifier: GPL-3.0-or-later
// Explicit SCF grid ownership; only a small result record returns to CPU control.
#include "grid_potential.hpp"
#include "execution.hpp"
#include "resident.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" int remag_gpu_enabled();
extern "C" int remag_grid_enabled();
extern "C" int remag_flow_active();
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
bool verify() {const char* v=std::getenv("REMAG_GRID_VERIFY_EPOCH");return v&&std::strcmp(v,"1")==0;}
struct Counts {
    unsigned long long setups=0,arms=0,solves=0,ends=0,h2d=0,d2h=0,checked=0,allocations=0;
    ~Counts() {
        if(FILE* f=std::fopen("remag-grid-flow.json","w")) {
            std::fprintf(f,"{\"geometry_setups\":%llu,\"arms\":%llu,\"solves\":%llu,\"ends\":%llu,"
                "\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,\"debug_host_validation_bytes\":%llu,"
                "\"bridge_allocations\":%llu,\"scope\":\"resident SCF density-to-potential arrays; temporary 224-byte result export; plan allocations excluded\"}\n",
                setups,arms,solves,ends,h2d,d2h,checked,allocations);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr;size_t bytes=0;
    void reserve(size_t n) {if(n<=bytes)return;if(p){check(cudaFree(p));p=nullptr;bytes=0;}check(cudaMalloc(&p,n));bytes=n;counts.allocations++;}
    void upload(const void* source,size_t n) {
        reserve(n);check(cudaMemcpy(p,source,n,cudaMemcpyHostToDevice));counts.h2d+=n;
        remag::transfer(remag::Transfer::grid_flow_geometry,true,n);
    }
    ~Buffer() {if(p)cudaFree(p);}
};
enum class Stage {idle,armed,density,potential,consumed};
struct Snapshot {
    const void* source=nullptr;std::vector<unsigned char> bytes;
    void capture(const void* p,size_t n) {
        source=p;bytes.resize(n);if(n)std::memcpy(bytes.data(),p,n);
    }
    bool matches() {
        if(!verify()||bytes.empty()) return true;
        counts.checked+=bytes.size();return std::memcmp(source,bytes.data(),bytes.size())==0;
    }
};
struct State {
    bool valid=false;uint64_t epoch=0;int size=0,spins=0,nsm=0,npcc=0;
    std::array<int,3> mesh{};
    std::array<const void*,4> sources{};
    std::array<Snapshot,4> constants;
    Snapshot rho_host,potential_host;
    const double *rho_identity=nullptr,*potential_identity=nullptr;
    Stage stage=Stage::idle;
    Buffer problem,atom,core,neutral,rho,potential,result;
    std::unique_ptr<remag::GridPotentialPlan> plan;
} state;
bool shape_valid(const int* mesh,int spins,int nsm,int npcc) {
    if(!mesh||mesh[0]<1||mesh[1]<1||mesh[2]<1||(spins!=1&&spins!=2&&spins!=4)||nsm<1||
       (npcc!=0&&npcc!=1)||static_cast<long long>(mesh[0])*mesh[1]>INT_MAX/mesh[2]||
       mesh[0]*mesh[1]*mesh[2]>INT_MAX/spins) return false;
    for(int i=0;i<3;i++) if(mesh[i]%nsm) return false;
    return true;
}
bool same_constants() {for(auto& s:state.constants) if(!s.matches()) return false;return true;}
void owned(Stage stage,const double* host,size_t bytes,bool density) {
    if(!state.valid||state.epoch!=remag::geometry_epoch()||state.stage!=stage||
       bytes!=size_t(state.size)*state.spins*sizeof(double)||
       host!=(density?state.rho_identity:state.potential_identity)||!same_constants()||
       !state.rho_host.matches()||!state.potential_host.matches())
        throw std::runtime_error("Resident grid consumer does not match its declared lifetime");
}
}
extern "C" int remag_flow_enabled() {
    const char* v=std::getenv("REMAG_RESIDENT_GRID");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_flow_active() {return remag_flow_enabled()&&remag::in_scf();}
extern "C" int remag_flow_needs_setup(const int* mesh,int spins,int nsm,int npcc,
                                       const double* cell,const double* atom,const double* core,const double* neutral) {
    if(!remag_flow_active()||!remag_gpu_enabled()||!remag_grid_enabled()||state.stage!=Stage::idle||
       !shape_valid(mesh,spins,nsm,npcc)||!cell||!atom||!core||!neutral||!remag::geometry_epoch()) return -190;
    const std::array<const void*,4> sources={cell,atom,core,neutral};
    if(!state.valid||state.epoch!=remag::geometry_epoch()||state.spins!=spins||state.nsm!=nsm||state.npcc!=npcc||
       state.sources!=sources||!std::equal(state.mesh.begin(),state.mesh.end(),mesh)) return 1;
    return same_constants()?0:-191;
}
extern "C" int remag_flow_prepare(const int* mesh,int spins,int nsm,int npcc,const double* cell,
        const double* atom,const double* core,const double* neutral,const double* reciprocal,double cutoff,
        const double* gradient,double volume,double dvol,double harris) {
    try {
        if(remag_flow_needs_setup(mesh,spins,nsm,npcc,cell,atom,core,neutral)!=1||!reciprocal||!gradient||
           !std::isfinite(cutoff)||cutoff<=0.||!std::isfinite(volume)||volume<=0.||
           !std::isfinite(dvol)||dvol<=0.||!std::isfinite(harris)) return -192;
        for(int i=0;i<9;i++) if(!std::isfinite(cell[i])||!std::isfinite(reciprocal[i])||!std::isfinite(gradient[i])) return -192;
        state.valid=false;const int size=mesh[0]*mesh[1]*mesh[2];const size_t bytes=size_t(size)*sizeof(double);
        remag::GridPotentialProblem p{};
        std::copy(mesh,mesh+3,p.hartree.mesh);std::copy(mesh,mesh+3,p.pbe.mesh);
        std::copy(reciprocal,reciprocal+9,p.hartree.reciprocal);std::copy(gradient,gradient+9,p.pbe.gradient);
        p.hartree.cutoff=cutoff;p.hartree.volume=volume;p.pbe.volume=volume;p.pbe.spins=spins;
        p.harris_energy=harris;p.dvol=dvol;p.nsm=nsm;p.partial_core=npcc;
        if(!state.plan||state.spins!=spins||!std::equal(state.mesh.begin(),state.mesh.end(),mesh))
            state.plan=std::make_unique<remag::GridPotentialPlan>(mesh[0],mesh[1],mesh[2],spins,nullptr);
        state.problem.upload(&p,sizeof(p));state.atom.upload(atom,bytes);state.neutral.upload(neutral,bytes);
        // The original no-core pointer has only one valid element. Keep a
        // valid device pointer without reading a nonexistent host core grid.
        if(npcc) state.core.upload(core,bytes);else state.core.reserve(sizeof(double));
        state.rho.reserve(bytes*spins);state.potential.reserve(bytes*spins);state.result.reserve(sizeof(remag::GridPotentialResult));
        state.constants[0].capture(cell,9*sizeof(double));state.constants[1].capture(atom,bytes);
        state.constants[2].capture(core,npcc?bytes:0);state.constants[3].capture(neutral,bytes);
        state.sources={cell,atom,core,neutral};std::copy(mesh,mesh+3,state.mesh.begin());
        state.size=size;state.spins=spins;state.nsm=nsm;state.npcc=npcc;
        state.epoch=remag::geometry_epoch();state.valid=true;counts.setups++;return 0;
    } catch(const std::exception& e) {state.valid=false;std::fprintf(stderr,"REMAG grid-flow setup: %s\n",e.what());return -193;}
}
extern "C" int remag_flow_arm(int size,int spins,const double* rho,const double* potential) {
    if(!remag_flow_active()||!state.valid||state.epoch!=remag::geometry_epoch()||state.stage!=Stage::idle||
       size!=state.size||spins!=state.spins||!rho||!potential||rho==potential||!same_constants()) return -194;
    state.rho_identity=rho;state.potential_identity=potential;
    // Host grids are opaque while armed. Byte snapshots are debug-only.
    if(verify()) {state.rho_host.capture(rho,size_t(size)*spins*sizeof(double));state.potential_host.capture(potential,size_t(size)*spins*sizeof(double));}
    state.stage=Stage::armed;counts.arms++;return 0;
}
double* remag_flow_density_target(double* host,size_t bytes) {
    if(!remag_flow_active()) return nullptr;
    owned(Stage::armed,host,bytes,true);return static_cast<double*>(state.rho.p);
}
void remag_flow_density_ready(double* host,size_t bytes) {
    owned(Stage::armed,host,bytes,true);state.stage=Stage::density;
}
extern "C" int remag_flow_solve(const double* rho,const double* potential,double* energies,double* hs,double* xs) {
    try {
        if(!remag_flow_active()||!energies||!hs||!xs||potential!=state.potential_identity) return -195;
        owned(Stage::density,rho,size_t(state.size)*state.spins*sizeof(double),true);
        int status=state.plan->apply(static_cast<const remag::GridPotentialProblem*>(state.problem.p),
            static_cast<const double*>(state.rho.p),static_cast<const double*>(state.atom.p),static_cast<const double*>(state.core.p),
            static_cast<const double*>(state.neutral.p),static_cast<double*>(state.potential.p),
            static_cast<remag::GridPotentialResult*>(state.result.p));
        if(status) throw std::runtime_error("Grid-potential dispatch failed");
        remag::GridPotentialResult r{};
        check(cudaMemcpy(&r,state.result.p,sizeof(r),cudaMemcpyDeviceToHost));counts.d2h+=sizeof(r);
        remag::transfer(remag::Transfer::grid_flow_result,false,sizeof(r));
        if(r.status) {state.stage=Stage::idle;return -196;}
        std::copy(r.energies,r.energies+9,energies);std::copy(r.hartree_stress,r.hartree_stress+9,hs);std::copy(r.xc_stress,r.xc_stress+9,xs);
        state.stage=Stage::potential;counts.solves++;return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"REMAG grid-flow solve: %s\n",e.what());return -197;}
}
const double* remag_flow_potential(const double* host,size_t bytes) {
    if(!remag_flow_active()) return nullptr;
    owned(Stage::potential,host,bytes,false);
    state.stage=Stage::consumed;return static_cast<const double*>(state.potential.p);
}
const double* remag_flow_dvol() {
    if(!remag_flow_active()) return nullptr;
    if(state.stage!=Stage::consumed) throw std::runtime_error("Missing resident potential consumer");
    return &static_cast<const remag::GridPotentialProblem*>(state.problem.p)->dvol;
}
const remag::GridPotentialResult* remag_flow_result() {
    if(!remag_flow_active()||!state.valid||state.epoch!=remag::geometry_epoch()||state.stage!=Stage::idle||!counts.solves)
        throw std::runtime_error("Grid energy result has no completed owner");
    return static_cast<const remag::GridPotentialResult*>(state.result.p);
}
extern "C" int remag_flow_end() {
    if(!remag_flow_active()) return -198;
    try {owned(Stage::consumed,state.potential_identity,size_t(state.size)*state.spins*sizeof(double),false);}
    catch(const std::exception&) {return -198;}
    state.stage=Stage::idle;counts.ends++;return 0;
}
