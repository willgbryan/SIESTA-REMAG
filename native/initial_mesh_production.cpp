// SPDX-License-Identifier: GPL-3.0-or-later
// Device producer ownership. No mesh/density/energy readback is needed: failed
// producer status leaves a NaN Harris energy and is rejected by the SCF grid.
#include "initial_mesh_state.hpp"
#include "orbital_basis_state.hpp"
#include "atomic_density.hpp"
#include "initial_harris.hpp"
#include "final_grid_state.hpp"
#include "execution.hpp"
#include "spin_rotation.hpp"
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

namespace {
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
bool option(const char* name){const char* v=std::getenv(name);return v&&std::strcmp(v,"1")==0;}
uint64_t topology(int np,int entries,const int* ends,const int* orbitals){
    require(ends&&orbitals&&ends[0]==0&&ends[np]==entries,"Missing mesh topology");
    uint64_t h=14695981039346656037ULL;
    auto hash=[&](const int* data,size_t n){
        auto p=reinterpret_cast<const unsigned char*>(data);
        for(size_t i=0;i<n*sizeof(int);i++){h^=p[i];h*=1099511628211ULL;}
    };
    hash(ends,size_t(np)+1);hash(orbitals,entries);return h;
}
struct Record {
    uint64_t epoch=0;std::array<int,8> shape{};
    size_t staged=0,allocated=0,workspace=0;
    bool launched=false,phi_bound=false,density_borrowed=false,density_released=false,forces_complete=false,passed=false;
    int cycle_status=-1,producer_status=-1,converged=0;
    bool poison_requested=false,host_poisoned=false,reused=false;
};
struct Records {
    std::vector<Record> rows;
    ~Records(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-initial-mesh-production.json","w");if(!f)return;
        std::fputs("{\"schema\":1,\"production\":true,\"scope\":\"GPU mesh values, atomic reference density and initial Hartree energy; producer errors flow into existing SCF grid validation\",\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++){
            const auto& r=rows[i];std::fprintf(f,"%s{\"epoch\":%llu,\"shape\":[",i?",":"",static_cast<unsigned long long>(r.epoch));
            for(size_t j=0;j<r.shape.size();j++)std::fprintf(f,"%s%d",j?",":"",r.shape[j]);
            std::fprintf(f,"],\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"hartree_workspace_bytes\":%zu,\"array_upload_bytes\":0,\"export_bytes\":0,\"poison_requested\":%s,\"host_poisoned\":%s,\"reused_geometry\":%s,\"launched\":%s,\"phi_bound\":%s,\"density_borrowed\":%s,\"density_released\":%s,\"forces_complete\":%s,\"passed\":%s,\"cycle_status\":%d,\"cycle_producer_status\":%d,\"converged\":%s}",
                r.staged,r.allocated,r.workspace,r.poison_requested?"true":"false",r.host_poisoned?"true":"false",r.reused?"true":"false",r.launched?"true":"false",r.phi_bound?"true":"false",
                r.density_borrowed?"true":"false",r.density_released?"true":"false",r.forces_complete?"true":"false",r.passed?"true":"false",
                r.cycle_status,r.producer_status,r.converged?"true":"false");
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
};
Records& records(){static Records r;return r;}
struct Owner {
    uint64_t epoch=0,topology_hash=0;size_t row=0;
    cudaStream_t stream=nullptr;
    std::vector<void*> memory;std::unique_ptr<remag::HartreePlan> plan;
    remag::HartreeProblem problem{};
    float* phi=nullptr;double* density=nullptr;int* status=nullptr;remag::HartreeResult* hartree=nullptr;
    Owner(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Owner(){if(stream)cudaStreamSynchronize(stream);plan.reset();for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* array(size_t n,const T* input=nullptr){
        T* p=nullptr;const size_t bytes=std::max(n,size_t(1))*sizeof(T);
        check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);records().rows.at(row).allocated+=bytes;
        if(n&&input){check(cudaMemcpyAsync(p,input,n*sizeof(T),cudaMemcpyHostToDevice,stream));
            records().rows.at(row).staged+=n*sizeof(T);remag::transfer(remag::Transfer::initial_mesh_setup,true,n*sizeof(T));}
        return p;
    }
};
std::unique_ptr<Owner> owner;
Record& current(uint64_t epoch){
    require(owner&&epoch==owner->epoch&&epoch==remag::geometry_epoch()&&!remag::in_scf(),"Invalid initial mesh owner epoch");
    auto& r=records().rows.at(owner->row);require(r.launched,"Initial mesh producer was not launched");return r;
}
}

namespace remag {
bool initial_mesh_enabled(){return option("REMAG_INITIAL_MESH");}
const float* initial_mesh_values(uint64_t epoch,int no,int np,int nsp,int entries,const int* ends,const int* orbitals){
    auto& r=current(epoch);const auto& c=r.shape;
    require((!r.poison_requested||r.host_poisoned)&&initial_mesh_enabled()&&no==c[4]&&np==c[6]&&nsp==c[3]*c[3]*c[3]&&entries==c[7]&&
        topology(np,entries,ends,orbitals)==owner->topology_hash&&!r.density_released,"Grid mesh topology differs from producer");
    // Preparation synchronized the producer without exporting any value.
    r.phi_bound=true;return owner->phi;
}
const double* borrow_initial_mesh_density(uint64_t epoch,const GridPotentialProblem& p,GridPotentialProblem* device,cudaStream_t stream){
    auto& r=current(epoch);const auto& h=owner->problem;
    require((!r.poison_requested||r.host_poisoned)&&initial_mesh_enabled()&&!r.density_borrowed&&device&&stream&&p.nsm==r.shape[3]&&
        std::equal(p.hartree.mesh,p.hartree.mesh+3,h.mesh)&&std::equal(p.hartree.reciprocal,p.hartree.reciprocal+9,h.reciprocal)&&
        p.hartree.cutoff==h.cutoff&&p.hartree.volume==h.volume,"Initial density/Hartree geometry differs from SCF");
    check(bind_initial_harris_device(owner->hartree,device,owner->status,stream));
    r.density_borrowed=true;return owner->density;
}
void complete_initial_mesh_forces(uint64_t epoch){
    auto& r=current(epoch);require(r.passed&&r.density_released&&!r.forces_complete&&final_grid_forces_complete(epoch),
        "Initial mesh final forces were not completed");r.forces_complete=true;
}
void release_initial_mesh_density(uint64_t epoch,int status,int producer,int converged){
    auto& r=current(epoch);
    require(r.phi_bound&&r.density_borrowed&&!r.density_released&&!status&&!producer&&converged,
        "Cannot accept initial mesh without successful SCF and consumers");
    r.density_released=true;r.cycle_status=status;r.producer_status=producer;r.converged=converged;r.passed=true;
}
}
extern "C" int remag_initial_mesh_enabled(){return remag::initial_mesh_enabled();}
extern "C" int remag_initial_mesh_prepare(const int* c,const double* reciprocal,double cutoff,double volume,
        const int* ends,const int* orbitals,const int* unit,const double* occupation){
    try {
        require(remag::initial_mesh_enabled()&&!option("REMAG_INITIAL_MESH_SHADOW")&&!remag::in_scf()&&
            option("REMAG_DEVICE_SCF")&&option("REMAG_GPU_INITIALIZATION")&&remag::orbital_basis_enabled()&&
            !remag::orbital_basis_shadow_enabled()&&remag::final_grid_state_enabled()&&remag::integral_force_enabled(),
            "Initial mesh production requires device SCF, initialization and retained force owners");
        require(c&&reciprocal&&ends&&orbitals&&unit&&occupation,"Missing initial mesh inputs");
        require(c[3]>=1&&c[3]<=16&&c[4]>0&&c[5]>0&&c[5]<=c[4]&&c[6]>0&&c[7]>=0,"Invalid initial mesh dimensions");
        for(int i=0;i<3;i++)require(c[i]>0&&c[i]%c[3]==0,"Invalid clustered mesh");
        require(size_t(c[0])*c[1]<=size_t(INT_MAX)/c[2],"Initial mesh exceeds integer range");
        const int points=c[0]*c[1]*c[2],nsp=c[3]*c[3]*c[3];const size_t values=size_t(c[7])*nsp;
        require(points/nsp==c[6]&&values<=size_t(INT_MAX)&&4*values+16*size_t(points)<(size_t(4)<<30),"Initial mesh capacity mismatch");
        require(std::isfinite(cutoff)&&cutoff>0.&&std::isfinite(volume)&&volume>0.,"Invalid initial Hartree constants");
        for(int i=0;i<9;i++)require(std::isfinite(reciprocal[i]),"Invalid reciprocal cell");
        const auto epoch=remag::geometry_epoch();require(epoch>0,"Absent initial mesh geometry epoch");
        if(owner){const auto& previous=records().rows.at(owner->row);
            require(owner->epoch<epoch&&previous.passed&&previous.density_released&&previous.forces_complete,
                "Cannot discard an unconsumed initial mesh geometry");}
        const auto basis=remag::borrow_orbital_mesh_basis(epoch,c[4],c[6],c[7],nsp);
        owner.reset();auto s=std::make_unique<Owner>();s->epoch=epoch;s->row=records().rows.size();
        Record row;row.epoch=epoch;row.poison_requested=option("REMAG_INITIAL_MESH_POISON_HOST");std::copy(c,c+8,row.shape.begin());records().rows.push_back(row);
        s->topology_hash=topology(c[6],c[7],ends,orbitals);
        std::copy(c,c+3,s->problem.mesh);std::copy(reciprocal,reciprocal+9,s->problem.reciprocal);
        s->problem.cutoff=cutoff;s->problem.volume=volume;
        auto p=s->array(1,&s->problem);int zero=0;s->status=s->array(1,&zero);
        s->phi=s->array<float>(values);s->density=s->array<double>(points);s->hartree=s->array<remag::HartreeResult>(1);
        auto potential=s->array<double>(points);
        remag::AtomicDensityView density{{c[0],c[1],c[2]},c[3],c[6],nsp,c[7],c[4],c[5],
            s->array(size_t(c[6])+1,ends),basis.geometry.entry_orbital,s->array(c[4],unit),s->array(c[5],occupation),s->phi};
        s->plan=std::make_unique<remag::HartreePlan>(c[0],c[1],c[2],s->stream);
        records().rows.at(s->row).workspace=s->plan->resident_bytes();
        check(remag::evaluate_orbital_mesh_device(basis.basis,basis.geometry,s->phi,s->status,s->stream));
        check(remag::initial_atomic_density_device(density,s->density,s->status,s->stream));
        require(s->plan->apply(p,s->density,potential,s->hartree)==0,"Initial Hartree launch failed");
        // Host staging buffers belong to the caller. Complete their reads
        // before returning, without reading producer values/status on host.
        check(cudaStreamSynchronize(s->stream));
        records().rows.at(s->row).launched=true;owner=std::move(s);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG initial mesh production: %s\n",e.what());return -291;}
}

extern "C" int remag_initial_mesh_poison_enabled(){return option("REMAG_INITIAL_MESH_POISON_HOST");}
extern "C" int remag_initial_mesh_host_poison(const float* phi,const double* density,double harris){
    try {
        auto& r=current(remag::geometry_epoch());const auto& c=r.shape;
        require(r.poison_requested&&!r.host_poisoned&&!r.phi_bound&&!r.density_borrowed&&phi&&density&&std::isnan(harris),
                "Invalid initial mesh host poison control");
        const size_t values=size_t(c[7])*c[3]*c[3]*c[3],points=size_t(c[0])*c[1]*c[2];
        require(std::all_of(phi,phi+values,[](float x){return std::isnan(x);})&&
                std::all_of(density,density+points,[](double x){return std::isnan(x);}),"Incomplete initial mesh host poison");
        r.host_poisoned=true;return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG initial mesh poison: %s\n",e.what());return -292;}
}

// Called only after the Fortran sweep verifies the crystal cell and positions.
// These unpolarized reference fields and orbital values do not depend on the
// electronic spin guess; preserve them across independently converged directions.
extern "C" int remag_reuse_initial_mesh() {
    if(!remag::initial_mesh_enabled())return 0;
    try {
        require(remag_orientation_retain_geometry()&&!remag::in_scf()&&owner,
            "Initial mesh reuse requires a retained magnetic geometry");
        const auto& old=records().rows.at(owner->row);
        require(old.passed&&old.density_released&&old.forces_complete&&
            remag::geometry_epoch()==old.epoch+1,"Initial mesh reuse precedes completed consumers");
        Record next;next.epoch=remag::geometry_epoch();next.shape=old.shape;
        next.launched=true;next.reused=true;next.poison_requested=old.poison_requested;next.host_poisoned=old.host_poisoned;
        owner->epoch=next.epoch;owner->row=records().rows.size();records().rows.push_back(next);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG initial mesh reuse: %s\n",e.what());return -293;}
}
