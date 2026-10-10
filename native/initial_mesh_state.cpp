// SPDX-License-Identifier: GPL-3.0-or-later
// Early geometry owner and full upstream shadow. Production consumers are not
// switched until this complete producer chain passes real-material comparison.
#include "orbital_basis_state.hpp"
#include "atomic_density.hpp"
#include "initial_harris.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Record {
    uint64_t epoch=0;std::array<int,8> shape{};
    size_t staged=0,allocated=0,workspace=0,exported=0,raw_bytes=0;
    size_t phi_bits=0,phi_numerical=0;int status=-1,hartree_status=-1;
    bool compared=false,passed=false,harris_exact=false;
    std::array<double,5> error{},scaled{};std::string file;
};
void number(FILE* f,double value){if(std::isfinite(value))std::fprintf(f,"%.17g",value);else std::fputs("null",f);}
struct Records {
    std::vector<Record> rows;
    ~Records(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-initial-mesh.json","w");if(!f)return;
        std::fputs("{\"schema\":1,\"scope\":\"early device mesh values, atomic density and initial Hartree full shadow\",\"production\":false,\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++){
            const auto& r=rows[i];std::fprintf(f,"%s{\"epoch\":%llu,\"shape\":[",i?",":"",static_cast<unsigned long long>(r.epoch));
            for(size_t j=0;j<r.shape.size();j++)std::fprintf(f,"%s%d",j?",":"",r.shape[j]);
            std::fprintf(f,"],\"setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"hartree_workspace_bytes\":%zu,\"shadow_export_bytes\":%zu,\"raw_bytes\":%zu,\"raw_file\":\"%s\",\"status\":%d,\"hartree_status\":%d,\"compared\":%s,\"passed\":%s,\"harris_bound_exactly\":%s,\"phi_bit_mismatches\":%zu,\"phi_numerical_mismatches\":%zu,\"max_absolute_error\":[",
                r.staged,r.allocated,r.workspace,r.exported,r.raw_bytes,r.file.c_str(),r.status,r.hartree_status,
                r.compared?"true":"false",r.passed?"true":"false",r.harris_exact?"true":"false",r.phi_bits,r.phi_numerical);
            for(int j=0;j<5;j++){if(j)std::fputs(",",f);number(f,r.error[j]);}
            std::fputs("],\"max_scaled_error\":[",f);
            for(int j=0;j<5;j++){if(j)std::fputs(",",f);number(f,r.scaled[j]);}
            std::fputs("],\"field_order\":[\"phi\",\"density\",\"potential\",\"energy\",\"stress\"]}",f);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
};
Records& records(){static Records value;return value;}
struct Owner {
    uint64_t epoch=0;size_t row=0;bool ready=false;
    cudaStream_t stream=nullptr;std::vector<void*> memory;
    float* phi=nullptr;double* density=nullptr;remag::HartreeResult* hartree=nullptr;
    Owner(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Owner(){if(stream)cudaStreamSynchronize(stream);for(auto p:memory)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* array(size_t count,const T* source=nullptr){
        T* p=nullptr;const size_t bytes=std::max(count,size_t(1))*sizeof(T);
        check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);records().rows[row].allocated+=bytes;
        if(count&&source){check(cudaMemcpyAsync(p,source,count*sizeof(T),cudaMemcpyHostToDevice,stream));
            records().rows[row].staged+=count*sizeof(T);remag::transfer(remag::Transfer::initial_mesh_setup,true,count*sizeof(T));}
        return p;
    }
    template<class T>void download(size_t count,T* target,const T* source){
        if(!count)return;
        check(cudaMemcpyAsync(target,source,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
        records().rows[row].exported+=count*sizeof(T);remag::transfer(remag::Transfer::initial_mesh_shadow,false,count*sizeof(T));
    }
};
std::unique_ptr<Owner> owner;
bool enabled(){const char* s=std::getenv("REMAG_INITIAL_MESH_SHADOW");return s&&std::strcmp(s,"1")==0;}
template<class T>void save(FILE* f,const T* data,size_t count){require(std::fwrite(data,sizeof(T),count,f)==count,"Cannot retain mesh shadow");}
template<class T>bool compare(size_t n,const T* actual,const T* expected,double atol,double rtol,double& error,double& scaled){
    bool passed=true;error=scaled=0.;
    for(size_t i=0;i<n;i++){
        const double a=actual[i],e=expected[i];
        if(!std::isfinite(a)||!std::isfinite(e)){passed=false;error=scaled=INFINITY;continue;}
        const double d=std::abs(a-e),s=d/(atol+rtol*std::abs(e));
        error=std::max(error,d);scaled=std::max(scaled,s);passed=passed&&s<=1.;
    }
    return passed;
}
}

extern "C" int remag_initial_mesh_shadow_enabled(){return enabled();}
// c = mesh[3], nsm, orbitals, unit orbitals, coarse points, entries.
extern "C" int remag_initial_mesh_shadow(const int* c,const double* reciprocal,double cutoff,double volume,
        const int* ends,const int* unit,const double* occupation,const float* reference_phi,
        const double* reference_density,const double* reference_potential,double reference_harris,const double* reference_stress){
    try {
        require(enabled()&&!remag::in_scf()&&remag::orbital_basis_enabled()&&!remag::orbital_basis_shadow_enabled(),
                "Initial mesh shadow requires production orbital basis outside SCF");
        require(c&&reciprocal&&ends&&unit&&occupation&&reference_phi&&reference_density&&reference_potential&&reference_stress,
                "Missing initial mesh inputs");
        require(c[3]>=1&&c[3]<=16&&c[4]>0&&c[5]>0&&c[5]<=c[4]&&c[6]>0&&c[7]>=0,"Invalid initial mesh dimensions");
        for(int i=0;i<3;i++)require(c[i]>0&&c[i]%c[3]==0,"Invalid clustered mesh");
        require(size_t(c[0])*c[1]<=size_t(INT_MAX)/c[2],"Initial mesh exceeds integer range");
        const int points=c[0]*c[1]*c[2],nsp=c[3]*c[3]*c[3];const size_t values=size_t(c[7])*nsp;
        require(points/nsp==c[6]&&values<=size_t(INT_MAX)&&4*values+16*size_t(points)<(size_t(4)<<30),"Initial mesh capacity mismatch");
        require(std::isfinite(cutoff)&&cutoff>0.&&std::isfinite(volume)&&volume>0.&&std::isfinite(reference_harris),"Invalid initial Hartree constants");
        for(int i=0;i<9;i++)require(std::isfinite(reciprocal[i]),"Invalid reciprocal cell");
        const auto epoch=remag::geometry_epoch();
        require(epoch>0&&(!owner||(owner->ready&&owner->epoch<epoch)),"Duplicate/unconsumed initial mesh epoch");
        const auto basis=remag::borrow_orbital_mesh_basis(epoch,c[4],c[6],c[7],nsp);
        owner.reset();auto s=std::make_unique<Owner>();s->epoch=epoch;s->row=records().rows.size();
        Record record;record.epoch=epoch;std::copy(c,c+8,record.shape.begin());records().rows.push_back(record);
        auto& r=records().rows.back();
        remag::HartreeProblem problem{};std::copy(c,c+3,problem.mesh);std::copy(reciprocal,reciprocal+9,problem.reciprocal);
        problem.cutoff=cutoff;problem.volume=volume;
        remag::GridPotentialProblem seed{};seed.hartree=problem;seed.nsm=c[3];seed.dvol=volume/points;
        std::copy(c,c+3,seed.pbe.mesh);seed.pbe.volume=volume;seed.pbe.spins=1;
        auto p=s->array(1,&problem);auto grid=s->array(1,&seed);int zero=0;auto status=s->array(1,&zero);
        s->phi=s->array<float>(values);s->density=s->array<double>(points);s->hartree=s->array<remag::HartreeResult>(1);
        auto potential=s->array<double>(points);
        remag::AtomicDensityView density{{c[0],c[1],c[2]},c[3],c[6],nsp,c[7],c[4],c[5],
            s->array(size_t(c[6])+1,ends),basis.geometry.entry_orbital,s->array(c[4],unit),s->array(c[5],occupation),s->phi};
        remag::HartreePlan plan(c[0],c[1],c[2],s->stream);r.workspace=plan.resident_bytes();
        check(remag::evaluate_orbital_mesh_device(basis.basis,basis.geometry,s->phi,status,s->stream));
        check(remag::initial_atomic_density_device(density,s->density,status,s->stream));
        require(plan.apply(p,s->density,potential,s->hartree)==0,"Initial Hartree launch failed");
        check(remag::bind_initial_harris_device(s->hartree,grid,status,s->stream));
        std::vector<float> actual_phi(values);std::vector<double> actual_density(points),actual_potential(points);
        remag::HartreeResult h{};double harris=NAN;
        s->download(values,actual_phi.data(),s->phi);s->download(points,actual_density.data(),s->density);
        s->download(points,actual_potential.data(),potential);s->download(1,&h,s->hartree);
        s->download(1,&harris,&grid->harris_energy);s->download(1,&r.status,status);check(cudaStreamSynchronize(s->stream));
        r.hartree_status=h.status;r.harris_exact=std::isfinite(harris)&&harris==h.energy;
        bool passed=r.status==0&&h.status==0&&r.harris_exact;
        for(size_t i=0;i<values;i++){
            uint32_t a,e;std::memcpy(&a,&actual_phi[i],4);std::memcpy(&e,&reference_phi[i],4);
            r.phi_bits+=a!=e;r.phi_numerical+=actual_phi[i]!=reference_phi[i];
        }
        passed=compare(values,actual_phi.data(),reference_phi,1.,0.,r.error[0],r.scaled[0])&&passed&&r.phi_bits==0;
        passed=compare(points,actual_density.data(),reference_density,1.e-12,1.e-12,r.error[1],r.scaled[1])&&passed;
        passed=compare(points,actual_potential.data(),reference_potential,1.e-10,1.e-11,r.error[2],r.scaled[2])&&passed;
        passed=compare(1,&h.energy,&reference_harris,1.e-10,1.e-11,r.error[3],r.scaled[3])&&passed;
        passed=compare(9,h.stress,reference_stress,1.e-12,1.e-11,r.error[4],r.scaled[4])&&passed;
        r.compared=true;r.passed=passed;
        r.file="remag-initial-mesh-"+std::to_string(epoch)+".bin";
        std::unique_ptr<FILE,decltype(&std::fclose)> raw(std::fopen(r.file.c_str(),"wb"),&std::fclose);
        require(bool(raw),"Cannot create mesh shadow evidence");
        save(raw.get(),"RMGMSH01",8);save(raw.get(),c,8);save(raw.get(),&r.status,1);save(raw.get(),&h.status,1);
        save(raw.get(),actual_phi.data(),values);save(raw.get(),reference_phi,values);
        save(raw.get(),actual_density.data(),points);save(raw.get(),reference_density,points);
        save(raw.get(),actual_potential.data(),points);save(raw.get(),reference_potential,points);
        save(raw.get(),&h.energy,1);save(raw.get(),&reference_harris,1);save(raw.get(),&harris,1);
        save(raw.get(),h.stress,9);save(raw.get(),reference_stress,9);
        r.raw_bytes=48+8*values+32*size_t(points)+21*sizeof(double);
        require(std::fflush(raw.get())==0&&std::ftell(raw.get())==long(r.raw_bytes),"Incomplete mesh shadow evidence");
        require(passed,"Initial mesh full shadow failed");s->ready=true;owner=std::move(s);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG initial mesh: %s\n",e.what());return -290;}
}
