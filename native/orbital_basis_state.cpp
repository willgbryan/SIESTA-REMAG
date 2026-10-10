// SPDX-License-Identifier: GPL-3.0-or-later
#include "orbital_basis_state.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
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
struct Record {
    uint64_t epoch=0;std::array<int,9> shape{};
    size_t staged=0,allocated=0,shadow_staged=0,shadow_exported=0,derivative_bytes=0;
    remag::OrbitalComparison comparison{};int status=-1;bool consumed=false,passed=false,shadow=false,generated=false;
};
struct Records {
    std::vector<Record> rows;
    ~Records(){
        if(rows.empty())return;
        FILE* f=std::fopen("remag-orbital-basis.json","w");if(!f)return;
        std::fputs("{\"scope\":\"geometry-owned device orbital derivatives; optional full shadow\",\"cycles\":[",f);
        for(size_t i=0;i<rows.size();i++) {
            const auto& r=rows[i];const auto& c=r.comparison;
            std::fprintf(f,"%s{\"epoch\":%llu,\"shape\":[",i?",":"",static_cast<unsigned long long>(r.epoch));
            for(int j=0;j<9;j++)std::fprintf(f,"%s%d",j?",":"",r.shape[j]);
            std::fprintf(f,"],\"compact_setup_bytes\":%zu,\"device_allocation_bytes\":%zu,\"derivative_bytes\":%zu,\"shadow_setup_bytes\":%zu,\"shadow_export_bytes\":%zu,\"status\":%d,\"consumed\":%s,\"passed\":%s,\"generated\":%s,\"comparison_performed\":%s,\"host_derivatives_authoritative\":%s,",
                r.staged,r.allocated,r.derivative_bytes,r.shadow_staged,r.shadow_exported,r.status,r.consumed?"true":"false",r.passed?"true":"false",
                r.generated?"true":"false",r.shadow&&r.consumed?"true":"false",r.shadow?"true":"false");
            if(r.shadow&&r.consumed)
                std::fprintf(f,"\"compared\":%llu,\"mismatches\":%llu,\"nonfinite\":%llu,\"max_absolute_error\":[%.17g,%.17g,%.17g],\"max_scaled_error\":[%.17g,%.17g,%.17g],\"atol\":1e-10,\"rtol\":1e-10,\"displacements_exact\":true}",
                    c.compared,c.mismatches,c.nonfinite,c.absolute[0],c.absolute[1],c.absolute[2],c.scaled[0],c.scaled[1],c.scaled[2]);
            else std::fputs("\"compared\":null,\"mismatches\":null,\"nonfinite\":null,\"max_absolute_error\":null,\"max_scaled_error\":null,\"atol\":null,\"rtol\":null,\"displacements_exact\":null}",f);
        }
        std::fputs("]}\n",f);std::fclose(f);
    }
};
Records& records(){static Records value;return value;}
struct Owner {
    uint64_t epoch=0;size_t record=0,staged=0,allocated=0;
    bool borrowed=false,complete=false,shadow=false,generated=false;
    cudaStream_t force_stream=nullptr;
    cudaStream_t staging_stream=nullptr;
    remag::OrbitalBasisDeviceView view{};
    std::vector<void*> memory;
    Owner(){check(cudaStreamCreateWithFlags(&staging_stream,cudaStreamNonBlocking));}
    ~Owner(){if(staging_stream)cudaStreamSynchronize(staging_stream);for(void* p:memory)cudaFree(p);if(staging_stream)cudaStreamDestroy(staging_stream);}
    template<class T>T* allocate(size_t n){T* p=nullptr;const size_t bytes=std::max(size_t(1),n)*sizeof(T);check(cudaMalloc(reinterpret_cast<void**>(&p),bytes));memory.push_back(p);allocated+=bytes;return p;}
    template<class T>T* upload(size_t n,const T* source){auto p=allocate<T>(n);if(n){check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,staging_stream));staged+=n*sizeof(T);remag::transfer(remag::Transfer::orbital_basis_setup,true,n*sizeof(T));}return p;}
};
std::unique_ptr<Owner> owner;
bool enabled(const char* name){const char* v=std::getenv(name);return v&&std::strcmp(v,"1")==0;}
void valid_mode(){require(!(remag::orbital_basis_enabled()&&remag::orbital_basis_shadow_enabled()),"Choose device or shadow orbital derivatives");require(enabled("REMAG_GRID_FORCE"),"Orbital derivatives require GPU grid forces");}
void valid_owner(uint64_t epoch){valid_mode();require(!remag::in_scf()&&owner&&owner->epoch==epoch&&epoch==remag::geometry_epoch()&&remag::orbital_basis_active()&&owner->shadow==remag::orbital_basis_shadow_enabled(),"Orbital basis phase/epoch/mode changed");}
}
bool remag::orbital_basis_shadow_enabled(){return enabled("REMAG_ORBITAL_BASIS_SHADOW");}
bool remag::orbital_basis_enabled(){return enabled("REMAG_ORBITAL_BASIS");}
bool remag::orbital_basis_active(){return orbital_basis_enabled()||orbital_basis_shadow_enabled();}
extern "C" int remag_orbital_basis_active(){return remag::orbital_basis_active();}
extern "C" int remag_orbital_basis_enabled(){return remag::orbital_basis_enabled();}
extern "C" int remag_orbital_basis_stage(const int* c,double pi,const int* species,const int* orbital,const double* radial,
        const double* values,const double* second,const int* entry_orbital,const int* entry_offset,const int* orbital_atom,
        const int* orbital_local,const int* atom_species,const double* offset,const double* subpoint,const double* atom_offset){
    try {
        valid_mode();
        require(remag::orbital_basis_active()&&!remag::in_scf()&&c,"Invalid orbital basis preparation phase");
        const auto epoch=remag::geometry_epoch();require(epoch>0&&(!owner||(owner->complete&&owner->epoch<epoch)),"Duplicate or unconsumed orbital basis epoch");
        require(c[0]>0&&c[0]<=10000&&c[1]>0&&c[1]<=1000000&&c[2]>=0&&c[3]>0&&c[4]>0&&c[4]<=4096&&
            c[5]>0&&c[6]>0&&c[7]>0&&c[8]>0&&size_t(c[3])*c[4]<=(size_t(4)<<30)/(7*sizeof(double)),"Invalid orbital basis dimensions");
        require(std::isfinite(pi)&&pi>3.14&&pi<3.15&&species&&orbital&&radial&&values&&second&&entry_orbital&&entry_offset&&
            orbital_atom&&orbital_local&&atom_species&&offset&&subpoint&&atom_offset,"Missing orbital basis inputs");
        // Validate all compact mappings on the host before their one-time upload.
        for(int i=0;i<c[0];i++)require(species[2*i]>=0&&species[2*i+1]>0&&species[2*i]<=c[1]-species[2*i+1],"Invalid species orbital range");
        for(int i=0;i<c[3];i++)require(entry_orbital[i]>0&&entry_orbital[i]<=c[5]&&entry_offset[i]>0&&entry_offset[i]<=c[7],"Invalid orbital mesh mapping");
        for(int i=0;i<c[6];i++)require(atom_species[i]>0&&atom_species[i]<=c[0],"Invalid atom species");
        for(int i=0;i<c[5];i++) {
            require(orbital_atom[i]>0&&orbital_atom[i]<=c[6],"Invalid orbital atom");
            const int is=atom_species[orbital_atom[i]-1]-1;
            require(orbital_local[i]>0&&orbital_local[i]<=species[2*is+1],"Invalid local orbital");
        }
        owner.reset();auto s=std::make_unique<Owner>();s->epoch=epoch;
        s->view.basis={c[0],c[1],c[2],pi,s->upload(size_t(2)*c[0],species),s->upload(size_t(4)*c[1],orbital),
            s->upload(size_t(2)*c[1],radial),s->upload(c[2],values),s->upload(c[2],second)};
        s->view.geometry={c[3],c[4],c[5],c[6],c[7],s->upload(c[3],entry_orbital),s->upload(c[3],entry_offset),s->upload(c[5],orbital_atom),
            s->upload(c[5],orbital_local),s->upload(c[6],atom_species),s->upload(size_t(3)*c[7],offset),s->upload(size_t(3)*c[4],subpoint),s->upload(size_t(3)*c[6],atom_offset)};
        check(cudaStreamSynchronize(s->staging_stream));
        s->shadow=remag::orbital_basis_shadow_enabled();
        Record r;r.shadow=s->shadow;r.epoch=epoch;std::copy(c,c+9,r.shape.begin());r.staged=s->staged;r.allocated=s->allocated;
        r.derivative_bytes=size_t(56)*c[3]*c[4];s->record=records().rows.size();records().rows.push_back(r);owner=std::move(s);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG orbital basis preparation: %s\n",e.what());return -260;}
}
extern "C" int remag_orbital_basis_prepared(int no,int np,int entries,int nsp){
    try {
        if(!owner||owner->epoch!=remag::geometry_epoch())return 0;
        valid_owner(remag::geometry_epoch());const auto& g=owner->view.geometry;
        require(!owner->borrowed&&!owner->complete&&g.orbitals==no&&g.entries==entries&&g.subpoints==nsp&&
            records().rows[owner->record].shape[8]==np,"Prepared orbital mesh shape changed");
        return 1;
    }catch(const std::exception& e){std::fprintf(stderr,"REMAG prepared orbital basis: %s\n",e.what());return -261;}
}
remag::OrbitalBasisDeviceView remag::borrow_orbital_mesh_basis(uint64_t epoch,int no,int np,int entries,int nsp){
    valid_owner(epoch);
    require(remag_orbital_basis_prepared(no,np,entries,nsp)==1,"Orbital mesh basis is not prepared");
    return owner->view;
}
remag::OrbitalBasisDeviceView remag::begin_orbital_basis(uint64_t epoch,int no,int np,int entries,int nsp,cudaStream_t stream){
    valid_owner(epoch);
    require(!owner->borrowed&&!owner->complete&&stream,"Invalid orbital basis borrow");
    auto& s=*owner;const auto& g=s.view.geometry;
    require(g.orbitals==no&&g.entries==entries&&g.subpoints==nsp&&records().rows[s.record].shape[8]==np,"Orbital basis geometry changed before force use");
    const size_t n=size_t(entries)*nsp;
    s.view.phi=s.allocate<double>(n);s.view.gradient=s.allocate<double>(3*n);s.view.displacement=s.allocate<double>(3*n);
    auto& r=records().rows[s.record];
    if(s.shadow) {
        s.view.comparison=s.allocate<OrbitalComparison>(1);OrbitalComparison zero{};
        check(cudaMemcpyAsync(s.view.comparison,&zero,sizeof(zero),cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
        transfer(Transfer::orbital_basis_shadow,true,sizeof(zero));r.shadow_staged=sizeof(zero);
    }
    r.allocated=s.allocated;s.borrowed=true;s.force_stream=stream;return s.view;
}
void remag::generate_orbital_basis(uint64_t epoch,int* status,cudaStream_t stream){
    valid_owner(epoch);auto& s=*owner;
    require(s.borrowed&&!s.generated&&!s.complete&&stream==s.force_stream&&status,"Invalid orbital basis generation");
    check(evaluate_orbital_geometry_device(s.view.basis,s.view.geometry,s.view.phi,s.view.gradient,s.view.displacement,status,stream));
    s.generated=true;records().rows[s.record].generated=true;
}
void remag::finish_orbital_basis(uint64_t epoch,int force_status){
    valid_owner(epoch);auto& s=*owner;
    require(s.borrowed&&s.generated&&!s.complete,"Invalid orbital basis finish");
    auto& r=records().rows[s.record];r.status=force_status;r.consumed=true;r.passed=force_status==0;
    if(s.shadow) {
        check(cudaMemcpy(&r.comparison,s.view.comparison,sizeof(r.comparison),cudaMemcpyDeviceToHost));
        transfer(Transfer::orbital_basis_shadow,false,sizeof(r.comparison));r.shadow_exported=sizeof(r.comparison);
        const auto& c=r.comparison;
        r.passed=r.passed&&c.compared==r.derivative_bytes/8&&c.mismatches==0&&c.nonfinite==0&&
            std::isfinite(c.absolute[0])&&std::isfinite(c.absolute[1])&&c.absolute[2]==0.&&
            c.scaled[0]<=1.&&c.scaled[1]<=1.&&c.scaled[2]==0.;
    }
    require(r.passed,"Device orbital derivatives or full shadow failed");s.complete=true;
}
