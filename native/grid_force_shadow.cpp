// SPDX-License-Identifier: GPL-3.0-or-later
// Post-SCF grid-force adapter and optional upstream shadow diagnostic.
// Optional device generation removes orbital derivative staging. Compact
// geometry and other-force contributions still stage from the host.
#include "grid_force.hpp"
#include "orbital_basis_state.hpp"
#include "atomic_basis_state.hpp"
#include "execution.hpp"
#include "final_grid_state.hpp"
#include "integral_force_state.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
struct Shadow {
    remag::GridContractionView grid{};
    remag::GridForceProblem problem{};
    int atoms=0,entries=0,nh=0;
    uint64_t epoch=0;
    bool resident=false,authoritative=false,basis_authoritative=false,basis_shadow=false;
    remag::FinalGridStateView final{};
    std::vector<double> density,occupation,potential,auxiliary;
    std::vector<int> orbital_atoms,seen;
    std::vector<double> phi,gradient,displacement,initial_force,initial_stress;
};
std::unique_ptr<Shadow> shadow;
struct DeferredGrid {
    uint64_t epoch=0;int atoms=0,entries=0,spins=0;
    remag::GridContractionShape shape{};int partial_core=0;
    size_t staged=0,nodes=0,retained=0;
    std::vector<remag::AtomicForceRecord> atomic;
};
std::unique_ptr<DeferredGrid> deferred;
struct Device {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;size_t staged=0,exported=0;bool owns_stream;
    explicit Device(cudaStream_t borrowed=nullptr):stream(borrowed),owns_stream(!borrowed) {
        if(owns_stream) check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    ~Device() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        for(void* p:memory) cudaFree(p);
        if(stream&&owns_stream) cudaStreamDestroy(stream);
    }
    template<class T> T* upload(size_t n,const T* source) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));memory.push_back(p);
        check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));staged+=n*sizeof(T);
        remag::transfer(remag::Transfer::grid_flow_geometry,true,n*sizeof(T));return p;
    }
    void download(double* host,const double* device,size_t n) {
        check(cudaMemcpy(host,device,n*sizeof(double),cudaMemcpyDeviceToHost));exported+=n*sizeof(double);
        remag::transfer(remag::Transfer::grid_output,false,n*sizeof(double));
    }
};
int failed(const std::exception& e) {
    std::fprintf(stderr,"REMAG grid-force shadow: %s\n",e.what());shadow.reset();return -230;
}
}
extern "C" int remag_force_shadow_enabled() {
    const char* value=std::getenv("REMAG_FORCE_SHADOW");return value&&std::strcmp(value,"1")==0;
}
extern "C" int remag_grid_force_enabled() {
    const char* value=std::getenv("REMAG_GRID_FORCE");return value&&std::strcmp(value,"1")==0;
}
extern "C" int remag_force_active() {return remag_force_shadow_enabled()||remag_grid_force_enabled();}
extern "C" int remag_force_shadow_begin(int no,int nuo,int np,int nsp,int capacity,int nh,int atoms,int entries,
        const int* orbital_atoms,const double* density,const double* occupation,const double* potential,const double* auxiliary,
        double dvol,double volume,int forces,int stress,const double* initial_force,const double* initial_stress) {
    try {
        require(remag_force_active()&&!remag::in_scf()&&!shadow&&!deferred,"Invalid grid-force lifetime");
        require(!(remag_force_shadow_enabled()&&remag_grid_force_enabled()),"Choose authoritative or shadow grid forces");
        require(no>0&&nuo>0&&np>0&&nsp>0&&capacity>0&&entries>0&&atoms>0&&atoms<=no&&
                (nh==1||nh==2||nh==4||nh==8)&&orbital_atoms&&density&&occupation&&potential&&
                auxiliary&&initial_force&&initial_stress,"Missing force-shadow inputs");
        require(size_t(entries)*nsp<=(size_t(4)<<30)/(7*sizeof(double)),"Force-shadow geometry exceeds 4 GiB diagnostic limit");
        auto s=std::make_unique<Shadow>();s->grid=remag::grid_contraction_view();const auto g=s->grid.shape;
        require(g.no==no&&g.nuo==nuo&&g.np==np&&g.nsp==nsp&&g.nnz==capacity,"Force-shadow topology mismatch");
        require(s->grid.entries==entries,"Force-shadow entry count mismatch");
        s->atoms=atoms;s->entries=entries;s->nh=nh;s->epoch=remag::geometry_epoch();s->problem={dvol,volume,forces,stress};
        s->basis_authoritative=remag::orbital_basis_enabled();s->basis_shadow=remag::orbital_basis_shadow_enabled();
        require(!(s->basis_authoritative&&s->basis_shadow),"Choose device or shadow orbital derivatives");
        s->orbital_atoms.assign(orbital_atoms,orbital_atoms+no);
        if(!s->basis_authoritative) {
            s->seen.resize(entries);s->phi.resize(size_t(entries)*nsp);
            s->gradient.resize(s->phi.size()*3);s->displacement.resize(s->gradient.size());
        }
        s->initial_force.assign(initial_force,initial_force+3*atoms);s->initial_stress.assign(initial_stress,initial_stress+9);
        // Fortran may materialize contiguous argument temporaries. Own the
        // snapshot across begin/orbital/finish instead of retaining pointers
        // whose lifetime may end when the first C call returns.
        s->occupation.assign(occupation,occupation+nuo);s->resident=remag::final_grid_state_enabled();s->authoritative=remag_grid_force_enabled();
        require(!s->basis_authoritative||s->authoritative,"Device derivatives require authoritative grid forces");
        require(!s->authoritative||s->resident,"Authoritative grid forces require retained electronic state");
        require(!remag::atomic_grid_force_enabled()||s->authoritative,"Shared atomic forces require authoritative dfscf");
        if(s->resident) {
            s->final=remag::borrow_final_grid_state(s->epoch);const auto f=s->final.shape;
            require(f.no==g.no&&f.np==g.np&&f.nsp==g.nsp&&f.nuo==g.nuo&&f.nnz==g.nnz&&f.cells==g.cells&&
                    s->final.h_spins==nh&&s->final.dvol==dvol&&s->final.volume==volume,"Retained force inputs differ in geometry/configuration");
        } else {
            s->density.assign(density,density+size_t(capacity)*nh);
            s->potential.assign(potential,potential+size_t(np)*nsp*std::min(nh,4));
            s->auxiliary.assign(auxiliary,auxiliary+size_t(np)*nsp);
        }
        shadow=std::move(s);return 0;
    } catch(const std::exception& e) {return failed(e);}
}
extern "C" int remag_force_shadow_orbital(int entry,const double* phi,const double* gradient,const double* displacement) {
    try {
        require(shadow&&!remag::in_scf()&&shadow->epoch==remag::geometry_epoch(),"Invalid force-shadow geometry lifetime");
        auto& s=*shadow;require(!s.basis_authoritative&&entry>0&&entry<=s.entries&&phi&&gradient&&displacement&&!s.seen[entry-1],"Duplicate/missing force-shadow orbital");
        const size_t offset=size_t(entry-1)*s.grid.shape.nsp;
        std::copy(phi,phi+s.grid.shape.nsp,s.phi.data()+offset);
        std::copy(gradient,gradient+3*s.grid.shape.nsp,s.gradient.data()+3*offset);
        std::copy(displacement,displacement+3*s.grid.shape.nsp,s.displacement.data()+3*offset);
        s.seen[entry-1]=1;return 0;
    } catch(const std::exception& e) {return failed(e);}
}
extern "C" int remag_force_shadow_finish(double* reference_force,double* reference_stress) {
    try {
        require(shadow&&!remag::in_scf()&&shadow->epoch==remag::geometry_epoch()&&reference_force&&reference_stress,"Invalid force-shadow completion");
        const auto& s=*shadow;
        require(s.basis_authoritative==remag::orbital_basis_enabled()&&s.basis_shadow==remag::orbital_basis_shadow_enabled(),"Orbital derivative mode changed during force use");
        require(s.basis_authoritative||std::all_of(s.seen.begin(),s.seen.end(),[](int v){return v==1;}),"Force-shadow geometry incomplete");
        if(s.authoritative) require(std::equal(s.initial_force.begin(),s.initial_force.end(),reference_force)&&
            std::equal(s.initial_stress.begin(),s.initial_stress.end(),reference_stress),"CPU contraction modified authoritative GPU force inputs");
        const bool shared=remag::atomic_grid_force_enabled();
        const auto accumulator=shared?remag::final_force_accumulator(s.epoch,s.atoms,s.problem.forces,s.problem.stress,
            reference_force,reference_stress):remag::FinalForceAccumulatorView{};
        const auto atomic_records=shared?remag::final_atomic_force_records(s.epoch):std::vector<remag::AtomicForceRecord>{};
        Device device(accumulator.stream);const auto g=s.grid.shape;int zero=0;
        const bool basis_active=s.basis_authoritative||s.basis_shadow;
        const auto basis=basis_active?remag::begin_orbital_basis(s.epoch,g.no,g.np,s.entries,g.nsp,device.stream):remag::OrbitalBasisDeviceView{};
        const size_t before_derivatives=device.staged;
        const auto phi=s.basis_authoritative?basis.phi:device.upload(s.phi.size(),s.phi.data());
        const auto gradient=s.basis_authoritative?basis.gradient:device.upload(s.gradient.size(),s.gradient.data());
        const auto displacement=s.basis_authoritative?basis.displacement:device.upload(s.displacement.size(),s.displacement.data());
        const size_t derivative_upload=device.staged-before_derivatives;
        remag::GridForceView view{s.grid,s.atoms,s.entries,device.upload(g.no,s.orbital_atoms.data()),phi,gradient,displacement,device.upload(g.nuo,s.occupation.data())};
        auto problem=device.upload(1,&s.problem);
        const double* density=s.resident?s.final.density:device.upload(s.density.size(),s.density.data());
        const double* potential=s.resident?s.final.potential:device.upload(s.potential.size(),s.potential.data());
        const double* auxiliary=s.resident?s.final.auxiliary:device.upload(s.auxiliary.size(),s.auxiliary.data());
        auto force=shared?accumulator.force:device.upload(s.initial_force.size(),s.initial_force.data());
        auto stress=shared?accumulator.stress:device.upload(9,s.initial_stress.data());
        auto status=shared?accumulator.status:device.upload(1,&zero);check(cudaStreamSynchronize(device.stream));
        check(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeGlobal));
        if(basis_active) remag::generate_orbital_basis(s.epoch,status,device.stream);
        if(s.basis_shadow) {
            check(remag::compare_orbital_geometry_device(size_t(s.entries)*g.nsp,basis.phi,basis.gradient,basis.displacement,
                view.phi,view.gradient,view.displacement,basis.comparison,status,device.stream));
        }
        check(remag::add_grid_force_device(view,s.nh,problem,density,potential,auxiliary,force,stress,status,device.stream));
        check(cudaStreamEndCapture(device.stream,&device.graph));size_t nodes=0;
        check(cudaGraphGetNodes(device.graph,nullptr,&nodes));require(nodes==(s.basis_shadow?size_t(3):s.basis_authoritative?size_t(2):size_t(1)),"Unexpected force-shadow graph");
        std::vector<cudaGraphNode_t> graph_nodes(nodes);check(cudaGraphGetNodes(device.graph,graph_nodes.data(),&nodes));
        for(auto node:graph_nodes) {cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"Forbidden force-shadow graph node");}
        check(cudaGraphInstantiate(&device.executable,device.graph,0));check(cudaGraphLaunch(device.executable,device.stream));
        if(remag::integral_force_enabled()) {
            require(shared&&s.authoritative&&s.basis_authoritative&&!s.basis_shadow&&remag::atomic_basis_device_enabled()&&
                !remag::atomic_basis_shadow_enabled()&&derivative_upload==0,"Production integral forces require device grid derivatives");
            auto pending=std::make_unique<DeferredGrid>();pending->epoch=s.epoch;pending->atoms=s.atoms;pending->entries=s.entries;pending->spins=s.nh;
            pending->staged=device.staged+accumulator.staged_bytes;pending->nodes=nodes;pending->retained=s.final.bytes;pending->atomic=atomic_records;
            pending->shape=g;pending->partial_core=s.final.partial_core;
            // Scheduling completion permits overlap on the same stream. The
            // electronic/basis owners remain held until terminal publication.
            remag::complete_final_grid_force(s.epoch);deferred=std::move(pending);shadow.reset();return 0;
        }
        check(cudaStreamSynchronize(device.stream));int code=-1;check(cudaMemcpy(&code,status,sizeof(int),cudaMemcpyDeviceToHost));
        remag::transfer(remag::Transfer::grid_output,false,sizeof(int));device.exported+=sizeof(int);
        std::vector<double> f(s.initial_force.size()),t(9);device.download(f.data(),force,f.size());device.download(t.data(),stress,9);
        double ferr=0.,terr=0.;bool finite=true;
        for(size_t i=0;i<f.size();i++) {finite&=std::isfinite(f[i])&&std::isfinite(reference_force[i]);if(!s.authoritative) ferr=std::max(ferr,std::abs(f[i]-reference_force[i]));}
        for(size_t i=0;i<t.size();i++) {finite&=std::isfinite(t[i])&&std::isfinite(reference_stress[i]);if(!s.authoritative) terr=std::max(terr,std::abs(t[i]-reference_stress[i]));}
        const bool passed=code==0&&finite&&(s.authoritative||(ferr<1.e-10&&terr<1.e-12));
        static int sequence=0;char filename[80];std::snprintf(filename,sizeof(filename),s.authoritative?"remag-grid-force-%04d.json":"remag-force-shadow-%04d.json",++sequence);
        FILE* file=std::fopen(filename,"w");require(file,"Cannot retain force-shadow evidence");
        std::fprintf(file,"{\"scope\":\"GPU grid-force contractions; optional device derivatives; other integral/analysis terms remain on CPU\",\"authoritative\":%s,\"comparison_performed\":%s,\"epoch\":%llu,\"atoms\":%d,\"entries\":%d,\"spins\":%d,\"status\":%d,\"passed\":%s,\"setup_bytes\":%zu,\"export_bytes\":%zu,\"graph_nodes\":%zu,\"atomic_terms_authoritative\":%s,\"shared_accumulator\":%s,\"atomic_intermediate_export_bytes\":%s,\"atomic_terms\":[",
            s.authoritative?"true":"false",s.authoritative?"false":"true",static_cast<unsigned long long>(s.epoch),s.atoms,s.entries,s.nh,code,passed?"true":"false",device.staged+accumulator.staged_bytes,device.exported,nodes,shared?"true":"false",shared?"true":"false",shared?"0":"null");
        for(size_t i=0;i<atomic_records.size();i++) {
            const auto& r=atomic_records[i];
            std::fprintf(file,"%s{\"kind\":%d,\"points\":%d,\"entries\":%d,\"diagonal_spins\":%d,\"graph_nodes\":%d,\"setup_bytes\":%zu,\"derivative_upload_bytes\":%zu,\"host_derivative_storage_bytes\":%zu}",
                i?",":"",r.kind,r.points,r.entries,r.diagonal_spins,r.graph_nodes,r.setup_bytes,r.derivative_upload_bytes,r.host_derivative_storage_bytes);
        }
        std::fprintf(file,"],\"atomic_basis_shadow\":%s,\"atomic_basis_device\":%s,\"orbital_basis_shadow\":%s,\"orbital_basis_device\":%s,\"derivative_upload_bytes\":%zu,\"host_derivative_storage_bytes\":%zu,",
            remag::atomic_basis_shadow_enabled()?"true":"false",remag::atomic_basis_device_enabled()?"true":"false",s.basis_shadow?"true":"false",s.basis_authoritative?"true":"false",derivative_upload,8*(s.phi.size()+s.gradient.size()+s.displacement.size()));
        std::fprintf(file,"\"topology_validation\":\"retained_geometry_owner\",\"retained_entries\":%d,\"topology_export_bytes\":0,",s.grid.entries);
        if(s.authoritative) std::fputs("\"max_force_error_ry_per_bohr\":null,\"max_stress_error_ry_per_bohr3\":null,\"initial_force\":[",file);
        else std::fprintf(file,"\"max_force_error_ry_per_bohr\":%.17g,\"max_stress_error_ry_per_bohr3\":%.17g,\"reference_force\":[",ferr,terr);
        for(size_t i=0;i<f.size();i++) std::fprintf(file,"%s%.17g",i?",":"",reference_force[i]);
        std::fputs("],\"device_force\":[",file);for(size_t i=0;i<f.size();i++) std::fprintf(file,"%s%.17g",i?",":"",f[i]);
        std::fputs(s.authoritative?"],\"initial_stress\":[":"],\"reference_stress\":[",file);for(int i=0;i<9;i++) std::fprintf(file,"%s%.17g",i?",":"",reference_stress[i]);
        std::fputs("],\"device_stress\":[",file);for(int i=0;i<9;i++) std::fprintf(file,"%s%.17g",i?",":"",t[i]);
        std::fprintf(file,"],\"electronic_inputs_resident\":%s,\"retained_electronic_bytes\":%zu,\"grid_shape\":[%d,%d,%d,%d,%d,%d],\"final_partial_core\":%d}\n",
            s.resident?"true":"false",s.resident?s.final.bytes:0,g.no,g.np,g.nsp,g.nuo,g.nnz,g.cells,s.resident?s.final.partial_core:-1);
        std::fclose(file);
        if(basis_active) remag::finish_orbital_basis(s.epoch,code);
        if(remag::atomic_basis_active())remag::finish_atomic_basis(s.epoch,code);
        if(passed&&s.authoritative) {
            std::copy(f.begin(),f.end(),reference_force);std::copy(t.begin(),t.end(),reference_stress);
        }
        if(passed&&s.resident) remag::complete_final_grid_force(s.epoch);
        shadow.reset();require(passed,"Grid-force computation or upstream shadow comparison failed");return 0;
    } catch(const std::exception& e) {return failed(e);}
}
namespace remag {
void finish_integral_grid_force(uint64_t epoch,int status){
    require(integral_force_enabled()&&!in_scf()&&deferred&&deferred->epoch==epoch&&epoch==geometry_epoch()&&status==0,
        "Deferred grid publication before successful final result");
    finish_orbital_basis(epoch,status);finish_atomic_basis(epoch,status);
    const auto& d=*deferred;static int sequence=0;char name[80];std::snprintf(name,sizeof(name),"remag-integral-grid-%04d.json",++sequence);
    FILE* f=std::fopen(name,"w");require(f,"Cannot retain deferred grid evidence");
    std::fprintf(f,"{\"scope\":\"grid contributions in shared terminal integral-force publication\",\"epoch\":%llu,\"atoms\":%d,\"entries\":%d,\"spins\":%d,\"status\":%d,\"setup_bytes\":%zu,\"export_bytes\":0,\"graph_nodes\":%zu,\"retained_electronic_bytes\":%zu,\"derivative_upload_bytes\":0,\"atomic_terms\":[",
        static_cast<unsigned long long>(epoch),d.atoms,d.entries,d.spins,status,d.staged,d.nodes,d.retained);
    for(size_t i=0;i<d.atomic.size();i++){const auto& r=d.atomic[i];std::fprintf(f,"%s{\"kind\":%d,\"points\":%d,\"entries\":%d,\"diagonal_spins\":%d,\"graph_nodes\":%d,\"setup_bytes\":%zu,\"derivative_upload_bytes\":%zu,\"host_derivative_storage_bytes\":%zu}",
        i?",":"",r.kind,r.points,r.entries,r.diagonal_spins,r.graph_nodes,r.setup_bytes,r.derivative_upload_bytes,r.host_derivative_storage_bytes);}
    const auto g=d.shape;
    std::fprintf(f,"],\"orbital_basis_shadow\":false,\"orbital_basis_device\":true,\"atomic_basis_shadow\":false,\"atomic_basis_device\":true,\"grid_shape\":[%d,%d,%d,%d,%d,%d],\"final_partial_core\":%d}\n",
        g.no,g.np,g.nsp,g.nuo,g.nnz,g.cells,d.partial_core);
    std::fclose(f);deferred.reset();
}
}
