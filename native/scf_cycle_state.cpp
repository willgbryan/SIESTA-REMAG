// SPDX-License-Identifier: GPL-3.0-or-later
// Production handoff. All scientific staging precedes the measured cycle;
// only device control executes between the two phase markers.
#include "scf_cycle.hpp"
#include "execution.hpp"
#include "population_state.hpp"
#include "final_grid_state.hpp"
#include "density_history_state.hpp"
#include "final_grid.hpp"
#include "initial_state.hpp"
#include "nonlocal_h0_state.hpp"
#include "initial_mesh_state.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" int remag_final_density_enabled();
extern "C" int remag_final_density_shadow_enabled();
extern "C" int remag_final_density_active();
extern "C" int remag_final_grid_enabled();
extern "C" int remag_gpu_enabled();
extern "C" int remag_grid_enabled();
extern "C" int remag_cycle_spectrum_deferred();
extern "C" int remag_cycle_h_export_gated();
extern "C" int remag_hs_prepare(int,int,int,int,int,int,int,const int*,const int*,const int*,const int*,const double*,const double*,const double*,const double*);
extern "C" int remag_hs_prepare_geometry(int,int,int,int,int,int,int,const int*,const int*,const int*,const int*,const double*,const double*,const double*);
extern "C" int remag_dm_begin(int,int,int,int,int,int,const int*,const int*,const int*,const int*,const double*);
extern "C" void remag_activity_graph(void*) __attribute__((weak));
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
void require(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
struct Record {
    uint64_t epoch=0;
    int geometry=0,iterations=0,converged=0,status=0,producer=0,stage=0,nodes=0,forbidden=0;
    int capacity=0,spins=0,energy_spins=0,bank_size=0,compact_exports=0,old_energy_host=0,energy_export_reasons=0,density_export_reasons=0;
    size_t matrix_exports[6]{}; // H(Dout), H(Dgen), D, Dold, E, Eold
    bool deferred_spectrum=false;
    bool gated_hamiltonian=false;int hamiltonian_export_reasons=1;
    size_t spectrum_exports[2]{}; // eigenvalues, occupations at cycle publication
    int eigensystem_jobs=0,solves_per_iteration=0,solver_lanes=1;bool retains_all_vectors=false,warm_start=false;size_t vector_bank_bytes=0;
    size_t staged=0,exported=0,workspace=0;double seconds=0.;
    bool device_initialization=false;size_t resident_h0_bytes=0,initial_mesh_density_bytes=0;
    remag::InitialDensityResult initial{};
    bool profiling=false;remag::ScfStageProfile profile{};
};
struct FinalGridRecord {
    uint64_t epoch=0;int capacity=0,spins=0;bool shadow=false,passed=false,hamiltonian_required=false;
    size_t hamiltonian_export_bytes=0;
    double matrix_error=0.,grid_error=0.;
    std::vector<double> values,reference;
    std::string matrix_file;
};
struct FinalDensityRecord {
    uint64_t epoch=0;int mesh[3]{},nsm=0,spins=0,kind=0;
    bool requested=false,exported=false,shadow=false,passed=false;
    size_t retained=0,downloaded=0;double error=0.;std::string file;
};
struct FinalSpectrumRecord {
    uint64_t epoch=0;int bank_size=0;
    size_t retained=0,exported=0;bool consumed=false,discarded=false;
};
struct Counts {
    std::vector<Record> cycles;
    std::vector<FinalSpectrumRecord> spectra;
    void publish_spectra() const {
        if(spectra.empty()) return;
        if(FILE* f=std::fopen("remag-final-spectrum.json","w")) {
            std::fputs("{\"schema\":1,\"scope\":\"Device eigenvalues retained until final analysis; occupations have no host consumer in this mode\",\"cycles\":[",f);
            for(size_t i=0;i<spectra.size();i++) {
                const auto& r=spectra[i];
                std::fprintf(f,"%s{\"epoch\":%llu,\"bank_size\":%d,\"retained_device_bytes\":%zu,\"eigenvalue_export_bytes\":%zu,\"occupation_export_bytes\":0,\"consumed\":%s,\"discarded\":%s}",
                    i?",":"",static_cast<unsigned long long>(r.epoch),r.bank_size,r.retained,r.exported,r.consumed?"true":"false",r.discarded?"true":"false");
            }
            std::fputs("]}\n",f);std::fclose(f);
        }
    }
    std::vector<FinalDensityRecord> densities;
    void publish_densities() const {
        if(densities.empty()) return;
        if(FILE* f=std::fopen("remag-final-density.json","w")) {
            std::fputs("{\"scope\":\"Optional sequential device density retained per geometry; explicit requested RHO/DRHO exports after all force consumers\",\"cycles\":[",f);
            for(size_t i=0;i<densities.size();i++) {
                const auto& r=densities[i];
                std::fprintf(f,"%s{\"epoch\":%llu,\"mesh\":[%d,%d,%d],\"nsm\":%d,\"spins\":%d,\"kind\":%d,\"requested\":%s,\"exported\":%s,\"shadow\":%s,\"retained_device_bytes\":%zu,\"host_to_device_bytes\":0,\"device_to_host_bytes\":%zu,\"passed\":%s,\"comparison_performed\":%s,",
                    i?",":"",static_cast<unsigned long long>(r.epoch),r.mesh[0],r.mesh[1],r.mesh[2],r.nsm,r.spins,r.kind,
                    r.requested?"true":"false",r.exported?"true":"false",r.shadow?"true":"false",r.retained,r.downloaded,
                    r.exported?(r.passed?"true":"false"):"null",r.exported&&r.shadow?"true":"false");
                if(r.exported&&r.shadow) std::fprintf(f,"\"max_absolute_error\":%.17g,\"array_file\":\"%s\"}",r.error,r.file.c_str());
                else std::fputs("\"max_absolute_error\":null,\"array_file\":null}",f);
            }
            std::fputs("]}\n",f);std::fclose(f);
        }
    }
    std::vector<FinalGridRecord> final_grids;
    void publish_final_grids() const {
        if(!final_grids.empty()) if(FILE* f=std::fopen("remag-final-grid.json","w")) {
            std::fputs("{\"scope\":\"Final force-grid reuse of published GPU diagnostics; optional pinned final-force Hamiltonian is generated/exported within cycle publication; legacy exports and density analysis remain\",\"cycles\":[",f);
            for(size_t i=0;i<final_grids.size();i++) {
                const auto& r=final_grids[i];
                std::fprintf(f,"%s{\"epoch\":%llu,\"capacity\":%d,\"spins\":%d,\"authoritative\":%s,\"comparison_performed\":%s,\"passed\":%s,\"host_to_device_bytes\":0,\"device_to_host_bytes\":0,\"hamiltonian_required\":%s,\"hamiltonian_cycle_export_bytes\":%zu,\"values\":[",
                    i?",":"",static_cast<unsigned long long>(r.epoch),r.capacity,r.spins,r.shadow?"false":"true",r.shadow?"true":"false",r.passed?"true":"false",r.hamiltonian_required?"true":"false",r.hamiltonian_export_bytes);
                for(size_t j=0;j<r.values.size();j++) std::fprintf(f,"%s%.17g",j?",":"",r.values[j]);
                std::fputs("],\"reference\":[",f);
                for(size_t j=0;j<r.reference.size();j++) std::fprintf(f,"%s%.17g",j?",":"",r.reference[j]);
                if(r.shadow) std::fprintf(f,"],\"matrix_file\":\"%s\",\"max_matrix_error\":%.17g,\"max_grid_error\":%.17g}",r.matrix_file.c_str(),r.matrix_error,r.grid_error);
                else std::fputs("],\"matrix_file\":null,\"max_matrix_error\":null,\"max_grid_error\":null}",f);
            }
            std::fputs("]}\n",f);std::fclose(f);
        }
    }
    ~Counts() {
        publish_spectra();
        publish_densities();
        publish_final_grids();
        if(cycles.empty()) return;
        if(FILE* f=std::fopen("remag-device-scf.json","w")) {
            std::fputs("{\"schema\":3,\"scope\":\"whole electronic cycles and explicit terminal exports; no individual iteration timings\",\"cycles\":[",f);
            for(size_t i=0;i<cycles.size();i++) {
                const auto& r=cycles[i];
                std::fprintf(f,"%s{\"epoch\":%llu,\"geometry\":%d,\"iterations\":%d,\"converged\":%s,\"status\":%d,\"producer_status\":%d,\"stage\":%d,\"seconds\":%.17g,\"graph_nodes\":%d,\"forbidden_nodes\":%d,\"setup_bytes\":%zu,\"export_bytes\":%zu,\"workspace_bytes\":%zu,",
                    i?",":"",static_cast<unsigned long long>(r.epoch),r.geometry,r.iterations,r.converged?"true":"false",r.status,r.producer,r.stage,r.seconds,r.nodes,r.forbidden,r.staged,r.exported,r.workspace);
                std::fprintf(f,"\"terminal\":{\"capacity\":%d,\"spins\":%d,\"energy_spins\":%d,\"bank_size\":%d,\"compact\":%s,\"old_energy_host\":%s,\"energy_export_reasons\":%d,\"density_export_reasons\":%d,\"matrix_bytes\":[",
                    r.capacity,r.spins,r.energy_spins,r.bank_size,r.compact_exports?"true":"false",r.old_energy_host?"true":"false",r.energy_export_reasons,r.density_export_reasons);
                for(int j=0;j<6;j++)std::fprintf(f,"%s%zu",j?",":"",r.matrix_exports[j]);
                std::fprintf(f,"],\"deferred_spectrum\":%s,\"spectrum_bytes\":[%zu,%zu],\"gated_hamiltonian\":%s,\"hamiltonian_export_reasons\":%d},\"initialization\":{\"device\":%s,\"status\":%d,\"charge\":%.17g,\"scale\":%.17g,\"mix_first\":%d,\"note\":%d,\"result_bytes\":%zu},",
                    r.deferred_spectrum?"true":"false",r.spectrum_exports[0],r.spectrum_exports[1],
                    r.gated_hamiltonian?"true":"false",r.hamiltonian_export_reasons,
                    r.device_initialization?"true":"false",r.initial.status,r.initial.charge,r.initial.scale,r.initial.mix_first,r.initial.note,
                    r.device_initialization?sizeof(r.initial):size_t(0));
                std::fprintf(f,"\"nonlocal_h0\":{\"device\":%s,\"retained_bytes\":%zu},",
                    r.resident_h0_bytes?"true":"false",r.resident_h0_bytes);
                std::fprintf(f,"\"initial_mesh\":{\"device\":%s,\"density_bytes\":%zu,\"harris_device_bound\":%s},",
                    r.initial_mesh_density_bytes?"true":"false",r.initial_mesh_density_bytes,r.initial_mesh_density_bytes?"true":"false");
                std::fprintf(f,"\"warm_start\":%s,",r.warm_start?"true":"false");
                std::fprintf(f,"\"eigensystems\":{\"jobs\":%d,\"solves_per_iteration\":%d,\"retains_all_vectors\":%s,\"vector_bank_bytes\":%zu,\"solver_lanes\":%d},\"stage_profile\":",
                    r.eigensystem_jobs,r.solves_per_iteration,r.retains_all_vectors?"true":"false",r.vector_bank_bytes,r.solver_lanes);
                if(!r.profiling)std::fputs("null",f);
                else {
                    std::fputs("{\"clock\":\"PTX globaltimer\",\"scope\":\"Serial guarded stages including graph scheduling and profiling overhead; diagnostic timings\",\"stages\":{",f);
                    const char* names[]={"assembly_and_eigensolver","occupations","density_and_normalization","hamiltonian","density_grid_hartree_xc","scalar_reductions","convergence_and_mixing"};
                    for(int j=0;j<7;j++)std::fprintf(f,"%s\"%s\":{\"seconds\":%.17g,\"calls\":%llu}",j?",":"",names[j],double(r.profile.nanoseconds[j])*1.e-9,r.profile.calls[j]);
                    std::fputs("}}",f);
                }
                std::fputc('}',f);
            }
            std::fputs("]}\n",f);std::fclose(f);
        }
    }
} counts;
enum class Stage {idle,requested,grid,ready,executed,exported,failed};
struct State {
    Stage stage=Stage::idle;uint64_t epoch=0;cudaStream_t stream=nullptr;
    bool hs_lease=false,dm_lease=false,grid_lease=false;
    int capacity=0,nh=0,eh=0,bank_size=0,dimension=0,complex_values=0;
    remag::GridPotentialProblem grid_problem{};remag::ScfCycleBindings bindings{};
    remag::GridContractionShape grid_shape{};remag::GridContractionView final_grid_view{};
    int soc=0,final_h_required=0;
    std::vector<void*> storage;std::unique_ptr<remag::ScfCyclePlan> plan;
    std::vector<remag::ScfCycleStep> diagnostics;remag::ScfCycleResult result{};
    remag::InitialDensityResult* initial=nullptr;
    std::vector<double> final_h;
    bool final_grid_pending=false,final_grid_used=false;
    bool initial_density_borrowed=false;bool h0_borrowed=false,initial_mesh_borrowed=false;
    int final_rho_requested=0;
    double* final_rho=nullptr;double* final_drho=nullptr;
    double* final_spectrum=nullptr;double* host_spectrum=nullptr;size_t spectrum_record=0;
    void clear_final_spectrum() {
        if(final_spectrum) {
            check(cudaFree(final_spectrum));final_spectrum=nullptr;
            counts.spectra.at(spectrum_record).discarded=true;
        }
        host_spectrum=nullptr;
    }
    void clear_final_rho() {if(final_rho) {check(cudaFree(final_rho));final_rho=nullptr;}
        if(final_drho) {check(cudaFree(final_drho));final_drho=nullptr;}}
    double *host_h=nullptr,*host_hold=nullptr,*host_d=nullptr,*host_dold=nullptr,*host_e=nullptr,*host_eold=nullptr;
    Record record{};
    template<class T> T* allocate(size_t n,const T* source=nullptr) {
        require(n>0&&n<=SIZE_MAX/sizeof(T),"Invalid cycle allocation");
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));storage.push_back(p);
        if(source) {
            check(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice));record.staged+=n*sizeof(T);
            remag::transfer(remag::Transfer::sparse_state_setup,true,n*sizeof(T));
        } else check(cudaMemsetAsync(p,0,n*sizeof(T),stream));
        return p;
    }
    void download(void* target,const void* source,size_t bytes) {
        require(target&&source,"Missing cycle export target");check(cudaMemcpy(target,source,bytes,cudaMemcpyDeviceToHost));
        record.exported+=bytes;remag::transfer(remag::Transfer::sparse_state_export,false,bytes);
    }
    void download_matrix(int role,double* target,const double* source,size_t bytes) {
        require(role>=0&&role<6&&!record.matrix_exports[role],"Invalid or repeated terminal matrix export");
        download(target,source,bytes);record.matrix_exports[role]=bytes;
    }
    void release() {
        if(stream) check(cudaStreamSynchronize(stream));
        plan.reset(); // graph references the leases and every buffer below
        if(h0_borrowed){remag::release_nonlocal_h0_matrices(epoch);h0_borrowed=false;}
        if(grid_lease) {remag::release_grid_contraction();grid_lease=false;}
        if(dm_lease) {remag::release_density_matrix();dm_lease=false;}
        if(hs_lease) {remag::release_hamiltonian_assembly();hs_lease=false;}
        for(auto p:storage) check(cudaFree(p));
        storage.clear();
        if(stream) {check(cudaStreamDestroy(stream));stream=nullptr;}
    }
    ~State() {try {release();clear_final_rho();clear_final_spectrum();} catch(...) {}}
} state;
void current(Stage stage) {
    require(!remag::in_scf()&&state.stage==stage&&state.epoch==remag::geometry_epoch(),"Invalid device-cycle lifetime/epoch");
}
int failure(const char* where,const std::exception& e) {
    state.stage=Stage::failed;std::fprintf(stderr,"REMAG device SCF %s: %s\n",where,e.what());return -220;
}
}
extern "C" int remag_cycle_enabled() {
    const char* v=std::getenv("REMAG_DEVICE_SCF");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_initialization_enabled() {
    const char* v=std::getenv("REMAG_GPU_INITIALIZATION");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_cycle_spectrum_deferred() {
    const char* v=std::getenv("REMAG_FINAL_SPECTRA");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_cycle_h_export_gated() {
    const char* v=std::getenv("REMAG_GATE_H_EXPORT");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_cycle_initial_grid() {
    return remag_initialization_enabled()&&state.stage==Stage::grid;
}
extern "C" int remag_cycle_request() {
    try {
        require(remag_cycle_enabled()&&remag_gpu_enabled()&&remag_grid_enabled()&&!remag::in_scf()&&
            (state.stage==Stage::idle||state.stage==Stage::exported)&&remag::geometry_epoch(),"Device-cycle request conflicts with active work");
        remag::reset_final_grid_state();
        state.clear_final_rho();
        state.clear_final_spectrum();
        require(!state.final_grid_pending||state.final_grid_used,"Final force-grid handoff missing before next geometry");
        state.final_h.clear();state.final_grid_pending=false;state.final_grid_used=false;
        require(!state.initial_mesh_borrowed,"Initial mesh cycle lease was not returned");
        state.release();state.epoch=remag::geometry_epoch();state.record={};state.result={};state.diagnostics.clear();
        state.record.epoch=state.epoch;state.initial=nullptr;
        state.record.device_initialization=remag_initialization_enabled();
        state.record.deferred_spectrum=remag_cycle_spectrum_deferred();
        require(!state.record.deferred_spectrum||(remag::final_grid_state_enabled()&&remag::integral_force_enabled()),
            "Deferred spectrum requires final device state and force ownership");
        state.bindings={};check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking));
        state.stage=Stage::requested;return 0;
    } catch(const std::exception& e) {return failure("request",e);}
}
extern "C" int remag_cycle_wants_grid() {return state.stage==Stage::requested;}
extern "C" int remag_cycle_grid(const int* mesh,int spins,int nsm,int npcc,const double* atom,const double* core,
        const double* neutral,const double* reciprocal,double cutoff,const double* gradient,double volume,double dvol,double harris) {
    try {
        current(Stage::requested);
        require(mesh&&atom&&core&&neutral&&reciprocal&&gradient&&mesh[0]>0&&mesh[1]>0&&mesh[2]>0&&
            (spins==1||spins==2||spins==4)&&nsm>0&&(npcc==0||npcc==1),"Invalid cycle mesh");
        const size_t size=size_t(mesh[0])*mesh[1]*mesh[2];require(size<=INT_MAX/size_t(spins),"Cycle mesh too large");
        for(int i=0;i<3;i++) require(mesh[i]%nsm==0,"Invalid mesh clustering");
        require(std::isfinite(cutoff)&&cutoff>0.&&std::isfinite(volume)&&volume>0.&&std::isfinite(dvol)&&dvol>0.&&(remag::initial_mesh_enabled()||std::isfinite(harris)),"Invalid cycle grid constants");
        auto& p=state.grid_problem;p={};std::copy(mesh,mesh+3,p.hartree.mesh);std::copy(mesh,mesh+3,p.pbe.mesh);
        std::copy(reciprocal,reciprocal+9,p.hartree.reciprocal);std::copy(gradient,gradient+9,p.pbe.gradient);
        p.hartree.cutoff=cutoff;p.hartree.volume=volume;p.pbe.volume=volume;p.pbe.spins=spins;
        p.harris_energy=remag::initial_mesh_enabled()?NAN:harris;p.dvol=dvol;p.nsm=nsm;p.partial_core=npcc;
        auto& b=state.bindings;auto device_grid=state.allocate<remag::GridPotentialProblem>(1,&p);b.grid=device_grid;
        if(remag::initial_mesh_enabled()) {
            b.atomic_density=remag::borrow_initial_mesh_density(state.epoch,p,device_grid,state.stream);
            state.initial_mesh_borrowed=true;state.record.initial_mesh_density_bytes=size*sizeof(double);
        } else b.atomic_density=state.allocate<double>(size,atom);
        b.core_density=state.allocate<double>(npcc?size:1,npcc?core:nullptr);
        b.neutral_potential=state.allocate<double>(size,neutral);state.stage=Stage::grid;return 0;
    } catch(const std::exception& e) {return failure("grid staging",e);}
}
// Integers: kind,n,capacity,nh,no,nk,history,fermi-spinor,soc,initial-EDM,
//           mix-first,criteria,minimum,maximum,final-H-required,density-output-mask (RHO=1,DRHO=2).
//           energy-export-reasons (legacy forces=1,TSDE=2,NetCDF EDM=4).
// Reals: weight,linear-weight,condition,temp,charge,norm-tol,occ-cutoff,
//        DM/H/EDM/Harris/FreeE tolerances,Ena,Eions,Emad.
extern "C" int remag_cycle_prepare(const int* c,const double* v,const int* num,const int* ptr,const int* cols,
        const int* mapping,const double* xyz,const double* kpoints,const double* weights,const double* overlap,
        const double* kinetic,const double* nonlocal_h,const double* onsite,const cuDoubleComplex* offsite,
        double* h,double* hold,double* density,double* old_density,double* energy_density,double* old_energy_density) {
    try {
        current(Stage::grid);require(c&&v&&num&&ptr&&cols&&mapping&&xyz&&kpoints&&weights&&overlap&&kinetic&&nonlocal_h&&
            h&&hold&&density&&old_density&&energy_density,"Missing cycle inputs");
        const int kind=c[0],n=c[1],cap=c[2],nh=c[3],no=c[4],nk=c[5],eh=std::min(nh,4);
        require(kind>=0&&kind<=3&&n>0&&n<=2048&&cap>0&&cap<=INT_MAX/8&&no>=n&&nk>0&&nk<=128&&
            (kind%2||nk==1)&&(kind<2?(nh==1||nh==2):(nh==4||nh==8))&&c[13]>0&&c[13]<=4096&&
            (c[9]==0||c[9]==1)&&(!c[9]||old_energy_density),"Invalid cycle dimensions/options");
        if(kind%2==0) for(int i=0;i<3;i++) require(kpoints[i]==0.,"Gamma cycle must use zero phases");
        require(state.grid_problem.pbe.spins==eh,"Cycle grid spin mismatch");
        remag::ScfControlProblem control{{c[11],c[12],c[13],0,{},state.epoch},
            {cap*nh,c[6],v[0],v[1],v[2],state.epoch},c[10],0};
        std::copy(v+7,v+12,control.convergence.tolerances);
        const bool initialize=state.record.device_initialization;
        const int hs_status=initialize?
            remag_hs_prepare_geometry(kind,n,cap,nh,no,nk,cap,num,ptr,cols,mapping,xyz,kpoints,overlap):
            remag_hs_prepare(kind,n,cap,nh,no,nk,cap,num,ptr,cols,mapping,xyz,kpoints,h,overlap);
        require(hs_status==0,"H/S geometry preparation failed");
        remag::ScfCycleShape shape{};shape.assembly=remag::lease_hamiltonian_assembly();state.hs_lease=true;
        require(remag_dm_begin(kind,n,cap,nh,eh,no,num,ptr,cols,mapping,xyz)==0,"Density geometry preparation failed");
        shape.density=remag::lease_density_matrix();state.dm_lease=true;
        shape.grid=remag::lease_grid_contraction();state.grid_lease=true;
        state.grid_shape=shape.grid.shape;
        state.final_grid_view=shape.grid;state.soc=c[8];state.final_h_required=c[14];
        require(c[14]==0||c[14]==1,"Invalid final Hamiltonian export request");
        require(c[15]>=0&&c[15]<=3,"Invalid density output request");state.final_rho_requested=c[15];
        require(c[16]>=0&&c[16]<=7&&bool(c[16]&1)==!remag::integral_force_enabled(),
            "Invalid energy-density export reasons");
        state.record.energy_export_reasons=c[16];
        require(c[17]>=0&&c[17]<=63&&bool(c[17]&1)==!remag::density_history_enabled(),
            "Invalid density export reasons");
        require(!remag::density_history_enabled()||remag::integral_force_enabled(),
            "GPU history requires device integral forces");
        state.record.density_export_reasons=c[17];
        state.record.gated_hamiltonian=remag_cycle_h_export_gated();
        require(c[18]>=0&&c[18]<=127&&
            (state.record.gated_hamiltonian?(remag::integral_force_enabled()&&!(c[18]&1)):c[18]==1),
            "Invalid Hamiltonian export reasons");
        state.record.hamiltonian_export_reasons=c[18];
        std::copy(state.grid_problem.pbe.mesh,state.grid_problem.pbe.mesh+3,shape.mesh);
        shape.normalization={cap,nh,eh,std::min(nh,2),v[4],v[5]};shape.occupation_cutoff=v[6];
        for(int i=0;i<nk;i++) shape.kpoints.push_back({kpoints[3*i],kpoints[3*i+1],kpoints[3*i+2]});
        const int dim=n*(kind>=2?2:1),spins=kind<2?nh:1;
        remag::FermiProblem fp{c[7],spins,nk,dim,dim,v[3],v[4]};
        remag::HamiltonianSeedsProblem hp{cap,nh,c[8],0};remag::ScfScalarsProblem sp{cap,nh,n,0};
        remag::ScfEnergySeeds constants{};constants.ena=v[12];constants.ions=v[13];constants.madelung=v[14];constants.temperature=v[3];
        auto& b=state.bindings;b.fermi=state.allocate<remag::FermiProblem>(1,&fp);b.weights=state.allocate<double>(nk,weights);
        b.hamiltonian=state.allocate<remag::HamiltonianSeedsProblem>(1,&hp);
        b.kinetic=state.allocate<double>(cap,kinetic);
        if(remag::nonlocal_h0_enabled()){
            require(initialize&&remag::integral_force_enabled(),"Resident H0 requires GPU initialization and integral forces");
            remag::validate_nonlocal_h0_topology(n,no,cap,num,ptr,cols);
            const auto retained=remag::borrow_nonlocal_h0_matrices(state.epoch,n,no,cap,c[8]);state.h0_borrowed=true;
            b.nonlocal_h=retained.scalar;b.offsite=reinterpret_cast<const cuDoubleComplex*>(retained.soc);
            state.record.resident_h0_bytes=size_t(cap)*(c[8]==2?72:8);
        }else b.nonlocal_h=state.allocate<double>(cap,nonlocal_h);
        if(c[8]==1) {require(onsite,"Missing onsite SOC");b.onsite=state.allocate<double>(size_t(cap)*6,onsite);}
        if(c[8]==2&&!remag::nonlocal_h0_enabled()) {require(offsite,"Missing offsite SOC");b.offsite=state.allocate<cuDoubleComplex>(size_t(cap)*4,offsite);}
        b.scalars=state.allocate<remag::ScfScalarsProblem>(1,&sp);b.constants=state.allocate<remag::ScfEnergySeeds>(1,&constants);
        b.num=state.allocate<int>(n,num);b.ptr=state.allocate<int>(n,ptr);b.overlap=shape.assembly.overlap;
        remag::bind_population_topology(state.epoch,b.num,b.ptr,b.overlap,state.stream);
        b.h=state.allocate<double>(size_t(cap)*nh,initialize?nullptr:h);
        state.initial_density_borrowed=initialize&&remag::initial_state_enabled();
        b.density=state.initial_density_borrowed?
            remag::borrow_initial_density(state.epoch,n,cap,nh,num,ptr,overlap,density):state.allocate<double>(size_t(cap)*nh,density);
        b.energy_density=state.allocate<double>(size_t(cap)*eh,!initialize&&c[9]?energy_density:nullptr);
        state.host_h=h;state.host_hold=hold;state.host_d=density;state.host_dold=old_density;
        state.host_e=energy_density;state.host_eold=old_energy_density;
        state.capacity=cap;state.nh=nh;state.eh=eh;state.bank_size=dim*spins*nk;state.dimension=dim;state.complex_values=kind!=0;state.diagnostics.resize(c[13]);
        state.record.capacity=cap;state.record.spins=nh;state.record.energy_spins=eh;state.record.bank_size=state.bank_size;
        state.record.compact_exports=remag::integral_force_enabled();state.record.old_energy_host=old_energy_density!=nullptr;
        // Legacy staging uses the default stream. All preparation completes
        // before the new nondefault-stream owner can execute.
        check(cudaDeviceSynchronize());state.plan=std::make_unique<remag::ScfCyclePlan>(shape,b,control,state.stream);
        if(initialize) {
            const remag::InitialDensityProblem initial{cap,nh,std::min(nh,2),c[10],v[4]};
            auto problem=state.allocate<remag::InitialDensityProblem>(1,&initial);
            auto dependency=state.allocate<int>(1);
            state.initial=state.allocate<remag::InitialDensityResult>(1);
            check(remag::initial_density_device(cap,nh,problem,dependency,b.overlap,b.density,b.density,b.energy_density,state.initial,state.stream));
            check(state.plan->initialize_hamiltonian(state.initial));
            // Initialization is outside the steady SCF interval. Its kernels
            // finish before that interval starts; no electronic readback.
            check(cudaStreamSynchronize(state.stream));
        }
        state.record.workspace=state.plan->workspace_bytes();state.record.nodes=state.plan->graph_nodes();
        state.record.eigensystem_jobs=state.plan->eigensystem_jobs();state.record.solves_per_iteration=state.plan->solves_per_iteration();
        state.record.solver_lanes=state.plan->solver_lanes();
        state.record.warm_start=state.plan->warm_start();
        state.record.retains_all_vectors=state.plan->retains_all_vectors();state.record.vector_bank_bytes=state.plan->vector_bank_bytes();
        state.record.forbidden=state.plan->forbidden_nodes();state.record.staged+=state.plan->setup_upload_bytes();
        remag::transfer(remag::Transfer::solver_graph_setup,true,state.plan->setup_upload_bytes());
        require(!state.record.forbidden,"Cycle graph contains forbidden nodes");
        if(remag_activity_graph) remag_activity_graph(state.plan->executable());
        state.stage=Stage::ready;return 0;
    } catch(const std::exception& e) {return failure("prepare",e);}
}
extern "C" int remag_cycle_execute(int geometry,int* iterations,int* converged) {
    bool entered=false;
    try {
        current(Stage::ready);require(iterations&&converged,"Missing cycle result targets");state.record.geometry=geometry;
        require(remag_phase_enter(1,geometry,1)==0,"Cannot enter device cycle");entered=true;
        const auto start=std::chrono::steady_clock::now();
        check(state.plan->launch());check(cudaStreamSynchronize(state.stream));
        const auto stop=std::chrono::steady_clock::now();
        require(remag_phase_enter(2,geometry,0)==0,"Cannot exit device cycle");entered=false;
        state.record.seconds=std::chrono::duration<double>(stop-start).count();
        if(const auto* profile=state.plan->stage_profile()) {
            state.record.profiling=true;state.download(&state.record.profile,profile,sizeof(state.record.profile));
        }
        state.download(&state.result,state.plan->result(),sizeof(state.result));const auto& r=state.result;
        if(state.initial) state.download(&state.record.initial,state.initial,sizeof(state.record.initial));
        state.record.iterations=r.iterations;state.record.converged=r.control.convergence.converged;
        state.record.status=r.status;state.record.producer=r.producer_status;state.record.stage=r.stage;
        if(r.status) {
            counts.cycles.push_back(state.record);
            remag::BlockJacobiLocalFailure local_failure{0,-1,-1,-1};
            const int local_dimension=state.plan->solver_local_dimension();
            // Upstream die aborts without C++ destructors. Persist this failed
            // status and the unchanged last assembled solver inputs now,
            // after closing SCF, so the failure remains reproducible.
            auto save=[&](const char* name,const void* device,size_t bytes) {
                std::vector<unsigned char> data(bytes);state.download(data.data(),device,bytes);
                FILE* f=std::fopen(name,"wb");require(f,"Cannot retain cycle failure evidence");
                const bool written=std::fwrite(data.data(),1,bytes,f)==bytes;std::fclose(f);
                require(written,"Incomplete cycle failure evidence");
            };
            if(r.stage==1) {
                const size_t bytes=size_t(state.dimension)*state.dimension*(state.complex_values?16:8);
                save("remag-failed-h.bin",state.plan->assembled_hamiltonian(r.block),bytes);
                save("remag-failed-s.bin",state.plan->assembled_overlap(r.block),bytes);
                save("remag-failed-solver.bin",state.plan->solver_diagnostic(r.block),sizeof(remag::BlockJacobiResult));
                state.download(&local_failure,state.plan->solver_local_failure(r.block),sizeof(local_failure));
                if(local_failure.info) save("remag-failed-local.bin",state.plan->solver_failed_local_input(r.block),size_t(local_dimension)*local_dimension*(state.complex_values?16:8));
            }
            if(FILE* f=std::fopen("remag-device-scf-failure.json","w")) {
                std::fprintf(f,"{\"status\":%d,\"producer_status\":%d,\"stage\":%d,\"block\":%d,\"pass\":%d,\"accepted_iterations\":%d,\"dimension\":%d,\"complex_values\":%s,\"local_solver_info\":%d,\"local_pair\":%d,\"local_round\":%d,\"local_sweep\":%d,\"local_dimension\":%d,\"matrix_layout\":\"Fortran column-major, last assembled solver input\"}\n",
                    r.status,r.producer_status,r.stage,r.block,r.pass,r.iterations,state.dimension,state.complex_values?"true":"false",
                    local_failure.info,local_failure.pair,local_failure.round,local_failure.sweep,local_dimension);
                std::fclose(f);
            }
            std::fprintf(stderr,"REMAG cycle producer failed: stage=%d block=%d pass=%d status=%d producer=%d\n",r.stage,r.block,r.pass,r.status,r.producer_status);
        }
        require(!r.status&&r.finished&&r.iterations>0&&size_t(r.iterations)<=state.diagnostics.size(),"Device cycle failed; scientific outputs are invalid");
        state.download(state.diagnostics.data(),state.plan->diagnostics(),size_t(r.iterations)*sizeof(remag::ScfCycleStep));
        *iterations=r.iterations;*converged=r.control.convergence.converged;state.stage=Stage::executed;return 0;
    } catch(const std::exception& e) {if(entered) remag_phase_enter(2,geometry,0);return failure("execute",e);}
}
extern "C" int remag_cycle_initial_result(double* values,int* mix,int* note) {
    try {
        current(Stage::executed);
        require(state.record.device_initialization&&values&&mix&&note,"Missing initial result");
        const auto& r=state.record.initial;require(!r.status,"Invalid initial density");
        values[0]=r.charge;values[1]=r.scale;*mix=r.mix_first;*note=r.note;return 0;
    } catch(const std::exception& e) {return failure("initial result",e);}
}
extern "C" int remag_cycle_export(int bank_size,double* energies,double* occupations) {
    try {
        current(Stage::executed);require(bank_size==state.bank_size,"Cycle spectrum shape mismatch");
        const auto& b=state.bindings;const size_t hbytes=size_t(state.capacity)*state.nh*sizeof(double),ebytes=size_t(state.capacity)*state.eh*sizeof(double);
        if(remag::final_grid_state_enabled()) {
            require(state.result.control.convergence.converged,"Final electronic state requires convergence");
            remag::retain_final_grid_state({state.grid_shape,state.nh,state.eh,state.epoch,
                state.grid_problem.dvol,state.grid_problem.pbe.volume,b.density,b.energy_density,
                state.plan->mesh_potential(),state.plan->mesh_auxiliary_potential(),state.plan->mesh_deformation_density(),0,
                {state.grid_problem.pbe.mesh[0],state.grid_problem.pbe.mesh[1],state.grid_problem.pbe.mesh[2]},
                state.grid_problem.nsm,state.grid_problem.partial_core,
                state.plan->diagnostics()[state.result.iterations-1].grid.hartree_stress,
                remag::density_history_enabled()?&state.plan->result()->control.convergence.converged:nullptr},state.stream);
        }
        require(bool(state.record.compact_exports)==remag::integral_force_enabled(),"Terminal export mode changed during SCF");
        require(state.record.gated_hamiltonian==bool(remag_cycle_h_export_gated()),"Hamiltonian export mode changed during SCF");
        if(state.record.compact_exports) {
            // The supported route mixes H and never mixes after convergence.
            // end_of_cycle_save_operations normally assigns H=Hold. Publish
            // that generator directly into H and skip the obsolete H(Dout),
            // Dold and Eold. Their consumers are inside the device cycle;
            // fresh geometry initialization reconstructs the next inputs.
            if(state.record.hamiltonian_export_reasons)
                state.download_matrix(1,state.host_h,state.plan->old_hamiltonian(),hbytes);
        } else {
            state.download_matrix(0,state.host_h,b.h,hbytes);
            state.download_matrix(1,state.host_hold,state.plan->old_hamiltonian(),hbytes);
            state.download_matrix(3,state.host_dold,state.plan->old_density(),hbytes);
            if(state.host_eold)state.download_matrix(5,state.host_eold,state.plan->old_energy_density(),ebytes);
        }
        // History reuses the retained device state. Only a recorded output,
        // diagnostic, or explicitly tracked legacy consumer needs host D.
        if(state.record.density_export_reasons)
            state.download_matrix(2,state.host_d,b.density,hbytes);
        // The force owner retains device E. Only a legacy force calculation
        // or an explicitly requested TSDE/NetCDF EDM needs a host copy.
        if(state.record.energy_export_reasons)
            state.download_matrix(4,state.host_e,b.energy_density,ebytes);
        const char* reuse=std::getenv("REMAG_FINAL_GRID");const char* shadow=std::getenv("REMAG_FINAL_GRID_SHADOW");
        if((reuse&&std::strcmp(reuse,"1")==0)||(shadow&&std::strcmp(shadow,"1")==0)) {
            require(remag::atomic_grid_force_enabled(),"Final grid reuse requires shared authoritative grid forces");
            state.final_h.clear();
            // H_new is unused by default. Generate/export it only for the
            // strict shadow or an actual compatibility/diagnostic consumer.
            // Its offsite SOC convention differs from the SCF H export.
            if((shadow&&std::strcmp(shadow,"1")==0)||state.final_h_required) {
                auto* h=state.allocate<double>(size_t(state.capacity)*state.nh);
                check(remag::final_grid_hamiltonian_device(state.final_grid_view,state.nh,state.soc,
                    b.kinetic,b.nonlocal_h,b.onsite,state.plan->mesh_potential(),state.grid_problem.dvol,h,state.stream));
                check(cudaStreamSynchronize(state.stream));state.final_h.resize(size_t(state.capacity)*state.nh);
                state.download(state.final_h.data(),h,hbytes);
            }
            state.final_grid_pending=true;state.final_grid_used=false;
        }
        if(remag_final_density_active()) {
            require(remag_final_grid_enabled()&&!(remag_final_density_enabled()&&remag_final_density_shadow_enabled())&&
                !state.final_rho&&!state.final_drho,"Requested density reuse requires final-grid production and an empty output buffer");
            for(int kind=1;kind<=2;kind++) {
                FinalDensityRecord r;r.epoch=state.epoch;r.spins=state.eh;r.nsm=state.grid_problem.nsm;r.kind=kind;
                std::copy(state.grid_problem.pbe.mesh,state.grid_problem.pbe.mesh+3,r.mesh);
                r.requested=(state.final_rho_requested&kind)!=0;r.shadow=remag_final_density_shadow_enabled();
                if(r.requested) {
                    r.retained=size_t(r.mesh[0])*r.mesh[1]*r.mesh[2]*r.spins*sizeof(double);
                    auto** target=kind==1?&state.final_rho:&state.final_drho;
                    check(cudaMalloc(reinterpret_cast<void**>(target),r.retained));
                }
                counts.densities.push_back(r);
            }
            if(state.final_rho_requested) {
                check(remag::final_density_sequential_device(state.grid_problem.pbe.mesh,state.grid_problem.nsm,state.eh,
                    state.plan->mesh_density(),state.bindings.atomic_density,state.final_rho,state.final_drho,state.stream));
                check(cudaStreamSynchronize(state.stream)); // post-cycle, no host data publication
            }
        }
        require(state.record.deferred_spectrum==bool(remag_cycle_spectrum_deferred()),"Spectrum mode changed during SCF");
        const size_t spectrum_bytes=size_t(bank_size)*sizeof(double);
        if(state.record.deferred_spectrum) {
            require(!state.final_spectrum&&energies&&occupations,"Missing or repeated retained spectrum");
            check(cudaMalloc(reinterpret_cast<void**>(&state.final_spectrum),spectrum_bytes));
            check(cudaMemcpy(state.final_spectrum,state.plan->energies(),spectrum_bytes,cudaMemcpyDeviceToDevice));
            state.host_spectrum=energies;state.spectrum_record=counts.spectra.size();
            counts.spectra.push_back({state.epoch,bank_size,spectrum_bytes,0,false,false});
        } else {
            state.download(energies,state.plan->energies(),spectrum_bytes);
            state.download(occupations,state.plan->occupations(),spectrum_bytes);
            state.record.spectrum_exports[0]=spectrum_bytes;state.record.spectrum_exports[1]=spectrum_bytes;
        }
        state.release();
        if(state.initial_mesh_borrowed){
            remag::release_initial_mesh_density(state.epoch,state.result.status,state.result.producer_status,state.result.control.convergence.converged);
            state.initial_mesh_borrowed=false;
        }
        if(state.initial_density_borrowed){remag::release_initial_density(state.epoch);state.initial_density_borrowed=false;}
        counts.cycles.push_back(state.record);state.stage=Stage::exported;return 0;
    } catch(const std::exception& e) {return failure("export",e);}
}
extern "C" int remag_cycle_analysis_spectrum(int bank_size,double* energies) {
    try {
        current(Stage::exported);
        require(state.record.deferred_spectrum&&remag_cycle_spectrum_deferred()&&state.final_spectrum&&
            energies==state.host_spectrum&&bank_size==state.bank_size&&remag::final_grid_forces_complete(state.epoch),
            "Invalid final-analysis spectrum consumer");
        auto& r=counts.spectra.at(state.spectrum_record);
        require(!r.consumed&&!r.discarded&&r.epoch==state.epoch&&r.bank_size==bank_size,"Stale or repeated spectrum consumer");
        check(cudaMemcpy(energies,state.final_spectrum,r.retained,cudaMemcpyDeviceToHost));
        remag::transfer(remag::Transfer::final_spectrum_output,false,r.retained);
        r.exported=r.retained;r.consumed=true;return 0;
    } catch(const std::exception& e) {return failure("analysis spectrum",e);}
}
extern "C" int remag_final_grid_enabled() {
    const char* v=std::getenv("REMAG_FINAL_GRID");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_final_grid_shadow_enabled() {
    const char* v=std::getenv("REMAG_FINAL_GRID_SHADOW");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_final_grid_active() {return remag_final_grid_enabled()||remag_final_grid_shadow_enabled();}
// Uses the existing terminal diagnostic publication and optionally the final
// force Hamiltonian. Host copies are an explicit bridge until all consumers of
// the legacy electronic exports have migrated. Only the force-grid call may
// claim this record, once per epoch; later density analysis remains separate.
extern "C" int remag_final_grid(int capacity,int spins,int grid_spins,const int* mesh,int nsm,int core,
        double dvol,double volume,const double* density,double* h,double* values,double* hs,double* xs) {
    try {
        current(Stage::exported);const auto& p=state.grid_problem;
        require(remag_final_grid_active()&&!(remag_final_grid_enabled()&&remag_final_grid_shadow_enabled())&&
            remag::atomic_grid_force_enabled()&&!state.final_grid_used&&state.result.control.convergence.converged&&
            capacity==state.capacity&&spins==state.nh&&grid_spins==state.eh&&mesh&&std::equal(mesh,mesh+3,p.pbe.mesh)&&
            nsm==p.nsm&&core==p.partial_core&&dvol==p.dvol&&volume==p.pbe.volume&&density&&h&&values&&hs&&xs&&
            state.final_grid_pending&&density==state.host_d&&
            ((remag_final_grid_shadow_enabled()||state.final_h_required)?state.final_h.size()==size_t(capacity)*spins:state.final_h.empty()),
            "Invalid final grid ownership/configuration or density identity");
        const auto& g=state.diagnostics[state.result.iterations-1].grid;
        require(!g.status,"Invalid final grid diagnostic");
        FinalGridRecord r;r.epoch=state.epoch;r.capacity=capacity;r.spins=spins;r.shadow=remag_final_grid_shadow_enabled();
        r.hamiltonian_required=state.final_h_required;r.hamiltonian_export_bytes=state.final_h.size()*sizeof(double);
        r.values.insert(r.values.end(),g.energies,g.energies+9);r.values.insert(r.values.end(),g.hartree_stress,g.hartree_stress+9);
        r.values.insert(r.values.end(),g.xc_stress,g.xc_stress+9);
        require(std::all_of(r.values.begin(),r.values.end(),[](double x){return std::isfinite(x);})&&
            std::all_of(state.final_h.begin(),state.final_h.end(),[](double x){return std::isfinite(x);}),"Nonfinite final grid publication");
        r.passed=true;
        if(r.shadow) {
            r.reference.insert(r.reference.end(),values,values+9);r.reference.insert(r.reference.end(),hs,hs+9);r.reference.insert(r.reference.end(),xs,xs+9);
            for(size_t i=0;i<r.values.size();i++) {
                require(std::isfinite(r.reference[i]),"Nonfinite final grid reference");
                double e=std::abs(r.values[i]-r.reference[i]);r.grid_error=std::max(r.grid_error,e);
                r.passed&=e<=1.e-9+1.e-10*std::abs(r.reference[i]);
            }
            for(size_t i=0;i<state.final_h.size();i++) {
                require(std::isfinite(h[i]),"Nonfinite final Hamiltonian reference");
                double e=std::abs(state.final_h[i]-h[i]);r.matrix_error=std::max(r.matrix_error,e);
                r.passed&=e<=1.e-10+1.e-10*std::abs(h[i]);
            }
            char name[96];std::snprintf(name,sizeof(name),"remag-final-grid-shadow-%04zu.bin",counts.final_grids.size()+1);r.matrix_file=name;
            FILE* f=std::fopen(name,"wb");require(f,"Cannot retain final Hamiltonian shadow");
            size_t n=state.final_h.size();bool written=std::fwrite(h,sizeof(double),n,f)==n&&std::fwrite(state.final_h.data(),sizeof(double),n,f)==n;
            std::fclose(f);require(written,"Incomplete final Hamiltonian shadow");
        } else {
            std::copy(state.final_h.begin(),state.final_h.end(),h);
            std::copy(g.energies,g.energies+9,values);std::copy(g.hartree_stress,g.hartree_stress+9,hs);std::copy(g.xc_stress,g.xc_stress+9,xs);
        }
        counts.final_grids.push_back(r);state.final_grid_used=true;
        if(!r.passed) counts.publish_final_grids(); // Fortran die aborts without C++ destructors.
        require(r.passed,"Final grid differs from pinned upstream recomputation");return 0;
    } catch(const std::exception& e) {return failure("final grid",e);}
}
extern "C" int remag_final_density_enabled() {
    const char* v=std::getenv("REMAG_FINAL_DENSITY");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_final_density_shadow_enabled() {
    const char* v=std::getenv("REMAG_FINAL_DENSITY_SHADOW");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_final_density_active() {return remag_final_density_enabled()||remag_final_density_shadow_enabled();}
extern "C" int remag_final_density(int kind,int capacity,int spins,int grid_spins,const int* mesh,int nsm,const double* density,double* rho) {
    try {
        current(Stage::exported);
        require(remag_final_density_active()&&remag_final_grid_enabled()&&state.final_grid_used&&
            remag::final_grid_forces_complete(state.epoch)&&(kind==1||kind==2)&&(state.final_rho_requested&kind)&&
            (kind==1?state.final_rho:state.final_drho)&&counts.densities.size()>=2&&capacity==state.capacity&&spins==state.nh&&grid_spins==state.eh&&
            mesh&&std::equal(mesh,mesh+3,state.grid_problem.pbe.mesh)&&nsm==state.grid_problem.nsm&&density&&rho&&
            state.final_grid_pending&&density==state.host_d,
            "Requested density export has stale/incomplete forces, shape mismatch or density identity mismatch");
        auto& r=counts.densities[counts.densities.size()-2+(kind-1)];
        require(r.epoch==state.epoch&&r.kind==kind&&r.requested&&!r.exported&&
            r.shadow==bool(remag_final_density_shadow_enabled()),"Duplicate or mismatched requested density export");
        const size_t n=r.retained/sizeof(double);
        std::vector<double> values(n);
        check(cudaMemcpy(values.data(),kind==1?state.final_rho:state.final_drho,r.retained,cudaMemcpyDeviceToHost));
        remag::transfer(remag::Transfer::final_density_output,false,r.retained);
        r.downloaded=r.retained;r.exported=true;r.passed=true;
        require(std::all_of(values.begin(),values.end(),[](double x){return std::isfinite(x);}),"Nonfinite requested density output");
        if(r.shadow) {
            for(size_t i=0;i<n;i++) {
                require(std::isfinite(rho[i]),"Nonfinite density output reference");
                double e=std::abs(values[i]-rho[i]);r.error=std::max(r.error,e);r.passed&=e<=1.e-10+1.e-10*std::abs(rho[i]);
            }
            char name[96];std::snprintf(name,sizeof(name),"remag-final-density-shadow-%04zu-%d.bin",counts.densities.size()/2,kind);r.file=name;
            FILE* f=std::fopen(name,"wb");require(f,"Cannot retain density shadow");
            bool written=std::fwrite(rho,sizeof(double),n,f)==n&&std::fwrite(values.data(),sizeof(double),n,f)==n;
            std::fclose(f);require(written,"Incomplete density shadow");
        } else std::copy(values.begin(),values.end(),rho);
        if(!r.passed) counts.publish_densities();
        require(r.passed,"Requested density differs from original recomputation");return 0;
    } catch(const std::exception& e) {return failure("final density",e);}
}
// Retained host diagnostics, callable only after final scientific publication.
// values: scalar[13],seed[3],fermi/entropy/charge[3],grid energies[9].
extern "C" int remag_cycle_diagnostic(int iteration,double* values,int* converged) {
    try {
        current(Stage::exported);require(values&&converged&&iteration>0&&iteration<=state.result.iterations,"Invalid cycle diagnostic request");
        const auto& d=state.diagnostics[iteration-1];std::copy(d.step.scalars.values,d.step.scalars.values+13,values);
        values[13]=d.hamiltonian.kinetic;values[14]=d.hamiltonian.nonlocal_energy;values[15]=d.hamiltonian.soc_energy;
        values[16]=d.fermi.fermi;values[17]=d.fermi.entropy;values[18]=d.fermi.charge;
        std::copy(d.grid.energies,d.grid.energies+9,values+19);*converged=d.step.control.convergence.converged;return 0;
    } catch(const std::exception& e) {return failure("diagnostic",e);}
}
