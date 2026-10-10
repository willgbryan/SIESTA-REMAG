// SPDX-License-Identifier: GPL-3.0-or-later
// Explicit sparse SCF ownership. Host convergence is a temporary scalar boundary.
#include "resident.hpp"
#include "sparse_state.hpp"
#include "execution.hpp"
#include "hamiltonian_seeds.hpp"
#include "scf_scalars.hpp"
#include "pulay.hpp"
#include "grid_potential.hpp"
#include "occupations.hpp"
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
extern "C" int remag_dm_enabled();
extern "C" int remag_dm_normalization_enabled();
extern "C" int remag_hs_enabled();
extern "C" int remag_spectra_enabled();
extern "C" int remag_flow_enabled();
extern "C" int remag_state_active();
namespace {
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
bool enabled(const char* name) {const char* v=std::getenv(name);return v&&std::strcmp(v,"1")==0;}
struct Counts {
    unsigned long long setups=0,begins=0,densities=0,seeds=0,grids=0,finishes=0,reductions=0,mixes=0,exports=0;
    unsigned long long h2d=0,d2h=0,allocations=0,live=0,peak=0,checked=0,hs_borrows=0,density_borrows=0,first_mix_bypasses=0;
    size_t plan_bytes=0;
    ~Counts() {
        if(FILE* f=std::fopen("remag-sparse-state.json","w")) {
            std::fprintf(f,"{\"setups\":%llu,\"iterations\":%llu,\"density_publications\":%llu,\"seed_prepares\":%llu,"
                "\"grid_commits\":%llu,\"hamiltonian_finishes\":%llu,\"scalar_reductions\":%llu,\"mixes\":%llu,"
                "\"exports\":%llu,\"host_to_device_bytes\":%llu,\"device_to_host_bytes\":%llu,"
                "\"bridge_allocations\":%llu,\"peak_owned_buffer_bytes\":%llu,\"plan_workspace_bytes\":%zu,"
                "\"debug_host_validation_bytes\":%llu,\"hs_borrows\":%llu,\"density_borrows\":%llu,\"first_mix_bypasses\":%llu,"
                "\"scope\":\"owned sparse H/D/E; temporary 144-byte scalar summary and 4-byte mixer status; final array exports\"}\n",
                setups,begins,densities,seeds,grids,finishes,reductions,mixes,exports,h2d,d2h,allocations,peak,
                plan_bytes,checked,hs_borrows,density_borrows,first_mix_bypasses);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr;size_t bytes=0;
    void reserve(size_t n) {
        if(n<=bytes) return;
        if(p) {check(cudaFree(p));counts.live-=bytes;p=nullptr;bytes=0;}
        check(cudaMalloc(&p,n));bytes=n;counts.live+=n;counts.peak=std::max(counts.peak,counts.live);counts.allocations++;
    }
    void upload(const void* source,size_t n) {
        reserve(n);check(cudaMemcpy(p,source,n,cudaMemcpyHostToDevice));counts.h2d+=n;
        remag::transfer(remag::Transfer::sparse_state_setup,true,n);
    }
    ~Buffer() {if(p) cudaFree(p);}
};
struct Snapshot {
    const void* source=nullptr;std::vector<unsigned char> bytes;
    void save(const void* p,size_t n) {
        source=p;bytes.clear();if(enabled("REMAG_GRID_VERIFY_EPOCH")&&n) {
            bytes.resize(n);std::memcpy(bytes.data(),p,n);
        }
    }
    bool matches() const {
        counts.checked+=bytes.size();return bytes.empty()||std::memcmp(source,bytes.data(),bytes.size())==0;
    }
};
enum class Stage {empty,ready,solving,density,seeded,grid_writing,grid_ready,hamiltonian,reduced,mixed,exported,failed};
struct State {
    Stage stage=Stage::empty;uint64_t epoch=0,lifetime=0;
    int n=0,spins=0,rows=0,history=0,iteration=0;bool initial_energy=false,mix_first=true,hs_used=false,rho_used=false;
    size_t matrix_bytes=0,energy_bytes=0;
    std::array<double*,6> host{}; // H, Hold, D, Dold, E, Eold
    std::array<Snapshot,12> snapshots; // six invariant arrays and all six opaque host arrays
    std::unique_ptr<remag::HamiltonianSeedsPlan> seed_plan;
    std::unique_ptr<remag::ScfScalarsPlan> scalar_plan;
    std::unique_ptr<remag::PulayPlan> mixer;
    Buffer h,hold,d,dold,e,eold,residual,kin,vkb,on,off,num,ptr,overlap;
    Buffer seed_problem,scalar_problem,mix_problem,constants,energy_seeds,seed_result,scalar_result,mix_result,summary;
} state;
struct Summary {double values[16];int status[4];};
static_assert(sizeof(Summary)==144,"Unexpected sparse summary ABI");
bool held() {return state.stage!=Stage::empty&&state.stage!=Stage::exported;}
void ownership(Stage stage) {
    if(!held()||state.stage!=stage||state.epoch!=remag::geometry_epoch())
        throw std::runtime_error("Sparse SCF owner stage/geometry mismatch");
    for(const auto& snapshot:state.snapshots) if(!snapshot.matches())
        throw std::runtime_error("Sparse SCF host input changed inside its lifetime");
}
void copy_device(Buffer& out,const Buffer& in,size_t bytes) {check(cudaMemcpyAsync(out.p,in.p,bytes,cudaMemcpyDeviceToDevice));}
template<class T> T* device(Buffer& b) {return static_cast<T*>(b.p);}

__global__ void compose_seeds(const remag::ScfEnergySeeds* constants,const remag::HamiltonianSeedsResult* h,
        const remag::GridPotentialResult* grid,const remag::FermiResult* fermi,remag::ScfEnergySeeds* out) {
    *out=*constants;
    out->kinetic=h->kinetic;out->nonlocal_energy=h->nonlocal_energy;out->soc=h->soc_energy;
    out->enascf=grid->energies[4];out->enaatm=grid->energies[3];out->hartree=grid->energies[1];
    out->xc=grid->energies[5]+grid->energies[6];out->entropy=fermi->entropy;
    // Keep component errors on device; the scalar plan will refuse the record.
    if(h->status||h->stage!=2||grid->status||fermi->status) out->kinetic=nan("");
}
__global__ void summarize(const remag::ScfScalarsResult* scalars,const remag::HamiltonianSeedsResult* h,
        const remag::GridPotentialResult* grid,const remag::FermiResult* fermi,Summary* result) {
    for(int i=0;i<13;i++) result->values[i]=scalars->values[i];
    result->values[13]=h->kinetic;result->values[14]=h->nonlocal_energy;result->values[15]=h->soc_energy;
    result->status[0]=scalars->status;result->status[1]=h->status;
    result->status[2]=grid->status;result->status[3]=fermi->status;
}
__global__ void undo_first_mix(size_t n,const double* input,const double* residual,double* output,
        const remag::PulayResult* result) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n&&result->status==0) output[i]=input[i]+residual[i];
}
}

cudaError_t remag::compose_scf_energy_seeds_device(const ScfEnergySeeds* constants,
        const HamiltonianSeedsResult* hamiltonian,const GridPotentialResult* grid,
        const FermiResult* fermi,ScfEnergySeeds* output,cudaStream_t stream) {
    if(!constants||!hamiltonian||!grid||!fermi||!output||constants==output) return cudaErrorInvalidValue;
    compose_seeds<<<1,1,0,stream>>>(constants,hamiltonian,grid,fermi,output);
    return cudaGetLastError();
}

extern "C" int remag_state_enabled() {return enabled("REMAG_RESIDENT_SCF");}
extern "C" int remag_state_active() {return remag_state_enabled()&&remag::in_scf();}

extern "C" int remag_state_prepare(int n,int spins,int rows,int soc,int history,double weight,double linear_weight,double condition,
        int initial_energy,int mix_first,const int* num,const int* ptr,const double* overlap,const double* kin,const double* vkb,
        const double* on,const cuDoubleComplex* off,const double* constants,
        double* h,double* hold,double* d,double* dold,double* e,double* eold) {
    try {
        if(!remag_state_enabled()||remag::in_scf()||
           (held()&&!(state.stage==Stage::failed&&state.epoch!=remag::geometry_epoch()))||!remag::geometry_epoch()||
           !remag_gpu_enabled()||!remag_dm_enabled()||!remag_dm_normalization_enabled()||!remag_hs_enabled()||
           !remag_spectra_enabled()||!remag_flow_enabled()||
           n<1||rows<1||rows>n||(spins!=1&&spins!=2&&spins!=4&&spins!=8)||n>INT_MAX/spins||
           (spins==8?(soc!=1&&soc!=2):soc!=0)||history<2||history>16||
           !std::isfinite(weight)||weight<=0.||weight>1.||!std::isfinite(linear_weight)||linear_weight<=0.||linear_weight>1.||
           !std::isfinite(condition)||condition<=0.||condition>=1.||
           (initial_energy!=0&&initial_energy!=1)||(mix_first!=0&&mix_first!=1)||!num||!ptr||!overlap||!kin||!vkb||!constants||
           (soc==1&&!on)||(soc==2&&!off)||!h||!hold||!d||!dold||!e||(!eold&&initial_energy)) return -210;
        std::array<double*,6> hosts{h,hold,d,dold,e,eold};
        for(size_t i=0;i<hosts.size();i++) for(size_t j=0;j<i;j++) if(hosts[i]&&hosts[i]==hosts[j]) return -210;
        for(int i=0;i<4;i++) if(!std::isfinite(constants[i])) return -210;
        if(constants[3]<0.) return -210;
        for(int i=0;i<rows;i++) if(num[i]<0||ptr[i]<0||static_cast<long long>(num[i])+ptr[i]>n) return -210;
        state.stage=Stage::failed;
        if(!state.seed_plan||n!=state.n||spins!=state.spins) {
            state.seed_plan=std::make_unique<remag::HamiltonianSeedsPlan>(n,spins,nullptr);
        }
        if(!state.scalar_plan||n!=state.n||spins!=state.spins||rows!=state.rows)
            state.scalar_plan=std::make_unique<remag::ScfScalarsPlan>(n,spins,rows,nullptr);
        if(!state.mixer||n!=state.n||spins!=state.spins||history!=state.history)
            state.mixer=std::make_unique<remag::PulayPlan>(n*spins,history,nullptr);
        counts.plan_bytes=state.seed_plan->workspace_bytes()+state.scalar_plan->workspace_bytes()+state.mixer->workspace_bytes();
        const size_t scalar_bytes=size_t(n)*sizeof(double),matrix=scalar_bytes*spins,energy=scalar_bytes*std::min(spins,4);
        remag::HamiltonianSeedsProblem hp{n,spins,soc,0};
        remag::ScfScalarsProblem sp{n,spins,rows,0};
        remag::PulayProblem mp{n*spins,history,weight,linear_weight,condition,++state.lifetime};
        remag::ScfEnergySeeds es{};es.ena=constants[0];es.ions=constants[1];es.madelung=constants[2];es.temperature=constants[3];
        state.seed_problem.upload(&hp,sizeof(hp));state.scalar_problem.upload(&sp,sizeof(sp));state.mix_problem.upload(&mp,sizeof(mp));
        state.constants.upload(&es,sizeof(es));state.energy_seeds.reserve(sizeof(es));
        state.seed_result.reserve(sizeof(remag::HamiltonianSeedsResult));state.scalar_result.reserve(sizeof(remag::ScfScalarsResult));
        state.mix_result.reserve(sizeof(remag::PulayResult));state.summary.reserve(sizeof(Summary));
        state.kin.upload(kin,scalar_bytes);state.vkb.upload(vkb,scalar_bytes);state.overlap.upload(overlap,scalar_bytes);
        state.num.upload(num,size_t(rows)*sizeof(int));state.ptr.upload(ptr,size_t(rows)*sizeof(int));
        if(soc==1) state.on.upload(on,6*scalar_bytes);
        if(soc==2) state.off.upload(off,8*scalar_bytes);
        state.h.upload(h,matrix);state.d.upload(d,matrix);
        state.hold.reserve(matrix);state.dold.reserve(matrix);state.eold.reserve(energy);state.residual.reserve(matrix);
        if(initial_energy) state.e.upload(e,energy);
        else {state.e.reserve(energy);check(cudaMemset(state.e.p,0,energy));}
        std::array<const void*,12> sources{kin,vkb,overlap,num,ptr,soc==1?static_cast<const void*>(on):static_cast<const void*>(off),h,hold,d,dold,e,eold};
        std::array<size_t,12> sizes{scalar_bytes,scalar_bytes,scalar_bytes,size_t(rows)*sizeof(int),size_t(rows)*sizeof(int),
                                   soc==1?6*scalar_bytes:soc==2?8*scalar_bytes:0,matrix,matrix,matrix,matrix,energy,eold?energy:0};
        for(size_t i=0;i<sources.size();i++) state.snapshots[i].save(sources[i],sizes[i]);
        state.host=hosts;state.n=n;state.spins=spins;state.rows=rows;state.history=history;state.initial_energy=initial_energy;
        state.mix_first=mix_first!=0;
        state.matrix_bytes=matrix;state.energy_bytes=energy;state.epoch=remag::geometry_epoch();state.iteration=0;
        state.stage=Stage::ready;counts.setups++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse prepare: %s\n",e.what());return -211;}
}

extern "C" int remag_state_begin(int iteration) {
    try {
        if(!remag_state_active()||iteration!=state.iteration+1||
           (state.stage!=Stage::ready&&state.stage!=Stage::mixed)) return -212;
        ownership(state.stage);
        copy_device(state.hold,state.h,state.matrix_bytes);copy_device(state.dold,state.d,state.matrix_bytes);
        copy_device(state.eold,state.e,state.energy_bytes);state.iteration=iteration;
        state.hs_used=false;state.rho_used=false;
        state.stage=Stage::solving;counts.begins++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse begin: %s\n",e.what());return -213;}
}

const double* remag_state_hamiltonian(const double* host,size_t bytes) {
    if(!remag_state_active()) return nullptr;
    ownership(Stage::solving);
    if(host!=state.host[0]||bytes!=state.matrix_bytes||state.hs_used) throw std::runtime_error("Sparse H borrower shape/identity mismatch");
    state.hs_used=true;
    counts.hs_borrows++;return device<double>(state.h);
}
bool remag_state_publish_density(double* host_d,double* host_e,const double* d,const double* e,size_t db,size_t eb) {
    if(!remag_state_active()) return false;
    ownership(Stage::solving);
    if(host_d!=state.host[2]||host_e!=state.host[4]||db!=state.matrix_bytes||eb!=state.energy_bytes||!d||!e||!state.hs_used)
        throw std::runtime_error("Sparse D/E publication shape/identity mismatch");
    check(cudaMemcpyAsync(state.d.p,d,db,cudaMemcpyDeviceToDevice));check(cudaMemcpyAsync(state.e.p,e,eb,cudaMemcpyDeviceToDevice));
    state.stage=Stage::density;counts.densities++;return true;
}
const double* remag_state_density(const double* host,size_t bytes) {
    if(!remag_state_active()) return nullptr;
    ownership(Stage::seeded);
    if(host!=state.host[2]||bytes!=state.matrix_bytes||state.rho_used) throw std::runtime_error("Sparse density borrower shape/identity mismatch");
    state.rho_used=true;
    counts.density_borrows++;return device<double>(state.d);
}
extern "C" int remag_state_seed() {
    try {
        if(!remag_state_active()) return -214;
        ownership(Stage::density);
        check(static_cast<cudaError_t>(state.seed_plan->prepare(device<remag::HamiltonianSeedsProblem>(state.seed_problem),
            device<double>(state.kin),device<double>(state.vkb),device<double>(state.on),device<cuDoubleComplex>(state.off),
            device<double>(state.d),device<double>(state.h),device<remag::HamiltonianSeedsResult>(state.seed_result))));
        state.stage=Stage::seeded;counts.seeds++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse seed: %s\n",e.what());return -215;}
}
double* remag_state_grid_target(double* host,size_t bytes) {
    if(!remag_state_active()) return nullptr;
    ownership(Stage::seeded);
    if(host!=state.host[0]||bytes!=state.matrix_bytes||!state.rho_used) throw std::runtime_error("Sparse grid borrower shape/identity mismatch");
    state.stage=Stage::grid_writing;return device<double>(state.h);
}
void remag_state_grid_ready(double* host,size_t bytes) {
    ownership(Stage::grid_writing);
    if(host!=state.host[0]||bytes!=state.matrix_bytes) throw std::runtime_error("Sparse grid commit shape/identity mismatch");
    state.stage=Stage::grid_ready;counts.grids++;
}
extern "C" int remag_state_finish_hamiltonian() {
    try {
        if(!remag_state_active()) return -216;
        ownership(Stage::grid_ready);
        check(static_cast<cudaError_t>(state.seed_plan->finish_grid(device<remag::HamiltonianSeedsProblem>(state.seed_problem),
            device<cuDoubleComplex>(state.off),device<double>(state.h),device<remag::HamiltonianSeedsResult>(state.seed_result))));
        state.stage=Stage::hamiltonian;counts.finishes++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse finish: %s\n",e.what());return -217;}
}
int remag_state_reduce_device(const remag::GridPotentialResult* grid,const remag::FermiResult* fermi,double* values) {
    try {
        if(!remag_state_active()||!values) return -218;
        ownership(Stage::hamiltonian);
        if(!grid||!fermi) throw std::runtime_error("Missing device energy producer");
        check(remag::compose_scf_energy_seeds_device(device<remag::ScfEnergySeeds>(state.constants),
            device<remag::HamiltonianSeedsResult>(state.seed_result),grid,fermi,
            device<remag::ScfEnergySeeds>(state.energy_seeds),nullptr));
        check(static_cast<cudaError_t>(state.scalar_plan->apply(device<remag::ScfScalarsProblem>(state.scalar_problem),
            device<remag::ScfEnergySeeds>(state.energy_seeds),device<int>(state.num),device<int>(state.ptr),device<double>(state.overlap),
            device<double>(state.h),device<double>(state.hold),device<double>(state.d),device<double>(state.dold),
            device<double>(state.e),device<double>(state.eold),device<double>(state.residual),device<remag::ScfScalarsResult>(state.scalar_result))));
        summarize<<<1,1>>>(device<remag::ScfScalarsResult>(state.scalar_result),device<remag::HamiltonianSeedsResult>(state.seed_result),
                           grid,fermi,device<Summary>(state.summary));check(cudaGetLastError());
        Summary result{};check(cudaMemcpy(&result,state.summary.p,sizeof(result),cudaMemcpyDeviceToHost));counts.d2h+=sizeof(result);
        remag::transfer(remag::Transfer::sparse_state_scalars,false,sizeof(result));
        for(int status:result.status) if(status) {
            state.stage=Stage::failed;
            std::fprintf(stderr,"REMAG sparse component status: scalars=%d Hamiltonian=%d grid=%d FD=%d\n",
                result.status[0],result.status[1],result.status[2],result.status[3]);
            return -219;
        }
        std::copy(result.values,result.values+16,values);state.stage=Stage::reduced;counts.reductions++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse scalars: %s\n",e.what());return -220;}
}
extern "C" int remag_state_scalars(double* values) {
    try {return remag_state_reduce_device(remag_flow_result(),remag_fermi_result(),values);}
    catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse producers: %s\n",e.what());return -220;}
}
extern "C" int remag_state_mix() {
    try {
        if(!remag_state_active()) return -221;
        ownership(Stage::reduced);
        check(static_cast<cudaError_t>(state.mixer->apply(device<remag::PulayProblem>(state.mix_problem),device<double>(state.hold),
            device<double>(state.residual),device<double>(state.h),device<remag::PulayResult>(state.mix_result))));
        if(state.iteration==1&&!state.mix_first) {
            // Upstream still records the first Pulay history, then reconstructs
            // Xout = Xin + F. Skipping apply would change all later mixing.
            size_t n=size_t(state.n)*state.spins;
            undo_first_mix<<<(n+255)/256,256>>>(n,device<double>(state.hold),device<double>(state.residual),
                device<double>(state.h),device<remag::PulayResult>(state.mix_result));
            check(cudaGetLastError());counts.first_mix_bypasses++;
        }
        int status=0;check(cudaMemcpy(&status,state.mix_result.p,sizeof(status),cudaMemcpyDeviceToHost));counts.d2h+=sizeof(status);
        remag::transfer(remag::Transfer::sparse_state_mixer_status,false,sizeof(status));
        if(status) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse Pulay status: %d\n",status);return -222;}
        state.stage=Stage::mixed;counts.mixes++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse mix: %s\n",e.what());return -223;}
}
extern "C" int remag_state_export() {
    try {
        if(!remag_state_enabled()||remag::in_scf()||
           (state.stage!=Stage::reduced&&state.stage!=Stage::mixed)) return -224;
        ownership(state.stage);
        std::array<Buffer*,6> arrays{&state.h,&state.hold,&state.d,&state.dold,&state.e,&state.eold};
        for(size_t i=0;i<arrays.size();i++) {
            // Upstream allocates host Eold only when EDM convergence is used.
            // Device history exists in both modes for the scalar plan.
            if(!state.host[i]) continue;
            const size_t bytes=i<4?state.matrix_bytes:state.energy_bytes;
            check(cudaMemcpy(state.host[i],arrays[i]->p,bytes,cudaMemcpyDeviceToHost));counts.d2h+=bytes;
            remag::transfer(remag::Transfer::sparse_state_export,false,bytes);
        }
        state.stage=Stage::exported;counts.exports++;return 0;
    } catch(const std::exception& e) {state.stage=Stage::failed;std::fprintf(stderr,"REMAG sparse export: %s\n",e.what());return -225;}
}
