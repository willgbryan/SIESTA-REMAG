// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_cycle.hpp"
#include "block_jacobi.hpp"
#include "sparse_state.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace remag {
namespace {
void check(cudaError_t c) {if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c));}
void applied(int c) {check(static_cast<cudaError_t>(c));}
__global__ void profile_stamp(ScfStageProfile* profile,int stage,bool begin) {
    unsigned long long now;
    // NVIDIA PTX global nanosecond timer, used only for optional profiling on
    // the measured target. No scientific decision depends on this register.
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if(begin) profile->start=now;
    else {profile->nanoseconds[stage-1]+=now-profile->start;profile->calls[stage-1]++;}
}
__global__ void identity(int n,int complex_values,double* h,double* s) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*n) {
        if(complex_values) {h[2*i]=s[2*i]=i%n==i/n?1.:0.;h[2*i+1]=s[2*i+1]=0.;}
        else h[i]=s[i]=i%n==i/n?1.:0.;
    }
}
__global__ void start(ScfCycleResult* r,HamiltonianAssemblyConfig expected,const HamiltonianAssemblyConfig* assembly,
        const FermiProblem* f,const HamiltonianSeedsProblem* h,const ScfScalarsProblem* s,
        int capacity,const double* onsite,const cuDoubleComplex* offsite,cudaGraphConditionalHandle loop) {
    if(!r->finished) {
        const auto a=*assembly;
        bool valid=a.kind==expected.kind&&a.n==expected.n&&a.nnz==expected.nnz&&a.nh==expected.nh&&
            a.no==expected.no&&a.nk==expected.nk&&a.ldh==expected.ldh&&a.pairs==expected.pairs;
        const int dim=expected.n*(expected.kind>=2?2:1),spins=expected.kind<2?expected.nh:1;
        valid=valid&&f->maxe==dim&&f->ne==dim&&f->nk==expected.nk&&f->maxspn==spins&&
            (expected.nh<4?f->nspinor==(expected.nh==1?1:2):(f->nspinor==1||f->nspinor==2))&&h->capacity==capacity&&h->spins==expected.nh&&
            h->reserved==0&&h->soc>=0&&h->soc<=2&&(h->soc==0||expected.nh==8)&&
            (h->soc!=1||onsite)&&(h->soc!=2||offsite)&&s->capacity==capacity&&s->spins==expected.nh&&
            s->rows==expected.n&&s->reserved==0;
        if(!valid) {r->status=8;r->producer_status=1;r->finished=1;}
    }
    cudaGraphSetConditional(loop,!r->finished&&!r->status);
}
__global__ void gate(ScfCycleResult* r,int stage,int block,int pass,cudaGraphConditionalHandle active) {
    if(!r->status) {r->stage=stage;r->block=block;r->pass=pass;}
    cudaGraphSetConditional(active,!r->status);
}
__global__ void status(ScfCycleResult* r,const int* code,int stage) {
    if(*code&&!r->status) {r->status=stage;r->producer_status=*code;}
}
// Only this join writes the shared cycle status. A failed wave is reported in
// job order, after all private solver work has stopped; later waves are gated.
__global__ void wave_status(ScfCycleResult* r,const BlockJacobiResult* solved,int first,int count) {
    for(int lane=0;lane<count;lane++) if(solved[lane].status) {
        r->status=1;r->producer_status=solved[lane].status;r->stage=1;r->block=first+lane;r->pass=1;
        return;
    }
}
// Device k loop (REMAG_SCF_K_LOOP=1). A cursor names the first job of the
// current wave; lane L serves job cursor+L. Jobs past the end are clamped to
// the last job for assembly and solving, and nothing they produce is kept.
__global__ void loop_begin(int* cursor,int jobs,const ScfCycleResult* r,cudaGraphConditionalHandle loop) {
    *cursor=0;cudaGraphSetConditional(loop,jobs>0&&!r->status);
}
__global__ void loop_advance(int* cursor,int step,int jobs,const ScfCycleResult* r,cudaGraphConditionalHandle loop) {
    *cursor+=step;cudaGraphSetConditional(loop,*cursor<jobs&&!r->status);
}
__global__ void loop_proceed(const ScfCycleResult* r,cudaGraphConditionalHandle active) {
    cudaGraphSetConditional(active,!r->status);
}
// Report the first failed solve of a wave in job order, as wave_status.
__global__ void loop_status(ScfCycleResult* r,const BlockJacobiResult* solved,const int* cursor,int count,int jobs,int pass) {
    if(r->status) return;
    for(int lane=0;lane<count;lane++) {
        const int job=*cursor+lane;
        if(job<jobs&&solved[lane].status) {
            r->status=1;r->producer_status=solved[lane].status;r->stage=1;r->block=job;r->pass=pass;return;
        }
    }
}
// First-pass spectrum (and, with the bank, vectors) of a successful real job.
__global__ void loop_publish(const int* cursor,int lane,int jobs,int dim,size_t words,const double* w,double* eigenvalues,
        const unsigned long long* z,unsigned long long* bank,const BlockJacobiResult* solved) {
    const int job=*cursor+lane;
    if(job>=jobs||solved->status) return;
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<words||i<size_t(dim);i+=size_t(gridDim.x)*blockDim.x) {
        if(i<size_t(dim)) eigenvalues[size_t(job)*dim+i]=w[i];
        if(bank&&i<words) bank[size_t(job)*words+i]=z[i];
    }
}
// Batched solver: first-pass spectrum and vectors of every real lane at once (lane = blockIdx.y).
__global__ void loop_publish_wave(const int* cursor,int jobs,int dim,size_t words,const double* w,double* eigenvalues,
        const unsigned long long* z,unsigned long long* bank,const BlockJacobiResult* solved) {
    const int lane=blockIdx.y,job=*cursor+lane;
    if(job>=jobs||solved[lane].status) return;
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<words||i<size_t(dim);i+=size_t(gridDim.x)*blockDim.x) {
        if(i<size_t(dim)) eigenvalues[size_t(job)*dim+i]=w[size_t(lane)*dim+i];
        if(bank&&i<words) bank[size_t(job)*words+i]=z[size_t(lane)*words+i];
    }
}
// Batched solver status in the block-Jacobi record: 1 Cholesky failure, 3 eigensolver failure.
__global__ void batched_status(int lanes,const int* cholesky,const int* eigen,BlockJacobiResult* solved) {
    const int lane=blockIdx.x*blockDim.x+threadIdx.x;
    if(lane<lanes) {solved[lane]=BlockJacobiResult{};solved[lane].status=cholesky[lane]?1:(eigen[lane]?3:0);}
}
// Warm-start basis of the lane's job, moved through lane-private storage so the
// seeded solver keeps fixed pointers. Clamped duplicates start unseeded.
__global__ void loop_seed_gather(const int* cursor,int lane,int jobs,size_t words,const unsigned long long* seeds,
        const int* valid,unsigned long long* seed,int* seed_valid) {
    const int job=*cursor+lane;
    if(job>=jobs) {if(!blockIdx.x&&!threadIdx.x) *seed_valid=0;return;}
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x) seed[i]=seeds[size_t(job)*words+i];
    if(!blockIdx.x&&!threadIdx.x) *seed_valid=valid[job];
}
__global__ void loop_seed_scatter(const int* cursor,int lane,int jobs,size_t words,const unsigned long long* seed,
        const int* seed_valid,unsigned long long* seeds,int* valid) {
    const int job=*cursor+lane;
    if(job>=jobs) return;
    for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x) seeds[size_t(job)*words+i]=seed[i];
    if(!blockIdx.x&&!threadIdx.x) valid[job]=*seed_valid;
}
// Occupations, energies, k and spin of the lane's job for the staged density
// build. A job past the end contributes nothing: its occupations are zero.
__global__ void loop_stage(const int* cursor,int lane,int jobs,int spins,int dim,const double* kpoints,const double* occupation,
        const double* eigenvalues,double* q,double* e,DensityJobStage* stage) {
    const int job=*cursor+lane,used=min(job,jobs-1);const bool valid=job<jobs;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<dim;i+=gridDim.x*blockDim.x) {
        q[i]=valid?occupation[size_t(used)*dim+i]:0.;e[i]=eigenvalues[size_t(used)*dim+i];
    }
    if(!blockIdx.x&&!threadIdx.x) {
        const int k=used/spins;
        *stage={{kpoints[3*k],kpoints[3*k+1],kpoints[3*k+2]},used%spins,valid?1:0};
    }
}
// Batched solver: stages, occupations and energies of every lane of a wave.
__global__ void loop_stage_wave(const int* cursor,int jobs,int spins,int dim,const double* kpoints,const double* occupation,
        const double* eigenvalues,double* q,double* e,DensityJobStage* stage) {
    const int lane=blockIdx.y,job=*cursor+lane,used=min(job,jobs-1);const bool valid=job<jobs;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<dim;i+=gridDim.x*blockDim.x) {
        q[size_t(lane)*dim+i]=valid?occupation[size_t(used)*dim+i]:0.;e[size_t(lane)*dim+i]=eigenvalues[size_t(used)*dim+i];
    }
    if(!blockIdx.x&&!threadIdx.x) {
        const int k=used/spins;
        stage[lane]={{kpoints[3*k],kpoints[3*k+1],kpoints[3*k+2]},used%spins,valid?1:0};
    }
}
__global__ void loop_transpose_wave(int n,const cuDoubleComplex* bank,const cuDoubleComplex* solved,const int* cursor,int jobs,
        cuDoubleComplex* target) {
    __shared__ cuDoubleComplex tile[32][33];
    const int lane=blockIdx.z;const size_t matrix=size_t(n)*n;
    const cuDoubleComplex* source=bank?bank+size_t(min(*cursor+lane,jobs-1))*matrix:solved+size_t(lane)*matrix;
    cuDoubleComplex* out=target+size_t(lane)*matrix;
    int row=blockIdx.x*32+threadIdx.x,state=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(row<n&&state+j<n) tile[threadIdx.y+j][threadIdx.x]=source[size_t(state+j)*n+row];
    __syncthreads();
    row=blockIdx.x*32+threadIdx.y;state=blockIdx.y*32+threadIdx.x;
    for(int j=0;j<32;j+=8) if(row+j<n&&state<n) out[size_t(row+j)*n+state]=tile[threadIdx.x][threadIdx.y+j];
}
// The density transpose of density_matrix.cu, reading the job's bank slot (or
// the lane's solve) through the device cursor.
template<class T> __global__ void loop_transpose(int n,int nev,const T* bank,const T* solved,const int* cursor,int lane,int jobs,T* target) {
    __shared__ T tile[32][33];
    const T* source=bank?bank+size_t(min(*cursor+lane,jobs-1))*n*nev:solved;
    int row=blockIdx.x*32+threadIdx.x,state=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(row<n&&state+j<nev) tile[threadIdx.y+j][threadIdx.x]=source[size_t(state+j)*n+row];
    __syncthreads();
    row=blockIdx.x*32+threadIdx.y;state=blockIdx.y*32+threadIdx.x;
    for(int j=0;j<32;j+=8) if(row+j<n&&state<nev) target[size_t(row+j)*nev+state]=tile[threadIdx.x][threadIdx.y+j];
}
__global__ void initial_control(ScfCycleResult* r,ScfControlProblem* p,const InitialDensityResult* initial) {
    if(initial->status) {r->status=8;r->producer_status=initial->status;r->stage=0;}
    else if(initial->mix_first!=0&&initial->mix_first!=1) {r->status=8;r->producer_status=1;r->stage=0;}
    else p->mix_first=initial->mix_first;
}
__global__ void finish(ScfCycleResult* r,const ScfControlResult* control,const ScfControlStep* history,
        const FermiResult* fermi,const DensityNormalizationResult* normalization,const HamiltonianSeedsResult* hamiltonian,
        const GridPotentialResult* grid,ScfCycleStep* diagnostics,cudaGraphConditionalHandle loop) {
    // Failed physics never reaches convergence/mixing. Preserve the last
    // successful control record, not uninitialized current producer data.
    if(!r->status) {
        r->control=*control;r->iterations=control->records;
        if(control->records) diagnostics[control->records-1]={history[control->records-1],*fermi,*normalization,*hamiltonian,*grid};
        if(control->status) {r->status=7;r->producer_status=control->status;}
    }
    r->finished=r->status||!r->control.continue_loop;
    cudaGraphSetConditional(loop,!r->finished);
}
void audit(cudaGraph_t graph,int& nodes,int& forbidden) {
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> list(n);
    check(cudaGraphGetNodes(graph,list.data(),&n));
    for(auto node:list) {
        nodes++;cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeGraph) {cudaGraph_t child;check(cudaGraphChildGraphNodeGetGraph(node,&child));audit(child,nodes,forbidden);}
        else if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms p{};check(cudaGraphMemcpyNodeGetParams(node,&p));if(p.kind!=cudaMemcpyDeviceToDevice) forbidden++;
        } else if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty) forbidden++;
    }
}
}
struct ScfCyclePlan::Impl {
    ScfCycleShape shape;ScfCycleBindings b;ScfControlProblem settings;cudaStream_t stream;
    int dim,spins,jobs,ng,capacity,nh,eh,lanes=1,nodes=0,forbidden=0;
    bool full_vector_bank=false,warm_start=false,k_loop=false,batched=false;
    bool initialized_hamiltonian=false,launched=false;
    size_t bytes=0,vector_bytes,seed_bytes=0;std::vector<void*> allocations;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    std::vector<std::unique_ptr<BlockJacobiPlan>> solvers;
    std::unique_ptr<GridPotentialPlan> grid;
    std::unique_ptr<HamiltonianSeedsPlan> seeds;
    std::unique_ptr<ScfScalarsPlan> scalars;
    std::unique_ptr<ScfControlPlan> control;
    ScfCycleResult* result=nullptr;ScfControlProblem* problem=nullptr;ScfControlResult* decision=nullptr;ScfControlStep* history=nullptr;ScfCycleStep* diagnostics=nullptr;
    BlockJacobiResult* solved=nullptr;FermiResult* occupied=nullptr;DensityNormalizationResult* normalized=nullptr;
    HamiltonianSeedsResult* seeded=nullptr;GridPotentialResult* gridded=nullptr;ScfScalarsResult* reduced=nullptr;
    ScfEnergySeeds* energy_seeds=nullptr;
    ScfStageProfile* profile=nullptr;
    unsigned char *dense_h=nullptr,*dense_s=nullptr,*vectors=nullptr,*transposed=nullptr,*seed_vectors=nullptr;
    int* seed_valid=nullptr;
    // Device k loop: wave cursors, lane-private solver outputs and seeds, and
    // the staged job of the density build.
    int *cursor=nullptr,*lane_valid=nullptr;unsigned char *lane_vectors=nullptr,*lane_seeds=nullptr;
    double *lane_values=nullptr,*stage_q=nullptr,*stage_e=nullptr;DensityJobStage* stage=nullptr;
    // Batched solver (REMAG_SCF_BATCHED_SOLVER=1): one wave of generalized
    // problems per batched Cholesky, triangular solves and XsyevBatched call;
    // its eigenvectors stay in dense_h. lane_z names where a lane's vectors are.
    unsigned char* lane_z=nullptr;
    cusolverDnHandle_t bsolver=nullptr;cublasHandle_t bblas=nullptr;cusolverDnParams_t bparams=nullptr;
    void *bwork=nullptr,*bblas_work=nullptr;size_t bwork_bytes=0;int *bcholesky=nullptr,*beigen=nullptr;
    cuDoubleComplex **hpointers=nullptr,**spointers=nullptr;
    // Wave density build: transposed vectors, occupations, energies and stages of every lane.
    unsigned char* wave_vectors=nullptr;double *wave_q=nullptr,*wave_e=nullptr,*wave_unique=nullptr;DensityJobStage* wave_stage=nullptr;
    double* unique_density=nullptr;
    double *eigenvalues=nullptr,*occupation=nullptr,*second_values=nullptr,*partial=nullptr;
    double *hold=nullptr,*dold=nullptr,*eold=nullptr,*residual=nullptr,*rho=nullptr,*potential=nullptr;
    template<class T> T* allocate(size_t n,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));allocations.push_back(p);bytes+=n*sizeof(T);
        if(source) check(cudaMemcpyAsync(p,source,n*sizeof(T),cudaMemcpyHostToDevice,stream));
        else check(cudaMemsetAsync(p,0,n*sizeof(T),stream));return p;
    }
    Impl(const ScfCycleShape& s,const ScfCycleBindings& inputs,const ScfControlProblem& c,cudaStream_t execution)
        :shape(s),b(inputs),settings(c),stream(execution) {
        const auto& h=s.assembly.shape;const auto& d=s.density.shape;const auto& g=s.grid.shape;
        const char* looped=std::getenv("REMAG_SCF_K_LOOP");
        if(looped&&std::strcmp(looped,"0")&&std::strcmp(looped,"1"))throw std::runtime_error("REMAG_SCF_K_LOOP must be 0 or 1");
        k_loop=looped&&std::strcmp(looped,"1")==0;
        const char* batching=std::getenv("REMAG_SCF_BATCHED_SOLVER");
        if(batching&&std::strcmp(batching,"0")&&std::strcmp(batching,"1"))throw std::runtime_error("REMAG_SCF_BATCHED_SOLVER must be 0 or 1");
        batched=batching&&std::strcmp(batching,"1")==0;
        if(batched&&!k_loop)throw std::runtime_error("The batched solver needs REMAG_SCF_K_LOOP=1");
        if(k_loop&&h.kind%2==0)throw std::runtime_error("The device k loop needs a sampled-k calculation");
        const int max_k=k_loop?(1<<20):128;
        if(!k_loop&&h.nk>128)throw std::runtime_error("Device SCF unrolls at most 128 k points; set REMAG_SCF_K_LOOP=1");
        if(h.n<1||h.n>2048||d.nnz<1||d.nnz>INT_MAX/8||h.nk<1||h.nk>max_k||
           (h.kind<2?(h.nh!=1&&h.nh!=2):(h.nh!=4&&h.nh!=8))||!s.assembly.config)
            throw std::runtime_error("Invalid device SCF capacity");
        if(s.density.unique_pairs<0||s.density.unique_pairs>d.pairs||
           (s.density.unique_pairs&&(!s.density.unique_rows||!s.density.unique_cols||!s.density.pair_unique)))
            throw std::runtime_error("Invalid unique density layout");
        dim=h.n*(h.kind>=2?2:1);spins=h.kind<2?h.nh:1;jobs=h.nk*spins;ng=std::min(h.nh,4);
        const char* cache=std::getenv("REMAG_DEVICE_VECTOR_CACHE");
        if(cache&&std::strcmp(cache,"0")&&std::strcmp(cache,"1"))throw std::runtime_error("Invalid device vector-cache mode");
        full_vector_bank=h.kind%2==0||(cache&&std::strcmp(cache,"1")==0);
        const char* parallel=std::getenv("REMAG_SCF_SOLVER_LANES");
        if(parallel) {
            const bool wide=k_loop&&(!std::strcmp(parallel,"16")||!std::strcmp(parallel,"32")||
                                     !std::strcmp(parallel,"64")||!std::strcmp(parallel,"128")||
                                     (batched&&(!std::strcmp(parallel,"256")||!std::strcmp(parallel,"512")||
                                                !std::strcmp(parallel,"1024"))));
            if(std::strcmp(parallel,"1")&&std::strcmp(parallel,"2")&&std::strcmp(parallel,"4")&&std::strcmp(parallel,"8")&&!wide)
                throw std::runtime_error(k_loop?"REMAG_SCF_SOLVER_LANES must be 1, 2, 4, 8, 16, 32, 64 or 128":
                                                "REMAG_SCF_SOLVER_LANES must be 1, 2, 4 or 8");
            lanes=std::min(std::atoi(parallel),jobs);
        } else if(batched) lanes=std::min(256,jobs);
        // The k loop gives every lane private solver output, so lanes no
        // longer need bank slots; the unrolled path writes straight into them.
        if(lanes>1&&!full_vector_bank&&!k_loop)
            throw std::runtime_error("Parallel SCF solves require the complete device vector bank");
        const char* warm=std::getenv("REMAG_SCF_WARM_START");
        if(warm&&std::strcmp(warm,"0")&&std::strcmp(warm,"1"))throw std::runtime_error("REMAG_SCF_WARM_START must be 0 or 1");
        warm_start=warm&&std::strcmp(warm,"1")==0;
        if(warm_start&&!full_vector_bank)throw std::runtime_error("Eigensolver warm start requires the complete vector bank");
        if(warm_start&&batched)throw std::runtime_error("The batched solver has no warm start");
        capacity=d.nnz;nh=d.nh;eh=d.eh;vector_bytes=size_t(dim)*dim*(h.kind?sizeof(cuDoubleComplex):sizeof(double));
        const auto& m=c.mixer;const auto& v=c.convergence;
        if(!stream||dim<1||dim>4096||h.nk<1||h.nk>max_k||h.kind<0||h.kind>3||
           (h.kind%2==0&&h.nk!=1)||d.kind!=h.kind||d.n!=h.n||h.ldh!=capacity||h.nh!=nh||
           eh!=ng||g.nuo!=h.n||g.nnz!=capacity||g.np<1||g.nsp<1||
           s.mesh[0]<1||s.mesh[1]<1||s.mesh[2]<1||
           size_t(s.mesh[0])*s.mesh[1]*s.mesh[2]!=size_t(g.np)*g.nsp||
           s.kpoints.size()!=size_t(h.nk)||!std::isfinite(s.occupation_cutoff)||s.occupation_cutoff<0.||
           s.normalization.rows!=capacity||s.normalization.nh!=nh||s.normalization.eh!=eh||
           s.normalization.spinor!=std::min(nh,2)||
           !std::isfinite(s.normalization.charge)||s.normalization.charge<=0.||
           !std::isfinite(s.normalization.tolerance)||s.normalization.tolerance<0.||
           m.size!=capacity*nh||m.history<1||v.maximum<1||v.maximum>4096||v.minimum<0||v.minimum>v.maximum||
           !v.epoch||m.epoch!=v.epoch||v.criteria<1||v.criteria>31||v.reserved||c.reserved||
           (c.mix_first!=0&&c.mix_first!=1)||!std::isfinite(m.weight)||m.weight<=0.||m.weight>1.||
           !std::isfinite(m.linear_weight)||m.linear_weight<=0.||m.linear_weight>1.||
           !std::isfinite(m.condition)||m.condition<=0.||m.condition>=1.||
           !b.fermi||!b.weights||!b.hamiltonian||!b.kinetic||!b.nonlocal_h||!b.grid||!b.atomic_density||
           !b.core_density||!b.neutral_potential||!b.scalars||!b.constants||!b.num||!b.ptr||!b.overlap||
           !b.h||!b.density||!b.energy_density||b.h==b.density||b.h==b.energy_density||b.density==b.energy_density||
           (b.constraint.sites&&(h.kind<2||!b.constraint.orbital_field)))
            throw std::runtime_error("Invalid device SCF cycle shape/bindings/settings");
        for(double t:v.tolerances) if(!std::isfinite(t)||t<0.) throw std::runtime_error("Invalid SCF tolerance");
        for(const auto& k:s.kpoints) for(double x:k) if(!std::isfinite(x)) throw std::runtime_error("Invalid SCF k point");
        try {
            const char* profiling=std::getenv("REMAG_SCF_STAGE_PROFILE");
            if(profiling&&std::strcmp(profiling,"0")&&std::strcmp(profiling,"1"))
                throw std::runtime_error("Invalid SCF stage-profile mode");
            if(profiling&&std::strcmp(profiling,"1")==0)profile=allocate<ScfStageProfile>(1);
            problem=allocate<ScfControlProblem>(1,&settings);result=allocate<ScfCycleResult>(1);
            decision=allocate<ScfControlResult>(1);history=allocate<ScfControlStep>(v.maximum);diagnostics=allocate<ScfCycleStep>(v.maximum);
            solved=allocate<BlockJacobiResult>(lanes);occupied=allocate<FermiResult>(1);normalized=allocate<DensityNormalizationResult>(1);
            seeded=allocate<HamiltonianSeedsResult>(1);gridded=allocate<GridPotentialResult>(1);reduced=allocate<ScfScalarsResult>(1);
            energy_seeds=allocate<ScfEnergySeeds>(1);
            dense_h=allocate<unsigned char>(vector_bytes*lanes);dense_s=allocate<unsigned char>(vector_bytes*lanes);
            if(!k_loop||full_vector_bank)vectors=allocate<unsigned char>(vector_bytes*(full_vector_bank?jobs:1));
            transposed=allocate<unsigned char>(vector_bytes);
            if(k_loop) {
                cursor=allocate<int>(2);
                if(!batched)lane_vectors=allocate<unsigned char>(vector_bytes*lanes);
                lane_z=batched?dense_h:lane_vectors;
                lane_values=allocate<double>(size_t(dim)*lanes);stage_q=allocate<double>(dim);stage_e=allocate<double>(dim);
                stage=allocate<DensityJobStage>(1);
            }
            eigenvalues=allocate<double>(size_t(dim)*jobs);occupation=allocate<double>(size_t(dim)*jobs);second_values=allocate<double>(dim);
            if(s.density.unique_pairs)unique_density=allocate<double>(density_unique_scratch_count(s.density));
            partial=allocate<double>((size_t(capacity)*std::min(nh,2)+255)/256);
            hold=allocate<double>(size_t(capacity)*nh);dold=allocate<double>(size_t(capacity)*nh);
            eold=allocate<double>(size_t(capacity)*eh);residual=allocate<double>(size_t(capacity)*nh);
            rho=allocate<double>(size_t(g.np)*g.nsp*ng);potential=allocate<double>(size_t(g.np)*g.nsp*ng);
            if(batched) prepare_batched();
            else for(int lane=0;lane<lanes;lane++)
                solvers.push_back(std::make_unique<BlockJacobiPlan>(dim,h.kind!=0,32,2.e-13,stream));
            if(warm_start) {
                seed_bytes=solvers[0]->seed_bytes();
                // State is keyed by the physical job, not the reusable lane.
                // Geometry reconstruction creates fresh zero publication flags.
                seed_vectors=allocate<unsigned char>(seed_bytes*jobs);seed_valid=allocate<int>(jobs);
                if(k_loop) {lane_seeds=allocate<unsigned char>(seed_bytes*lanes);lane_valid=allocate<int>(lanes);}
            }
            grid=std::make_unique<GridPotentialPlan>(s.mesh[0],s.mesh[1],s.mesh[2],ng,stream);
            seeds=std::make_unique<HamiltonianSeedsPlan>(capacity,nh,stream);
            scalars=std::make_unique<ScfScalarsPlan>(capacity,nh,h.n,stream);
            control=std::make_unique<ScfControlPlan>(capacity*nh,m.history,v.maximum,stream);
            // Warm up only private scratch. No accepted H/D/E or convergence
            // history is advanced outside the measured graph region.
            for(int lane=0;lane<lanes;lane++) {
                auto h=dense_h+size_t(lane)*vector_bytes,s=dense_s+size_t(lane)*vector_bytes;
                identity<<<(dim*dim+255)/256,256,0,stream>>>(dim,shape.assembly.shape.kind!=0,reinterpret_cast<double*>(h),reinterpret_cast<double*>(s));
                check(cudaGetLastError());if(!batched) solvers[lane]->warmup(h,s,solved+lane);
            }
            if(batched) {batched_solve();check(cudaStreamSynchronize(stream));}
            applied(grid->apply(b.grid,rho,b.atomic_density,b.core_density,b.neutral_potential,potential,gridded));
            check(cudaStreamSynchronize(stream));build();
            for(const auto& solver:solvers)bytes+=solver->workspace_bytes();
            bytes+=grid->workspace_bytes()+seeds->workspace_bytes()+scalars->workspace_bytes()+control->workspace_bytes();
        } catch(...) {release();throw;}
    }
    void release() {
        if(stream) cudaStreamSynchronize(stream);
        if(bparams) cusolverDnDestroyParams(bparams);if(bsolver) cusolverDnDestroy(bsolver);if(bblas) cublasDestroy(bblas);
        bparams=nullptr;bsolver=nullptr;bblas=nullptr;
        if(exec) cudaGraphExecDestroy(exec);if(graph) cudaGraphDestroy(graph);exec=nullptr;graph=nullptr;
        control.reset();scalars.reset();seeds.reset();grid.reset();solvers.clear();
        for(void* p:allocations) cudaFree(p);allocations.clear();
    }
    ~Impl() {release();}
    cudaGraphNode_t capture(cudaGraph_t dst,cudaGraphNode_t dep,const std::function<void()>& work) {
        cudaGraph_t captured=nullptr;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        try {work();} catch(...) {cudaStreamEndCapture(stream,&captured);if(captured) cudaGraphDestroy(captured);throw;}
        check(cudaStreamEndCapture(stream,&captured));audit(captured,nodes,forbidden);
        if(forbidden) {cudaGraphDestroy(captured);throw std::runtime_error("Forbidden SCF cycle node");}
        cudaGraphNode_t node;auto code=cudaGraphAddChildGraphNode(&node,dst,dep?&dep:nullptr,dep?1:0,captured);
        cudaGraphDestroy(captured);check(code);nodes++;return node;
    }
    cudaGraphNode_t guarded(cudaGraph_t dst,cudaGraphNode_t dep,int stage,int block,int pass,
            const std::function<void(cudaGraph_t)>& work) {
        if(profile)dep=capture(dst,dep,[&]{profile_stamp<<<1,1,0,stream>>>(profile,stage,true);check(cudaGetLastError());});
        cudaGraphConditionalHandle active;check(cudaGraphConditionalHandleCreate(&active,graph,0,cudaGraphCondAssignDefault));
        auto entered=capture(dst,dep,[&]{gate<<<1,1,0,stream>>>(result,stage,block,pass,active);check(cudaGetLastError());});
        cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=active;
        p.conditional.type=cudaGraphCondTypeIf;p.conditional.size=1;cudaGraphNode_t node;
        check(cudaGraphAddNode(&node,dst,&entered,1,&p));nodes++;work(p.conditional.phGraph_out[0]);
        if(profile)node=capture(dst,node,[&]{profile_stamp<<<1,1,0,stream>>>(profile,stage,false);check(cudaGetLastError());});
        return node;
    }
    cudaGraphNode_t solve(cudaGraph_t dst,cudaGraphNode_t dep,int job,int pass) {
        return guarded(dst,dep,1,job,pass,[&](cudaGraph_t body){
            const int spin=job%spins,k=job/spins;
            auto assembled=capture(body,nullptr,[&]{check(assemble_hamiltonian_device(shape.assembly,b.h,spin,k,dense_h,dense_s,stream));});
            auto z=vectors+(full_vector_bank?size_t(job)*vector_bytes:0);
            auto w=pass==1?eigenvalues+size_t(job)*dim:second_values;
            auto done=warm_start?solvers[0]->append_seeded(graph,body,assembled,dense_h,dense_s,z,w,solved,
                seed_vectors+size_t(job)*seed_bytes,seed_valid+job):
                solvers[0]->append(graph,body,assembled,dense_h,dense_s,z,w,solved);
            capture(body,done,[&]{status<<<1,1,0,stream>>>(result,&solved->status,1);check(cudaGetLastError());});
        });
    }
    cudaGraphNode_t solve_wave(cudaGraph_t dst,cudaGraphNode_t dep,int first) {
        return guarded(dst,dep,1,first,1,[&](cudaGraph_t body){
            const int count=std::min(lanes,jobs-first);
            std::vector<cudaGraphNode_t> completed;
            for(int lane=0;lane<count;lane++) {
                const int job=first+lane,spin=job%spins,k=job/spins;
                auto h=dense_h+size_t(lane)*vector_bytes,s=dense_s+size_t(lane)*vector_bytes;
                // Capture on the construction stream; DAG dependencies, not
                // the capture stream, order execution. Each branch owns all
                // writable solver/assembly storage and its spectral output.
                auto assembled=capture(body,nullptr,[&]{check(assemble_hamiltonian_device(shape.assembly,b.h,spin,k,h,s,stream));});
                auto z=vectors+size_t(job)*vector_bytes;auto w=eigenvalues+size_t(job)*dim;
                completed.push_back(warm_start?solvers[lane]->append_seeded(graph,body,assembled,h,s,z,w,solved+lane,
                    seed_vectors+size_t(job)*seed_bytes,seed_valid+job):
                    solvers[lane]->append(graph,body,assembled,h,s,z,w,solved+lane));
            }
            cudaGraphNode_t joined;
            check(cudaGraphAddEmptyNode(&joined,body,completed.data(),completed.size()));nodes++;
            capture(body,joined,[&]{wave_status<<<1,1,0,stream>>>(result,solved,first,count);check(cudaGetLastError());});
        });
    }
    cudaGraphNode_t accumulate(cudaGraph_t dst,cudaGraphNode_t dep,int job) {
        return guarded(dst,dep,3,job,2,[&](cudaGraph_t body){capture(body,nullptr,[&]{
            const auto k=shape.kpoints[job/spins];auto z=vectors+(full_vector_bank?size_t(job)*vector_bytes:0);
            check(transpose_vectors_device(dim,dim,shape.assembly.shape.kind!=0,z,transposed,stream));
            check(accumulate_density_device(shape.density,dim,job%spins,transposed,occupation+size_t(job)*dim,
                eigenvalues+size_t(job)*dim,shape.occupation_cutoff,k[0],k[1],k[2],b.density,b.energy_density,stream,unique_density));
        });});
    }
    // Handles, pointer arrays and zeroed workspace for one wave; staged before capture.
    void prepare_batched() {
        if(shape.assembly.shape.kind%2==0) throw std::runtime_error("The batched solver needs complex k-point matrices");
        auto solver_ok=[](cusolverStatus_t c){if(c!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER batched setup failed");};
        auto blas_ok=[](cublasStatus_t c){if(c!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS batched setup failed");};
        solver_ok(cusolverDnCreate(&bsolver));solver_ok(cusolverDnSetStream(bsolver,stream));
        solver_ok(cusolverDnCreateParams(&bparams));
        blas_ok(cublasCreate(&bblas));blas_ok(cublasSetStream(bblas,stream));
        bblas_work=allocate<unsigned char>(4u<<20);blas_ok(cublasSetWorkspace(bblas,bblas_work,4u<<20));
        bcholesky=allocate<int>(lanes);beigen=allocate<int>(lanes);
        std::vector<cuDoubleComplex*> hp(lanes),sp(lanes);
        for(int lane=0;lane<lanes;lane++) {
            hp[lane]=reinterpret_cast<cuDoubleComplex*>(dense_h+size_t(lane)*vector_bytes);
            sp[lane]=reinterpret_cast<cuDoubleComplex*>(dense_s+size_t(lane)*vector_bytes);
        }
        hpointers=allocate<cuDoubleComplex*>(lanes,hp.data());spointers=allocate<cuDoubleComplex*>(lanes,sp.data());
        size_t device=0,host=0;
        solver_ok(cusolverDnXsyevBatched_bufferSize(bsolver,bparams,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,dim,CUDA_C_64F,
            dense_h,dim,CUDA_R_64F,lane_values,CUDA_C_64F,&device,&host,lanes));
        if(host) throw std::runtime_error("The batched eigensolver asks for host workspace");
        bwork_bytes=std::max<size_t>(device,8);bwork=allocate<unsigned char>(bwork_bytes);
        wave_vectors=allocate<unsigned char>(vector_bytes*lanes);
        wave_q=allocate<double>(size_t(dim)*lanes);wave_e=allocate<double>(size_t(dim)*lanes);
        wave_stage=allocate<DensityJobStage>(lanes);
        if(shape.density.unique_pairs) wave_unique=allocate<double>(density_unique_scratch_count(shape.density)*lanes);
    }
    // S = L L^H, C = L^-1 H L^-H, C y = e y, z = L^-H y for every lane; the
    // workspace is zeroed before each call (NVIDIA's complex batch note for CUDA < 12.8).
    void batched_solve() {
        const cuDoubleComplex one=make_cuDoubleComplex(1.,0.);
        auto solver_ok=[](cusolverStatus_t c){if(c!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER batched call failed");};
        auto blas_ok=[](cublasStatus_t c){if(c!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS batched call failed");};
        solver_ok(cusolverDnZpotrfBatched(bsolver,CUBLAS_FILL_MODE_LOWER,dim,spointers,dim,bcholesky,lanes));
        blas_ok(cublasZtrsmBatched(bblas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,dim,dim,&one,
            spointers,dim,hpointers,dim,lanes));
        blas_ok(cublasZtrsmBatched(bblas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_C,CUBLAS_DIAG_NON_UNIT,dim,dim,&one,
            spointers,dim,hpointers,dim,lanes));
        check(cudaMemsetAsync(bwork,0,bwork_bytes,stream));
        solver_ok(cusolverDnXsyevBatched(bsolver,bparams,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,dim,CUDA_C_64F,dense_h,dim,
            CUDA_R_64F,lane_values,CUDA_C_64F,bwork,bwork_bytes,nullptr,0,beigen,lanes));
        blas_ok(cublasZtrsmBatched(bblas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_C,CUBLAS_DIAG_NON_UNIT,dim,dim,&one,
            spointers,dim,hpointers,dim,lanes));
        batched_status<<<(lanes+127)/128,128,0,stream>>>(lanes,bcholesky,beigen,solved);check(cudaGetLastError());
    }
    // Device WHILE loop over waves of `lanes` jobs; wave(body) appends one wave.
    cudaGraphNode_t k_loop_node(cudaGraph_t dst,int slot,const std::function<void(cudaGraph_t,cudaGraphConditionalHandle)>& wave) {
        cudaGraphConditionalHandle loop;check(cudaGraphConditionalHandleCreate(&loop,graph,0,cudaGraphCondAssignDefault));
        auto begun=capture(dst,nullptr,[&]{loop_begin<<<1,1,0,stream>>>(cursor+slot,jobs,result,loop);check(cudaGetLastError());});
        cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=loop;
        p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;cudaGraphNode_t node;
        check(cudaGraphAddNode(&node,dst,&begun,1,&p));nodes++;wave(p.conditional.phGraph_out[0],loop);return node;
    }
    // Assemble and solve the wave's jobs in parallel lanes into lane storage.
    cudaGraphNode_t loop_solves(cudaGraph_t body,int slot,int pass) {
        const int* c=cursor+slot;const size_t words=vector_bytes/8,seed_words=seed_bytes/8;
        if(batched) {
            auto solved_wave=capture(body,nullptr,[&]{
                check(assemble_hamiltonian_wave_device(shape.assembly,b.h,c,lanes,jobs,dense_h,dense_s,stream));
                batched_solve();
                if(pass==1) {
                    loop_publish_wave<<<dim3(16,lanes),256,0,stream>>>(c,jobs,dim,full_vector_bank?words:0,lane_values,eigenvalues,
                        reinterpret_cast<const unsigned long long*>(dense_h),reinterpret_cast<unsigned long long*>(vectors),solved);
                    check(cudaGetLastError());
                }
            });
            return capture(body,solved_wave,[&]{loop_status<<<1,1,0,stream>>>(result,solved,c,lanes,jobs,pass);check(cudaGetLastError());});
        }
        const bool seeded=warm_start&&pass==1;
        std::vector<cudaGraphNode_t> completed;
        for(int lane=0;lane<lanes;lane++) {
            auto h=dense_h+size_t(lane)*vector_bytes,sm=dense_s+size_t(lane)*vector_bytes;
            auto z=lane_vectors+size_t(lane)*vector_bytes;auto w=lane_values+size_t(lane)*dim;
            auto seed=seeded?lane_seeds+size_t(lane)*seed_bytes:nullptr;auto valid=seeded?lane_valid+lane:nullptr;
            auto assembled=capture(body,nullptr,[&]{
                if(seeded) {
                    loop_seed_gather<<<64,256,0,stream>>>(c,lane,jobs,seed_words,reinterpret_cast<const unsigned long long*>(seed_vectors),
                        seed_valid,reinterpret_cast<unsigned long long*>(seed),valid);check(cudaGetLastError());
                }
                check(assemble_hamiltonian_indexed_device(shape.assembly,b.h,c,lane,jobs,h,sm,stream));
            });
            auto done=seeded?solvers[lane]->append_seeded(graph,body,assembled,h,sm,z,w,solved+lane,seed,valid):
                solvers[lane]->append(graph,body,assembled,h,sm,z,w,solved+lane);
            if(pass==1) done=capture(body,done,[&]{
                loop_publish<<<64,256,0,stream>>>(c,lane,jobs,dim,full_vector_bank?words:0,w,eigenvalues,
                    reinterpret_cast<const unsigned long long*>(z),reinterpret_cast<unsigned long long*>(vectors),solved+lane);
                check(cudaGetLastError());
                if(seeded) {
                    loop_seed_scatter<<<64,256,0,stream>>>(c,lane,jobs,seed_words,reinterpret_cast<const unsigned long long*>(seed),
                        valid,reinterpret_cast<unsigned long long*>(seed_vectors),seed_valid);check(cudaGetLastError());
                }
            });
            completed.push_back(done);
        }
        cudaGraphNode_t joined;check(cudaGraphAddEmptyNode(&joined,body,completed.data(),completed.size()));nodes++;
        return capture(body,joined,[&]{loop_status<<<1,1,0,stream>>>(result,solved,c,lanes,jobs,pass);check(cudaGetLastError());});
    }
    // Accumulate the wave's jobs one after another, in job order, as the
    // unrolled path does; skipped entirely after a failure.
    cudaGraphNode_t loop_accumulate(cudaGraph_t body,cudaGraphNode_t dep,int slot) {
        const int* c=cursor+slot;
        cudaGraphConditionalHandle active;check(cudaGraphConditionalHandleCreate(&active,graph,0,cudaGraphCondAssignDefault));
        auto entered=capture(body,dep,[&]{loop_proceed<<<1,1,0,stream>>>(result,active);check(cudaGetLastError());});
        cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=active;
        p.conditional.type=cudaGraphCondTypeIf;p.conditional.size=1;cudaGraphNode_t node;
        check(cudaGraphAddNode(&node,body,&entered,1,&p));nodes++;
        capture(p.conditional.phGraph_out[0],nullptr,[&]{
            const bool complex_values=shape.assembly.shape.kind!=0;
            if(batched) {
                // One pass for the whole wave, in job order (density_wave_kernels.inc).
                loop_stage_wave<<<dim3((dim+255)/256,lanes),256,0,stream>>>(c,jobs,spins,dim,shape.assembly.kpoints,occupation,
                    eigenvalues,wave_q,wave_e,wave_stage);check(cudaGetLastError());
                const dim3 block(32,8),tiles((dim-1)/32+1,(dim-1)/32+1,lanes);
                loop_transpose_wave<<<tiles,block,0,stream>>>(dim,full_vector_bank?reinterpret_cast<const cuDoubleComplex*>(vectors):nullptr,
                    reinterpret_cast<const cuDoubleComplex*>(lane_z),c,jobs,reinterpret_cast<cuDoubleComplex*>(wave_vectors));
                check(cudaGetLastError());
                check(accumulate_density_wave_device(shape.density,dim,lanes,wave_vectors,size_t(dim)*dim,wave_q,wave_e,
                    shape.occupation_cutoff,wave_stage,b.density,b.energy_density,stream,wave_unique));
                return;
            }
            const dim3 block(32,8),tiles((dim-1)/32+1,(dim-1)/32+1);
            for(int lane=0;lane<lanes;lane++) {
                loop_stage<<<(dim+255)/256,256,0,stream>>>(c,lane,jobs,spins,dim,shape.assembly.kpoints,occupation,eigenvalues,
                    stage_q,stage_e,stage);check(cudaGetLastError());
                auto solved_z=lane_z+size_t(lane)*vector_bytes;
                if(complex_values) loop_transpose<<<tiles,block,0,stream>>>(dim,dim,
                    full_vector_bank?reinterpret_cast<const cuDoubleComplex*>(vectors):nullptr,
                    reinterpret_cast<const cuDoubleComplex*>(solved_z),c,lane,jobs,reinterpret_cast<cuDoubleComplex*>(transposed));
                else loop_transpose<<<tiles,block,0,stream>>>(dim,dim,full_vector_bank?reinterpret_cast<const double*>(vectors):nullptr,
                    reinterpret_cast<const double*>(solved_z),c,lane,jobs,reinterpret_cast<double*>(transposed));
                check(cudaGetLastError());
                check(accumulate_density_staged_device(shape.density,dim,transposed,stage_q,stage_e,shape.occupation_cutoff,stage,
                    b.density,b.energy_density,stream,unique_density));
            }
        });
        return node;
    }
    void build() {
        check(cudaGraphCreate(&graph,0));cudaGraphConditionalHandle loop;
        check(cudaGraphConditionalHandleCreate(&loop,graph,0,cudaGraphCondAssignDefault));
        auto entered=capture(graph,nullptr,[&]{start<<<1,1,0,stream>>>(result,shape.assembly.shape,shape.assembly.config,b.fermi,b.hamiltonian,b.scalars,capacity,b.onsite,b.offsite,loop);check(cudaGetLastError());});
        cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=loop;
        p.conditional.type=cudaGraphCondTypeWhile;p.conditional.size=1;cudaGraphNode_t node;
        check(cudaGraphAddNode(&node,graph,&entered,1,&p));nodes++;auto body=p.conditional.phGraph_out[0];
        auto tail=capture(body,nullptr,[&]{
            check(cudaMemcpyAsync(hold,b.h,size_t(capacity)*nh*sizeof(double),cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(dold,b.density,size_t(capacity)*nh*sizeof(double),cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(eold,b.energy_density,size_t(capacity)*eh*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        });
        if(k_loop) tail=guarded(body,tail,1,-1,1,[&](cudaGraph_t stage){
            k_loop_node(stage,0,[&](cudaGraph_t wave,cudaGraphConditionalHandle loop){
                auto done=loop_solves(wave,0,1);
                capture(wave,done,[&]{loop_advance<<<1,1,0,stream>>>(cursor,lanes,jobs,result,loop);check(cudaGetLastError());});
            });
        });
        else if(lanes==1) for(int job=0;job<jobs;job++) tail=solve(body,tail,job,1);
        else for(int first=0;first<jobs;first+=lanes) tail=solve_wave(body,tail,first);
        tail=guarded(body,tail,2,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            if(b.tetra) applied(b.tetra->apply(b.fermi,eigenvalues,occupation,occupied));
            else check(fermi_device(b.fermi,eigenvalues,b.weights,occupation,occupied,stream));
            status<<<1,1,0,stream>>>(result,&occupied->status,2);check(cudaGetLastError());
        });});
        tail=guarded(body,tail,3,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            check(clear_density_device(shape.density.shape,b.density,b.energy_density,stream));
        });});
        if(k_loop) tail=guarded(body,tail,3,-1,2,[&](cudaGraph_t stage){
            k_loop_node(stage,1,[&](cudaGraph_t wave,cudaGraphConditionalHandle loop){
                auto done=full_vector_bank?nullptr:loop_solves(wave,1,2);
                done=loop_accumulate(wave,done,1);
                capture(wave,done,[&]{loop_advance<<<1,1,0,stream>>>(cursor+1,lanes,jobs,result,loop);check(cudaGetLastError());});
            });
        });
        else for(int job=0;job<jobs;job++) {
            if(!full_vector_bank) tail=solve(body,tail,job,2);
            tail=accumulate(body,tail,job);
        }
        tail=guarded(body,tail,3,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            check(normalize_density_device(shape.normalization,b.overlap,partial,normalized,b.density,b.energy_density,stream));
            status<<<1,1,0,stream>>>(result,&normalized->status,3);check(cudaGetLastError());
        });});
        tail=guarded(body,tail,4,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            applied(seeds->prepare(b.hamiltonian,b.kinetic,b.nonlocal_h,b.onsite,b.offsite,b.density,b.h,seeded));
            status<<<1,1,0,stream>>>(result,&seeded->status,4);check(cudaGetLastError());
        });});
        tail=guarded(body,tail,5,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            check(contract_density_device(shape.grid,ng,nh,b.density,rho,stream));
            applied(grid->apply(b.grid,rho,b.atomic_density,b.core_density,b.neutral_potential,potential,gridded));
            status<<<1,1,0,stream>>>(result,&gridded->status,5);check(cudaGetLastError());
        });});
        tail=guarded(body,tail,4,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            check(add_grid_potential_device(shape.grid,ng,nh,potential,0.,&b.grid->dvol,b.h,stream));
            applied(seeds->finish_grid(b.hamiltonian,b.offsite,b.h,seeded));
            if(b.constraint.sites) {
                if(b.constraint.penalty>0.) check(spin_constraint_penalty_field_device(shape.density.shape.n,nh,capacity,
                    b.num,b.ptr,b.overlap,b.density,b.constraint,stream));
                check(spin_constraint_field_device(shape.assembly,nh,b.constraint.orbital_field,b.h,stream));
            }
            status<<<1,1,0,stream>>>(result,&seeded->status,4);check(cudaGetLastError());
        });});
        tail=guarded(body,tail,6,-1,0,[&](cudaGraph_t stage){capture(stage,nullptr,[&]{
            check(compose_scf_energy_seeds_device(b.constants,seeded,gridded,occupied,energy_seeds,stream));
            applied(scalars->apply(b.scalars,energy_seeds,b.num,b.ptr,b.overlap,b.h,hold,b.density,dold,b.energy_density,eold,residual,reduced));
            status<<<1,1,0,stream>>>(result,&reduced->status,6);check(cudaGetLastError());
        });});
        tail=guarded(body,tail,7,-1,0,[&](cudaGraph_t stage){
            control->append(graph,stage,nullptr,problem,reduced,hold,residual,b.h,decision,history);
        });
        capture(body,tail,[&]{finish<<<1,1,0,stream>>>(result,decision,history,occupied,normalized,seeded,gridded,diagnostics,loop);check(cudaGetLastError());});
        for(const auto& solver:solvers) {nodes+=solver->graph_nodes();forbidden+=solver->forbidden_nodes();}
        nodes+=control->graph_nodes();forbidden+=control->forbidden_nodes();
        if(forbidden) throw std::runtime_error("Forbidden nested SCF cycle node");
        check(cudaGraphInstantiate(&exec,graph,0));check(cudaGraphUpload(exec,stream));check(cudaStreamSynchronize(stream));
    }
};
ScfCyclePlan::ScfCyclePlan(const ScfCycleShape& s,const ScfCycleBindings& b,const ScfControlProblem& p,cudaStream_t stream)
    :impl_(new Impl(s,b,p,stream)) {}
ScfCyclePlan::~ScfCyclePlan()=default;
cudaError_t ScfCyclePlan::launch() {impl_->launched=true;return cudaGraphLaunch(impl_->exec,impl_->stream);}
cudaError_t ScfCyclePlan::initialize_hamiltonian(const InitialDensityResult* initial) {
    auto& s=*impl_;
    if(!initial||s.launched||s.initialized_hamiltonian)return cudaErrorInvalidValue;
    s.initialized_hamiltonian=true;
    initial_control<<<1,1,0,s.stream>>>(s.result,s.problem,initial);
    auto& b=s.b;
    auto code=s.seeds->prepare(b.hamiltonian,b.kinetic,b.nonlocal_h,b.onsite,b.offsite,b.density,b.h,s.seeded);
    if(code)return static_cast<cudaError_t>(code);
    status<<<1,1,0,s.stream>>>(s.result,&s.seeded->status,4);
    auto error=contract_density_device(s.shape.grid,s.ng,s.nh,b.density,s.rho,s.stream);if(error!=cudaSuccess)return error;
    code=s.grid->apply(b.grid,s.rho,b.atomic_density,b.core_density,b.neutral_potential,s.potential,s.gridded);
    if(code)return static_cast<cudaError_t>(code);
    status<<<1,1,0,s.stream>>>(s.result,&s.gridded->status,5);
    error=add_grid_potential_device(s.shape.grid,s.ng,s.nh,s.potential,0.,&b.grid->dvol,b.h,s.stream);if(error!=cudaSuccess)return error;
    code=s.seeds->finish_grid(b.hamiltonian,b.offsite,b.h,s.seeded);if(code)return static_cast<cudaError_t>(code);
    if(b.constraint.sites) {
        if(b.constraint.penalty>0.) {
            error=spin_constraint_penalty_field_device(s.shape.density.shape.n,s.nh,s.capacity,b.num,b.ptr,b.overlap,
                b.density,b.constraint,s.stream);if(error!=cudaSuccess)return error;
        }
        error=spin_constraint_field_device(s.shape.assembly,s.nh,b.constraint.orbital_field,b.h,s.stream);if(error!=cudaSuccess)return error;
    }
    status<<<1,1,0,s.stream>>>(s.result,&s.seeded->status,4);
    return cudaGetLastError();
}
cudaGraphExec_t ScfCyclePlan::executable() const {return impl_->exec;}
const ScfCycleResult* ScfCyclePlan::result() const {return impl_->result;}
const ScfControlStep* ScfCyclePlan::history() const {return impl_->history;}
const ScfCycleStep* ScfCyclePlan::diagnostics() const {return impl_->diagnostics;}
const ScfStageProfile* ScfCyclePlan::stage_profile() const {return impl_->profile;}
const BlockJacobiResult* ScfCyclePlan::solver_diagnostic(int job) const {return impl_->solved+job%impl_->lanes;}
const BlockJacobiLocalFailure* ScfCyclePlan::solver_local_failure(int job) const {
    return impl_->solvers.empty()?nullptr:impl_->solvers.at(job%impl_->lanes)->local_failure();}
const void* ScfCyclePlan::solver_failed_local_input(int job) const {
    return impl_->solvers.empty()?nullptr:impl_->solvers.at(job%impl_->lanes)->failed_local_input();}
int ScfCyclePlan::solver_local_dimension() const {return impl_->solvers.empty()?0:impl_->solvers[0]->local_dimension();}
const void* ScfCyclePlan::assembled_hamiltonian(int job) const {return impl_->dense_h+size_t(job%impl_->lanes)*impl_->vector_bytes;}
const void* ScfCyclePlan::assembled_overlap(int job) const {return impl_->dense_s+size_t(job%impl_->lanes)*impl_->vector_bytes;}
const double* ScfCyclePlan::old_hamiltonian() const {return impl_->hold;}
const double* ScfCyclePlan::old_density() const {return impl_->dold;}
const double* ScfCyclePlan::old_energy_density() const {return impl_->eold;}
const double* ScfCyclePlan::mesh_density() const {return impl_->rho;}
const double* ScfCyclePlan::mesh_potential() const {return impl_->potential;}
const double* ScfCyclePlan::mesh_auxiliary_potential() const {return impl_->grid->auxiliary_potential();}
const double* ScfCyclePlan::mesh_deformation_density() const {return impl_->grid->deformation_density();}
const double* ScfCyclePlan::energies() const {return impl_->eigenvalues;}
const double* ScfCyclePlan::occupations() const {return impl_->occupation;}
const FermiResult* ScfCyclePlan::fermi_result() const {return impl_->occupied;}
size_t ScfCyclePlan::workspace_bytes() const {return impl_->bytes;}
size_t ScfCyclePlan::setup_upload_bytes() const {
    size_t bytes=sizeof(ScfControlProblem);
    for(const auto& solver:impl_->solvers)bytes+=solver->setup_upload_bytes();
    return bytes;
}
bool ScfCyclePlan::warm_start() const {return impl_->warm_start;}
bool ScfCyclePlan::k_loop() const {return impl_->k_loop;}
bool ScfCyclePlan::batched_solver() const {return impl_->batched;}
int ScfCyclePlan::solver_lanes() const {return impl_->lanes;}
int ScfCyclePlan::graph_nodes() const {return impl_->nodes;}
size_t ScfCyclePlan::vector_bank_bytes() const {return impl_->vector_bytes*(impl_->full_vector_bank?impl_->jobs:1);}
int ScfCyclePlan::eigensystem_jobs() const {return impl_->jobs;}
int ScfCyclePlan::solves_per_iteration() const {return impl_->jobs*(impl_->full_vector_bank?1:2);}
bool ScfCyclePlan::retains_all_vectors() const {return impl_->full_vector_bank;}
int ScfCyclePlan::forbidden_nodes() const {return impl_->forbidden;}
}
