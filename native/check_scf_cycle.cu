// SPDX-License-Identifier: GPL-3.0-or-later
// Independent owned-fixture adapter; does not borrow production global owners.
#include "scf_cycle.hpp"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

extern "C" int remag_phase_enter(int,int,int);
extern "C" void remag_activity_finish() __attribute__((weak));
extern "C" void remag_activity_graph(void*) __attribute__((weak));
struct Problem {
    int n,kind,nh,nk,maximum,minimum,mix_first,soc,fault,converge;
    double charge;
    remag::GridPotentialProblem grid;
};
struct Check {size_t workspace_bytes,staged_bytes,exported_bytes;int graph_nodes,forbidden_nodes,replay_equal;};
static_assert(sizeof(Problem)==272&&sizeof(Check)==40,"Unexpected SCF cycle checker ABI");
namespace {
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
void phase(int code,int iteration) {if(remag_phase_enter(code,1,iteration)) throw std::runtime_error("Invalid cycle phase");}
struct Context {
    cudaStream_t stream=nullptr;std::vector<void*> buffers;size_t staged=0;
    Context() {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context() {if(stream) cudaStreamSynchronize(stream);for(auto p:buffers) cudaFree(p);if(stream) cudaStreamDestroy(stream);}
    template<class T> T* allocate(size_t n,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));buffers.push_back(p);
        if(source) {check(cudaMemcpy(p,source,n*sizeof(T),cudaMemcpyHostToDevice));staged+=n*sizeof(T);}
        else check(cudaMemsetAsync(p,0,n*sizeof(T),stream));return p;
    }
};
}
// inputs: H,D,E,S,kinetic,nonlocal,onsite,offsite,phi,kpoints,weights,xyz,
//         atomic,core,neutral,energy_constants. All sizes fixed by Problem.
// outputs: H,D,E,Ebank,Qbank,FermiResult,CycleResult,history,diagnostics.
extern "C" int remag_check_scf_cycle(const Problem* p,const void* const* input,void* const* output,Check* info) {
    try {
        if(!p||!input||!output||!info||p->n<1||p->n>64||p->maximum<1||p->maximum>20||p->nk<1||p->nk>8||
           p->kind<0||p->kind>3||(p->kind<2?(p->nh!=1&&p->nh!=2):(p->nh!=4&&p->nh!=8))) return -1;
        *info={};phase(0,0);Context c;const int n=p->n,nh=p->nh,eh=nh<4?nh:4,used=n*n,cap=used+2;
        const int dim=n*(p->kind>=2?2:1),jobs=p->nk*(p->kind<2?nh:1);
        const auto& mesh=p->grid.pbe.mesh;const int size=mesh[0]*mesh[1]*mesh[2],nsp=p->grid.nsm*p->grid.nsm*p->grid.nsm,points=size/nsp;
        if(size<1||size>4096||nsp<1||size%nsp) return -1;
        std::vector<int> num(n,n),ptr(n),rows(used),cols(used),index(used),ends(points+1),orbs(points*n),lookup(used);
        std::vector<remag::HamiltonianPair> pairs(used);
        for(int i=0;i<n;i++) ptr[i]=i*n;
        for(int i=0;i<used;i++) {rows[i]=i/n;cols[i]=i%n;index[i]=i;pairs[i]={i%n,i/n,i,i+1};lookup[i]=i;}
        for(int i=0;i<=points;i++) ends[i]=i*n;
        for(int i=0;i<points*n;i++) orbs[i]=i%n+1;
        int zero=0;auto ds=c.allocate<double>(cap,static_cast<const double*>(input[3]));
        remag::ScfCycleShape s{};remag::HamiltonianAssemblyConfig hc{p->kind,n,used,nh,n,p->nk,cap,used};
        s.assembly={hc,c.allocate<remag::HamiltonianAssemblyConfig>(1,&hc),c.allocate<remag::HamiltonianPair>(used,pairs.data()),
            c.allocate<int>(used,index.data()),ds,c.allocate<double>(3*used,static_cast<const double*>(input[11])),
            c.allocate<double>(3*p->nk,static_cast<const double*>(input[9])),nullptr};
        s.density={{p->kind,n,cap,nh,eh,used},c.allocate<int>(used,rows.data()),c.allocate<int>(used,cols.data()),
            s.assembly.indices,s.assembly.xij};
        s.grid={{n,points,nsp,n,cap,1},c.allocate<int>(points+1,ends.data()),c.allocate<int>(points*n,orbs.data()),
            c.allocate<int>(1,&zero),c.allocate<int>(used,lookup.data()),c.allocate<float>(size*n,static_cast<const float*>(input[8]))};
        for(int a=0;a<3;a++) s.mesh[a]=mesh[a];
        auto k=static_cast<const double*>(input[9]);for(int i=0;i<p->nk;i++) s.kpoints.push_back({k[3*i],k[3*i+1],k[3*i+2]});
        s.normalization={cap,nh,eh,std::min(nh,2),p->charge*(p->fault==3?1.2:1.),1.e-6};s.occupation_cutoff=1.e-12;
        remag::FermiProblem fp{nh==1||p->kind==2?1:2,p->kind<2?nh:1,p->nk,dim,dim,.03,p->charge};
        if(p->fault==2) fp.charge=4.*dim;if(p->fault==7) fp.maxe++;
        remag::HamiltonianSeedsProblem hp{cap,nh,p->soc,0};remag::ScfScalarsProblem sp{cap,nh,n,0};
        if(p->fault==8) sp.rows++;
        remag::ScfCycleBindings b{};
        b.fermi=c.allocate<remag::FermiProblem>(1,&fp);b.weights=c.allocate<double>(p->nk,static_cast<const double*>(input[10]));
        b.hamiltonian=c.allocate<remag::HamiltonianSeedsProblem>(1,&hp);
        b.kinetic=c.allocate<double>(cap,static_cast<const double*>(input[4]));b.nonlocal_h=c.allocate<double>(cap,static_cast<const double*>(input[5]));
        b.onsite=c.allocate<double>(6*cap,static_cast<const double*>(input[6]));b.offsite=c.allocate<cuDoubleComplex>(4*cap,static_cast<const cuDoubleComplex*>(input[7]));
        b.grid=c.allocate<remag::GridPotentialProblem>(1,&p->grid);
        b.atomic_density=c.allocate<double>(size,static_cast<const double*>(input[12]));b.core_density=c.allocate<double>(size,static_cast<const double*>(input[13]));
        b.neutral_potential=c.allocate<double>(size,static_cast<const double*>(input[14]));b.scalars=c.allocate<remag::ScfScalarsProblem>(1,&sp);
        b.constants=c.allocate<remag::ScfEnergySeeds>(1,static_cast<const remag::ScfEnergySeeds*>(input[15]));
        b.num=c.allocate<int>(n,num.data());b.ptr=c.allocate<int>(n,ptr.data());b.overlap=ds;
        b.h=c.allocate<double>(size_t(cap)*nh,static_cast<const double*>(input[0]));
        b.density=c.allocate<double>(size_t(cap)*nh,static_cast<const double*>(input[1]));
        b.energy_density=c.allocate<double>(size_t(cap)*eh,static_cast<const double*>(input[2]));
        remag::ScfControlProblem settings{{7,p->minimum,p->maximum,0,{},1},{cap*nh,3,.05,.13,1.e-8,1},p->mix_first,0};
        for(double& t:settings.convergence.tolerances) t=p->converge?1.e3:0.;
        check(cudaStreamSynchronize(c.stream));
        remag::ScfCyclePlan plan(s,b,settings,c.stream);
        if(p->fault==9) {
            remag::InitialDensityProblem first{cap,nh,std::min(nh,2),p->mix_first,p->charge};
            auto initial=c.allocate<remag::InitialDensityResult>(1);
            check(remag::initial_density_device(cap,nh,c.allocate(1,&first),c.allocate(1,&zero),b.overlap,
                b.density,b.density,b.energy_density,initial,c.stream));
            check(plan.initialize_hamiltonian(initial));
            // The pre-step may be installed exactly once, before launch.
            if(plan.initialize_hamiltonian(initial)!=cudaErrorInvalidValue)throw std::runtime_error("Duplicate initial Hamiltonian accepted");
        }
        info->workspace_bytes=plan.workspace_bytes();info->graph_nodes=plan.graph_nodes();info->forbidden_nodes=plan.forbidden_nodes();
        info->staged_bytes=c.staged+plan.setup_upload_bytes();
        std::vector<size_t> sizes{size_t(cap)*nh*sizeof(double),size_t(cap)*nh*sizeof(double),size_t(cap)*eh*sizeof(double),
            size_t(dim)*jobs*sizeof(double),size_t(dim)*jobs*sizeof(double),sizeof(remag::FermiResult),sizeof(remag::ScfCycleResult),
            size_t(p->maximum)*sizeof(remag::ScfControlStep),size_t(p->maximum)*sizeof(remag::ScfCycleStep)};
        std::vector<const void*> values{b.h,b.density,b.energy_density,plan.energies(),plan.occupations(),plan.fermi_result(),plan.result(),plan.history(),plan.diagnostics()};
        std::vector<unsigned char*> copies;for(auto bytes:sizes) copies.push_back(c.allocate<unsigned char>(bytes));
        if(remag_activity_graph) remag_activity_graph(plan.executable());
        phase(1,1);
        check(plan.launch());
        for(size_t i=0;i<sizes.size();i++) check(cudaMemcpyAsync(copies[i],values[i],sizes[i],cudaMemcpyDeviceToDevice,c.stream));
        check(plan.launch());check(cudaStreamSynchronize(c.stream));
        phase(2,0);
        info->replay_equal=1;
        for(size_t i=0;i<sizes.size();i++) {
            check(cudaMemcpy(output[i],values[i],sizes[i],cudaMemcpyDeviceToHost));
            std::vector<unsigned char> first(sizes[i]);check(cudaMemcpy(first.data(),copies[i],sizes[i],cudaMemcpyDeviceToHost));
            if(std::memcmp(first.data(),output[i],sizes[i])) info->replay_equal=0;
            info->exported_bytes+=2*sizes[i];
        }
        phase(3,0);
        if(remag_activity_finish) remag_activity_finish();return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"SCF cycle check: %s\n",e.what());if(remag_activity_finish) remag_activity_finish();return -1;
    }
}
