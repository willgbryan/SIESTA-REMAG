// SPDX-License-Identifier: GPL-3.0-or-later
// Synthetic setup-only diagnostic. Links the existing REMAG library; never
// launches an SCF cycle, reads material inputs, or invokes a reference solver.
#include "scf_cycle.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t code) {
    if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Storage {
    cudaStream_t stream=nullptr;
    std::vector<void*> allocations;
    Storage() {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Storage() {
        if(stream) cudaStreamSynchronize(stream);
        for(auto p:allocations) cudaFree(p);
        if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t count,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));
        allocations.push_back(p);
        if(source) check(cudaMemcpy(p,source,count*sizeof(T),cudaMemcpyHostToDevice));
        else check(cudaMemsetAsync(p,0,count*sizeof(T),stream));
        return p;
    }
};
int number(const char* text) {
    char* end=nullptr;long value=std::strtol(text,&end,10);
    if(!*text||*end||value<1||value>1000) throw std::runtime_error("Invalid diagnostic dimension");
    return int(value);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: probe_scf_setup dimension kpoints mesh");
        const int n=number(argv[1]),nk=number(argv[2]),side=number(argv[3]);
        if((n!=30&&n!=76)||(nk!=132&&nk!=948)||(side!=4&&side!=40))
            throw std::runtime_error("Outside the registered synthetic diagnostic shapes");
        int devices=0,runtime=0,driver=0;
        check(cudaGetDeviceCount(&devices));
        if(devices!=1) throw std::runtime_error("Exactly one visible GPU required");
        check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
        std::printf("{\"stage\":\"setup\",\"n\":%d,\"nk\":%d,\"mesh\":%d,\"runtime\":%d,\"driver\":%d}\n",n,nk,side,runtime,driver);
        std::fflush(stdout);
        const int nh=2,used=n*n,cap=used+2,size=side*side*side,zero=0;
        const double charge=n*.5;
        Storage c;
        std::vector<int> num(n,n),ptr(n),rows(used),cols(used),index(used),ends(size+1),orbs(size*n),lookup(used);
        std::vector<remag::HamiltonianPair> pairs(used);
        std::vector<double> overlap(cap),h(cap*nh),xyz(3*used),kpoints(3*nk),weights(nk,1./nk);
        for(int i=0;i<n;i++) {ptr[i]=i*n;overlap[i*n+i]=1.;for(int spin=0;spin<nh;spin++)h[spin*cap+i*n+i]=1.+i*.01;}
        for(int i=0;i<used;i++) {rows[i]=i/n;cols[i]=i%n;index[i]=lookup[i]=i;pairs[i]={i%n,i/n,i,i+1};}
        for(int i=0;i<=size;i++) ends[i]=i*n;
        for(int i=0;i<size*n;i++) orbs[i]=i%n+1;
        // Fixed distinct sampled-k labels. Zero displacements make S identity.
        for(int k=0;k<nk;k++) kpoints[3*k]=double(k)/nk;
        auto ds=c.allocate<double>(cap,overlap.data());
        remag::ScfCycleShape s{};
        remag::HamiltonianAssemblyConfig hc{1,n,used,nh,n,nk,cap,used};
        s.assembly={hc,c.allocate(1,&hc),c.allocate(used,pairs.data()),c.allocate(used,index.data()),ds,
            c.allocate(xyz.size(),xyz.data()),c.allocate(kpoints.size(),kpoints.data()),nullptr};
        s.density={{1,n,cap,nh,nh,used},c.allocate(used,rows.data()),c.allocate(used,cols.data()),s.assembly.indices,s.assembly.xij};
        s.density.unique_pairs=used;s.density.unique_rows=s.density.rows;s.density.unique_cols=s.density.cols;s.density.pair_unique=s.assembly.indices;
        s.grid={{n,size,1,n,cap,1},c.allocate(ends.size(),ends.data()),c.allocate(orbs.size(),orbs.data()),
            c.allocate(1,&zero),c.allocate(used,lookup.data()),c.allocate<float>(size_t(size)*n),size*n};
        for(int axis=0;axis<3;axis++)s.mesh[axis]=side;
        for(int k=0;k<nk;k++)s.kpoints.push_back({kpoints[3*k],0.,0.});
        s.normalization={cap,nh,nh,nh,charge,1.e-6};s.occupation_cutoff=1.e-12;
        remag::FermiProblem fp{2,nh,nk,n,n,.03,charge};
        remag::HamiltonianSeedsProblem hp{cap,nh,0,0};
        remag::ScfScalarsProblem sp{cap,nh,n,0};
        remag::ScfEnergySeeds energies{};
        remag::GridPotentialProblem gp{};
        gp.nsm=1;gp.dvol=216./size;gp.hartree.volume=gp.pbe.volume=216.;gp.hartree.cutoff=1.e30;gp.pbe.spins=nh;
        for(int a=0;a<3;a++) {
            gp.hartree.mesh[a]=gp.pbe.mesh[a]=side;
            gp.hartree.reciprocal[4*a]=2.*std::acos(-1.)/6.;gp.pbe.gradient[4*a]=side/6.;
        }
        remag::ScfCycleBindings b{};
        b.fermi=c.allocate(1,&fp);b.weights=c.allocate(weights.size(),weights.data());
        b.hamiltonian=c.allocate(1,&hp);b.kinetic=c.allocate<double>(cap);b.nonlocal_h=c.allocate<double>(cap);
        b.grid=c.allocate(1,&gp);b.atomic_density=c.allocate<double>(size);b.core_density=c.allocate<double>(size);b.neutral_potential=c.allocate<double>(size);
        b.scalars=c.allocate(1,&sp);b.constants=c.allocate(1,&energies);
        b.num=c.allocate(num.size(),num.data());b.ptr=c.allocate(ptr.size(),ptr.data());b.overlap=ds;
        b.h=c.allocate(h.size(),h.data());b.density=c.allocate<double>(cap*nh);b.energy_density=c.allocate<double>(cap*nh);
        remag::ScfControlProblem control{{7,1,2,0,{},1},{cap*nh,3,.05,.13,1.e-8,1},0,0};
        check(cudaStreamSynchronize(c.stream));
        {std::ifstream maps("/proc/self/maps");std::ofstream out("probe-maps.txt");out<<maps.rdbuf();}
        std::puts("{\"stage\":\"construct_begin\",\"cycle_launch_authorized\":false}");std::fflush(stdout);
        remag::ScfCyclePlan plan(s,b,control,c.stream);
        std::printf("{\"stage\":\"constructed\",\"graph_nodes\":%d,\"forbidden_nodes\":%d,\"jobs\":%d,\"lanes\":%d,\"workspace_bytes\":%zu}\n",
            plan.graph_nodes(),plan.forbidden_nodes(),plan.eigensystem_jobs(),plan.solver_lanes(),plan.workspace_bytes());
        std::fflush(stdout);
        return plan.forbidden_nodes()?3:0;
    } catch(const std::exception& error) {std::fprintf(stderr,"Setup diagnostic: %s\n",error.what());return 2;}
}
