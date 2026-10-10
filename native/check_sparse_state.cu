// SPDX-License-Identifier: GPL-3.0-or-later
// Coupled production-owner check. Mock D/E and grid producers are explicitly
// staged before SCF. They do not qualify real density/grid physics.
#include "sparse_state.hpp"
#include "execution.hpp"
#include "grid_potential.hpp"
#include "occupations.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ok(int code) {if(code) throw std::runtime_error("Sparse checker call failed: "+std::to_string(code));}
void gpu(cudaError_t code) {if(code) throw std::runtime_error(cudaGetErrorString(code));}
void require(bool condition) {if(!condition) throw std::runtime_error("Sparse checker assertion failed");}
template<class F> void reject(F f,int& count) {
    bool caught=false;try {f();} catch(const std::runtime_error&) {caught=true;}
    require(caught);count++;
}
struct Memory {
    void* p=nullptr;
    Memory(size_t n,const void* source=nullptr) {
        gpu(cudaMalloc(&p,n));if(source) gpu(cudaMemcpy(p,source,n,cudaMemcpyHostToDevice));
    }
    ~Memory() {if(p) cudaFree(p);}
    double* data() {return static_cast<double*>(p);}
};
__global__ void add_grid(int n,const double* grid,double* h) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) h[i]+=grid[i];
}
}

extern "C" int remag_check_sparse_state(int n,int spins,int soc,int cycles,int initial_energy,int mix_first,int failure,
    const int* num,const int* ptr,const double* geometry,const double* constants,const double* initial,
    const double* sequence,const double* producer_values,double* scalars,double* borrowed,double* final,int* rejected) {
    try {
        require(n>=19&&cycles>=2&&failure>=-2&&failure<=5);
        for(const char* name:{"REMAG_GPU","REMAG_DM","REMAG_NORMALIZE","REMAG_HS",
                "REMAG_OCCUPATIONS","REMAG_RESIDENT_SPECTRA","REMAG_RESIDENT_GRID","REMAG_RESIDENT_SCF","REMAG_GRID_VERIFY_EPOCH"})
            setenv(name,"1",1);
        const size_t m=size_t(n)*spins,e=size_t(n)*std::min(spins,4),stride=2*m+e;
        const size_t bytes=m*sizeof(double),ebytes=e*sizeof(double);
        std::vector<double> host(initial,initial+4*m+2*e),snapshot=host;
        double *h=host.data(),*hold=h+m,*d=hold+m,*dold=d+m,*ed=dold+m,*eold=ed+e;
        const double *overlap=geometry,*kin=geometry+n,*vkb=geometry+2*n,*on=geometry+3*n;
        const auto* off=reinterpret_cast<const cuDoubleComplex*>(geometry+9*n);
        auto prepare=[&](double* old_h) {return remag_state_prepare(n,spins,4,soc,3,.05,.13,1.e-8,initial_energy,mix_first,
            num,ptr,overlap,kin,vkb,on,off,constants,h,old_h,d,dold,ed,failure==-2?nullptr:eold);};
        ok(remag_phase_enter(0,0,0));
        if(failure!=-1) remag_geometry_changed();
        require(prepare(h)==-210);(*rejected)++;
        require(remag_state_prepare(n,spins,4,soc,3,.05,.13,1.e-8,1,mix_first,
            num,ptr,overlap,kin,vkb,on,off,constants,h,hold,d,dold,ed,nullptr)==-210);(*rejected)++;
        require(remag_state_prepare(n,spins,4,soc,3,.05,.13,1.e-8,initial_energy,2,
            num,ptr,overlap,kin,vkb,on,off,constants,h,hold,d,dold,ed,eold)==-210);(*rejected)++;
        Memory inputs(cycles*stride*sizeof(double),sequence),history(cycles*bytes);
        std::vector<remag::GridPotentialResult> grid(cycles);
        std::vector<remag::FermiResult> fermi(cycles);
        for(int k=0;k<cycles;k++) {
            std::copy(producer_values+10*k,producer_values+10*k+9,grid[k].energies);
            fermi[k].entropy=producer_values[10*k+9];
        }
        if(failure==3) grid[0].status=2;
        if(failure==4) fermi[0].status=2;
        Memory grid_results(grid.size()*sizeof(grid[0]),grid.data()),fermi_results(fermi.size()*sizeof(fermi[0]),fermi.data());
        ok(prepare(hold));require(prepare(hold)==-210);(*rejected)++;
        require(remag_state_begin(1)==-212);(*rejected)++;
        if(failure==1) h[0]+=1.;
        if(failure==2) remag_geometry_changed();
        for(int k=0;k<cycles;k++) {
            ok(remag_phase_enter(1,static_cast<int>(remag::geometry_epoch()),k+1));
            if(failure==1||failure==2) {
                require(remag_state_begin(1)==-213);(*rejected)++;
                h[0]=snapshot[0];require(host==snapshot);return 0;
            }
            require(remag_state_begin(k+2)==-212);(*rejected)++;
            ok(remag_state_begin(k+1));require(remag_state_export()==-224);(*rejected)++;
            const double* input=inputs.data()+k*stride;
            reject([&]{remag_state_publish_density(d,ed,input,input+m,bytes,ebytes);},*rejected);
            reject([&]{remag_state_density(d,bytes);},*rejected);
            reject([&]{remag_state_hamiltonian(hold,bytes);},*rejected);
            reject([&]{remag_state_hamiltonian(h,bytes-8);},*rejected);
            const double* current=remag_state_hamiltonian(h,bytes);require(current);
            gpu(cudaMemcpyAsync(history.data()+k*m,current,bytes,cudaMemcpyDeviceToDevice));
            reject([&]{remag_state_hamiltonian(h,bytes);},*rejected);
            reject([&]{remag_state_publish_density(dold,ed,input,input+m,bytes,ebytes);},*rejected);
            require(remag_state_publish_density(d,ed,input,input+m,bytes,ebytes));ok(remag_state_seed());
            reject([&]{remag_state_grid_target(h,bytes);},*rejected);
            require(remag_state_density(d,bytes));
            reject([&]{remag_state_density(d,bytes);},*rejected);
            double* target=remag_state_grid_target(h,bytes);
            add_grid<<<(m+255)/256,256>>>(static_cast<int>(m),input+m+e,target);gpu(cudaGetLastError());
            reject([&]{remag_state_grid_ready(hold,bytes);},*rejected);
            remag_state_grid_ready(h,bytes);ok(remag_state_finish_hamiltonian());
            int code=remag_state_reduce_device(static_cast<remag::GridPotentialResult*>(grid_results.p)+k,
                static_cast<remag::FermiResult*>(fermi_results.p)+k,scalars+16*k);
            if(failure>=3) {require(code==-219);(*rejected)++;require(host==snapshot);return 0;}
            ok(code);if(k+1<cycles) ok(remag_state_mix());
            require(host==snapshot);
        }
        ok(remag_phase_enter(2,static_cast<int>(remag::geometry_epoch()),0));ok(remag_state_export());
        require(remag_state_export()==-224);(*rejected)++;
        gpu(cudaMemcpy(borrowed,history.p,cycles*bytes,cudaMemcpyDeviceToHost));
        std::copy(host.begin(),host.end(),final);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Sparse ownership check: %s\n",e.what());return -1;}
}
