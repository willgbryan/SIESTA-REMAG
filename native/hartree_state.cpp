// SPDX-License-Identifier: GPL-3.0-or-later
// Transitional SIESTA host adapter; the reusable device component is hartree.cu.
#include "hartree.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

extern "C" int remag_gpu_enabled();
namespace {
bool verify() {const char* v=std::getenv("REMAG_GRID_VERIFY_EPOCH");return v&&std::strcmp(v,"1")==0;}
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
struct Counts {
    unsigned long long calls=0,setups=0,h2d=0,d2h=0,checked=0,allocations=0;
    double seconds=0.;
    ~Counts() {
        if(FILE* f=std::fopen("remag-hartree.json","w")) {
            std::fprintf(f,"{\"calls\":%llu,\"geometry_setups\":%llu,\"host_to_device_bytes\":%llu,"
                "\"device_to_host_bytes\":%llu,\"debug_host_validation_bytes\":%llu,\"bridge_allocations\":%llu,"
                "\"wall_seconds\":%.9f,\"scope\":\"periodic FP64 Hartree; temporary density upload and potential/scalar exports; plan allocations excluded\"}\n",
                calls,setups,h2d,d2h,checked,allocations,seconds);
            std::fclose(f);
        }
    }
} counts;
struct Buffer {
    void* p=nullptr;size_t capacity=0;
    void reserve(size_t bytes) {
        if(bytes<=capacity) return;
        if(p) {check(cudaFree(p));p=nullptr;capacity=0;}
        check(cudaMalloc(&p,bytes));capacity=bytes;counts.allocations++;
    }
    void upload(const void* source,size_t bytes,remag::Transfer boundary) {
        reserve(bytes);check(cudaMemcpy(p,source,bytes,cudaMemcpyHostToDevice));
        counts.h2d+=bytes;remag::transfer(boundary,true,bytes);
    }
    void download(void* target,size_t bytes,remag::Transfer boundary) {
        check(cudaMemcpy(target,p,bytes,cudaMemcpyDeviceToHost));
        counts.d2h+=bytes;remag::transfer(boundary,false,bytes);
    }
    ~Buffer() {if(p) cudaFree(p);}
};
struct State {
    bool valid=false;
    uint64_t epoch=0;
    int size=0;
    std::array<int,3> mesh{};
    const double* host_cell=nullptr;
    std::array<unsigned char,9*sizeof(double)> cell{};
    Buffer problem,density,potential,result;
    std::unique_ptr<remag::HartreePlan> plan;
} state;
bool valid_mesh(const int* mesh) {
    return mesh&&mesh[0]>0&&mesh[1]>0&&mesh[2]>0&&
           static_cast<long long>(mesh[0])*mesh[1]<=INT_MAX/mesh[2];
}
}
extern "C" int remag_hartree_enabled() {
    const char* v=std::getenv("REMAG_HARTREE");return v&&std::strcmp(v,"1")==0;
}
extern "C" int remag_hartree_needs_setup(const int* mesh,const double* cell) {
    if(!remag_hartree_enabled()||!remag_gpu_enabled()||!valid_mesh(mesh)||!cell||remag::geometry_epoch()==0) return -180;
    if(!state.valid||state.epoch!=remag::geometry_epoch()||cell!=state.host_cell||
       !std::equal(state.mesh.begin(),state.mesh.end(),mesh)) return 1;
    if(verify()) {
        counts.checked+=state.cell.size();
        if(std::memcmp(cell,state.cell.data(),state.cell.size())!=0) return -181;
    }
    return 0;
}
extern "C" int remag_hartree_prepare(const int* mesh,const double* cell,const double* reciprocal,double cutoff,double volume) {
    try {
        if(remag_hartree_needs_setup(mesh,cell)!=1||!reciprocal||!std::isfinite(cutoff)||cutoff<=0.||
           !std::isfinite(volume)||volume<=0.) return -182;
        for(int i=0;i<9;i++) if(!std::isfinite(cell[i])||!std::isfinite(reciprocal[i])) return -182;
        state.valid=false;
        remag::HartreeProblem problem{};
        std::copy(mesh,mesh+3,problem.mesh);std::copy(reciprocal,reciprocal+9,problem.reciprocal);
        problem.cutoff=cutoff;problem.volume=volume;
        // Mesh changes require a new FFT plan; cell-only changes reuse its work.
        if(!state.plan||!std::equal(state.mesh.begin(),state.mesh.end(),mesh))
            state.plan=std::make_unique<remag::HartreePlan>(mesh[0],mesh[1],mesh[2],nullptr);
        state.problem.upload(&problem,sizeof(problem),remag::Transfer::hartree_geometry);
        state.size=mesh[0]*mesh[1]*mesh[2];size_t bytes=size_t(state.size)*sizeof(double);
        state.density.reserve(bytes);state.potential.reserve(bytes);state.result.reserve(sizeof(remag::HartreeResult));
        std::copy(mesh,mesh+3,state.mesh.begin());std::memcpy(state.cell.data(),cell,state.cell.size());
        state.host_cell=cell;state.epoch=remag::geometry_epoch();state.valid=true;counts.setups++;return 0;
    } catch(const std::exception& error) {
        state.valid=false;std::fprintf(stderr,"REMAG Hartree prepare: %s\n",error.what());return -183;
    }
}
extern "C" int remag_hartree_solve(const int* mesh,const double* cell,const double* density,
                                   double* energy,double* potential,double* stress) {
    try {
        if(remag_hartree_needs_setup(mesh,cell)!=0||!density||!energy||!potential||!stress) return -184;
        auto start=std::chrono::steady_clock::now();size_t bytes=size_t(state.size)*sizeof(double);
        state.density.upload(density,bytes,remag::Transfer::hartree_density);
        int status=state.plan->apply(static_cast<const remag::HartreeProblem*>(state.problem.p),
            static_cast<const double*>(state.density.p),static_cast<double*>(state.potential.p),
            static_cast<remag::HartreeResult*>(state.result.p));
        if(status) throw std::runtime_error("Hartree device dispatch failed: "+std::to_string(status));
        remag::HartreeResult result{};
        state.result.download(&result,sizeof(result),remag::Transfer::hartree_result);
        if(result.status) return -185;
        state.potential.download(potential,bytes,remag::Transfer::hartree_potential);
        *energy=result.energy;std::copy(result.stress,result.stress+9,stress);
        counts.calls++;counts.seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return 0;
    } catch(const std::exception& error) {
        std::fprintf(stderr,"REMAG Hartree solve: %s\n",error.what());return -186;
    }
}
