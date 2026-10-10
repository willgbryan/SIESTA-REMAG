// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include "resident.hpp"

namespace remag {
struct ScfEnergySeeds;
struct HamiltonianSeedsResult;
// Compose the supported energy terms without exporting producer scalars.
// All arguments are device pointers; only a kernel is enqueued on stream.
// A failed producer poisons the record so the scalar plan rejects it.
cudaError_t compose_scf_energy_seeds_device(const ScfEnergySeeds* constants,
    const HamiltonianSeedsResult* hamiltonian,const GridPotentialResult* grid,
    const FermiResult* fermi,ScfEnergySeeds* output,cudaStream_t stream);
}

extern "C" int remag_state_enabled();
extern "C" int remag_state_active();
// Host eold may be null only when EDM convergence/initial_energy is disabled.
// Device Eold remains allocated in either mode; absent host storage is not exported.
extern "C" int remag_state_prepare(int n,int spins,int rows,int soc,int history,
    double weight,double linear_weight,double condition,int initial_energy,int mix_first,
    const int* num,const int* ptr,const double* overlap,const double* kin,const double* vkb,
    const double* on,const cuDoubleComplex* off,const double* constants,
    double* h,double* hold,double* d,double* dold,double* e,double* eold);
extern "C" int remag_state_begin(int iteration);
extern "C" int remag_state_seed();
extern "C" int remag_state_finish_hamiltonian();
extern "C" int remag_state_scalars(double* values);
extern "C" int remag_state_mix();
extern "C" int remag_state_export();
// Device producer records, also used by the coupled ownership checker.
// Production obtains these from its declared grid/spectrum lifetimes.
int remag_state_reduce_device(const remag::GridPotentialResult* grid,
    const remag::FermiResult* fermi,double* values);
