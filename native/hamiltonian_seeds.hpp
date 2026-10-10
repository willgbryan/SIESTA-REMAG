// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstddef>

namespace remag {
struct HamiltonianSeedsProblem {int capacity,spins,soc,reserved;}; // soc: 0 none, 1 onsite, 2 offsite
struct HamiltonianSeedsResult {
    double kinetic,nonlocal_energy,soc_energy;
    int status,active,stage,prepared_soc; // stage: 1 seed committed, 2 post-grid SOC committed
};
static_assert(sizeof(HamiltonianSeedsProblem)==16&&sizeof(HamiltonianSeedsResult)==40,"Unexpected Hamiltonian seed ABI");

// Literal supported setup_hamiltonian arithmetic. All arrays are device-owned,
// in Fortran (capacity,component) layout. Onsite has six real components;
// offsite has four complex components. Unused SOC input pointers may be null.
// prepare initializes H and retains kinetic/nonlocal/SOC energies. After the
// grid has accumulated into H, finish_grid adds offsite SOC in original order.
// Each operation commits H only on success; a failed preparation cannot feed
// finish_grid. No copies, allocations, synchronization or I/O in these methods.
class HamiltonianSeedsPlan {
public:
    HamiltonianSeedsPlan(int capacity,int spins,cudaStream_t stream);
    ~HamiltonianSeedsPlan();
    HamiltonianSeedsPlan(const HamiltonianSeedsPlan&)=delete;
    HamiltonianSeedsPlan& operator=(const HamiltonianSeedsPlan&)=delete;
    int prepare(const HamiltonianSeedsProblem* p,const double* kinetic,const double* nonlocal_h,
                const double* onsite,const cuDoubleComplex* offsite,const double* density,
                double* h,HamiltonianSeedsResult* result);
    int finish_grid(const HamiltonianSeedsProblem* p,const cuDoubleComplex* offsite,
                    double* h,HamiltonianSeedsResult* result);
    size_t workspace_bytes() const;
private:
    int capacity_,spins_,blocks_;
    cudaStream_t stream_;
    double *trial_=nullptr,*partial_=nullptr;
    void release();
};
}
