// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace remag {
struct ScfScalarsProblem {int capacity,spins,rows,reserved;};
// Rydberg units, except dimensionless Entropy. The supported conventional path
// has zero DFT+U, dispersion, external-field, meta, MM and bulk-bias corrections.
struct ScfEnergySeeds {
    double ena,kinetic,nonlocal_energy,soc,ions,enascf,enaatm,hartree,xc,entropy,temperature,madelung;
};
struct ScfScalarsResult {
    // Ebs, DEharr, Etot, Eharrs, FreeE, DEna, dDmax, dHmax, dEmax, Q(1:4).
    double values[13];
    int status,active; // 0 success, 1 configuration mismatch, 2 nonfinite/invalid topology
};
static_assert(sizeof(ScfScalarsProblem)==16&&sizeof(ScfEnergySeeds)==96&&sizeof(ScfScalarsResult)==112,
              "Unexpected SCF scalar ABI");

// Serial Hamiltonian mixing, spin H/D widths 1/2/4/8 and E width min(spins,4).
// Matrices have Fortran (capacity,spin) layout. Energies and residuals consume
// the full capacity; charge consumes only the entries selected by num/ptr.
// All apply arguments are device pointers. No allocations, copies, host
// scientific branches, synchronization or I/O occur inside apply. The output
// residual is H-Hold, ready for stable Pulay, and is unchanged on device error.
class ScfScalarsPlan {
public:
    ScfScalarsPlan(int capacity,int spins,int rows,cudaStream_t stream);
    ~ScfScalarsPlan();
    ScfScalarsPlan(const ScfScalarsPlan&)=delete;
    ScfScalarsPlan& operator=(const ScfScalarsPlan&)=delete;
    int apply(const ScfScalarsProblem* problem,const ScfEnergySeeds* seeds,
              const int* num,const int* ptr,const double* overlap,
              const double* h,const double* hold,const double* density,const double* old_density,
              const double* energy_density,const double* old_energy_density,
              double* h_residual,ScfScalarsResult* result);
    size_t workspace_bytes() const;
private:
    int capacity_,spins_,rows_,blocks_,row_blocks_;
    cudaStream_t stream_;
    double *partial_=nullptr,*charge_partial_=nullptr,*residual_=nullptr;
    void release();
};
}
