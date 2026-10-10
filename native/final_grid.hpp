// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "grid.hpp"
namespace remag {
// Pure permutation from SIESTA's clustered mesh to sequential output order.
// Host shape, device arrays; no allocation, host transfer or synchronization.
cudaError_t final_density_sequential_device(const int mesh[3],int nsm,int spins,
    const double* clustered,const double* atomic,double* sequential,double* deformation,cudaStream_t);
// Pinned final-force Hamiltonian convention, which differs from the SCF
// convention for offsite SOC. Device pointers only, no transfers or allocation.
cudaError_t final_grid_hamiltonian_device(const GridContractionView&,int nh,int soc,
    const double* kinetic,const double* nonlocal_h,const double* onsite,const double* potential,
    double dvol,double* h,cudaStream_t);
}
