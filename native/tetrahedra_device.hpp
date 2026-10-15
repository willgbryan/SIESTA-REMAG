// SPDX-License-Identifier: GPL-3.0-or-later
// Tetrahedron occupations on the GPU, for the device SCF cycle: the same
// grid (built on the host by tetra_grid), corner energies, Fermi search and
// weights as tetra_occupations in tetrahedra.cpp, the host reference. The
// Fermi level is found in one thread block, so its sums have a fixed order;
// each listed k point gathers its corner weights in tetrahedron order, as the
// host accumulates them, without atomics.
#pragma once
#include "tetrahedra.hpp"
#include "occupations.hpp"
#include <cuda_runtime_api.h>
#include <cstddef>

namespace remag {
class TetraDevicePlan {
public:
    // energies and occupations are (maxe,maxspn,nk) as fermid.F.
    TetraDevicePlan(const TetraGrid& grid,int method,int nspinor,int maxspn,int maxe,int ne,cudaStream_t stream);
    ~TetraDevicePlan();
    TetraDevicePlan(const TetraDevicePlan&)=delete;
    TetraDevicePlan& operator=(const TetraDevicePlan&)=delete;
    // Device pointers. The charge is read from problem->charge. Writes the
    // occupations (k weight times occupation) and a FermiResult with zero
    // entropy; status is a TetraStatus on failure. Capturable: kernels only.
    int apply(const FermiProblem* problem,const double* energies,double* occupations,FermiResult* result);
    size_t workspace_bytes() const;
private:
    int method_,nspinor_,maxspn_,maxe_,ne_,nk_,corners_,tetrahedra_,blocks_;
    size_t items_,bytes_=0;
    cudaStream_t stream_;
    int *stencil_=nullptr,*point_=nullptr,*incidence_=nullptr,*offsets_=nullptr;
    double *sorted_=nullptr,*corner_=nullptr,*bounds_=nullptr,*fermi_=nullptr;
    unsigned char* order_=nullptr;
    void release();
};
}
