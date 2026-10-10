// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>

namespace remag {
struct GridContractionShape {int no,np,nsp,nuo,nnz,cells;};
struct GridContractionView {
    GridContractionShape shape;
    const int *ends,*orbs,*translations,*lookup;
    const float* phi;
    // Retained with the validated geometry, including its epoch and fingerprint.
    int entries=0;
};
// Borrow a staged geometry before graph construction. Complete outstanding
// work and destroy its graph before invalidating geometry or staging another
// grid layout. Basis storage retains upstream's FP32 representation.
GridContractionView grid_contraction_view();
// Validate CPU query topology against the immutable geometry that indexes the
// retained device density. Scans metadata only; no transfers or electronic reads.
void validate_grid_sparse_topology(int rows,int columns,int capacity,const int* num,const int* ptr,const int* list);
// Exclusive whole-cycle handoff. Acquire only after preparation and before
// any legacy numerical consumer. Legacy consumers/staging reject while leased.
// Finish GPU work and destroy all dependent graphs before releasing. Release
// closes this preparation lifetime without inventing legacy per-step counters.
GridContractionView lease_grid_contraction();
void release_grid_contraction();
// All arrays are device pointers. These operations enqueue one kernel on the
// caller's stream; no allocation, copies, synchronization or I/O occurs here.
cudaError_t contract_density_device(const GridContractionView& view,int ng,int nh,
    const double* density,double* rho,cudaStream_t stream);
// Adds to H in-place and preserves the SOC offdiagonal block duplication.
// device_dvol is a device scalar; a null pointer selects the fixed host_dvol
// value supplied when constructing the graph (legacy adapter compatibility).
cudaError_t add_grid_potential_device(const GridContractionView& view,int ng,int nh,
    const double* potential,double host_dvol,const double* device_dvol,double* h,cudaStream_t stream);
}
