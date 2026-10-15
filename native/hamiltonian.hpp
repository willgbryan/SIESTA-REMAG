// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>

namespace remag {
struct HamiltonianAssemblyConfig {int kind,n,nnz,nh,no,nk,ldh,pairs;};
struct HamiltonianPair {int row,col,begin,end;};
struct HamiltonianAssemblyView {
    HamiltonianAssemblyConfig shape;
    const HamiltonianAssemblyConfig* config;
    const HamiltonianPair* pairs;
    const int* indices;
    const double *overlap,*xij,*kpoints,*hamiltonian;
};
// Acquire during the declared sparse-assembly lifetime, before graph capture.
// The view borrows immutable geometry storage: end/invalidation of that owner
// invalidates the view and any graph containing it. The caller must finish
// device work and destroy its graph before changing the geometry lifetime.
HamiltonianAssemblyView hamiltonian_assembly_view();
// Exclusive whole-cycle handoff. Acquire only after preparation and before
// any legacy numerical consumer. Legacy consumers/staging reject while leased.
// Finish GPU work and destroy all dependent graphs before releasing. Release
// closes this preparation lifetime without inventing legacy per-step counters.
HamiltonianAssemblyView lease_hamiltonian_assembly();
void release_hamiltonian_assembly();
// Fixed spin/k selection is resolved when building a graph. Sparse H can
// change in-place between iterations. All pointer arguments refer to device
// storage. Only device memset and kernels are enqueued on the supplied stream;
// no allocation, synchronization, copies to/from the host or I/O occur here.
cudaError_t assemble_hamiltonian_device(const HamiltonianAssemblyView& view,const double* sparse_h,
    int spin,int k,void* dense_h,void* dense_s,cudaStream_t stream);
// Sampled-k kinds only: the same assembly for job min(*cursor+lane, jobs-1),
// spin = job % spin blocks and k = job / spin blocks, with the cursor read on
// the device so one captured graph can serve every k point.
cudaError_t assemble_hamiltonian_indexed_device(const HamiltonianAssemblyView& view,const double* sparse_h,
    const int* cursor,int lane,int jobs,void* dense_h,void* dense_s,cudaStream_t stream);
// The same for lanes 0..lanes-1 at once, into consecutive n-by-n matrices.
cudaError_t assemble_hamiltonian_wave_device(const HamiltonianAssemblyView& view,const double* sparse_h,
    const int* cursor,int lanes,int jobs,void* dense_h,void* dense_s,cudaStream_t stream);
}
