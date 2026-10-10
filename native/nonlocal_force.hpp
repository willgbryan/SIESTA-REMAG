// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// One-based orbital/atom indices; zero-based row offsets. Sparse density
// components have a capacity stride, including padding between components.
struct NonlocalForceView {
    int entries,atoms,rows,columns,capacity,components,spinor;
    const int *query_rows,*query_columns,*atom_pairs,*num,*ptr,*list;
    // scalar query: epsilon, S_left, S_right, gradient_left[3].
    // angular query: the 33-value soc_projector_device output.
    const double *scalar,*operators,*xyz;
};
// kind: 0 scalar, 1 SOC l=0, 2 SOC l>0. split only changes l=0 energy
// reporting here; angular input already carries the selected energy split.
struct NonlocalForceProblem {double volume;int kind,split;};
static_assert(sizeof(NonlocalForceProblem)==16,"Unexpected nonlocal-force ABI");
// Adds to caller-owned energy[9], force[3*atoms], stress[9]. Energy layout:
// Enl, real(Eso11,Eso22,Eso12,Eso21), imag(Eso11,Eso22,Eso12,Eso21).
// Scalar queries sum repeated sparse entries; SOC queries use the last.
// No allocation, transfers, synchronization, status polling or I/O.
// Sticky status: 1 invalid configuration/index, 2 nonfinite arithmetic or
// accumulator overflow. Nonzero status invalidates all physical outputs.
cudaError_t add_nonlocal_force_device(const NonlocalForceView&,const NonlocalForceProblem*,
    const double* density,double* energy,double* force,double* stress,int* status,cudaStream_t);
}
