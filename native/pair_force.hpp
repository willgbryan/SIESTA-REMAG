// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// One-based orbital/atom indices; row offsets are zero-based. Query order is
// the upstream neighbor order. Density components use a capacity stride.
struct PairForceView {
    int entries,atoms,rows,columns,capacity,spinor;
    const int *query_rows,*query_columns,*atom_pairs,*num,*ptr,*list;
    const double *integrals,*xyz; // interleaved value/dx/dy/dz and x/y/z
};
struct PairForceProblem {double volume;int kind,padding;}; // kind 1 kinetic, 4 overlap
static_assert(sizeof(PairForceProblem)==16,"Unexpected pair-force problem ABI");
// Add kinetic energy/forces/stress or overlap forces/stress. Caller initializes
// the kinetic energy accumulator to zero for a complete kinetic calculation.
// D or E is already resident; only its first spinor components are read.
// No allocation, copies, synchronization, scalar polling or I/O. Status is
// sticky: 1 invalid device configuration/index, 2 nonfinite arithmetic.
cudaError_t add_pair_force_device(const PairForceView&,const PairForceProblem*,
    const double* density,double* energy,double* force,double* stress,int* status,cudaStream_t);
}
