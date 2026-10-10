// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "nonlocal_compact.hpp"
namespace remag {
struct NonlocalH0View {
    int neighbors,centers,blocks,queries,rows,columns,capacity,soc,split,local_neighbors;
    const NonlocalCenter* center;
    const NonlocalNeighbor* neighbor;
    const NonlocalBlock* block;
    const int *num,*ptr,*list;
    // Stable row lists ordered by (center, left-neighbor index). Geometry
    // ownership constructs these once, outside SCF. Every local left orbital
    // appears exactly once; row==0 neighbors are right operands only.
    const int *row_ptr,*row_neighbors;
    // Four doubles per two-center query: value, gradient[3]. Only value is read.
    const double* integrals;
};
struct H0Complex {double real,imag;};
static_assert(sizeof(H0Complex)==16,"Complex H0 storage ABI");
// Adds to scalar H0[capacity] and optional SOC H0_off[4*capacity], in upstream
// 11,22,12,21 component order. Scalar duplicate columns update the FIRST CSR
// entry; SOC updates EVERY duplicate. Caller retains all device storage.
// Two kernels: validate topology, then independently gather each sparse entry
// in upstream center/left/right/projector order. No floating-point atomics,
// allocation, transfers, synchronization, host status polling or I/O.
// Sticky status: 1 topology, 2 nonfinite consumed value/arithmetic, 3 SOC guard.
// Nonzero status invalidates the complete output, including partial writes.
cudaError_t add_nonlocal_h0_device(const NonlocalH0View&,double* h,H0Complex* soc,
    int* status,cudaStream_t);
}
