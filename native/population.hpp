// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime.h>

namespace remag {
struct PopulationProblem {int capacity,rows,atoms,spins;};
// Read sparse D/S, skipping unused capacity and hermitizing SOC components
// 3/7 and 4/8 exactly as dm_charge and Mulliken population option 1.
// Outputs are row/atom-major with min(spins,4) components, then total charge.
// Status: 0 success, 1 shape/topology, 2 nonfinite input/arithmetic. Any error
// invalidates all outputs. Every launch initializes outputs and status; no
// allocation, synchronization, host readback or I/O occurs here. All pointer
// arguments reference device memory. Input/output storage must not overlap.
cudaError_t population_device(int capacity,int rows,int atoms,int spins,
    const PopulationProblem*,const int* num,const int* ptr,const int* lasto,
    const double* overlap,const double* density,double* orbital,double* atomic,
    double* total,int* status,cudaStream_t);
}
