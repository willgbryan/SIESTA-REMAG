// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>

namespace remag {
// Initial staging owns this immutable problem description. All pointers passed
// to fermi_device below refer to device memory, including scalar state.
struct FermiProblem {
    int nspinor,maxspn,nk,maxe,ne;
    double temperature,charge;
};
struct FermiResult {
    double fermi,entropy,charge;
    int status,iterations;
};
// FD occupations, variable total spin, no single excitation. Same bracket,
// tolerance, temperature floor and entropy convention as upstream fermid.F.
// Unsupported settings must be rejected by the integrating caller.
// status: 0 success; 1 invalid shape; 2 nonfinite/invalid data or weights;
//         3 insufficient bands; 4 iteration limit. No scalar is read back.
cudaError_t fermi_device(const FermiProblem* problem,const double* energies,
                        const double* weights,double* occupations,
                        FermiResult* result,cudaStream_t stream);
}
