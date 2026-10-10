// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
struct InitialDensityProblem {int capacity,spins,spinor,mix_first;double target_charge;};
struct InitialDensityResult {double charge,scale;int status,mix_first,note,reserved;};
// Pinned normalize_dm(first=.true.) followed by state_init's zero EDM. Input
// density may be a GPU history prediction or an explicitly staged initial DM.
// All pointers are device-owned; output may alias input density. A failed
// validation leaves D/E outputs untouched. The dependency status is sticky:
// extrapolation errors must never be converted into an accepted initialization.
// Computes the full-capacity trace in upstream serial order, permits a large
// initial charge mismatch, and disables first-step mixing above 1e-4 electrons.
// This differs intentionally from the tighter within-SCF normalization gate.
cudaError_t initial_density_device(int capacity,int spins,const InitialDensityProblem*,
    const int* dependency_status,const double* overlap,const double* input,
    double* density,double* energy_density,InitialDensityResult*,cudaStream_t);
}
