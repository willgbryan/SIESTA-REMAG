// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "grid_potential.hpp"
namespace remag {
// Bind initial Hartree energy directly into the device grid configuration.
// Invalid producer status leaves a NaN energy, so GridPotentialPlan also
// rejects the handoff. All pointers are device-owned; one kernel, no copies.
cudaError_t bind_initial_harris_device(const HartreeResult*,GridPotentialProblem*,
    int* status,cudaStream_t);
}
