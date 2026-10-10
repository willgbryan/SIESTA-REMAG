// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime.h>

namespace remag {
// Indices are one-based, including repeated periodic atom images and points.
// Point indices refer to the input field's ordering: clustered for potentials,
// sequential for retained deformation density. The caller owns that mapping.
struct AtomicGridForceView {
    int points,entries,atoms;
    const int *point_index,*atom_index;
    const double *gradient,*displacement; // xyz, then entry
    const int* active=nullptr; // optional 0/1 mask; inactive entries are never read
};
struct AtomicGridForceProblem {
    double dvol,volume;
    int kind,diagonal_spins,forces,stress; // kind: 1 partial core, 2 neutral atom
};
// Adds only the mesh atomic contribution. No allocation, transfer, host status
// readback, synchronization or I/O. Sticky status: 1 configuration, 2 nonfinite.
cudaError_t add_atomic_grid_force_device(const AtomicGridForceView&,const AtomicGridForceProblem*,
    const double* potential,const double* auxiliary,const double* deformation,
    double* force,double* stress,int* status,cudaStream_t);
}
