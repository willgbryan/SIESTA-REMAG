// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "grid.hpp"

namespace remag {
// Additional immutable geometry for dfscf's FP64 force contraction. Values
// must be recomputed with upstream all_phi, not promoted from the SCF cache.
// Orbital entries follow grid.ends/orbs. phi is [nsp,entries], gradient and
// displacement are [3,nsp,entries], all Fortran order. Atom indices are one
// based and cover every full-cell orbital; occupations cover unit-cell ones.
struct GridForceView {
    GridContractionView grid;
    int atoms,entries;
    const int* orbital_atoms;
    const double *phi,*gradient,*displacement,*atomic_occupation;
};
struct GridForceProblem {double dvol,volume;int forces,stress;};
static_assert(sizeof(GridForceProblem)==24,"Unexpected grid-force problem ABI");
// Pure device-pointer operation. Adds to force[3,atoms] and stress[9]. Stress
// order is ix*3+iy, matching upstream xgC's loops. Potentials use clustered
// [nsp,np,grid_spins] and [nsp,np] ordering. status must initially be zero;
// errors are sticky (1 bad geometry/configuration, 2 nonfinite numerical data).
// No transfer/allocation/synchronization/I/O. Caller retains every view buffer
// through execution and must reject all partial outputs when status is nonzero.
// This covers dfscf only, not core/neutral, kinetic, nonlocal or overlap forces.
cudaError_t add_grid_force_device(const GridForceView&,int density_spins,const GridForceProblem*,
    const double* density,const double* potential,const double* auxiliary,
    double* forces,double* stress,int* status,cudaStream_t stream);
}
