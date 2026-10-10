// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime.h>
namespace remag {
struct PopulationObservableProblem {int rows,atoms,species,components;};
// All numerical inputs and outputs are device pointers. Species indices are
// one-based. orbital/atomic values use interleaved component order.
// Outputs: 5 per orbital (charge, spin, xyz); 6 per atom (net, charge, spin, xyz);
// 5 per species (upstream cumulative species-order totals); 10 final scalars
// (total charge/spin/xyz, atomic net-charge/spin/xyz). Status is sticky.
cudaError_t population_observables_device(int rows,int atoms,int species,int components,
    const PopulationObservableProblem*,const int* atom_species,const double* valence,
    const double* orbital,const double* atomic,const double* total,
    double* orbital_view,double* atomic_view,double* species_view,double* summary,
    int* status,cudaStream_t);
}
