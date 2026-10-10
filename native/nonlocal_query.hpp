// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "nonlocal_compact.hpp"
namespace remag {
struct NonlocalQueryView {
    int centers,neighbors,queries,projectors;
    const NonlocalCenter* center;
    const NonlocalNeighbor* neighbor;
    // One projector registry list per center, in upstream projector order.
    const int *projector_ptr,*projector_registry;
    // One orbital registry and displacement per neighbor, instead of per query.
    const int* orbital_registry;
    const double* displacement;
};
// Expand to the existing two-center/force representation on device. The output
// order is center, neighbor, projector; every query copies its neighbor's xyz.
// Two stream-ordered kernels validate complete coverage, then expand. There is
// no allocation, copy, synchronization or I/O. Sticky status: 1 invalid layout
// or registry, 2 nonfinite displacement. Invalid input leaves outputs untouched.
cudaError_t expand_nonlocal_queries_device(const NonlocalQueryView&,int* pairs,
    double* xyz,int* status,cudaStream_t);
}
