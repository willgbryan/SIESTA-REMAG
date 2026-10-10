// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// Pinned matel_table S/T tables. All maps and registry/query indices preserve
// upstream's one-based numbering. FFR layout is [1025,2,radials] in Fortran.
struct TwoCenterView {
    int functions,radials,terms,rows,cols,base1,base2,first1,last1,first2,last2;
    double delta,pi;
    const int *ind,*term_offsets,*ilm,*radial_index;
    const double *cutoffs,*radial,*coefficients;
};
// pairs[2,count], xyz[3,count], result[4,count] (value then xyz gradient).
// One kernel; borrowed device pointers; no allocations, copies, waits or I/O.
// Status is sticky: 1 invalid map/index, 2 nonfinite/interpolation range.
// Invalid queries produce zero outputs. A pre-existing status never resets.
// Supports product angular orders 0..10 (registered pure functions l<=5).
cudaError_t evaluate_two_center_device(const TwoCenterView&,int count,const int* pairs,
    const double* xyz,double* result,int* status,cudaStream_t);
}
