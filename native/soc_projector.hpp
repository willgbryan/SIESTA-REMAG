// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// Each query contains both j=l-1/2,l+1/2 radial blocks and real harmonics
// m=-l,...,+l. Left derivatives have a trailing contiguous xyz dimension.
// l=0 is a separate scalar contraction in pristine SIESTA.
struct SocProjectorView {
    int queries,l,split;
    const double *epsilon,*left,*right,*left_gradient;
};
// Output per query: Vion, real(V)[4], imag(V)[4], real(F)[12], imag(F)[12].
// V and F use the pristine Fortran component ordering. F always includes
// both scalar-relativistic and spin-orbit terms, independent of split.
// Caller supplies all storage. No copies, allocation, synchronization or I/O.
// Sticky status: 2 nonfinite input/arithmetic, 3 upstream symmetry guard.
// On any per-query error all its output elements are zero. A nonzero status
// invalidates the entire batch; other query outputs must not be consumed.
cudaError_t soc_projector_device(const SocProjectorView&,double* output,int* status,cudaStream_t);
}
