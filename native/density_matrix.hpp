// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace remag {
struct DensityMatrixShape {int kind,n,nnz,nh,eh,pairs;};
struct DensityMatrixView {
    DensityMatrixShape shape;
    const int *rows,*cols,*indices;
    const double* xyz;
    // Optional geometry-owned grouping of periodic copies by orbital pair.
    // Maps have the same lifetime as rows/cols/indices; zero selects the legacy layout.
    int unique_pairs=0;
    const int *unique_rows=nullptr,*unique_cols=nullptr,*pair_unique=nullptr;
};
// Device-resident selection of one sampled-k job for a looped density build:
// Cartesian k, output spin block and whether the job exists (a partial final
// wave stages zero occupations).
struct DensityJobStage {double k[3];int outspin,valid;};
static_assert(sizeof(DensityJobStage)==32,"Unexpected density job stage ABI");
struct DensityNormalizationConfig {int rows,nh,eh,spinor;double charge,tolerance;};
struct DensityNormalizationResult {double charge,scale;int status;};
static_assert(sizeof(DensityNormalizationResult)==24,"Unexpected density normalization ABI");
// Borrow before capture, during an active density builder. Finish device work
// and destroy graphs before ending/replacing that builder's geometry lifetime.
// D/E and all numerical scratch are supplied explicitly by the caller.
DensityMatrixView density_matrix_view();
// Exclusive whole-cycle handoff. Acquire only after preparation and before
// any legacy numerical consumer. Legacy consumers/staging reject while leased.
// Finish GPU work and destroy all dependent graphs before releasing. Release
// closes this preparation lifetime without inventing legacy per-step counters.
DensityMatrixView lease_density_matrix();
void release_density_matrix();
// Pure device operations: no allocation, host/device copy, synchronization or
// I/O. Pointer arguments name nonaliasing device storage. Geometry, shapes,
// spin/k selections and normalization settings are fixed when building a graph.
cudaError_t clear_density_device(const DensityMatrixShape& shape,double* d,double* e,cudaStream_t stream);
// Nonconjugate column-major n-by-nev -> orbital-major n-by-nev transpose.
cudaError_t transpose_vectors_device(int n,int nev,bool complex_values,const void* source,void* target,cudaStream_t stream);
cudaError_t accumulate_density_device(const DensityMatrixView& view,int nev,int outspin,
    const void* vectors,const double* occupations,const double* energies,double cutoff,
    double kx,double ky,double kz,double* d,double* e,cudaStream_t stream,double* unique_scratch=nullptr);
// Sampled-k kinds: the same accumulation with k and the output spin read from
// a device stage, so a single captured node serves every job of a k loop.
cudaError_t accumulate_density_staged_device(const DensityMatrixView& view,int nev,const void* vectors,
    const double* occupations,const double* energies,double cutoff,const DensityJobStage* stage,
    double* d,double* e,cudaStream_t stream,double* unique_scratch=nullptr);
// A wave of `lanes` sampled-k jobs in one pass, adding each job's contribution
// in lane order, as that many accumulate_density_staged_device calls would.
// Lane L's transposed vectors start at vectors + L*stride (elements), its
// occupations and energies at L*nev, its stage at stages[L]; with grouping,
// unique_scratch holds lanes * density_unique_scratch_count(view) doubles.
cudaError_t accumulate_density_wave_device(const DensityMatrixView& view,int nev,int lanes,const void* vectors,
    size_t stride,const double* occupations,const double* energies,double cutoff,const DensityJobStage* stages,
    double* d,double* e,cudaStream_t stream,double* unique_scratch=nullptr);
// Private per-cycle scratch; reuse serially across k/spin jobs. No dense n*n bank.
inline size_t density_unique_scratch_count(const DensityMatrixView& v) {
    return size_t(v.unique_pairs)*2*(v.shape.kind==0?1:(v.shape.kind==1?2:8));
}
// Scratch: ceil(rows*spinor/256) doubles and one result. A failed charge leaves
// D/E unchanged and records status -134. The device consumer must check status.
cudaError_t normalize_density_device(const DensityNormalizationConfig& config,const double* overlap,
    double* partial,DensityNormalizationResult* result,double* d,double* e,cudaStream_t stream);
}
