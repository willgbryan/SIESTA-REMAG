// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// Two radial functions per species: core then neutral potential. Indices are
// [n, sample offset]; parameters are [cutoff, delta]. z<0 is a floating species.
struct AtomicBasisView {
    int species,samples;
    const int *atomic_number,*index;
    const double *parameters,*values,*second;
};
// kind 1 calls chcore_sub; kind 2 calls phiatm(is,0). With mesh_filter=1,
// additionally apply the exact enclosing meshsubs radius selection. active
// describes this outer selection, including selected floating/empty functions.
// Query species are zero-based; result[4,query] is value and xyz gradient.
// Sticky status: 1 invalid descriptor/index, 2 nonfinite/interpolation range.
// One kernel, no allocation, copies, synchronization or I/O.
cudaError_t evaluate_atomic_queries_device(const AtomicBasisView&,int kind,int mesh_filter,
    int count,const int* species,const double* xyz,double* result,int* active,int* status,cudaStream_t);
// Geometry pairs retain full atom image, offset and clustered mesh point,
// before radial selection. Original mappings are one-based. nsp=nsm^3.
struct AtomicGeometryView {
    int pairs,nsp,atoms,offsets,unit_atoms,points,nsm,mesh[3];
    const int *pair_atom,*pair_offset,*pair_cluster,*atom_species,*unit_atom;
    const double *offset,*subpoint,*atom_offset;
};
// Outputs [nsp,pair] and [xyz,nsp,pair]. Point indices are clustered for core,
// sequential for neutral; unit atom indices never replace the geometry atom.
cudaError_t evaluate_atomic_geometry_device(const AtomicBasisView&,const AtomicGeometryView&,int kind,
    double* value,double* gradient,double* displacement,int* active,int* point,int* unit_atom,int* status,cudaStream_t);
struct AtomicBasisValues {
    const double *value,*gradient,*displacement;
    const int *active,*point,*atom;
};
struct AtomicBasisComparison {
    unsigned long long points,selected,reference_selected,compared,index_compared,mismatches,nonfinite,mask_mismatches,index_mismatches;
    double absolute[3],scaled[3];
};
static_assert(sizeof(AtomicBasisComparison)==120,"Unexpected atomic comparison ABI");
// Check every selection and every value/gradient, including inactive zeros.
// Displacements and field/atom indices are checked at every CPU-selected point.
// Numerical gates: 1e-10+1e-10*abs(reference), exact displacements/indices/masks.
// Sticky status 3 mismatch, 2 nonfinite. Caller initializes the record.
cudaError_t compare_atomic_geometry_device(size_t points,const AtomicBasisValues&,const AtomicBasisValues&,
    AtomicBasisComparison*,int* status,cudaStream_t);
}
