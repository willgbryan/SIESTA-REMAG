// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
namespace remag {
// Original species order. species[2,is] = zero-based orbital begin/count;
// orbitals[4,io] = radial sample count/offset/l/m; radial[2,io] = cutoff/delta.
// Radial samples are shared by magnetic copies of the same nl orbital.
struct OrbitalBasisView {
    int species_count,orbital_count,sample_count;
    double pi;
    const int *species,*orbitals;
    const double *radial,*values,*second;
};
// All query indices are zero-based. xyz[3,n], result[4,n] = phi,grad(3).
// Literal all_phi/rad_get/rlylm arithmetic through upstream atmfuncs' l<=5.
// status: sticky 1 invalid shape/index/table, 2 nonfinite/range error.
// No allocation, copies, synchronization or I/O; one device kernel.
cudaError_t evaluate_orbital_queries_device(const OrbitalBasisView&,int count,
    const int* species,const int* orbital,const double* xyz,double* result,int* status,cudaStream_t);
// Exact dfscf geometry reconstruction: xdop(:,listp2(entry))+xdsp(:,isp)-dxa(:,ia).
// Original Fortran mappings are one-based. All arrays are device pointers.
struct OrbitalGeometryView {
    int entries,subpoints,orbitals,atoms,offsets;
    const int *entry_orbital,*entry_offset,*orbital_atom,*orbital_local,*atom_species;
    const double *offset,*subpoint,*atom_offset;
};
cudaError_t evaluate_orbital_geometry_device(const OrbitalBasisView&,const OrbitalGeometryView&,
    double* phi,double* gradient,double* displacement,int* status,cudaStream_t);
// PhiOnMesh uses phiatm rather than all_phi: selected-orbital lmax and the
// cutoff-minus-1e-12 guard must be preserved before rounding to phi_grid_p.
// Output is the FP32 array consumed by existing sparse grid contractions.
// One kernel, no allocation/transfers/synchronization/I/O; empty entries are
// a no-op. Existing force-derivative semantics above are unchanged.
cudaError_t evaluate_orbital_mesh_device(const OrbitalBasisView&,const OrbitalGeometryView&,
    float* phi,int* status,cudaStream_t);
struct OrbitalComparison {
    unsigned long long compared,mismatches,nonfinite;
    double absolute[3],scaled[3];
};
static_assert(sizeof(OrbitalComparison)==72,"Unexpected orbital comparison ABI");
// Checks every phi/gradient/displacement element. Value/gradient tolerance is
// 1e-10 + 1e-10*abs(reference); displacements must agree exactly. Status 3
// indicates mismatch, 2 nonfinite data. Caller initializes record to zero.
cudaError_t compare_orbital_geometry_device(size_t points,const double* phi,const double* gradient,const double* displacement,
    const double* reference_phi,const double* reference_gradient,const double* reference_displacement,
    OrbitalComparison*,int* status,cudaStream_t);
}
