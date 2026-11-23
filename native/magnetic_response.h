// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stddef.h>

// Host C ABI. Complex inputs are interleaved double real/imag, each matrix
// column-major; banks are [k][matrix]. All energies and kBT are in Ry.
// A plan uploads immutable matrices once and retains all Cholesky factors.
// No numerical CPU fallback exists. Nonzero status invalidates that call.
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double chemical_potential, charge, band_energy, free_energy;
    double delta_band, delta_free, delta_canonical_free;
    double spin[3], torque[3];
    double max_residual, max_orthogonality_error;
    double seconds;
    size_t workspace_bytes;
} RemagMftResult;
const char* remag_magnetic_error(void);
int remag_mft_create(int n, int nk, const void* base, const void* magnetic,
    const void* soc, const void* overlap, const double* weights,
    double temperature, double electrons, size_t memory_limit, void** handle);
int remag_mft_sweep(void* handle, const double quaternion[4], double soc_scale,
    int set_reference, RemagMftResult* result, double* energies,
    double* occupations, double* orbital_population);
void remag_mft_destroy(void* handle);
#ifdef __cplusplus
}
#endif
