// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>

// Explicit one-shot request made by a patched upstream density builder.
// Unarmed eigensolver calls keep the original host-output contract.
bool remag_vector_request(int& slot, bool& export_host);
void remag_capture_vectors(int slot, int n, int nev, bool complex_values, const void* device_vectors);
const double* remag_resident_density(const double* host, size_t bytes);
// A declared SCF spectrum bank captures producer output without host upload.
void remag_capture_spectrum(const double* host_values,const double* device_values,int n);
// Only an explicitly armed, resident SCF bank can suppress host eigenvalues.
bool remag_spectrum_export_required();
bool remag_retained_occupations(const double* host_occupations,const double* host_values,int n,
                                const double*& occupations,const double*& values);
// A one-shot sparse H/S selection fills the solver's device matrices directly.
bool remag_assemble_hs(int n,bool complex_values,const void* host_h,const void* host_s,
                      void* device_h,void* device_s);
// Declared resident-grid lifetime: rho producer -> potential flow -> vmat.
double* remag_flow_density_target(double* host,size_t bytes);
void remag_flow_density_ready(double* host,size_t bytes);
const double* remag_flow_potential(const double* host,size_t bytes);
const double* remag_flow_dvol();
namespace remag {struct GridPotentialResult;struct FermiResult;}
const remag::GridPotentialResult* remag_flow_result();
const remag::FermiResult* remag_fermi_result();
// A declared sparse SCF lifetime makes host H/D/E opaque until final export.
const double* remag_state_hamiltonian(const double* host,size_t bytes);
bool remag_state_publish_density(double* host_d,double* host_e,const double* d,const double* e,size_t db,size_t eb);
const double* remag_state_density(const double* host,size_t bytes);
double* remag_state_grid_target(double* host,size_t bytes);
void remag_state_grid_ready(double* host,size_t bytes);
