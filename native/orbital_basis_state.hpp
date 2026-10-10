// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "orbital_basis.hpp"
#include <cstdint>
namespace remag {
bool orbital_basis_shadow_enabled();
bool orbital_basis_enabled();
bool orbital_basis_active();
struct OrbitalBasisDeviceView {
    OrbitalBasisView basis;
    OrbitalGeometryView geometry;
    double *phi,*gradient,*displacement;
    OrbitalComparison* comparison;
};
// Borrow compact geometry during setup; derivative storage remains force-owned.
OrbitalBasisDeviceView borrow_orbital_mesh_basis(uint64_t epoch,int no,int np,int entries,int nsp);
// Initial compact staging occurs before the electronic cycle. Borrow once
// after convergence; both generation and comparison use the force stream.
OrbitalBasisDeviceView begin_orbital_basis(uint64_t epoch,int no,int np,int entries,int nsp,cudaStream_t);
// Enqueue once on the borrowed stream, including during capture.
void generate_orbital_basis(uint64_t epoch,int* status,cudaStream_t);
// Consume the existing force status after its stream completes. Production
// adds no export; shadow exports a compact record and rejects partial coverage.
void finish_orbital_basis(uint64_t epoch,int force_status);
}

extern "C" int remag_orbital_basis_stage(const int*,double,const int*,const int*,const double*,const double*,const double*,
    const int*,const int*,const int*,const int*,const int*,const double*,const double*,const double*);
