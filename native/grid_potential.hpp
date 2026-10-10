// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "hartree.hpp"
#include "pbe.hpp"
#include <memory>

namespace remag {
// All geometry/scientific configuration is staged once and device-owned.
struct GridPotentialProblem {
    HartreeProblem hartree;
    PbeProblem pbe;
    double harris_energy,dvol;
    int nsm,partial_core;
};
struct GridPotentialResult {
    // Uscf, DUscf, Uatm, Enaatm, Enascf, Ex, Ec, DEx, DEc (Rydbergs).
    double energies[9],hartree_stress[9],xc_stress[9];
    int status; // 0 valid, 1 plan shape mismatch, 2 invalid data/configuration
};
static_assert(sizeof(GridPotentialProblem)==224&&sizeof(GridPotentialResult)==224,"Unexpected grid-potential ABI");

// Full periodic, built-in PBE, nonrelativistic XC, 1/2/4 grid spin channels.
// Electron density and output potential are in original clustered ordering;
// atomic/core densities and neutral-atom potential use sequential ordering.
// No applied fields, doping, additional charge/potential or force work here.
// apply uses device pointers only, with no explicit transfers, allocations,
// synchronization or I/O. Consumers must check device result status.
class GridPotentialPlan {
public:
    GridPotentialPlan(int n1,int n2,int n3,int spins,cudaStream_t stream);
    ~GridPotentialPlan();
    GridPotentialPlan(const GridPotentialPlan&)=delete;
    GridPotentialPlan& operator=(const GridPotentialPlan&)=delete;
    int apply(const GridPotentialProblem* problem,const double* electron_density,
              const double* atomic_density,const double* core_density,const double* neutral_potential,
              double* potential,GridPotentialResult* result);
    size_t workspace_bytes() const;
    // Borrowed final-force inputs, valid after a successful completed apply
    // until the next apply or destruction. No transfer or recomputation.
    const double* auxiliary_potential() const; // clustered Hartree + neutral
    const double* deformation_density() const; // sequential spin-summed rho - rho_atom
private:
    int mesh_[3],spins_,size_,blocks_;
    cudaStream_t stream_;
    std::unique_ptr<HartreePlan> hartree_;
    std::unique_ptr<PbePlan> pbe_;
    double *charge_=nullptr,*xc_density_=nullptr,*hartree_potential_=nullptr,*xc_potential_=nullptr,*auxiliary_=nullptr,*partial_=nullptr;
    HartreeResult* hartree_result_=nullptr;
    PbeResult* pbe_result_=nullptr;
    void release();
};
}
