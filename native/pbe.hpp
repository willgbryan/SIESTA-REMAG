// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace remag {
struct PbePoint;
// Geometry-owned configuration, staged once. Mesh-gradient vectors are
// column-major inverse-transpose(cell / mesh); they do NOT include 2*pi.
struct PbeProblem {int mesh[3],spins;double gradient[9],volume;};
struct PbeResult {
    double energies[4],stress[9]; // Ex,Ec,Dx,Dc in Ry; stress in Ry/Bohr^3
    int status; // 0 valid, 1 plan/config shape mismatch, 2 invalid/nonfinite data
};
static_assert(sizeof(PbeProblem)==96&&sizeof(PbeResult)==112,"Unexpected PBE ABI layout");
// Built-in PBE, irel=0, full serial periodic FP64 grid, 1/2/4 spin components.
// All apply inputs/outputs are device pointers. No copies, allocations,
// synchronization or file operations in apply. Initial/final staging belongs
// to the caller. Input density and output potential must not alias.
// Device consumers must honor result.status; outputs are invalid on failure.
class PbePlan {
public:
    PbePlan(int n1,int n2,int n3,int spins,cudaStream_t stream);
    ~PbePlan();
    PbePlan(const PbePlan&)=delete;
    PbePlan& operator=(const PbePlan&)=delete;
    int apply(const PbeProblem* problem,const double* density,double* potential,PbeResult* result);
    size_t workspace_bytes() const;
private:
    int mesh_[3],spins_,size_,blocks_;
    cudaStream_t stream_;
    PbePoint* points_=nullptr;
    double* partial_=nullptr;
    void release();
};
}
