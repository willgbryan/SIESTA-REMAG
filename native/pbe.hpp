// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <vector>

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
// Noncollinear (spins=4) treatment, chosen before plans are built:
// 0 libgridxc's legacy branch (default); 2 multicollinear (Pu et al. 2023),
// as the host REMAG.XC.NoncollinearGGA multicollinear, with the directions
// set by remag_set_xc_multicollinear; 1 local frame, as the host
// REMAG.XC.NoncollinearGGA local-frame (native/remag_xc.F90.in): collinear
// PBE on n_up/dn = (n +- s|m|)/2 mapped back with the chain rule. Whether s
// follows a fixed axis (moments collinear up to sign) is decided once per
// process, from the first density with nonzero magnetization, by the host or
// the device, whichever evaluates first; the other adopts that decision.
extern "C" void remag_set_xc_frame(int mode);
// Gauss-Legendre polar nodes per hemisphere and azimuths (<0: 4*polar).
extern "C" void remag_set_xc_multicollinear(int polar,int azimuth);
extern "C" int remag_xc_frame_mode();
// Host adoption/publication of the decision (decided 0/1, signed 0/1, axis).
extern "C" void remag_query_xc_frame(int* decided,int* signed_axis,double* axis);
extern "C" void remag_publish_xc_frame(int signed_axis,const double* axis);
// Test only: forget the decision.
extern "C" void remag_reset_xc_frame();
class PbePlan {
public:
    PbePlan(int n1,int n2,int n3,int spins,cudaStream_t stream);
    ~PbePlan();
    PbePlan(const PbePlan&)=delete;
    PbePlan& operator=(const PbePlan&)=delete;
    int apply(const PbeProblem* problem,const double* density,double* potential,PbeResult* result);
    size_t workspace_bytes() const;
private:
    int mesh_[3],spins_,size_,blocks_,frame_=0;
    cudaStream_t stream_;
    PbePoint* points_=nullptr;
    double* partial_=nullptr;
    // Local frame: channel densities and potentials, directions, the spins=2
    // problem and per-block magnetization moments for the axis decision.
    double *channels_=nullptr,*channel_potential_=nullptr,*directions_=nullptr,*moments_=nullptr;
    PbeProblem* channel_problem_=nullptr;
    // Multicollinear: directions and weights, n and m, potentials of the three
    // l values, accumulated V_n, V_m and V_s s, three results and the sums.
    int mc_count_=0;
    double *mc_directions_=nullptr,*base_=nullptr,*stencil_=nullptr,*accumulated_=nullptr,*sums_=nullptr;
    PbeResult* stencil_results_=nullptr;
    std::vector<double> host_weights_;
    int apply_multicollinear(const PbeProblem* problem,const double* density,double* potential,PbeResult* result);
    void release();
};
}
