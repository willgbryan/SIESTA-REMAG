// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "hamiltonian.hpp"
#include <cuComplex.h>
#include <cstddef>
#include <memory>

namespace remag {
struct FrozenSpectrumStatus {
    // 0 success, 1 invalid sparse values/rotation, 2 eigensolver failure,
    // 3 nonfinite spectrum or invalid eigenvector normalization.
    int status=0, k=-1, solver_info=0, band=-1;
    double norm=0., spin_squared=0.;
};

// Internal, experimental full-offsite-SOC evaluator; not an SCF driver.
// The caller must supply a converged generator H and its ORIGINAL lattice SOC
// operator, sharing the assembly view's (ldh,component) sparse layout. H, SOC,
// geometry and stream are borrowed, immutable, and must outlive this plan.
// reference_has_soc=false declares a genuinely SOC-free SCF generator; frozen
// targets then add the supplied full lattice SOC once, after exchange rotation.
// Construction is ordered on stream after the reference producer. No state is
// retained across a geometry change, and this class never changes the reference.
//
// One dense H/S workspace is reused for every k point. Only eigenvalues (and
// optional three spin expectations per band) are retained across the mesh.
// Dense vector storage therefore does not grow with nk. This initial executor
// uses one solver lane; it does not imply a general dense-k capacity guarantee.
class FrozenPotentialPlan {
public:
    FrozenPotentialPlan(const HamiltonianAssemblyView&, const double* generator_h,
        const cuDoubleComplex* offsite_soc, bool spin_expectations,
        size_t workspace_limit_bytes, cudaStream_t, bool reference_has_soc=true);
    ~FrozenPotentialPlan();
    FrozenPotentialPlan(const FrozenPotentialPlan&)=delete;
    FrozenPotentialPlan& operator=(const FrozenPotentialPlan&)=delete;

    // Active quaternion (w,x,y,z), rotating exchange relative to lattice SOC.
    // target_has_soc=false is the explicit SOC-free reference control.
    // Asynchronously enqueues rotation, assembly and solves. Results are valid
    // only after stream completion AND status()==0. A subsequent call replaces
    // every output. No host numerical readback occurs inside the k loop.
    // Invalid arguments throw before launch; asynchronous numerical failures
    // remain in status(). Never read a partial spectrum as a completed result.
    void solve(const double quaternion[4], bool target_has_soc=true);
    // Diagnostic only, after a successful solve: assemble one selected k into
    // caller-owned device storage [generator H, S, lattice SOC, rotated H],
    // four column-major complex matrices of bands()^2 elements. Keeps BOTH
    // independently assembled triangles for an offline rotation/solver check.
    // Reuses scratch; leaves spectra, spin and job count unchanged. Output
    // must not alias any borrowed input or plan storage and outlive the stream.
    void capture_matrices(int k, const double quaternion[4], cuDoubleComplex* output);
    const FrozenSpectrumStatus* status() const;
    const double* energies() const; // [k][band], full spinors, Ry
    // [k][band][xyz], dimensionless Pauli expectations with C^H S C=1.
    // nullptr in eigenvalues-only mode. Occupations/weights are NOT applied.
    const double* spin_expectations() const;
    // First failed spin check only; column-major dense C and original S.
    // Valid after completion when status()==3 and status()->band>=0.
    const cuDoubleComplex* failed_vectors() const;
    const cuDoubleComplex* failed_overlap() const;
    int bands() const;
    int kpoints() const;
    int eigensystem_jobs() const; // cuSOLVER calls enqueued in the latest solve
    // Explicit plan-owned device bytes, including cuSOLVER's requested work.
    // Excludes borrowed inputs and unreported CUDA/library internal storage.
    size_t workspace_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
