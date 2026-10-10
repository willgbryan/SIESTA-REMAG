// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cstddef>

namespace remag {
// Geometry-owned scientific data, staged once by the integrating caller.
// Reciprocal vectors are column-major and include 2*pi; units match poison.F.
struct HartreeProblem {
    int mesh[3];
    double reciprocal[9],cutoff,volume;
};
struct HartreeResult {
    double energy,stress[9];
    // 0 success, 1 shape/configuration mismatch, 2 invalid/nonfinite data.
    int status;
};

// A serial full-periodic-grid Poisson component. Construction/destruction are
// geometry/setup work. apply() uses only device pointers and existing buffers;
// it performs no explicit host/device copies, allocations or synchronization.
// Library-internal activity still needs tracing before a residency claim.
class HartreePlan {
public:
    HartreePlan(int n1,int n2,int n3,cudaStream_t stream);
    ~HartreePlan();
    HartreePlan(const HartreePlan&)=delete;
    HartreePlan& operator=(const HartreePlan&)=delete;
    int apply(const HartreeProblem* problem,const double* density,
              double* potential,HartreeResult* result);
    size_t workspace_bytes() const {return work_bytes_;}
    size_t resident_bytes() const;
private:
    int mesh_[3],size_,blocks_;
    size_t work_bytes_=0;
    cudaStream_t stream_;
    cufftHandle plan_=0;
    void* work_=nullptr;
    cufftDoubleComplex* spectrum_=nullptr;
    double* partial_=nullptr;
    void release();
};
}
