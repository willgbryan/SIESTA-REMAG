// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "scf_scalars.hpp"
#include <cstdint>

namespace remag {
struct ScfConvergenceProblem {
    int criteria,minimum,maximum,reserved; // bits: DM,H,EDM,Harris,FreeE
    double tolerances[5];
    uint64_t epoch;
};
struct ScfConvergenceResult {
    int status,iteration,criteria_satisfied,converged,mix,continue_loop,exhausted,energy_flags;
};
static_assert(sizeof(ScfConvergenceProblem)==64&&sizeof(ScfConvergenceResult)==32,"Unexpected convergence ABI");

// All apply pointers are device pointers. One kernel evaluates the unchanged
// conventional criteria, including the upstream energy-converger latch.
// The caller controls termination and must honor status/continue_loop.
// Calling again after convergence is allowed for criterion diagnostics; calling
// past maximum poisons the state. A changed nonzero epoch resets all history.
// No allocation, copies, synchronization or file I/O inside apply.
class ScfConvergencePlan {
public:
    explicit ScfConvergencePlan(cudaStream_t stream);
    ~ScfConvergencePlan();
    ScfConvergencePlan(const ScfConvergencePlan&)=delete;
    ScfConvergencePlan& operator=(const ScfConvergencePlan&)=delete;
    int apply(const ScfConvergenceProblem* problem,const ScfScalarsResult* scalars,ScfConvergenceResult* result);
    size_t workspace_bytes() const;
private:
    cudaStream_t stream_;
    void* state_=nullptr;
};
}
