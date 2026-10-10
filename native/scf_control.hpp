// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "scf_convergence.hpp"
#include "pulay.hpp"
#include <memory>

namespace remag {
struct ScfControlProblem {
    ScfConvergenceProblem convergence;
    PulayProblem mixer;
    int mix_first,reserved;
};
struct ScfControlResult {
    ScfConvergenceResult convergence;
    PulayResult mixer;
    // status: 0 success, 1 invalid/changed configuration, 2 convergence
    // producer failure, 3 Pulay failure. Errors stay sticky within an epoch.
    int status,records,mixed,first_mix_bypassed,continue_loop,finished,reserved[2];
};
struct ScfControlStep {ScfScalarsResult scalars;ScfControlResult control;};
static_assert(sizeof(ScfControlProblem)==112&&sizeof(ScfControlResult)==128&&sizeof(ScfControlStep)==240,
              "Unexpected device SCF control ABI");

// Appends device convergence, conditional stable Pulay, first-mix semantics and
// bounded diagnostic retention to an existing iteration graph. Every pointer
// is a stable device pointer; no copies, allocation or host callbacks execute
// inside the appended graph. The caller supplies upstream-equivalent scalars,
// Hold and H-Hold and honors result.continue_loop in its outer device loop.
// A new nonzero epoch resets all control and mixer history. Re-entry after a
// completed/failed epoch is inert. History is exported only after the cycle.
// This component does not compute density, grid physics or final observables.
class ScfControlPlan {
public:
    ScfControlPlan(int size,int mixer_history,int diagnostic_capacity,cudaStream_t stream);
    ~ScfControlPlan();
    ScfControlPlan(const ScfControlPlan&)=delete;
    ScfControlPlan& operator=(const ScfControlPlan&)=delete;
    cudaGraphNode_t append(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,
        const ScfControlProblem* problem,const ScfScalarsResult* scalars,
        const double* input,const double* residual,double* output,ScfControlResult* result,ScfControlStep* history);
    size_t workspace_bytes() const;
    int graph_nodes() const;
    int forbidden_nodes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
