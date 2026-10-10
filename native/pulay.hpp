// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace remag {
// Stable Pulay, all flattened spin components, fixed weights within an epoch.
// A changed epoch explicitly clears history, even if the shape is unchanged.
struct PulayProblem {
    int size,history;
    double weight,linear_weight,condition;
    uint64_t epoch;
};
struct PulayResult {
    int status,iteration,history_size,method,rank;
    int oldest,next,active; // internal device control, exported only in checks
    double gram_max,projection_max; // device diagnostics for history conditioning
    double inverse_first,coefficient_first;
};
static_assert(sizeof(PulayProblem)==40&&sizeof(PulayResult)==64,"Unexpected Pulay ABI");

class PulayPlan {
public:
    PulayPlan(int size,int history,cudaStream_t stream);
    ~PulayPlan();
    PulayPlan(const PulayPlan&)=delete;
    PulayPlan& operator=(const PulayPlan&)=delete;
    // Device pointers only. xin, residual and next must not alias. A failure
    // leaves next untouched and poisons history until an explicit epoch reset.
    // No transfers, allocation, host synchronization, file I/O or scalar readback.
    int apply(const PulayProblem* problem,const double* xin,const double* residual,
              double* next,PulayResult* result);
    size_t workspace_bytes() const;
private:
    int size_,history_,blocks_,fields_;
    cudaStream_t stream_;
    void* state_=nullptr;
    double *previous_residual_=nullptr,*previous_output_=nullptr,*delta_x_=nullptr,
           *delta_f_=nullptr,*partial_=nullptr,*trial_=nullptr,*coefficients_=nullptr,
           *gram_=nullptr,*projection_=nullptr;
    void release();
};
}
