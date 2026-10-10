// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>

namespace remag {
struct BlockJacobiResult {
    // 0 success, 1 Cholesky failure, 2 nonfinite data, 3 local eigensolver
    // failure, 4 exhausted sweeps. Outputs are usable only at status zero.
    int status,sweeps;
    double relative_offdiagonal,history[32];
};
static_assert(sizeof(BlockJacobiResult)==272,"Unexpected block-Jacobi result ABI");
// First local library failure in the current solve, retained entirely on the
// device. pair/round/sweep are zero based; info==0 means no local failure.
struct BlockJacobiLocalFailure {int info,pair,round,sweep;};
static_assert(sizeof(BlockJacobiLocalFailure)==16,"Unexpected local diagnostic ABI");

// Experimental conventional generalized FP64 eigensolver. Before construction,
// REMAG_BLOCK_WIDTH selects 32/64 (default 64); REMAG_BLOCK_LOCAL_SOLVER
// selects syev/jacobi (default syev; batched Jacobi requires width 32).
// These preparation-only choices never change during a plan lifetime.
// The graph builder
// allocates/captures outside execution; all scientific inputs/outputs are device
// pointers. Appending into a conditional body permits nested device control.
// The lower triangle is authoritative, matching the existing LAPACK/cuSOLVER
// convention; upper-triangle and imaginary diagonal storage is ignored.
// The plan and its buffers must outlive every execution of the constructed graph.
class BlockJacobiPlan {
public:
    BlockJacobiPlan(int n,bool complex_values,int maximum,double tolerance,cudaStream_t stream);
    ~BlockJacobiPlan();
    BlockJacobiPlan(const BlockJacobiPlan&)=delete;
    BlockJacobiPlan& operator=(const BlockJacobiPlan&)=delete;
    void warmup(const void* h,const void* s,BlockJacobiResult* result);
    cudaGraphNode_t append(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,
        const void* h,const void* s,void* z,double* w,BlockJacobiResult* result);
    // Optional per-job persistent orthogonal basis; flag is initialized to zero
    // by the owner and published only after a successful complete solve.
    cudaGraphNode_t append_seeded(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,
        const void* h,const void* s,void* z,double* w,BlockJacobiResult* result,void* seed,int* valid);
    size_t seed_bytes() const;
    size_t workspace_bytes() const;
    size_t host_workspace_bytes() const;
    size_t setup_upload_bytes() const;
    size_t setup_upload_calls() const;
    size_t device_allocation_count() const;
    const BlockJacobiLocalFailure* local_failure() const;
    const void* failed_local_input() const;
    int local_dimension() const;
    int graph_nodes() const;
    int forbidden_nodes() const;
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};
}
