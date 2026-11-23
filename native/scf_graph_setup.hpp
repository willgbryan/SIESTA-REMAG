// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cuda_runtime_api.h>
#include <cstdio>
#include "scf_runtime_policy.hpp"

namespace remag {
// Setup evidence must reach disk before a driver call can terminate the process.
// This is outside graph capture/execution and does not change the graph or retry.
struct ScfGraphSetup {
    int dimension, spins, jobs, lanes, nodes, forbidden;
    int runtime_version=-1, driver_cuda_version=-1;
    int runtime_query_status=-1, driver_query_status=-1;

    void checkpoint(const char* phase,int status=-1) const noexcept {
        auto* file=std::fopen("remag-scf-setup.log","a");
        if(!file) {std::fputs("REMAG: cannot open SCF setup checkpoint\n",stderr);return;}
        const int written=std::fprintf(file,
            "{\"schema\":1,\"phase\":\"%s\",\"status\":%d,\"dimension\":%d,"
            "\"spins\":%d,\"jobs\":%d,\"lanes\":%d,\"graph_nodes\":%d,\"forbidden_nodes\":%d,"
            "\"runtime_version\":%d,\"driver_cuda_version\":%d,"
            "\"runtime_query_status\":%d,\"driver_query_status\":%d}\n",
            phase,status,dimension,spins,jobs,lanes,nodes,forbidden,
            runtime_version,driver_cuda_version,runtime_query_status,driver_query_status);
        // fclose flushes stdio before the next CUDA call. This survives process
        // failure, not necessarily host power loss; no destructor is required.
        const int closed=std::fclose(file);
        if(written<0||closed)std::fputs("REMAG: failed to write SCF setup checkpoint\n",stderr);
    }
};

inline cudaError_t instantiate_scf_graph(cudaGraphExec_t* exec,cudaGraph_t graph,
        cudaStream_t stream,ScfGraphSetup evidence) {
    evidence.checkpoint("version_query_begin");
    evidence.runtime_query_status=cudaRuntimeGetVersion(&evidence.runtime_version);
    evidence.driver_query_status=cudaDriverGetVersion(&evidence.driver_cuda_version);
    if(evidence.runtime_query_status!=cudaSuccess)evidence.runtime_version=-1;
    if(evidence.driver_query_status!=cudaSuccess)evidence.driver_cuda_version=-1;
    if(!supported_scf_runtime(evidence.runtime_version,evidence.driver_cuda_version)) {
        evidence.checkpoint("runtime_rejected",cudaErrorNotSupported);
        std::fputs("REMAG: device SCF runtime profile is not admitted; no graph instantiation or fallback\n",stderr);
        return cudaErrorNotSupported;
    }
    evidence.checkpoint("instantiate_begin");
    auto status=cudaGraphInstantiate(exec,graph,0);
    evidence.checkpoint("instantiate_end",status);
    if(status!=cudaSuccess)return status;
    evidence.checkpoint("upload_begin");
    status=cudaGraphUpload(*exec,stream);
    evidence.checkpoint("upload_end",status);
    if(status!=cudaSuccess)return status;
    evidence.checkpoint("synchronize_begin");
    status=cudaStreamSynchronize(stream);
    evidence.checkpoint("synchronize_end",status);
    return status;
}
}
