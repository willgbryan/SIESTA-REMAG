// SPDX-License-Identifier: GPL-3.0-or-later
// Diagnostic interposer, linked only with probe_cycle_barrier.cpp. Retain the
// exact production library, graph and flags. Dump before forwarding once.
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* executable,cudaGraph_t graph,unsigned long long flags) {
    using Call=cudaError_t (*)(cudaGraphExec_t*,cudaGraph_t,unsigned long long);
    static std::atomic<int> count{0};const int serial=++count;
    if(serial>64)std::_Exit(92); // Bound diagnostic disk use and construction.
    auto call=reinterpret_cast<Call>(dlsym(RTLD_NEXT,"cudaGraphInstantiate"));
    if(!call)std::_Exit(93);
    char path[80];std::snprintf(path,sizeof(path),"graph-before-%03d.dot",serial);
    const auto dumped=cudaGraphDebugDotPrint(graph,path,cudaGraphDebugDotFlagsVerbose);
    int runtime=-1,driver=-1;const int runtime_status=cudaRuntimeGetVersion(&runtime),driver_status=cudaDriverGetVersion(&driver);
    auto log=[&](const char* phase,int status) {
        auto* f=std::fopen("graph-interpose.log","a");if(!f)std::_Exit(94);
        const int written=std::fprintf(f,"{\"phase\":\"%s\",\"serial\":%d,\"status\":%d,\"dump_status\":%d,\"flags\":%llu,\"runtime\":%d,\"driver\":%d,\"runtime_query_status\":%d,\"driver_query_status\":%d}\n",
            phase,serial,status,int(dumped),flags,runtime,driver,runtime_status,driver_status);
        const int closed=std::fclose(f);if(written<0||closed)std::_Exit(94);
    };
    log("instantiate_begin",-1);
    const auto result=call(executable,graph,flags);
    log("instantiate_end",int(result));
    return result;
}
