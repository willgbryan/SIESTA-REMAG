// SPDX-License-Identifier: GPL-3.0-or-later
// Generalized FP64 eigensolve, optionally retaining eigenvectors for sparse density construction.
// Transfers at the remaining host boundaries are counted explicitly.
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include "resident.hpp"
#include "execution.hpp"
#include "block_jacobi.hpp"

extern "C" void remag_activity_graph(void*) __attribute__((weak));

namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void check(cusolverStatus_t code) {
    if (code != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cuSOLVER status " + std::to_string(code));
}
struct Stats {
    unsigned long long calls=0, h2d=0, d2h=0, allocations=0, live_bytes=0, peak_bytes=0;
    unsigned long long retained_calls=0, avoided_vector_d2h=0, avoided_value_d2h=0;
    unsigned long long block_calls=0,block_setups=0,block_workspace_peak=0;
    double seconds=0;
    ~Stats() {
        const char* path = std::getenv("REMAG_GPU_STATS");
        if (!path) path = "remag-gpu.json";
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "{\"calls\":%llu,\"host_to_device_bytes\":%llu,"
                "\"device_to_host_bytes\":%llu,\"device_allocation_count\":%llu,"
                "\"peak_bridge_device_bytes\":%llu,\"wall_seconds\":%.9f,"
                "\"retained_vector_calls\":%llu,\"avoided_eigenvector_device_to_host_bytes\":%llu,"
                "\"avoided_eigenvalue_device_to_host_bytes\":%llu,"
                "\"experimental_block_calls\":%llu,\"experimental_block_setups\":%llu,"
                "\"experimental_block_workspace_peak_bytes\":%llu,"
                "\"kernel_launch_count\":null,\"scope\":\"bridge only; excludes library internals\"}\n",
                calls,h2d,d2h,allocations,peak_bytes,seconds,retained_calls,avoided_vector_d2h,avoided_value_d2h,
                block_calls,block_setups,block_workspace_peak);
            std::fclose(f);
        }
    }
};
Stats stats;
std::mutex solver_mutex;
struct Buffer {
    void* ptr=nullptr;
    size_t capacity=0;
    void reserve(size_t bytes) {
        if (bytes <= capacity) return;
        if (ptr) { check(cudaFree(ptr)); stats.live_bytes -= capacity; ptr=nullptr; capacity=0; }
        check(cudaMalloc(&ptr, bytes));
        capacity=bytes; stats.allocations++; stats.live_bytes+=bytes;
        if (stats.live_bytes > stats.peak_bytes) stats.peak_bytes=stats.live_bytes;
    }
    ~Buffer() { if (ptr) cudaFree(ptr); }
};
struct BlockExecutor {
    cudaStream_t stream=nullptr;
    cudaEvent_t inputs_ready=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    std::unique_ptr<remag::BlockJacobiPlan> plan;
    Buffer vectors,result;
    int dimension=0;
    bool complex_values=false;
    void clear() {
        if(executable) {cudaGraphExecDestroy(executable);executable=nullptr;}
        if(graph) {cudaGraphDestroy(graph);graph=nullptr;}
        if(plan) stats.live_bytes-=plan->workspace_bytes();
        plan.reset();
    }
    ~BlockExecutor() {
        clear();
        if(inputs_ready) cudaEventDestroy(inputs_ready);
        if(stream) cudaStreamDestroy(stream);
    }
    int solve(int n,bool complex_input,const void* h,const void* s,double* w) {
        if(!stream) check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        if(!inputs_ready) check(cudaEventCreateWithFlags(&inputs_ready,cudaEventDisableTiming));
        // Existing matrix assembly uses the default stream. Order this
        // experimental adapter explicitly; the future whole-SCF executor will
        // use one declared stream and device status instead of this host call.
        check(cudaEventRecord(inputs_ready,nullptr));
        check(cudaStreamWaitEvent(stream,inputs_ready,0));
        if(!executable||dimension!=n||complex_values!=complex_input) {
            clear();dimension=n;complex_values=complex_input;
            vectors.reserve(size_t(n)*n*(complex_input?sizeof(cuDoubleComplex):sizeof(double)));
            result.reserve(sizeof(remag::BlockJacobiResult));
            plan=std::make_unique<remag::BlockJacobiPlan>(n,complex_input,20,2.e-13,stream);
            stats.live_bytes+=plan->workspace_bytes();
            stats.allocations+=plan->device_allocation_count();
            if(stats.live_bytes>stats.peak_bytes) stats.peak_bytes=stats.live_bytes;
            stats.h2d+=plan->setup_upload_bytes();
            remag::transfer(remag::Transfer::solver_graph_setup,true,plan->setup_upload_bytes(),plan->setup_upload_calls());
            auto status=static_cast<remag::BlockJacobiResult*>(result.ptr);
            plan->warmup(h,s,status);
            check(cudaGraphCreate(&graph,0));
            plan->append(graph,graph,nullptr,h,s,vectors.ptr,w,status);
            if(plan->host_workspace_bytes()||plan->forbidden_nodes())
                throw std::runtime_error("Experimental block solver has a forbidden host workspace or graph node");
            check(cudaGraphInstantiate(&executable,graph,0));
            if(remag_activity_graph) remag_activity_graph(executable);
            check(cudaGraphUpload(executable,stream));
            stats.block_setups++;
            if(plan->workspace_bytes()>stats.block_workspace_peak) stats.block_workspace_peak=plan->workspace_bytes();
        }
        check(cudaGraphLaunch(executable,stream));
        int status=0;
        // Temporary four-byte adapter boundary; loop-internal status remains
        // on device. This is a numerical integration experiment, not a
        // scalar-free SCF path or a qualified activity trace.
        check(cudaMemcpyAsync(&status,result.ptr,sizeof(status),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        stats.d2h+=sizeof(status);
        remag::transfer(remag::Transfer::solver_status,false,sizeof(status));
        stats.block_calls++;
        return status;
    }
};
struct Context {
    cusolverDnHandle_t handle=nullptr;
    Buffer a,b,w,work,info;
    BlockExecutor block;
    Context() {
        int count=0;
        check(cudaGetDeviceCount(&count));
        if (count != 1) throw std::runtime_error("REMAG requires exactly one visible GPU");
        check(cudaSetDevice(0));
        check(cusolverDnCreate(&handle));
        info.reserve(sizeof(int));
    }
    ~Context() { if (handle) cusolverDnDestroy(handle); }
};

template<typename T> int solve(int n, int neig, int ld, const T* h, const T* s, T* z, double* w) {
    if (n < 1 || ld < n || neig > n || neig < -n) return -100;
    std::lock_guard<std::mutex> guard(solver_mutex);
    auto start=std::chrono::steady_clock::now();
    int slot=-1;
    bool export_host=true;
    const bool retain=remag_vector_request(slot,export_host);
    try {
        if (retain && neig <= 0) throw std::runtime_error("Resident request requires eigenvectors");
        // Real and complex paths each reuse buffers. No reallocation for fixed SCF dimensions.
        static Context c;
        const size_t matrix_bytes=size_t(n)*n*sizeof(T);
        c.a.reserve(matrix_bytes); c.b.reserve(matrix_bytes); c.w.reserve(size_t(n)*sizeof(double));
        auto a=static_cast<T*>(c.a.ptr), b=static_cast<T*>(c.b.ptr);
        auto dw=static_cast<double*>(c.w.ptr);
        auto mode=neig>0 ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        const char* option=std::getenv("REMAG_BLOCK_SOLVER");
        const bool blocked=option&&std::strcmp(option,"1")==0;
        int lwork=0;
        if(!blocked) {
          if constexpr (sizeof(T)==sizeof(double)) {
            check(cusolverDnDsygvd_bufferSize(c.handle,CUSOLVER_EIG_TYPE_1,mode,CUBLAS_FILL_MODE_LOWER,
                n,a,n,b,n,dw,&lwork));
          } else {
            check(cusolverDnZhegvd_bufferSize(c.handle,CUSOLVER_EIG_TYPE_1,mode,CUBLAS_FILL_MODE_LOWER,
                n,a,n,b,n,dw,&lwork));
          }
          c.work.reserve(size_t(lwork)*sizeof(T));
        }
        if(!remag_assemble_hs(n,sizeof(T)!=sizeof(double),h,s,a,b)) {
            check(cudaMemcpy2D(a,size_t(n)*sizeof(T),h,size_t(ld)*sizeof(T),size_t(n)*sizeof(T),n,cudaMemcpyHostToDevice));
            remag::transfer(remag::Transfer::solver_matrices,true,matrix_bytes);
            check(cudaMemcpy2D(b,size_t(n)*sizeof(T),s,size_t(ld)*sizeof(T),size_t(n)*sizeof(T),n,cudaMemcpyHostToDevice));
            remag::transfer(remag::Transfer::solver_matrices,true,matrix_bytes);
            stats.h2d += 2*matrix_bytes;
        }
        int info=0;
        if(blocked) {
            info=c.block.solve(n,sizeof(T)!=sizeof(double),a,b,dw);
            if(info) throw std::runtime_error("Experimental block solver device status "+std::to_string(info));
            a=static_cast<T*>(c.block.vectors.ptr);
        } else {
          if constexpr (sizeof(T)==sizeof(double)) {
            check(cusolverDnDsygvd(c.handle,CUSOLVER_EIG_TYPE_1,mode,CUBLAS_FILL_MODE_LOWER,
                n,a,n,b,n,dw,static_cast<T*>(c.work.ptr),lwork,static_cast<int*>(c.info.ptr)));
          } else {
            check(cusolverDnZhegvd(c.handle,CUSOLVER_EIG_TYPE_1,mode,CUBLAS_FILL_MODE_LOWER,
                n,a,n,b,n,dw,static_cast<T*>(c.work.ptr),lwork,static_cast<int*>(c.info.ptr)));
          }
          check(cudaMemcpy(&info,c.info.ptr,sizeof(int),cudaMemcpyDeviceToHost));
          stats.d2h += sizeof(int);
          remag::transfer(remag::Transfer::solver_status,false,sizeof(int));
        }
        if (info) return info;
        if(remag_spectrum_export_required()) {
            check(cudaMemcpy(w,dw,size_t(n)*sizeof(double),cudaMemcpyDeviceToHost));
            stats.d2h += size_t(n)*sizeof(double);
            remag::transfer(remag::Transfer::solver_values,false,size_t(n)*sizeof(double));
        } else {
            if(neig>0&&(!retain||export_host))
                throw std::runtime_error("Resident spectral bank requires a retained-only vector consumer");
            stats.avoided_value_d2h+=size_t(n)*sizeof(double);
        }
        remag_capture_spectrum(w,dw,n);
        if (neig > 0) {
            if (retain) {
                remag_capture_vectors(slot,n,neig,sizeof(T)!=sizeof(double),a);
                stats.retained_calls++;
            }
            if (export_host) {
                check(cudaMemcpy2D(z,size_t(ld)*sizeof(T),a,size_t(n)*sizeof(T),size_t(n)*sizeof(T),neig,cudaMemcpyDeviceToHost));
                stats.d2h += size_t(n)*neig*sizeof(T);
                remag::transfer(remag::Transfer::solver_vectors,false,size_t(n)*neig*sizeof(T));
            } else {
                stats.avoided_vector_d2h += size_t(n)*neig*sizeof(T);
            }
        }
        stats.calls++;
        stats.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"REMAG GPU failure: %s\n",e.what());
        return -101;
    }
}
} // namespace

extern "C" int remag_gpu_enabled() {
    const char* value=std::getenv("REMAG_GPU");
    return !value || std::strcmp(value,"0")!=0;
}
extern "C" int remag_solve_real(int n,int neig,int ld,const double* h,const double* s,double* z,double* w) {
    return solve(n,neig,ld,h,s,z,w);
}
extern "C" int remag_solve_complex(int n,int neig,int ld,const cuDoubleComplex* h,const cuDoubleComplex* s,cuDoubleComplex* z,double* w) {
    return solve(n,neig,ld,h,s,z,w);
}
