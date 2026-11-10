// SPDX-License-Identifier: GPL-3.0-or-later
// Isolated timing adapter for the unchanged production BlockJacobiPlan.
#include "block_jacobi.hpp"
#include <cuComplex.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Batch {
    int n, count, lanes;
    size_t elements, seed_size = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    cuDoubleComplex *h = nullptr, *s = nullptr, *z = nullptr;
    double* w = nullptr;
    remag::BlockJacobiResult* results = nullptr;
    unsigned char* seeds = nullptr;
    int* valid = nullptr;
    std::vector<void*> allocations;
    std::vector<std::unique_ptr<remag::BlockJacobiPlan>> plans;

    Batch(int dimension, int jobs) : n(dimension), count(jobs), lanes(std::min(8, jobs)),
                                    elements(size_t(n)*n*jobs) {}
    ~Batch() {
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        plans.clear();
        for (void* pointer : allocations) cudaFree(pointer);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t size) {
        T* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), size*sizeof(T)));
        allocations.push_back(pointer);
        return pointer;
    }
    void upload(const void* host_h, const void* host_s) {
        check(cudaMemcpyAsync(h, host_h, elements*sizeof(*h), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(s, host_s, elements*sizeof(*s), cudaMemcpyHostToDevice, stream));
    }
    void initialize(const void* host_h, const void* host_s) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        h = allocate<cuDoubleComplex>(elements); s = allocate<cuDoubleComplex>(elements);
        z = allocate<cuDoubleComplex>(elements); w = allocate<double>(size_t(n)*count);
        results = allocate<remag::BlockJacobiResult>(count);
        valid = allocate<int>(count);
        upload(host_h, host_s);
        check(cudaMemsetAsync(valid, 0, count*sizeof(int), stream));
        for (int lane=0; lane<lanes; ++lane) {
            plans.emplace_back(std::make_unique<remag::BlockJacobiPlan>(n, true, 32, 2e-13, stream));
            // Library priming is untimed and does not publish a warm basis.
            plans.back()->warmup(h, s, results+lane);
        }
        seed_size = plans[0]->seed_bytes();
        seeds = allocate<unsigned char>(seed_size*count);
        check(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t previous = nullptr;
        // Match scf_cycle.cu: independent lane branches, then a wave join.
        // The captured H/S bank replaces assembly; assembly is not timed here.
        for (int first=0; first<count; first+=lanes) {
            std::vector<cudaGraphNode_t> completed;
            for (int lane=0; lane<std::min(lanes, count-first); ++lane) {
                const int job = first+lane;
                const size_t offset = size_t(job)*n*n;
                completed.push_back(plans[lane]->append_seeded(graph, graph, previous,
                    h+offset, s+offset, z+offset, w+size_t(job)*n, results+job,
                    seeds+size_t(job)*seed_size, valid+job));
            }
            check(cudaGraphAddEmptyNode(&previous, graph, completed.data(), completed.size()));
        }
        check(cudaGraphInstantiate(&executable, graph, 0));
        check(cudaGraphUpload(executable, stream));
        check(cudaStreamSynchronize(stream));
    }
};
}

extern "C" void* remag_compare_create(int n, int count, const void* h, const void* s) {
    try {
        if ((n != 96 && n != 112) || count < 1 || count > 216 || !h || !s)
            throw std::runtime_error("Unsupported comparison shape");
        auto batch = std::make_unique<Batch>(n, count);
        batch->initialize(h, s);
        return batch.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CUDA comparison setup: %s\n", e.what()); return nullptr;
    }
}
extern "C" void remag_compare_destroy(void* pointer) { delete static_cast<Batch*>(pointer); }

// resident=1 leaves inputs and eigenpairs on the device during timing. Fetch
// below is then used for independent validation outside the timed region.
extern "C" int remag_compare_solve(void* pointer, int cold, int resident,
        const void* h, const void* s, void* z, double* w,
        remag::BlockJacobiResult* results, double* graph_seconds) {
    try {
        if (!pointer || !graph_seconds || !results || (cold != 0 && cold != 1) || (resident != 0 && resident != 1))
            throw std::runtime_error("Invalid comparison call");
        auto& b = *static_cast<Batch*>(pointer);
        if (!resident) {
            if (!h || !s || !z || !w || !results) throw std::runtime_error("Missing host arrays");
            b.upload(h, s);
        }
        if (cold) check(cudaMemsetAsync(b.valid, 0, b.count*sizeof(int), b.stream));
        check(cudaEventRecord(b.begin, b.stream));
        check(cudaGraphLaunch(b.executable, b.stream));
        check(cudaEventRecord(b.end, b.stream));
        if (!resident) {
            check(cudaMemcpyAsync(z, b.z, b.elements*sizeof(*b.z), cudaMemcpyDeviceToHost, b.stream));
            check(cudaMemcpyAsync(w, b.w, size_t(b.n)*b.count*sizeof(double), cudaMemcpyDeviceToHost, b.stream));
        }
        // Preserve and check solver statuses in both boundaries.
        check(cudaMemcpyAsync(results, b.results, b.count*sizeof(*b.results), cudaMemcpyDeviceToHost, b.stream));
        check(cudaStreamSynchronize(b.stream));
        float milliseconds;
        check(cudaEventElapsedTime(&milliseconds, b.begin, b.end));
        *graph_seconds = milliseconds*1e-3;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CUDA comparison solve: %s\n", e.what()); return -1;
    }
}
extern "C" int remag_compare_fetch(void* pointer, void* z, double* w) {
    try {
        if (!pointer || !z || !w) return -1;
        auto& b = *static_cast<Batch*>(pointer);
        check(cudaMemcpy(z, b.z, b.elements*sizeof(*b.z), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(w, b.w, size_t(b.n)*b.count*sizeof(double), cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CUDA comparison fetch: %s\n", e.what()); return -1;
    }
}
