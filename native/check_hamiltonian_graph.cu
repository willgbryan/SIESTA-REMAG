// SPDX-License-Identifier: GPL-3.0-or-later
#include "hamiltonian.hpp"
#include <cuComplex.h>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t c) {if(c!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(c));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> allocations;
    Context() {check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context() {
        if(executable) cudaGraphExecDestroy(executable);if(graph) cudaGraphDestroy(graph);
        for(void* p:allocations) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t count,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));allocations.push_back(p);
        if(source) check(cudaMemcpy(p,source,count*sizeof(T),cudaMemcpyHostToDevice));return p;
    }
};
__global__ void select(int size,const int* index,const double* all,double* values) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<size) values[i]=all[size_t(*index)*size+i];
}
__global__ void advance(int* index) {++*index;}
__global__ void save(size_t bytes,int block,int blocks,const int* index,const unsigned char* matrix,unsigned char* all) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<bytes) all[(size_t(*index)*blocks+block)*bytes+i]=matrix[i];
}
}
struct Check {size_t staged_bytes,exported_bytes;int iterations,graph_nodes,forbidden_nodes;};
extern "C" int remag_check_hamiltonian_graph(int cycles,const double* hs,void* output_h,void* output_s,Check* info) {
    try {
        if(cycles<1||cycles>8||!hs||!output_h||!output_s||!info) return -1;
        auto view=remag::hamiltonian_assembly_view();const auto& p=view.shape;
        if(p.ldh>INT_MAX/p.nh||p.n>1024||p.nk>64) return -1;
        *info={};Context c;const int sparse_size=p.ldh*p.nh,spins=p.kind<2?p.nh:1,blocks=p.nk*spins;
        const size_t dim=size_t(p.n)*(p.kind>=2?2:1),bytes=dim*dim*(p.kind?sizeof(cuDoubleComplex):sizeof(double));
        auto all=c.allocate<double>(size_t(sparse_size)*cycles,hs);auto h=c.allocate<double>(sparse_size);
        auto dense_h=c.allocate<unsigned char>(bytes),dense_s=c.allocate<unsigned char>(bytes);
        auto all_h=c.allocate<unsigned char>(bytes*blocks*cycles),all_s=c.allocate<unsigned char>(bytes*blocks*cycles);
        int zero=0;auto index=c.allocate<int>(1,&zero);
        // All borrowed geometry uploads and staging precede capture/launch.
        check(cudaDeviceSynchronize());check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
        select<<<(sparse_size-1)/256+1,256,0,c.stream>>>(sparse_size,index,all,h);check(cudaGetLastError());
        for(int k=0;k<p.nk;k++) for(int spin=0;spin<spins;spin++) {
            check(remag::assemble_hamiltonian_device(view,h,spin,k,dense_h,dense_s,c.stream));
            save<<<(bytes+255)/256,256,0,c.stream>>>(bytes,k*spins+spin,blocks,index,dense_h,all_h);
            save<<<(bytes+255)/256,256,0,c.stream>>>(bytes,k*spins+spin,blocks,index,dense_s,all_s);check(cudaGetLastError());
        }
        advance<<<1,1,0,c.stream>>>(index);check(cudaGetLastError());check(cudaStreamEndCapture(c.stream,&c.graph));
        size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(c.graph,nodes.data(),&count));info->graph_nodes=int(count);
        for(auto node:nodes) {
            cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty) info->forbidden_nodes++;
        }
        if(info->forbidden_nodes) throw std::runtime_error("Forbidden matrix assembly graph node");
        check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphUpload(c.executable,c.stream));
        for(int i=0;i<cycles;i++) check(cudaGraphLaunch(c.executable,c.stream));
        check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(output_h,all_h,bytes*blocks*cycles,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output_s,all_s,bytes*blocks*cycles,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&info->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));
        info->staged_bytes=size_t(sparse_size)*cycles*sizeof(double)+sizeof(int);
        info->exported_bytes=2*bytes*blocks*cycles+sizeof(int);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Captured matrix assembly: %s\n",e.what());return -1;}
}
