// SPDX-License-Identifier: GPL-3.0-or-later
#include "grid.hpp"
#include <algorithm>
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
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);if(graph) cudaGraphDestroy(graph);
        for(void* p:allocations) cudaFree(p);if(stream) cudaStreamDestroy(stream);
    }
    template<class T> T* allocate(size_t count,const T* source=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));allocations.push_back(p);
        if(source) check(cudaMemcpy(p,source,count*sizeof(T),cudaMemcpyHostToDevice));return p;
    }
};
__global__ void select(size_t ns,size_t ng,const int* index,const double* all_d,const double* all_v,
        const double* all_h,const double* all_volume,double* d,double* v,double* h,double* volume) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,k=*index;
    if(i<ns) {d[i]=all_d[k*ns+i];h[i]=all_h[k*ns+i];}
    if(i<ng) v[i]=all_v[k*ng+i];if(i==0) *volume=all_volume[k];
}
__global__ void save(size_t count,const int* index,const double* values,double* all) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count) all[size_t(*index)*count+i]=values[i];
}
__global__ void advance(int* index) {++*index;}
}
struct Check {size_t staged_bytes,exported_bytes;int iterations,graph_nodes,forbidden_nodes;};
static_assert(sizeof(Check)==32,"Unexpected grid checker ABI");
extern "C" int remag_check_grid_graph(int cycles,int ng,int nh,const double* host_d,const double* host_v,
        const double* host_h,const double* host_volume,double* host_rho,double* output_h,Check* info) {
    try {
        if(cycles<1||cycles>8||ng<1||ng>4||ng>nh||!host_d||!host_v||!host_h||!host_volume||
           !host_rho||!output_h||!info) return -1;
        auto view=remag::grid_contraction_view();const auto& p=view.shape;
        *info={};Context c;
        size_t ns=size_t(p.nnz)*nh,grid_size=size_t(p.np)*p.nsp*ng;
        auto all_d=c.allocate<double>(ns*cycles,host_d),all_v=c.allocate<double>(grid_size*cycles,host_v);
        auto all_h=c.allocate<double>(ns*cycles,host_h),all_volume=c.allocate<double>(cycles,host_volume);
        auto d=c.allocate<double>(ns),v=c.allocate<double>(grid_size),h=c.allocate<double>(ns),volume=c.allocate<double>(1);
        auto rho=c.allocate<double>(grid_size),saved_rho=c.allocate<double>(grid_size*cycles),saved_h=c.allocate<double>(ns*cycles);
        int zero=0;auto index=c.allocate<int>(1,&zero);
        check(cudaDeviceSynchronize());check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
        select<<<(std::max(ns,grid_size)+255)/256,256,0,c.stream>>>(ns,grid_size,index,all_d,all_v,all_h,all_volume,d,v,h,volume);
        check(cudaGetLastError());check(remag::contract_density_device(view,ng,nh,d,rho,c.stream));
        check(remag::add_grid_potential_device(view,ng,nh,v,0.,volume,h,c.stream));
        save<<<(grid_size+255)/256,256,0,c.stream>>>(grid_size,index,rho,saved_rho);
        save<<<(ns+255)/256,256,0,c.stream>>>(ns,index,h,saved_h);
        advance<<<1,1,0,c.stream>>>(index);check(cudaGetLastError());check(cudaStreamEndCapture(c.stream,&c.graph));
        size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(c.graph,nodes.data(),&count));info->graph_nodes=int(count);
        for(auto node:nodes) {
            cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeEmpty) info->forbidden_nodes++;
        }
        if(info->forbidden_nodes) throw std::runtime_error("Forbidden grid graph node");
        check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphUpload(c.executable,c.stream));
        for(int i=0;i<cycles;i++) check(cudaGraphLaunch(c.executable,c.stream));
        check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(host_rho,saved_rho,grid_size*cycles*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output_h,saved_h,ns*cycles*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&info->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));
        info->staged_bytes=((2*ns+grid_size+1)*cycles)*sizeof(double)+sizeof(int);
        info->exported_bytes=(ns+grid_size)*cycles*sizeof(double)+sizeof(int);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Captured grid: %s\n",e.what());return -1;}
}
