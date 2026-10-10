// SPDX-License-Identifier: GPL-3.0-or-later
#include "density_matrix.hpp"
#include <cuComplex.h>
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
__global__ void select(size_t count,const int* index,const double* all,double* values) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count) values[i]=all[size_t(*index)*count+i];
}
__global__ void save(size_t count,const int* index,const double* values,double* all) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count) all[size_t(*index)*count+i]=values[i];
}
__global__ void save_result(const int* index,const remag::DensityNormalizationResult* value,
        remag::DensityNormalizationResult* all) {all[*index]=*value;}
__global__ void advance(int* index) {++*index;}
}
struct Check {size_t staged_bytes,exported_bytes;int iterations,graph_nodes,forbidden_nodes;};
static_assert(sizeof(Check)==32,"Unexpected density checker ABI");
extern "C" int remag_check_density_graph(int cycles,int jobs,int nev,const int* spins,const double* kpoints,
        const void* host_vectors,const double* occupations,const double* energies,double cutoff,
        const double* overlap,double charge,double tolerance,double* output_d,double* output_e,
        remag::DensityNormalizationResult* output_results,Check* info) {
    try {
        if(cycles<1||cycles>8||jobs<1||jobs>128||!spins||!kpoints||!host_vectors||!occupations||
           !energies||!output_d||!output_e||!output_results||!info) return -1;
        auto view=remag::density_matrix_view();const auto& p=view.shape;
        const int dim=p.n*(p.kind>=2?2:1);
        if(dim>2048||nev<1||nev>dim) return -1;
        *info={};Context c;
        const size_t vector_bytes=size_t(dim)*nev*(p.kind?sizeof(cuDoubleComplex):sizeof(double));
        const size_t nq=size_t(jobs)*nev,nd=size_t(p.nnz)*p.nh,ne=size_t(p.nnz)*p.eh;
        auto input=c.allocate<unsigned char>(vector_bytes*jobs,static_cast<const unsigned char*>(host_vectors));
        auto transposed=c.allocate<unsigned char>(vector_bytes);
        auto qs=c.allocate<double>(nq*cycles,occupations),q=c.allocate<double>(nq);
        auto ee=c.allocate<double>(nq,energies);
        auto d=c.allocate<double>(nd),e=c.allocate<double>(ne);
        auto all_d=c.allocate<double>(nd*cycles),all_e=c.allocate<double>(ne*cycles);
        auto result=c.allocate<remag::DensityNormalizationResult>(1);
        auto all_results=c.allocate<remag::DensityNormalizationResult>(cycles);
        double *s=nullptr,*partial=nullptr;
        if(overlap) {s=c.allocate<double>(p.nnz,overlap);partial=c.allocate<double>((size_t(p.nnz)*2+255)/256);}
        int zero=0;auto index=c.allocate<int>(1,&zero);
        check(cudaDeviceSynchronize());check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
        check(remag::clear_density_device(p,d,e,c.stream));
        select<<<(nq+255)/256,256,0,c.stream>>>(nq,index,qs,q);check(cudaGetLastError());
        for(int job=0;job<jobs;job++) {
            check(remag::transpose_vectors_device(dim,nev,p.kind!=0,input+size_t(job)*vector_bytes,transposed,c.stream));
            check(remag::accumulate_density_device(view,nev,spins[job],transposed,q+size_t(job)*nev,ee+size_t(job)*nev,
                cutoff,kpoints[3*job],kpoints[3*job+1],kpoints[3*job+2],d,e,c.stream));
        }
        if(overlap) check(remag::normalize_density_device({p.nnz,p.nh,p.eh,2,charge,tolerance},s,partial,result,d,e,c.stream));
        else check(cudaMemsetAsync(result,0,sizeof(*result),c.stream));
        save<<<(nd+255)/256,256,0,c.stream>>>(nd,index,d,all_d);
        save<<<(ne+255)/256,256,0,c.stream>>>(ne,index,e,all_e);
        save_result<<<1,1,0,c.stream>>>(index,result,all_results);
        advance<<<1,1,0,c.stream>>>(index);check(cudaGetLastError());
        check(cudaStreamEndCapture(c.stream,&c.graph));
        size_t count=0;check(cudaGraphGetNodes(c.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(c.graph,nodes.data(),&count));info->graph_nodes=int(count);
        for(auto node:nodes) {
            cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeMemset&&type!=cudaGraphNodeTypeEmpty) info->forbidden_nodes++;
        }
        if(info->forbidden_nodes) throw std::runtime_error("Forbidden density graph node");
        check(cudaGraphInstantiate(&c.executable,c.graph,0));check(cudaGraphUpload(c.executable,c.stream));
        for(int i=0;i<cycles;i++) check(cudaGraphLaunch(c.executable,c.stream));
        check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(output_d,all_d,nd*cycles*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output_e,all_e,ne*cycles*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output_results,all_results,cycles*sizeof(*result),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&info->iterations,index,sizeof(int),cudaMemcpyDeviceToHost));
        info->staged_bytes=vector_bytes*jobs+nq*(cycles+1)*sizeof(double)+sizeof(int)+(overlap?size_t(p.nnz)*sizeof(double):0);
        info->exported_bytes=(nd+ne)*cycles*sizeof(double)+cycles*sizeof(*result)+sizeof(int);
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"Captured density: %s\n",e.what());return -1;}
}
