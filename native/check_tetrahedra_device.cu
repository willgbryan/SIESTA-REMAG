// SPDX-License-Identifier: GPL-3.0-or-later
// Test-only adapter: tetrahedron occupations on the GPU for host-supplied
// bands, run directly or as a captured graph launched twice. Never called by SIESTA.
#include "tetrahedra_device.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
}
extern "C" int remag_check_tetra_device(int method,const int* grid,const double* displacement,const double* cell,int nk,
        const double* kpoints,const double* weights,int trs,int nspinor,int maxspn,int maxe,int ne,const double* energies,
        double charge,double* occupations,double* fermi,int graph_mode,int* status) {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;std::vector<void*> buffers;
    int code=0;
    try {
        remag::TetraGrid g;
        *status=remag::tetra_grid(method,grid,displacement,cell,nk,kpoints,weights,trs,g);
        if(*status) return 0;
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        remag::TetraDevicePlan plan(g,method,nspinor,maxspn,maxe,ne,stream);
        const size_t size=size_t(maxe)*maxspn*nk;
        auto alloc=[&](size_t bytes) {void* p=nullptr;check(cudaMalloc(&p,bytes));buffers.push_back(p);return p;};
        auto e=static_cast<double*>(alloc(size*sizeof(double))),q=static_cast<double*>(alloc(size*sizeof(double)));
        auto p=static_cast<remag::FermiProblem*>(alloc(sizeof(remag::FermiProblem)));
        auto r=static_cast<remag::FermiResult*>(alloc(sizeof(remag::FermiResult)));
        const remag::FermiProblem problem{nspinor,maxspn,nk,maxe,ne,0.,charge};
        check(cudaMemcpy(p,&problem,sizeof(problem),cudaMemcpyHostToDevice));
        check(cudaMemcpy(e,energies,size*sizeof(double),cudaMemcpyHostToDevice));
        check(cudaMemset(q,0xff,size*sizeof(double)));
        if(graph_mode) {
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            const int applied=plan.apply(p,e,q,r);
            check(cudaStreamEndCapture(stream,&graph));
            if(applied) throw std::runtime_error("tetrahedron apply failed during capture");
            check(cudaGraphInstantiate(&exec,graph,0));
            for(int i=0;i<2;i++) check(cudaGraphLaunch(exec,stream));
        } else if(const int applied=plan.apply(p,e,q,r))
            throw std::runtime_error(std::string("tetrahedron apply failed: ")+cudaGetErrorString(cudaError_t(applied)));
        check(cudaStreamSynchronize(stream));
        remag::FermiResult result{};
        check(cudaMemcpy(&result,r,sizeof(result),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(occupations,q,size*sizeof(double),cudaMemcpyDeviceToHost));
        *fermi=result.fermi;*status=result.status;
    } catch(const std::exception& error) {std::fprintf(stderr,"REMAG tetrahedron check: %s\n",error.what());code=-1;}
    if(exec) cudaGraphExecDestroy(exec);
    if(graph) cudaGraphDestroy(graph);
    for(void* b:buffers) cudaFree(b);
    if(stream) cudaStreamDestroy(stream);
    return code;
}
