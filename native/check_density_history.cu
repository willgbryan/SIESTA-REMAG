// SPDX-License-Identifier: GPL-3.0-or-later
#include "density_history.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Context {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    std::vector<void*> buffers;
    Context(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Context(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
        for(auto p:buffers)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    template<class T>T* buffer(size_t n,const T* data=nullptr) {
        T* p=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&p),(n?n:1)*sizeof(T)));buffers.push_back(p);
        if(data&&n)check(cudaMemcpyAsync(p,data,n*sizeof(T),cudaMemcpyHostToDevice,stream));return p;
    }
};
__global__ void resize_result(remag::DensityHistoryResult* result,int status) {
    *result=remag::DensityHistoryResult{};result->status=status;
}
}
// A whole sequence is staged once. Optional capture contains operations and
// device snapshots only. Downloads happen after the complete sequence.
extern "C" int remag_check_density_history(const remag::DensityHistoryConfig* config,int steps,int captured,int initial_capacity,
        const int* operations,const remag::DensityHistoryCommand* commands,const int* convergence,
        const int* counts,const int* starts,const int* columns,const double* densities,const double* xyz,
        double* outputs,double* coeff,double* history,double* positions,int* topology,
        remag::DensityHistoryResult* results,remag::DensityHistoryResult* snapshots,int* allocations,size_t* report) {
    try {
        if(!config||!operations||!commands||!convergence||!counts||!starts||!columns||!densities||!xyz||
           !outputs||!coeff||!history||!positions||!topology||!results||!snapshots||!allocations||!report||steps<1||steps>1000||
           captured<0||captured>2)return -1;
        const auto c=*config;
        if(c.capacity<1||c.capacity>1000000||c.atoms<1||c.atoms>10000||c.rows<1||c.rows>100000||
           c.depth<0||c.depth>32||c.spins<1||c.spins>8||initial_capacity<1||initial_capacity>c.capacity)return -1;
        const size_t slot=size_t(c.capacity)*c.spins,dims=size_t(3)*c.atoms,bank=slot*c.depth,geom=dims*c.depth;
        const size_t topology_size=size_t(2)*c.rows+c.capacity;
        Context context;auto initial=c;initial.capacity=initial_capacity;
        remag::DensityHistoryPlan plan(initial,context.stream);
        auto p=context.buffer(steps,commands);auto converged=context.buffer(steps,convergence);
        auto num=context.buffer(size_t(steps)*c.rows,counts),ptr=context.buffer(size_t(steps)*c.rows,starts);
        auto col=context.buffer(size_t(steps)*c.capacity,columns);
        auto d=context.buffer(size_t(steps)*slot,densities),x=context.buffer(size_t(steps)*dims,xyz);
        auto out=context.buffer<double>(size_t(steps)*slot),co=context.buffer<double>(size_t(steps)*c.depth);
        auto hist=context.buffer<double>(size_t(steps)*bank),pos=context.buffer<double>(size_t(steps)*geom);
        auto topo=context.buffer<int>(size_t(steps)*topology_size);
        auto result=context.buffer<remag::DensityHistoryResult>(steps),snapshot=context.buffer<remag::DensityHistoryResult>(steps);
        check(cudaMemsetAsync(out,0xff,size_t(steps)*slot*8,context.stream));
        check(cudaMemsetAsync(co,0xff,size_t(steps)*c.depth*8,context.stream));
        check(cudaStreamSynchronize(context.stream));
        auto apply=[&](int i) {
                int code;
                switch(operations[i]) {
                case 0:code=plan.reset(p+i,result+i);break;
                case 1:code=plan.advance(p+i,num+size_t(i)*c.rows,ptr+size_t(i)*c.rows,col+size_t(i)*c.capacity,result+i);break;
                case 2:code=plan.push(p+i,d+size_t(i)*slot,x+size_t(i)*dims,converged+i,result+i);break;
                case 3:code=plan.extrapolate(p+i,x+size_t(i)*dims,out+size_t(i)*slot,co+size_t(i)*c.depth,result+i);break;
                case 4:
                    code=plan.reserve(commands[i].capacity);
                    resize_result<<<1,1,0,context.stream>>>(result+i,code);
                    code=int(cudaGetLastError());break;
                default:throw std::runtime_error("Invalid probe operation");
                }
                check(static_cast<cudaError_t>(code));
                allocations[i]=plan.allocated_capacity();
                auto t=topo+size_t(i)*topology_size;
                check(static_cast<cudaError_t>(plan.inspect(hist+size_t(i)*bank,pos+size_t(i)*geom,t,t+c.rows,t+2*c.rows,snapshot+i,c.capacity)));
        };
        report[0]=report[1]=0;
        auto capture=[&](int first,int end) {
            check(cudaStreamBeginCapture(context.stream,cudaStreamCaptureModeGlobal));
            for(int i=first;i<end;i++)apply(i);
            check(cudaStreamEndCapture(context.stream,&context.graph));
            size_t count=0;check(cudaGraphGetNodes(context.graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);
            check(cudaGraphGetNodes(context.graph,nodes.data(),&count));report[0]+=count;
            for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)report[1]++;}
            check(cudaGraphInstantiate(&context.executable,context.graph,0));check(cudaGraphLaunch(context.executable,context.stream));
            check(cudaStreamSynchronize(context.stream));
            check(cudaGraphExecDestroy(context.executable));context.executable=nullptr;
            check(cudaGraphDestroy(context.graph));context.graph=nullptr;
        };
        if(captured==1)capture(0,steps);
        else for(int i=0;i<steps;i++) {
            // Geometry-boundary resizing is outside graphs. Subsequent
            // operations capture new pointers after the old graph is retired.
            if(captured==2&&operations[i]!=4)capture(i,i+1);else apply(i);
        }
        check(cudaStreamSynchronize(context.stream));
        report[2]=plan.workspace_bytes();
        auto copy=[](auto* h,const auto* gpu,size_t n){if(n)check(cudaMemcpy(h,gpu,n*sizeof(*h),cudaMemcpyDeviceToHost));};
        copy(outputs,out,size_t(steps)*slot);copy(coeff,co,size_t(steps)*c.depth);copy(history,hist,size_t(steps)*bank);
        copy(positions,pos,size_t(steps)*geom);copy(topology,topo,size_t(steps)*topology_size);
        copy(results,result,steps);copy(snapshots,snapshot,steps);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"Density history probe: %s\n",e.what());return -1;}
}
