// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_control.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace remag {
namespace {
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
struct State {ScfControlProblem problem;int seen;};
__device__ bool same(const ScfControlProblem& a,const ScfControlProblem& b) {
    const auto& x=a.convergence;const auto& y=b.convergence;
    bool equal=x.criteria==y.criteria&&x.minimum==y.minimum&&x.maximum==y.maximum&&x.reserved==y.reserved&&x.epoch==y.epoch;
    for(int i=0;i<5;i++) equal=equal&&x.tolerances[i]==y.tolerances[i];
    const auto& m=a.mixer;const auto& n=b.mixer;
    return equal&&m.size==n.size&&m.history==n.history&&m.weight==n.weight&&m.linear_weight==n.linear_weight&&
        m.condition==n.condition&&m.epoch==n.epoch&&a.mix_first==b.mix_first&&a.reserved==b.reserved;
}
__global__ void enter(int size,int mixer_history,int capacity,const ScfControlProblem* p,State* s,
        ScfControlResult* out,PulayResult* mix,cudaGraphConditionalHandle active) {
    if(!s->seen) *out={};
    if(p->convergence.epoch&&(!s->seen||p->convergence.epoch!=s->problem.convergence.epoch)) {
        s->problem=*p;s->seen=1;*out={};
    }
    const auto& c=p->convergence;const auto& m=p->mixer;
    bool valid=c.epoch&&c.epoch==m.epoch&&c.criteria>0&&c.criteria<32&&c.reserved==0&&
        c.minimum>=0&&c.maximum>0&&c.minimum<=c.maximum&&c.maximum<=capacity&&
        m.size==size&&m.history==mixer_history&&isfinite(m.weight)&&m.weight>0.&&m.weight<=1.&&
        isfinite(m.linear_weight)&&m.linear_weight>0.&&m.linear_weight<=1.&&
        isfinite(m.condition)&&m.condition>0.&&m.condition<1.&&
        (p->mix_first==0||p->mix_first==1)&&p->reserved==0;
    for(double t:c.tolerances) valid=valid&&isfinite(t)&&t>=0.;
    if(!valid||!s->seen||!same(*p,s->problem)) {
        out->status=out->status?out->status:1;out->finished=1;out->continue_loop=0;
    }
    if(!out->finished&&out->records>=capacity) {out->status=1;out->finished=1;out->continue_loop=0;}
    if(!out->finished) *mix={};
    cudaGraphSetConditional(active,out->finished?0:1);
}
__global__ void decide_mix(const ScfConvergenceResult* decision,cudaGraphConditionalHandle mix) {
    cudaGraphSetConditional(mix,!decision->status&&decision->mix?1:0);
}
__global__ void first_mix(int size,const ScfControlProblem* p,const ScfConvergenceResult* c,const PulayResult* m,
        const double* input,const double* residual,double* output) {
    if(p->mix_first||c->iteration!=1||m->status) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) output[i]=input[i]+residual[i];
}
__global__ void finish(const ScfControlProblem* p,const ScfScalarsResult* scalars,const ScfConvergenceResult* c,
        const PulayResult* m,ScfControlResult* out,ScfControlStep* history) {
    out->convergence=*c;out->mixer=*m;
    out->status=c->status?2:((c->mix&&m->status)?3:0);
    out->mixed=!out->status&&c->mix;
    out->first_mix_bypassed=out->mixed&&c->iteration==1&&!p->mix_first;
    out->continue_loop=!out->status&&c->continue_loop;
    out->finished=!out->continue_loop;
    const int index=out->records++;
    history[index]={*scalars,*out};
}
cudaGraphNode_t conditional(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,
        cudaGraphConditionalHandle* handle,cudaGraph_t* body) {
    check(cudaGraphConditionalHandleCreate(handle,root,0,cudaGraphCondAssignDefault));
    cudaGraphNodeParams p{};p.type=cudaGraphNodeTypeConditional;p.conditional.handle=*handle;
    p.conditional.type=cudaGraphCondTypeIf;p.conditional.size=1;cudaGraphNode_t node;
    check(cudaGraphAddNode(&node,destination,dependency?&dependency:nullptr,dependency?1:0,&p));
    *body=p.conditional.phGraph_out[0];return node;
}
}
struct ScfControlPlan::Impl {
    int size,mixer_history,capacity;cudaStream_t stream;
    int nodes=0,forbidden=0;
    ScfConvergencePlan convergence;PulayPlan mixer;
    State* state=nullptr;ScfConvergenceResult* decision=nullptr;PulayResult* mixed=nullptr;
    Impl(int n,int h,int cap,cudaStream_t s):size(n),mixer_history(h),capacity(cap),stream(s),convergence(s),mixer(n,h,s) {
        if(!s||cap<1||cap>4096) throw std::runtime_error("Invalid device SCF control stream/history capacity");
        try {
            check(cudaMalloc(&state,sizeof(State)));check(cudaMemsetAsync(state,0,sizeof(State),stream));
            check(cudaMalloc(&decision,sizeof(*decision)));check(cudaMalloc(&mixed,sizeof(*mixed)));
        } catch(...) {release();throw;}
    }
    void release() {if(state) cudaFree(state);if(decision) cudaFree(decision);if(mixed) cudaFree(mixed);}
    ~Impl() {release();}
    cudaGraphNode_t capture(cudaGraph_t destination,cudaGraphNode_t dependency,const std::function<void()>& work) {
        cudaGraph_t captured=nullptr;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        try {work();} catch(...) {cudaStreamEndCapture(stream,&captured);if(captured) cudaGraphDestroy(captured);throw;}
        check(cudaStreamEndCapture(stream,&captured));cudaGraphNode_t node;
        size_t count=0;check(cudaGraphGetNodes(captured,nullptr,&count));std::vector<cudaGraphNode_t> records(count);
        check(cudaGraphGetNodes(captured,records.data(),&count));
        for(auto recorded:records) {
            cudaGraphNodeType type;check(cudaGraphNodeGetType(recorded,&type));nodes++;
            if(type!=cudaGraphNodeTypeKernel&&type!=cudaGraphNodeTypeEmpty) forbidden++;
        }
        if(forbidden) {cudaGraphDestroy(captured);throw std::runtime_error("Forbidden captured SCF control node");}
        auto code=cudaGraphAddChildGraphNode(&node,destination,dependency?&dependency:nullptr,dependency?1:0,captured);
        cudaGraphDestroy(captured);check(code);nodes++;return node;
    }
};
ScfControlPlan::ScfControlPlan(int n,int h,int capacity,cudaStream_t stream):impl_(new Impl(n,h,capacity,stream)) {}
ScfControlPlan::~ScfControlPlan()=default;
int ScfControlPlan::graph_nodes() const {return impl_->nodes;}
int ScfControlPlan::forbidden_nodes() const {return impl_->forbidden;}
size_t ScfControlPlan::workspace_bytes() const {
    return impl_->convergence.workspace_bytes()+impl_->mixer.workspace_bytes()+sizeof(State)+sizeof(ScfConvergenceResult)+sizeof(PulayResult);
}
cudaGraphNode_t ScfControlPlan::append(cudaGraph_t root,cudaGraph_t destination,cudaGraphNode_t dependency,
        const ScfControlProblem* p,const ScfScalarsResult* scalars,const double* input,const double* residual,double* output,
        ScfControlResult* result,ScfControlStep* history) {
    if(!root||!destination||!p||!scalars||!input||!residual||!output||!result||!history||
       input==residual||input==output||residual==output) throw std::runtime_error("Invalid device SCF control bindings");
    auto& i=*impl_;cudaGraphConditionalHandle active=0,mix=0;cudaGraph_t body,mix_body;
    // Create the condition node first, then add the captured setter dependency.
    auto node=conditional(root,destination,nullptr,&active,&body);
    i.nodes++;
    auto entered=i.capture(destination,dependency,[&]{
        enter<<<1,1,0,i.stream>>>(i.size,i.mixer_history,i.capacity,p,i.state,result,i.mixed,active);check(cudaGetLastError());
    });
    check(cudaGraphAddDependencies(destination,&entered,&node,1));
    auto mixing=conditional(root,body,nullptr,&mix,&mix_body);
    i.nodes++;
    auto decided=i.capture(body,nullptr,[&]{
        check(static_cast<cudaError_t>(i.convergence.apply(&p->convergence,scalars,i.decision)));
        decide_mix<<<1,1,0,i.stream>>>(i.decision,mix);check(cudaGetLastError());
    });
    check(cudaGraphAddDependencies(body,&decided,&mixing,1));
    i.capture(mix_body,nullptr,[&]{
        check(static_cast<cudaError_t>(i.mixer.apply(&p->mixer,input,residual,output,i.mixed)));
        first_mix<<<(i.size-1)/256+1,256,0,i.stream>>>(i.size,p,i.decision,i.mixed,input,residual,output);check(cudaGetLastError());
    });
    i.capture(body,mixing,[&]{
        finish<<<1,1,0,i.stream>>>(p,scalars,i.decision,i.mixed,result,history);check(cudaGetLastError());
    });
    return node;
}
}
