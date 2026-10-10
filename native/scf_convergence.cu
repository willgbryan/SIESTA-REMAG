// SPDX-License-Identifier: GPL-3.0-or-later
#include "scf_convergence.hpp"
#include <cmath>
#include <stdexcept>

namespace remag {
namespace {
struct State {uint64_t epoch;double harris,free_energy;int iteration,energy_flags,status;};
void check(cudaError_t code) {if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));}
__global__ void evaluate(const ScfConvergenceProblem* p,const ScfScalarsResult* in,State* s,ScfConvergenceResult* out) {
    *out={};
    if(p->epoch&&p->epoch!=s->epoch) {*s={};s->epoch=p->epoch;}
    out->iteration=s->iteration;out->energy_flags=s->energy_flags;
    if(s->status) {out->status=s->status;return;}
    bool valid=p->epoch&&p->criteria>0&&p->criteria<32&&p->reserved==0&&p->minimum>=0&&
               p->maximum>0&&p->minimum<=p->maximum;
    for(double t:p->tolerances) valid=valid&&isfinite(t)&&t>=0.;
    if(!valid) {out->status=s->status=1;return;}
    if(s->iteration>=p->maximum) {out->status=s->status=3;return;}
    bool finite=in->status==0;
    for(double v:in->values) finite=finite&&isfinite(v);
    if(!finite) {out->status=s->status=2;return;}
    if(s->iteration) {
        if(fabs(s->harris-in->values[3])<p->tolerances[3]) s->energy_flags|=1;
        if(fabs(s->free_energy-in->values[4])<p->tolerances[4]) s->energy_flags|=2;
    }
    s->harris=in->values[3];s->free_energy=in->values[4];s->iteration++;
    bool matched=true;
    for(int i=0;i<3;i++) if(p->criteria&(1<<i)) matched=matched&&in->values[6+i]<p->tolerances[i];
    if(p->criteria&8) matched=matched&&(s->energy_flags&1);
    if(p->criteria&16) matched=matched&&(s->energy_flags&2);
    out->iteration=s->iteration;out->energy_flags=s->energy_flags;out->criteria_satisfied=matched;
    out->converged=matched&&s->iteration>=p->minimum;
    out->mix=!out->converged;
    out->continue_loop=!out->converged&&s->iteration<p->maximum;
    out->exhausted=!out->converged&&s->iteration==p->maximum;
}
}
ScfConvergencePlan::ScfConvergencePlan(cudaStream_t stream):stream_(stream) {
    check(cudaMalloc(&state_,sizeof(State)));
    try {check(cudaMemsetAsync(state_,0,sizeof(State),stream_));}
    catch(...) {cudaFree(state_);state_=nullptr;throw;}
}
ScfConvergencePlan::~ScfConvergencePlan() {if(state_) cudaFree(state_);}
size_t ScfConvergencePlan::workspace_bytes() const {return sizeof(State);}
int ScfConvergencePlan::apply(const ScfConvergenceProblem* problem,const ScfScalarsResult* scalars,ScfConvergenceResult* result) {
    if(!problem||!scalars||!result) return int(cudaErrorInvalidValue);
    evaluate<<<1,1,0,stream_>>>(problem,scalars,static_cast<State*>(state_),result);
    return int(cudaGetLastError());
}
}
