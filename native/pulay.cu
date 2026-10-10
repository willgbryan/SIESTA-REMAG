// SPDX-License-Identifier: GPL-3.0-or-later
// Device-owned stable Pulay history, matching pinned m_mixing.F90 stack order.
#include "pulay.hpp"
#include "pulay_math.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace remag {
namespace {
constexpr int threads=256;
struct State {PulayProblem problem;int iteration,count,next,status;};
void check(cudaError_t error) {if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));}
__global__ void initialize(int size,int history,const PulayProblem* p,State* state,PulayResult* r) {
    *r=PulayResult{};
    if(p->size!=size||p->history!=history||!p->epoch||
       !isfinite(p->weight)||p->weight<=0.||p->weight>1.||
       !isfinite(p->linear_weight)||p->linear_weight<=0.||p->linear_weight>1.||
       !isfinite(p->condition)||p->condition<0.||p->condition>=1.) {r->status=1;return;}
    if(p->epoch!=state->problem.epoch) {
        *state=State{};state->problem=*p;
    }
    if(state->problem.weight!=p->weight||state->problem.linear_weight!=p->linear_weight||
       state->problem.condition!=p->condition) {r->status=1;return;}
    if(state->status) {r->status=state->status;return;}
    if(state->iteration==INT_MAX) {r->status=1;return;}
    const int capacity=history-1;
    r->iteration=state->iteration+1;
    r->history_size=state->iteration?min(state->count+1,capacity):0;
    r->next=state->iteration?(state->next+1)%capacity:0;
    r->oldest=r->history_size==capacity?r->next:0;
    r->active=1;
}
__global__ void update_history(int size,const State* state,const PulayResult* r,
        const double* xin,const double* residual,const double* previous_f,const double* previous_out,
        double* dx,double* df,int* status) {
    if(!r->active) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        double x=xin[i],f=residual[i];
        if(!isfinite(x)||!isfinite(f)||!isfinite(x+f)) {atomicMax(status,2);x=f=0.;}
        if(state->iteration) {
            const size_t index=size_t(state->next)*size+i;
            // Same cancellation/order as the original retained stack.
            dx[index]=previous_f[i]-previous_out[i]+x;df[index]=f-previous_f[i];
            if(!isfinite(dx[index])||!isfinite(df[index])) atomicMax(status,2);
        }
    }
}
__device__ void pair(int field,int capacity,int& row,int& col) {
    const int triangular=capacity*(capacity+1)/2;
    if(field>=triangular) {row=field-triangular;col=-1;return;}
    row=0;while(field>row) {field-=row+1;row++;}col=field;
}
__global__ void inner_products(int size,int capacity,int blocks,const PulayResult* r,
                               const double* df,const double* residual,double* partial) {
    __shared__ double sums[threads];
    int row,col;pair(blockIdx.y,capacity,row,col);
    double sum=0.;
    if(r->active&&row<r->history_size&&(col<0||col<r->history_size)) {
        const int a=(r->oldest+row)%capacity,b=col<0?0:(r->oldest+col)%capacity;
        for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x)
            sum+=df[size_t(a)*size+i]*(col<0?residual[i]:df[size_t(b)*size+i]);
    }
    sums[threadIdx.x]=sum;__syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) sums[threadIdx.x]+=sums[threadIdx.x+step];
        __syncthreads();
    }
    if(!threadIdx.x) partial[size_t(blockIdx.y)*blocks+blockIdx.x]=sums[0];
}
__global__ void collect_products(int capacity,int blocks,int fields,const double* partial,
                                 double* gram,double* projection,PulayResult* r) {
    r->active=0;
    if(r->status) return;
    const int n=r->history_size;
    for(int field=0;field<fields;field++) {
        int row,col;pair(field,capacity,row,col);
        if(row>=n||(col>=n)) continue;
        double sum=0.;for(int block=0;block<blocks;block++) sum+=partial[size_t(field)*blocks+block];
        if(col<0) {projection[row]=sum;r->projection_max=fmax(r->projection_max,fabs(sum));}
        else {gram[row*n+col]=sum;gram[col*n+row]=sum;r->gram_max=fmax(r->gram_max,fabs(sum));}
    }
}
__global__ void coefficients(const PulayProblem* p,const double* gram,const double* projection,
                             double* c,PulayResult* r) {
    if(r->status) return;
    auto solve=pulay_coefficients(r->history_size,gram,projection,p->condition,c);
    r->status=solve.status;r->method=solve.method;r->rank=solve.rank;r->active=solve.status==0;
    r->inverse_first=solve.inverse_first;r->coefficient_first=solve.coefficient_first;
}
__global__ void create_trial(int size,int capacity,const PulayProblem* p,const double* xin,const double* residual,
        const double* dx,const double* df,const double* c,double* trial,PulayResult* r) {
    if(!r->active) return;
    const double weight=r->history_size?p->weight:p->linear_weight;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        double value=xin[i]+weight*residual[i];
        for(int j=0;j<r->history_size;j++) {
            const size_t index=size_t((r->oldest+j)%capacity)*size+i;
            value+=c[j]*(dx[index]+weight*df[index]);
        }
        trial[i]=value;if(!isfinite(value)) atomicMax(&r->status,2);
    }
}
__global__ void commit(int size,const PulayResult* r,const double* xin,const double* residual,const double* trial,
                       double* next,double* previous_f,double* previous_out) {
    if(r->status) return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        next[i]=trial[i];previous_f[i]=residual[i];previous_out[i]=xin[i]+residual[i];
    }
}
__global__ void finish(const PulayResult* r,State* s) {
    s->status=r->status;
    if(!r->status) {s->iteration=r->iteration;s->count=r->history_size;s->next=r->next;}
}
}

PulayPlan::PulayPlan(int size,int history,cudaStream_t stream):size_(size),history_(history),stream_(stream) {
    if(size<1||history<2||history>pulay_max_history) throw std::runtime_error("Invalid Pulay plan shape/history");
    blocks_=std::min(256,(size-1)/threads+1);const int capacity=history-1;
    fields_=capacity*(capacity+1)/2+capacity;
    auto allocate=[](auto** p,size_t n) {check(cudaMalloc(reinterpret_cast<void**>(p),n));};
    try {
        const size_t bytes=size_t(size)*sizeof(double);
        allocate(&state_,sizeof(State));check(cudaMemsetAsync(state_,0,sizeof(State),stream_));
        allocate(&previous_residual_,bytes);allocate(&previous_output_,bytes);allocate(&trial_,bytes);
        allocate(&delta_x_,bytes*capacity);allocate(&delta_f_,bytes*capacity);
        allocate(&coefficients_,capacity*sizeof(double));allocate(&partial_,size_t(fields_)*blocks_*sizeof(double));
        allocate(&gram_,size_t(capacity)*capacity*sizeof(double));allocate(&projection_,capacity*sizeof(double));
    } catch(...) {release();throw;}
}
void PulayPlan::release() {
    void* pointers[]={state_,previous_residual_,previous_output_,delta_x_,delta_f_,coefficients_,partial_,trial_,gram_,projection_};
    for(auto p:pointers) if(p) cudaFree(p);
}
PulayPlan::~PulayPlan() {release();}
size_t PulayPlan::workspace_bytes() const {
    const size_t capacity=history_-1;
    return sizeof(State)+(size_t(size_)*(3+2*capacity)+size_t(fields_)*blocks_+2*capacity+capacity*capacity)*sizeof(double);
}
int PulayPlan::apply(const PulayProblem* p,const double* xin,const double* residual,double* next,PulayResult* r) {
    if(!p||!xin||!residual||!next||!r||xin==residual||next==xin||next==residual) return int(cudaErrorInvalidValue);
    const int capacity=history_-1;auto state=static_cast<State*>(state_);
    initialize<<<1,1,0,stream_>>>(size_,history_,p,state,r);
    update_history<<<blocks_,threads,0,stream_>>>(size_,state,r,xin,residual,previous_residual_,previous_output_,delta_x_,delta_f_,&r->status);
    inner_products<<<dim3(blocks_,fields_),threads,0,stream_>>>(size_,capacity,blocks_,r,delta_f_,residual,partial_);
    collect_products<<<1,1,0,stream_>>>(capacity,blocks_,fields_,partial_,gram_,projection_,r);
    coefficients<<<1,1,0,stream_>>>(p,gram_,projection_,coefficients_,r);
    create_trial<<<blocks_,threads,0,stream_>>>(size_,capacity,p,xin,residual,delta_x_,delta_f_,coefficients_,trial_,r);
    commit<<<blocks_,threads,0,stream_>>>(size_,r,xin,residual,trial_,next,previous_residual_,previous_output_);
    finish<<<1,1,0,stream_>>>(r,state);
    return int(cudaGetLastError());
}
}
