// SPDX-License-Identifier: GPL-3.0-or-later
// Device implementation of the FD branch of pinned SIESTA Src/fermid.F.
// One block owns bisection state; its stopping decision never leaves the GPU.
// The opt-in SCF bridge manages staging, retained inputs and temporary exports.
#include "occupations.hpp"
#include <cfloat>
#include <climits>
#include <cmath>

namespace {
constexpr int threads=256;
constexpr double tolerance=1.e-10;
template<int operation> __device__ double reduce(double value,double* scratch) {
    int tid=threadIdx.x;
    scratch[tid]=value;
    __syncthreads();
    for(int offset=threads/2;offset;offset/=2) {
        if(tid<offset) {
            if constexpr(operation==0) scratch[tid]+=scratch[tid+offset];
            else if constexpr(operation==1) scratch[tid]=fmin(scratch[tid],scratch[tid+offset]);
            else scratch[tid]=fmax(scratch[tid],scratch[tid+offset]);
        }
        __syncthreads();
    }
    double answer=scratch[0];
    __syncthreads();
    return answer;
}
__device__ double probability(double x) {
    if(x>100.) return 0.;
    if(x< -100.) return 1.;
    return 1./(1.+exp(x));
}
__global__ void fermi(const remag::FermiProblem* input,const double* energies,
                     const double* weights,double* occupations,remag::FermiResult* result) {
    __shared__ remag::FermiProblem p;
    __shared__ double scratch[threads],lower,upper,ef,total,temperature,inverse_temperature;
    __shared__ int done,status,iterations;
    int tid=threadIdx.x;
    if(tid==0) {
        p=*input;
        *result={0.,0.,0.,1,0};
        done=0;status=0;iterations=0;
    }
    __syncthreads();
    if(p.nk<1||p.maxe<1||p.ne<1||p.ne>p.maxe||p.maxspn<1||p.maxspn>2||
       (p.nspinor!=1&&p.nspinor!=2)||(p.nspinor==1&&p.maxspn!=1)||
       static_cast<long long>(p.maxe)*p.maxspn*p.nk>INT_MAX) return;
    const int count=p.ne*p.maxspn*p.nk,size=p.maxe*p.maxspn*p.nk;
    const double degeneracy=p.nspinor==1?2.:1.;
    for(int i=tid;i<size;i+=threads) occupations[i]=0.;
    double lo=DBL_MAX,hi=-DBL_MAX,bad=0.,sum_weights=0.;
    for(int k=tid;k<p.nk;k+=threads) {
        // Zero k weights make upstream's entropy division undefined; these
        // settings are outside this routine's explicit admissible contract.
        if(!isfinite(weights[k])||weights[k]<=0.) bad+=1.;
        sum_weights+=weights[k];
    }
    for(int i=tid;i<count;i+=threads) {
        int band=i%p.ne,block=i/p.ne;
        double e=energies[band+p.maxe*block];
        if(!isfinite(e)) bad+=1.;
        lo=fmin(lo,e);hi=fmax(hi,e);
    }
    bad=reduce<0>(bad,scratch);
    sum_weights=reduce<0>(sum_weights,scratch);
    lo=reduce<1>(lo,scratch);hi=reduce<2>(hi,scratch);
    if(tid==0) {
        if(bad||!isfinite(p.temperature)||!isfinite(p.charge)||p.charge<0.||!isfinite(sum_weights)||sum_weights<=0.)
            status=2;
        total=sum_weights*degeneracy*p.maxspn*p.ne;
        if(!status) {
            if(fabs(total-p.charge)<tolerance) done=1;
            else if(total<p.charge) status=3;
        }
        temperature=fmax(p.temperature,1.e-6);
        inverse_temperature=1./temperature;
        double range=temperature*sqrt(-log(tolerance*.01));
        lower=lo-range;upper=hi+range;ef=hi;
    }
    __syncthreads();
    if(status) {
        if(tid==0) result->status=status;
        return;
    }
    if(done) {
        for(int i=tid;i<count;i+=threads) {
            int band=i%p.ne,block=i/p.ne,k=block/p.maxspn;
            occupations[band+p.maxe*block]=degeneracy*weights[k];
        }
        if(tid==0) *result={ef,0.,total,0,0};
        return;
    }
    for(int iteration=1;iteration<=150;iteration++) {
        if(tid==0) ef=.5*(lower+upper);
        __syncthreads();
        double charge=0.;
        for(int i=tid;i<count;i+=threads) {
            int band=i%p.ne,block=i/p.ne,k=block/p.maxspn,index=band+p.maxe*block;
            double q=degeneracy*weights[k]*probability((energies[index]-ef)*inverse_temperature);
            occupations[index]=q;charge+=q;
        }
        charge=reduce<0>(charge,scratch);
        if(tid==0) {
            total=charge;iterations=iteration;
            done=fabs(charge-p.charge)<tolerance;
            if(charge<=p.charge) lower=ef;
            if(charge>=p.charge) upper=ef;
        }
        __syncthreads();
        if(done) break;
    }
    if(!done) {
        if(tid==0) *result={ef,0.,total,4,iterations};
        return;
    }
    double entropy=0.;
    for(int i=tid;i<count;i+=threads) {
        int band=i%p.ne,block=i/p.ne,k=block/p.maxspn,index=band+p.maxe*block;
        double w=(p.nspinor==1 ? .5*occupations[index] : occupations[index])/weights[k];
        double occupied=fmax(w,1.e-15),empty=fmax(1.-occupied,1.e-15);
        entropy+=degeneracy*weights[k]*(-occupied*log(occupied)-empty*log(empty));
    }
    entropy=reduce<0>(entropy,scratch);
    if(tid==0) *result={ef,entropy,total,0,iterations};
}
}
cudaError_t remag::fermi_device(const FermiProblem* problem,const double* energies,
                              const double* weights,double* occupations,
                              FermiResult* result,cudaStream_t stream) {
    fermi<<<1,threads,0,stream>>>(problem,energies,weights,occupations,result);
    return cudaGetLastError();
}
