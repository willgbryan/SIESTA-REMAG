// SPDX-License-Identifier: GPL-3.0-or-later
// Experimental complex FP32 seed solver. One threadgroup owns one matrix.
// A and V use device storage: 96/112-state matrices exceed threadgroup memory.
#include <metal_stdlib>
using namespace metal;

float2 mul(float2 a, float2 b) {
    return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
float2 adj(float2 a) { return float2(a.x, -a.y); }
struct Params { uint n, count, maximum; float tolerance; };

kernel void jacobi(device float2* matrices [[buffer(0)]],
                   device float2* vectors [[buffer(1)]],
                   device float* values [[buffer(2)]],
                   device int* status [[buffer(3)]],
                   device uint* sweeps [[buffer(4)]],
                   device float* residual [[buffer(5)]],
                   constant Params& cfg [[buffer(6)]],
                   uint job [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    const uint n=cfg.n;
    device float2* a=matrices+job*n*n;
    device float2* v=vectors+job*n*n;
    threadgroup float2 reduction[256], sine[64];
    threadgroup float cosine[64], dp[64], dq[64], scale;
    threadgroup uint p[64], q[64], invalid[256];
    threadgroup int done;
    float largest=0;
    uint bad=0;
    for(uint i=tid;i<n*n;i+=256) {
        float2 x=a[i];
        bad |= uint(!all(isfinite(x)));
        largest=max(largest,length(x));
        v[i]=float2(i%n==i/n ? 1.f : 0.f,0);
    }
    reduction[tid]=float2(largest,0); invalid[tid]=bad;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for(uint step=128;step;step/=2) {
        if(tid<step) {
            reduction[tid].x=max(reduction[tid].x,reduction[tid+step].x);
            invalid[tid] |= invalid[tid+step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) {
        scale=reduction[0].x>0 ? reduction[0].x : 1.f;
        done=invalid[0] ? -1 : 0;
        status[job]=1; sweeps[job]=0; residual[job]=NAN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(done<0) return;
    for(uint i=tid;i<n*n;i+=256) a[i]/=scale;
    threadgroup_barrier(mem_flags::mem_device);
    for(uint sweep=0;sweep<=cfg.maximum;++sweep) {
        float total=0,off=0;
        for(uint i=tid;i<n*n;i+=256) {
            float x=dot(a[i],a[i]); total+=x;
            if(i%n!=i/n) off+=x;
        }
        reduction[tid]=float2(total,off);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint step=128;step;step/=2) {
            if(tid<step) reduction[tid]+=reduction[tid+step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(tid==0) {
            float2 norms=reduction[0];
            done=!all(isfinite(norms)) ? -1 :
                (norms.y<=cfg.tolerance*cfg.tolerance*norms.x ? 1 : 0);
            residual[job]=norms.x>0 ? sqrt(norms.y/norms.x) : 0;
            sweeps[job]=sweep;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(done || sweep==cfg.maximum) break;
        for(uint round=0;round<n-1;++round) {
            if(tid<n/2) {
                uint ip=tid ? (round+tid)%(n-1) : n-1;
                uint iq=tid ? (round+n-1-tid)%(n-1) : round;
                p[tid]=min(ip,iq); q[tid]=max(ip,iq);
                ip=p[tid]; iq=q[tid];
                float app=a[ip*n+ip].x, aqq=a[iq*n+iq].x;
                float2 b=a[iq*n+ip];
                float ab=length(b),delta=aqq-app;
                float t=ab>0 ? copysign(2*ab/(abs(delta)+length(float2(delta,2*ab))),delta) : 0;
                float c=1/sqrt(1+t*t);
                cosine[tid]=c; sine[tid]=ab>0 ? b*(t*c/ab) : float2(0);
                dp[tid]=app-t*ab; dq[tid]=aqq+t*ab;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Disjoint column pairs, followed by disjoint row pairs. Device
            // barriers are required: threadgroup-only barriers are insufficient.
            for(uint i=tid;i<n*n/2;i+=256) {
                uint pair=i/n,row=i%n,ip=p[pair],iq=q[pair];
                float c=cosine[pair]; float2 s=sine[pair];
                float2 x=a[ip*n+row],y=a[iq*n+row];
                a[ip*n+row]=c*x-mul(adj(s),y); a[iq*n+row]=mul(s,x)+c*y;
                x=v[ip*n+row]; y=v[iq*n+row];
                v[ip*n+row]=c*x-mul(adj(s),y); v[iq*n+row]=mul(s,x)+c*y;
            }
            threadgroup_barrier(mem_flags::mem_device);
            for(uint i=tid;i<n*n/2;i+=256) {
                uint pair=i/n,col=i%n,ip=p[pair],iq=q[pair];
                float c=cosine[pair]; float2 s=sine[pair];
                float2 x=a[col*n+ip],y=a[col*n+iq];
                a[col*n+ip]=c*x-mul(s,y); a[col*n+iq]=mul(adj(s),x)+c*y;
            }
            threadgroup_barrier(mem_flags::mem_device);
            if(tid<n/2) {
                uint ip=p[tid],iq=q[tid];
                a[ip*n+ip]=float2(dp[tid],0); a[iq*n+iq]=float2(dq[tid],0);
                a[ip*n+iq]=float2(0); a[iq*n+ip]=float2(0);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
    for(uint i=tid;i<n;i+=256) {
        // Repeated FP32 rotations accumulate column norm drift. Normalize the
        // seed; mutual orthogonality is still checked/corrected in FP64.
        float norm=0;
        for(uint row=0;row<n;++row) norm+=dot(v[i*n+row],v[i*n+row]);
        float inverse=1/sqrt(norm);
        for(uint row=0;row<n;++row) v[i*n+row]*=inverse;
        values[job*n+i]=a[i*n+i].x*scale;
    }
    if(tid==0) status[job]=done==1 ? 0 : (done<0 ? 1 : 2);
}
