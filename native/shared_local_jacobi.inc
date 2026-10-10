// SPDX-License-Identifier: GPL-3.0-or-later
// One CTA owns a 32-state real/Hermitian problem throughout cyclic Jacobi.
// Padded column-major shared storage avoids the transposed row-pass bank stride.
// All arithmetic is FP64. Local tolerance is machine epsilon, as in cuSOLVER.
template<class T> __device__ T local_scale(T x,double a) {
    if constexpr(std::is_same_v<T,double>)return x*a;
    else return make_cuDoubleComplex(x.x*a,x.y*a);
}
template<class T> __device__ T local_combine(T x,T alpha,T y,T beta) {
    return accumulate_product(accumulate_product(value<T>(0.),x,alpha),y,beta);
}
template<class T> __global__ void shared_local_jacobi(T* matrix,double* eigenvalues,int* info) {
    __shared__ T a[32][33],v[32][33],sine[16];
    __shared__ double cosine[16],diagonal_p[16],diagonal_q[16],scale;
    __shared__ int p[16],q[16],order[32],done,bad;
    __shared__ double2 reduction[512];
    const int tid=threadIdx.x,pair=tid/32,row=tid%32;
    T* local=matrix+size_t(blockIdx.x)*1024;
    if(tid==0)bad=0;
    __syncthreads();
    double largest=0.;
    for(int i=tid;i<1024;i+=512) {
        const T x=lower_element(local,i%32,i/32,32);
        if(!finite(x))atomicExch(&bad,1);
        largest=fmax(largest,magnitude(x));
        a[i/32][i%32]=x;v[i/32][i%32]=value<T>(i%32==i/32?1.:0.);
    }
    reduction[tid].x=largest;
    __syncthreads();
    for(int stride=256;stride;stride/=2) {
        if(tid<stride)reduction[tid].x=fmax(reduction[tid].x,reduction[tid+stride].x);
        __syncthreads();
    }
    if(tid==0)scale=reduction[0].x>0.?reduction[0].x:1.;
    __syncthreads();
    if(bad) {
        for(int i=tid;i<1024;i+=512)local[i]=v[i/32][i%32];
        if(tid<32)eigenvalues[blockIdx.x*32+tid]=NAN;
        if(tid==0)info[blockIdx.x]=33;
        return;
    }
    for(int i=tid;i<1024;i+=512)a[i/32][i%32]=local_scale(a[i/32][i%32],1./scale);
    __syncthreads();
    for(int sweep=0;sweep<=100;sweep++) {
        double total=0.,off=0.;
        for(int i=tid;i<1024;i+=512) {
            const double x=squared(a[i/32][i%32]);total+=x;if(i%32!=i/32)off+=x;
        }
        reduction[tid]=make_double2(total,off);
        __syncthreads();
        for(int stride=256;stride;stride/=2) {
            if(tid<stride) {reduction[tid].x+=reduction[tid+stride].x;reduction[tid].y+=reduction[tid+stride].y;}
            __syncthreads();
        }
        if(tid==0) {
            constexpr double epsilon=2.2204460492503130808472633361816e-16;
            done=(!isfinite(reduction[0].x)||!isfinite(reduction[0].y))?-1:
                (reduction[0].y<=epsilon*epsilon*reduction[0].x?1:0);
        }
        __syncthreads();
        if(done||sweep==100)break;
        for(int round=0;round<31;round++) {
            if(tid<16) {
                int ip=tid?((round+tid)%31):31,iq=tid?((round+31-tid)%31):round;
                p[tid]=min(ip,iq);q[tid]=max(ip,iq);ip=p[tid];iq=q[tid];
                const double app=real(a[ip][ip]),aqq=real(a[iq][iq]);
                const T b=a[iq][ip];const double ab=magnitude(b),delta=aqq-app;
                double t=0.;
                if(ab>0.)t=copysign((2.*ab)/(fabs(delta)+hypot(delta,2.*ab)),delta);
                const double c=1./sqrt(1.+t*t);
                cosine[tid]=c;sine[tid]=ab>0.?local_scale(b,t*c/ab):value<T>(0.);
                diagonal_p[tid]=app-t*ab;diagonal_q[tid]=aqq+t*ab;
            }
            __syncthreads();
            const int ip=p[pair],iq=q[pair];const T c=value<T>(cosine[pair]),s=sine[pair];
            T x=a[ip][row],y=a[iq][row];
            a[ip][row]=local_combine(x,c,y,local_scale(adjoint(s),-1.));
            a[iq][row]=local_combine(x,s,y,c);
            x=v[ip][row];y=v[iq][row];
            v[ip][row]=local_combine(x,c,y,local_scale(adjoint(s),-1.));
            v[iq][row]=local_combine(x,s,y,c);
            __syncthreads();
            x=a[row][ip];y=a[row][iq];
            a[row][ip]=local_combine(x,c,y,local_scale(s,-1.));
            a[row][iq]=local_combine(x,adjoint(s),y,c);
            __syncthreads();
            if(tid<16) {
                const int i=p[tid],j=q[tid];
                a[i][i]=value<T>(diagonal_p[tid]);a[j][j]=value<T>(diagonal_q[tid]);
                a[i][j]=value<T>(0.);a[j][i]=value<T>(0.);
            }
            __syncthreads();
        }
    }
    // Rank stable ties so repeated/degenerate eigenvalues retain distinct columns.
    if(tid<32) {
        const double w=real(a[tid][tid]);int rank=0;
        for(int j=0;j<32;j++) {const double other=real(a[j][j]);rank+=other<w||(other==w&&j<tid);}
        order[tid]=rank;eigenvalues[blockIdx.x*32+rank]=w*scale;
    }
    __syncthreads();
    for(int i=tid;i<1024;i+=512)local[i%32+order[i/32]*32]=v[i/32][i%32];
    if(tid==0)info[blockIdx.x]=done==1?0:33;
}
