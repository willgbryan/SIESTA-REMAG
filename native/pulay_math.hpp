// SPDX-License-Identifier: GPL-3.0-or-later
// Stable-Pulay coefficient contract from pinned SIESTA m_mixing.F90.
// Upstream SIESTA is GPL-3.0-or-later; see the repository LICENSE.
// Small device-capable linear algebra; no host/library calls or allocation.
// The symmetric pseudoinverse is computed by Jacobi diagonalization, with
// the same singular-value cutoff as upstream's DGELSS fallback. This is a
// different numerical implementation and requires sequence/SCF validation.
#pragma once
#include <cmath>
#include <cfloat>
#ifdef __CUDACC__
#define REMAG_PULAY_HD __host__ __device__
#else
#define REMAG_PULAY_HD
#endif

namespace remag {
constexpr int pulay_max_history=16;
struct PulaySolve {
    int status=0; // 1 bad configuration, 2 nonfinite data/arithmetic, 3 Jacobi failure
    int method=0; // 0 no history, 1 pivoted LU, 2 original simple inverse, 3 pseudoinverse
    int rank=0;
    double inverse_first=0.,coefficient_first=0.;
};
namespace pulay_detail {
constexpr int capacity=pulay_max_history-1;
REMAG_PULAY_HD inline bool finite(double x) {return x==x&&fabs(x)<=DBL_MAX;}
REMAG_PULAY_HD inline void swap(double& x,double& y) {double z=x;x=y;y=z;}
REMAG_PULAY_HD inline bool inverse_ok(int n,const double* a,const double* b) {
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        double value=0.;for(int k=0;k<n;k++) value+=a[i*n+k]*b[k*n+j];
        value-=i==j?1.:0.;
        if(!finite(value)||fabs(value)>1.e-4) return false;
    }
    return true;
}
REMAG_PULAY_HD inline bool lu_inverse(int n,const double* a,double* inverse) {
    double lu[capacity*capacity],rhs[capacity];int pivot[capacity];
    for(int i=0;i<n*n;i++) lu[i]=a[i];
    for(int k=0;k<n;k++) {
        int p=k;for(int i=k+1;i<n;i++) if(fabs(lu[i*n+k])>fabs(lu[p*n+k])) p=i;
        pivot[k]=p;
        if(lu[p*n+k]==0.||!finite(lu[p*n+k])) return false;
        if(p!=k) for(int j=0;j<n;j++) swap(lu[k*n+j],lu[p*n+j]);
        for(int i=k+1;i<n;i++) {
            lu[i*n+k]/=lu[k*n+k];
            for(int j=k+1;j<n;j++) lu[i*n+j]-=lu[i*n+k]*lu[k*n+j];
        }
    }
    for(int col=0;col<n;col++) {
        for(int i=0;i<n;i++) rhs[i]=i==col?1.:0.;
        for(int k=0;k<n;k++) if(pivot[k]!=k) swap(rhs[k],rhs[pivot[k]]);
        for(int i=0;i<n;i++) for(int j=0;j<i;j++) rhs[i]-=lu[i*n+j]*rhs[j];
        for(int i=n-1;i>=0;i--) {
            for(int j=i+1;j<n;j++) rhs[i]-=lu[i*n+j]*rhs[j];
            rhs[i]/=lu[i*n+i];inverse[i*n+col]=rhs[i];
        }
    }
    return inverse_ok(n,a,inverse);
}
REMAG_PULAY_HD inline bool simple_inverse(int n,const double* a,double* b) {
    // Literal row/column arithmetic of upstream's second inversion attempt.
    for(int i=0;i<n*n;i++) b[i]=a[i];
    for(int i=0;i<n;i++) {
        if(b[i*n+i]==0.||!finite(b[i*n+i])) return false;
        double x=1./b[i*n+i];b[i*n+i]=1.;
        for(int j=0;j<n;j++) b[j*n+i]*=x;
        for(int k=0;k<n;k++) if(k!=i) {
            x=b[i*n+k];b[i*n+k]=0.;
            for(int j=0;j<n;j++) b[j*n+k]-=b[j*n+i]*x;
        }
    }
    return inverse_ok(n,a,b);
}
REMAG_PULAY_HD inline bool pseudoinverse(int n,const double* a,double condition,double* inverse,int& rank) {
    double b[capacity*capacity],v[capacity*capacity];
    double scale=0.;for(int i=0;i<n*n;i++) scale=fmax(scale,fabs(a[i]));
    rank=0;
    if(scale==0.) {for(int i=0;i<n*n;i++) inverse[i]=0.;return true;}
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {b[i*n+j]=a[i*n+j]/scale;v[i*n+j]=i==j?1.:0.;}
    bool converged=false;
    for(int sweep=0;sweep<100;sweep++) {
        double diagonal=0.,off=0.;
        for(int i=0;i<n;i++) {
            diagonal=fmax(diagonal,fabs(b[i*n+i]));
            for(int j=i+1;j<n;j++) off=fmax(off,fabs(b[i*n+j]));
        }
        if(off<=DBL_EPSILON*diagonal) {converged=true;break;}
        for(int p=0;p<n;p++) for(int q=p+1;q<n;q++) {
            const double value=b[p*n+q];
            if(fabs(value)<=DBL_EPSILON*diagonal) continue;
            const double tau=(b[q*n+q]-b[p*n+p])/(2*value);
            const double t=copysign(1.,tau)/(fabs(tau)+hypot(1.,tau));
            const double c=1./sqrt(1.+t*t),s=t*c;
            b[p*n+p]-=t*value;b[q*n+q]+=t*value;b[p*n+q]=b[q*n+p]=0.;
            for(int k=0;k<n;k++) {
                if(k!=p&&k!=q) {
                    const double x=b[k*n+p],y=b[k*n+q];
                    b[k*n+p]=b[p*n+k]=c*x-s*y;b[k*n+q]=b[q*n+k]=s*x+c*y;
                }
                const double x=v[k*n+p],y=v[k*n+q];v[k*n+p]=c*x-s*y;v[k*n+q]=s*x+c*y;
            }
        }
    }
    if(!converged) return false;
    double maximum=0.;for(int k=0;k<n;k++) maximum=fmax(maximum,fabs(b[k*n+k]));
    for(int i=0;i<n*n;i++) inverse[i]=0.;
    for(int k=0;k<n;k++) if(fabs(b[k*n+k])>condition*maximum) {
        rank++;
        double reciprocal=(1./b[k*n+k])/scale;
        for(int i=0;i<n;i++) for(int j=0;j<n;j++) inverse[i*n+j]+=v[i*n+k]*reciprocal*v[j*n+k];
    }
    for(int i=0;i<n*n;i++) if(!finite(inverse[i])) return false;
    return true;
}
}

// Gram and residual projections are row-major, symmetric and device-owned.
// Every output coefficient is checked; callers must propagate status instead
// of silently mixing a failed or nonfinite history.
REMAG_PULAY_HD inline PulaySolve pulay_coefficients(int n,const double* gram,const double* projection,
                                                   double condition,double* coefficients) {
    using namespace pulay_detail;
    PulaySolve info;
    if(n<0||n>capacity||!finite(condition)||condition<0.||condition>=1.) {info.status=1;return info;}
    if(!n) return info;
    for(int i=0;i<n;i++) {
        if(!finite(projection[i])) {info.status=2;return info;}
        for(int j=0;j<n;j++) {
            if(!finite(gram[i*n+j])) {info.status=2;return info;}
            if(gram[i*n+j]!=gram[j*n+i]) {info.status=1;return info;}
        }
    }
    if(n==1) {
        // Keep the one-entry solve scalar. It needs no matrix workspace or
        // fallback and uses the same reciprocal/product order as upstream.
        const double reciprocal=1./gram[0];info.method=1;info.rank=1;
        info.inverse_first=reciprocal;info.coefficient_first=gram[0];
        if(!finite(reciprocal)) {info.status=2;return info;}
        double value=0.;value-=reciprocal*projection[0];info.coefficient_first=value;
        if(!finite(value)) {info.status=2;return info;}
        coefficients[0]=value;return info;
    }
    double inverse[capacity*capacity];
    if(lu_inverse(n,gram,inverse)) {info.method=1;info.rank=n;}
    else if(simple_inverse(n,gram,inverse)) {info.method=2;info.rank=n;}
    else {
        info.method=3;
        if(!pseudoinverse(n,gram,condition,inverse,info.rank)) {info.status=3;return info;}
    }
    double output[capacity]{};
    for(int j=0;j<n;j++) for(int i=0;i<n;i++) output[i]-=inverse[i*n+j]*projection[j];
    info.inverse_first=inverse[0];info.coefficient_first=output[0];
    for(int i=0;i<n;i++) if(!finite(output[i])) {info.status=2;return info;}
    for(int i=0;i<n;i++) coefficients[i]=output[i];
    return info;
}
}
#undef REMAG_PULAY_HD
