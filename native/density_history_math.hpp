// SPDX-License-Identifier: GPL-3.0-or-later
// Coordinate extrapolation from pinned SIESTA m_new_dm.F90 and inver.f.
// Workspaces are caller-owned (device memory in production), not large thread
// local arrays. This alternative linear algebra requires numerical validation.
#pragma once
#include <cmath>
#include <cfloat>
#ifdef __CUDACC__
#define REMAG_HISTORY_HD __host__ __device__
#else
#define REMAG_HISTORY_HD
#endif
namespace remag {
constexpr int density_history_limit=32;
constexpr int density_history_matrix_limit=density_history_limit+1;
struct HistorySolve {int status=0,rank=0,fallback=0;};
struct HistoryMathWorkspace {
    double matrix[33*33],inverse[33*33],work[33*33],vectors[33*33];
    double rhs[33],solution[33],temporary[33];
    int pivots[33];
};
namespace history_detail {
REMAG_HISTORY_HD inline bool finite(double x) {return x==x&&fabs(x)<=DBL_MAX;}
REMAG_HISTORY_HD inline void swap(double& a,double& b) {double t=a;a=b;b=t;}
REMAG_HISTORY_HD inline void add_product(double a,double b,double& sum,double& error) {
    const double product=a*b,product_error=fma(a,b,-product),next=sum+product;
    error+=(fabs(sum)>=fabs(product)?(sum-next)+product:(product-next)+sum)+product_error;
    sum=next;
}
REMAG_HISTORY_HD inline bool inverse_ok(int n,const double* a,const double* b) {
    double error=0.;
    for(int i=0;i<n;i++)for(int j=0;j<n;j++) {
        double left=0.,right=0.;
        for(int k=0;k<n;k++){left+=a[i*n+k]*b[k*n+j];right+=b[i*n+k]*a[k*n+j];}
        error+=fabs(left-(i==j?1.:0.));error+=fabs(right-(i==j?1.:0.));
    }
    return finite(error)&&error/n<=1.e-4;
}
REMAG_HISTORY_HD inline bool inverse(int n,HistoryMathWorkspace& w) {
    auto a=w.matrix;auto b=w.inverse;auto lu=w.work;auto rhs=w.temporary;
    for(int i=0;i<n*n;i++)lu[i]=a[i];
    bool ok=true;
    for(int k=0;k<n;k++) {
        int p=k;for(int i=k+1;i<n;i++)if(fabs(lu[i*n+k])>fabs(lu[p*n+k]))p=i;
        w.pivots[k]=p;
        if(lu[p*n+k]==0.||!finite(lu[p*n+k])){ok=false;break;}
        if(p!=k)for(int j=0;j<n;j++)swap(lu[k*n+j],lu[p*n+j]);
        for(int i=k+1;i<n;i++) {
            lu[i*n+k]/=lu[k*n+k];
            for(int j=k+1;j<n;j++)lu[i*n+j]-=lu[i*n+k]*lu[k*n+j];
        }
    }
    if(ok)for(int col=0;col<n;col++) {
        for(int i=0;i<n;i++)rhs[i]=i==col?1.:0.;
        for(int k=0;k<n;k++)if(w.pivots[k]!=k)swap(rhs[k],rhs[w.pivots[k]]);
        for(int i=0;i<n;i++)for(int j=0;j<i;j++)rhs[i]-=lu[i*n+j]*rhs[j];
        for(int i=n-1;i>=0;i--) {
            for(int j=i+1;j<n;j++)rhs[i]-=lu[i*n+j]*rhs[j];
            rhs[i]/=lu[i*n+i];b[i*n+col]=rhs[i];
        }
    }
    if(ok&&inverse_ok(n,a,b))return true;
    for(int i=0;i<n*n;i++)b[i]=a[i];
    for(int i=0;i<n;i++) {
        if(b[i*n+i]==0.||!finite(b[i*n+i]))return false;
        double x=b[i*n+i];b[i*n+i]=1.;
        for(int j=0;j<n;j++)b[j*n+i]/=x;
        for(int k=0;k<n;k++)if(k!=i) {
            x=b[i*n+k];b[i*n+k]=0.;
            for(int j=0;j<n;j++)b[j*n+k]-=b[j*n+i]*x;
        }
    }
    return inverse_ok(n,a,b);
}
// Symmetric eigen decomposition gives the signed inverse and the same
// singular-value cutoff (1e-6 of the largest) as the upstream default SVD.
REMAG_HISTORY_HD inline bool svd(int n,HistoryMathWorkspace& w,int& rank) {
    auto a=w.matrix;auto b=w.work;auto v=w.vectors;
    double scale=0.;rank=0;
    for(int i=0;i<n*n;i++)scale=fmax(scale,fabs(a[i]));
    for(int i=0;i<n;i++)w.solution[i]=0.;
    if(scale==0.)return true;
    for(int i=0;i<n;i++)for(int j=0;j<n;j++) {
        b[i*n+j]=a[i*n+j]/scale;v[i*n+j]=i==j?1.:0.;
    }
    for(int refinement=0;refinement<3;refinement++) {
    bool converged=false;
    for(int sweep=0;sweep<100;sweep++) {
        bool small=true;
        for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)
            if(fabs(b[i*n+j])>DBL_EPSILON*sqrt(fabs(b[i*n+i]))*sqrt(fabs(b[j*n+j])))small=false;
        if(small){converged=true;break;}
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++) {
            double x=b[p*n+q];
            if(fabs(x)<=DBL_EPSILON*sqrt(fabs(b[p*n+p]))*sqrt(fabs(b[q*n+q])))continue;
            double tau=(b[q*n+q]-b[p*n+p])/(2*x);
            double t=copysign(1.,tau)/(fabs(tau)+hypot(1.,tau));
            double c=1./sqrt(1.+t*t),s=t*c;
            b[p*n+p]-=t*x;b[q*n+q]+=t*x;b[p*n+q]=b[q*n+p]=0.;
            for(int k=0;k<n;k++) {
                if(k!=p&&k!=q) {
                    double u=b[k*n+p],z=b[k*n+q];
                    b[k*n+p]=b[p*n+k]=c*u-s*z;b[k*n+q]=b[q*n+k]=s*u+c*z;
                }
                double u=v[k*n+p],z=v[k*n+q];v[k*n+p]=c*u-s*z;v[k*n+q]=s*u+c*z;
            }
        }
    }
    if(!converged)return false;
    // Re-form the complete projected matrix with compensated products, then
    // refine its eigenvectors. This corrects accumulated rotation errors near
    // the SVD cutoff, where merely recomputing diagonal Rayleigh quotients is
    // insufficient. Both original and projected matrices stay device-owned.
    for(int k=0;k<n;k++)for(int i=0;i<n;i++) {
        double sum=0.,error=0.;
        for(int j=0;j<n;j++)add_product(a[i*n+j]/scale,v[j*n+k],sum,error);
        w.inverse[i*n+k]=sum+error;
    }
    for(int i=0;i<n;i++)for(int j=0;j<=i;j++) {
        double sum=0.,error=0.;
        for(int k=0;k<n;k++)add_product(v[k*n+i],w.inverse[k*n+j],sum,error);
        b[i*n+j]=b[j*n+i]=sum+error;
    }
    }
    double largest=0.;for(int i=0;i<n;i++)largest=fmax(largest,fabs(b[i*n+i]));
    for(int k=0;k<n;k++)if(fabs(b[k*n+k])>1.e-6*largest) {
        rank++;double projection=0.;for(int i=0;i<n;i++)projection+=v[i*n+k]*w.rhs[i];
        projection=(projection/b[k*n+k])/scale;
        for(int i=0;i<n;i++)w.solution[i]+=v[i*n+k]*projection;
    }
    for(int i=0;i<n;i++)if(!finite(w.solution[i]))return false;
    return true;
}
// x is dimension-major within each oldest-to-newest history slot.
REMAG_HISTORY_HD inline double vector(int mode,int entry,int dim,int dims,int n,
        const double* x,const double* current) {
    if(mode==0) {
        double mean=0.;for(int j=0;j<n;j++)mean+=x[j*dims+dim];
        return x[entry*dims+dim]-mean/n;
    }
    if(mode==3)return x[(entry+1)*dims+dim]-(entry==n-2?current[dim]:0.)-x[entry*dims+dim];
    return x[entry*dims+dim]-current[dim];
}
REMAG_HISTORY_HD inline void build(int mode,int dims,int n,int m,bool constraint,
        const double* x,const double* current,HistoryMathWorkspace& w) {
    const int vectors=(constraint&&mode!=0)?m-1:m;
    for(int i=0;i<vectors;i++) {
        w.rhs[i]=0.;
        for(int j=0;j<=i;j++) {
            double value=0.;
            for(int k=0;k<dims;k++)value+=vector(mode,i,k,dims,n,x,current)*vector(mode,j,k,dims,n,x,current);
            w.matrix[i*m+j]=w.matrix[j*m+i]=value;
        }
        if(mode==0||mode==3)for(int k=0;k<dims;k++) {
            double target;
            if(mode==0){double mean=0.;for(int j=0;j<n;j++)mean+=x[j*dims+k];target=current[k]-mean/n;}
            else target=-(x[(n-1)*dims+k]-current[k]);
            w.rhs[i]+=vector(mode,i,k,dims,n,x,current)*target;
        }
    }
    if(constraint) {
        w.rhs[m-1]=1.;
        for(int j=0;j<m;j++)w.matrix[(m-1)*m+j]=1.;
        if(mode!=0){for(int i=0;i<m-1;i++)w.matrix[i*m+m-1]=1.;w.matrix[m*m-1]=0.;}
    }
}
}
// mode: 0 original, 1 inverse DIIS, 2 default DIIS/SVD, 3 experimental SVD.
// On invalid input/arithmetic, c is untouched. fallback records upstream's
// explicit last-entry policy; a numerical solver failure is an error instead
// of consuming undefined beta from the experimental upstream error branch.
REMAG_HISTORY_HD inline HistorySolve density_history_coefficients(int dims,int n,int mode,
        const double* x,const double* current,double* c,HistoryMathWorkspace& w) {
    using namespace history_detail;HistorySolve r;
    if(dims<1||n<1||n>density_history_limit||mode<0||mode>3||!x||!current||!c){r.status=1;return r;}
    for(int i=0;i<dims;i++)if(!finite(current[i])){r.status=2;return r;}
    for(int j=0;j<n;j++)for(int i=0;i<dims;i++)if(!finite(x[j*dims+i])){r.status=2;return r;}
    if(n==1){c[0]=1.;r.rank=1;return r;}
    int m=mode==0?n-1:(mode==3?n-1:n);
    if(mode<2) {
        for(;m>0;m--) {
            build(mode,dims,n,m,false,x,current,w);
            bool good=true;for(int i=0;i<m*m;i++)good=good&&finite(w.matrix[i]);
            if(!good){r.status=2;return r;}
            if(inverse(m,w))break;
        }
        r.rank=m;
        if(m==0||(mode==1&&m<n))r.fallback=1;
        else {
            m++;build(mode,dims,n,m,true,x,current,w);
            for(int i=0;i<m*m;i++)if(!finite(w.matrix[i])){r.status=2;return r;}
            for(int i=0;i<m;i++)if(!finite(w.rhs[i])){r.status=2;return r;}
            if(!inverse(m,w))r.fallback=1;
            else for(int i=0;i<m;i++) {
                double value=0.;for(int j=0;j<m;j++)value+=w.inverse[i*m+j]*w.rhs[j];
                w.solution[i]=value;
            }
        }
    }else {
        if(mode==2)m=n+1;
        build(mode,dims,n,m,mode==2,x,current,w);
        for(int i=0;i<m*m;i++)if(!finite(w.matrix[i])){r.status=2;return r;}
        for(int i=0;i<m;i++)if(!finite(w.rhs[i])){r.status=2;return r;}
        if(!svd(m,w,r.rank)){r.status=3;return r;}
    }
    // Use scratch for the final conversion so a late nonfinite value cannot
    // partially overwrite the caller's coefficients.
    for(int i=0;i<n;i++) {
        double value;
        if(r.fallback)value=i==n-1?1.:0.;
        else if(mode==3)value=i==n-1?1.:(i==0?-w.solution[0]:w.solution[i-1]-w.solution[i]);
        else value=i<m?w.solution[i]:0.;
        if(!finite(value)){r.status=2;return r;}w.temporary[i]=value;
    }
    for(int i=0;i<n;i++)c[i]=w.temporary[i];
    return r;
}
}
#undef REMAG_HISTORY_HD
