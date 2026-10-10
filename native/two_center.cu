// SPDX-License-Identifier: GPL-3.0-or-later
// Pinned matel_table:get_matel, interpolation:splint and spher_harm:rlylm.
#include "two_center.hpp"
#include <cmath>
#include <climits>
namespace remag {
namespace {
constexpr int samples=1025,max_l=10;
__device__ void fail(int* status,int code){atomicMax(status,code);}
__device__ double normalization(int l,int m,double pi) {
    m=m<0?-m:m;double fac=(2*l+1)/(4*pi);
    for(int i=l-m+1;i<=l+m;i++) fac=fac/i;
    double c=sqrt(fac);if(m) c=c*sqrt(2.);return c;
}
__device__ void harmonic(int lmax,int l,int signed_m,const double* r,double pi,double& value,double* grad) {
    const int m=signed_m<0?-signed_m:signed_m,ilm=l*l+l+signed_m;
    const double c=normalization(l,m,pi);value=0.;grad[0]=grad[1]=grad[2]=0.;
    if(lmax<=2) {
        switch(ilm) {
        case 0:value=c;break;
        case 1:value=-(c*r[1]);grad[1]=-c;break;
        case 2:value=c*r[2];grad[2]=c;break;
        case 3:value=-(c*r[0]);grad[0]=-c;break;
        case 4:value=c*6*r[0]*r[1];grad[0]=c*6*r[1];grad[1]=c*6*r[0];break;
        case 5:value=(-c)*3*r[1]*r[2];grad[1]=(-c)*3*r[2];grad[2]=(-c)*3*r[1];break;
        case 6:value=c*.5*(2*r[2]*r[2]-r[0]*r[0]-r[1]*r[1]);grad[0]=(-c)*r[0];grad[1]=(-c)*r[1];grad[2]=c*2*r[2];break;
        case 7:value=(-c)*3*r[0]*r[2];grad[0]=(-c)*3*r[2];grad[2]=(-c)*3*r[0];break;
        case 8:value=c*3*(r[0]*r[0]-r[1]*r[1]);grad[0]=c*6*r[0];grad[1]=(-c)*6*r[1];break;
        }
        return;
    }
    const double size=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if(size<1.e-14) {
        if(ilm==0)value=c;
        if(ilm==1)grad[1]=-c;
        if(ilm==2)grad[2]=c;
        if(ilm==3)grad[0]=-c;
        return;
    }
    double rx=r[0]/size,ry=r[1]/size,rz=r[2]/size,rxy=sqrt(rx*rx+ry*ry);
    if(rxy<1.e-14) {rx=1.e-14;rxy=sqrt(rx*rx+ry*ry);}
    double p[12][12]{},zp[12][12]{};
    for(int order=lmax;order>=0;order--) {
        p[order][order+1]=0.;p[order][order]=1.;double fac=1.;
        for(int i=1;i<=order;i++) {p[order][order]=-(p[order][order]*fac*rxy);fac=fac+2.;}
        p[order+1][order]=rz*(2*order+1)*p[order][order];
        for(int ll=order+2;ll<=lmax;ll++)
            p[ll][order]=(rz*(2*ll-1)*p[ll-1][order]-(ll+order-1)*p[ll-2][order])/(ll-order);
        for(int ll=order;ll<=lmax;ll++) zp[ll][order]=-((order*p[ll][order]*rz/rxy+p[ll][order+1])/rxy);
    }
    double rl[12]{};rl[0]=0.;rl[1]=1.;for(int ll=1;ll<=lmax;ll++)rl[ll+1]=rl[ll]*size;
    const double cosphi=rx/rxy,sinphi=ry/rxy;double cosm=1.,sinm=0.;
    for(int mm=0;mm<m;mm++) {double oldc=cosm,olds=sinm;cosm=oldc*cosphi-olds*sinphi;sinm=oldc*sinphi+olds*cosphi;}
    double gy[3],yy;
    if(signed_m<0) {
        gy[0]=(-zp[l][m])*rx*rz*sinm-p[l][m]*m*cosm*sinphi/rxy;
        gy[1]=(-zp[l][m])*ry*rz*sinm+p[l][m]*m*cosm*cosphi/rxy;
        gy[2]=zp[l][m]*rxy*rxy*sinm;yy=c*p[l][m]*sinm;
    } else {
        gy[0]=(-zp[l][m])*rx*rz*cosm+p[l][m]*m*sinm*sinphi/rxy;
        gy[1]=(-zp[l][m])*ry*rz*cosm-p[l][m]*m*sinm*cosphi/rxy;
        gy[2]=zp[l][m]*rxy*rxy*cosm;yy=c*p[l][m]*cosm;
    }
    value=rl[l+1]*yy;
    grad[0]=rx*l*rl[l]*yy+rl[l+1]*gy[0]*c/size;
    grad[1]=ry*l*rl[l]*yy+rl[l+1]*gy[1]*c/size;
    grad[2]=rz*l*rl[l]*yy+rl[l+1]*gy[2]*c/size;
}

__device__ bool evaluate(const TwoCenterView& t,int a,int b,const double* xyz,double* out,int* status) {
    if(a<t.first1||a>t.last1||b<t.first2||b>t.last2){fail(status,1);return false;}
    double x[3]={xyz[0],xyz[1],xyz[2]};
    if(!isfinite(x[0])||!isfinite(x[1])||!isfinite(x[2])){fail(status,2);return false;}
    double r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    if(r<1.e-12){x[2]=1.e-12;r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
    const double c1=t.cutoffs[a-1],c2=t.cutoffs[b-1];
    if(!isfinite(c1)||!isfinite(c2)||c1<0.||c2<0.){fail(status,1);return false;}
    if(!isfinite(r)||!isfinite(c1+c2)){fail(status,2);return false;}
    if(r>c1+c2)return true;
    if(r>1024*t.delta+t.delta*1.e-6){fail(status,2);return false;}
    const int lo=min(1023,max(0,int(floor(r/t.delta)))),hi=lo+1;
    const double xl=lo*t.delta,xh=hi*t.delta,dx=xh-xl,A=(xh-r)/dx,B=(r-xl)/dx;
    const double da=-1./dx,db=1./dx;
    const int first1=t.ind[a-1],end1=t.ind[a],first2=t.ind[b-1],end2=t.ind[b];
    // Check before subtraction so malformed bases/indices cannot overflow.
    if(first1<1||first2<1||first1>end1||first2>end2||first1<=t.base1||first2<=t.base2||
       static_cast<long long>(end1)-1>static_cast<long long>(t.base1)+t.rows||
       static_cast<long long>(end2)-1>static_cast<long long>(t.base2)+t.cols){fail(status,1);return false;}
    for(int f1=first1;f1<end1;f1++)for(int f2=first2;f2<end2;f2++) {
        const size_t pair=size_t(f1-t.base1-1)+size_t(f2-t.base2-1)*t.rows;
        const int first=t.term_offsets[pair],end=t.term_offsets[pair+1];
        if(first<1||first>end||static_cast<long long>(end)>static_cast<long long>(t.terms)+1){fail(status,1);return false;}
        int lmax=0;
        for(int i=first-1;i<end-1;i++) {
            const int ilm=t.ilm[i];
            if(ilm<1||ilm>(max_l+1)*(max_l+1)){fail(status,1);return false;}
            lmax=max(lmax,int(floor(sqrt(double(ilm-1)))));
        }
        for(int i=first-1;i<end-1;i++) {
            const int radial=t.radial_index[i],ilm=t.ilm[i]-1,l=int(floor(sqrt(double(ilm)))),m=ilm-l*l-l;
            if(radial<1||radial>t.radials){fail(status,1);return false;}
            const double* v=t.radial+size_t(radial-1)*2*samples;
            const double yl=v[lo],yh=v[hi],d2l=v[samples+lo],d2h=v[samples+hi],coefficient=t.coefficients[i];
            if(!isfinite(yl)||!isfinite(yh)||!isfinite(d2l)||!isfinite(d2h)||!isfinite(coefficient)){fail(status,2);return false;}
            const double sr=A*yl+B*yh+((A*A*A-A)*d2l+(B*B*B-B)*d2h)*(dx*dx)/6.;
            const double ds=da*yl+db*yh+(da*(3*(A*A)-1.)*d2l+db*(3*(B*B)-1.)*d2h)*(dx*dx)/6.;
            double y,dy[3];harmonic(lmax,l,m,x,t.pi,y,dy);
            out[0]=out[0]+sr*coefficient*y;
            for(int k=0;k<3;k++)out[k+1]=out[k+1]+ds*coefficient*y*x[k]/r+sr*coefficient*dy[k];
        }
    }
    for(int k=0;k<4;k++)if(!isfinite(out[k])){fail(status,2);return false;}
    return true;
}
__global__ void queries(TwoCenterView t,int count,const int* pairs,const double* xyz,double* result,int* status) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        double v[4]={};const bool ok=evaluate(t,pairs[2*size_t(i)],pairs[2*size_t(i)+1],xyz+3*size_t(i),v,status);
        for(int k=0;k<4;k++)result[4*size_t(i)+k]=ok?v[k]:0.;
    }
}
}
cudaError_t evaluate_two_center_device(const TwoCenterView& t,int count,const int* pairs,const double* xyz,double* result,int* status,cudaStream_t stream) {
    if(t.functions<1||t.functions>1000000||t.radials<0||t.radials>1000000||t.terms<0||t.terms>100000000||
       t.rows<1||t.cols<1||size_t(t.rows)*t.cols>100000000||t.base1<0||t.base2<0||
       t.first1<1||t.last1<t.first1||t.last1>t.functions||t.first2<1||t.last2<t.first2||t.last2>t.functions||
       !std::isfinite(t.delta)||t.delta<=0.||!std::isfinite(t.pi)||t.pi<3.14||t.pi>3.15||
       count<1||count>INT_MAX/4||!t.ind||!t.term_offsets||!t.ilm||!t.radial_index||!t.cutoffs||!t.radial||!t.coefficients||
       !pairs||!xyz||!result||!status||!stream)return cudaErrorInvalidValue;
    const int blocks=(count+127)/128;
    queries<<<blocks>1024?1024:blocks,128,0,stream>>>(t,count,pairs,xyz,result,status);return cudaGetLastError();
}
}
