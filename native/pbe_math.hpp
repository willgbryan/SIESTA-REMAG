// SPDX-License-Identifier: GPL-3.0-or-later
// Translation of pinned GridXC ggaxc.F90 (GGAXC, PBEXC, PBEformXC) and
// ldaxc.F90 (PW92C, nonrelativistic EXCHNG). Original routines by
// L.C. Balbas and J.M. Soler; NC extensions by V.M. Garcia-Suarez.
// GridXC copyright and redistribution terms are retained in GRIDXC_NOTICE.
// Preserve the active old_scheme NC rotation and original numerical floors.
// This supports built-in PBE, irel=0 only. Energies here are in Hartrees.
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define REMAG_PBE_HD __host__ __device__
#else
#define REMAG_PBE_HD
#endif

namespace remag {
struct PbePoint {
    double x,c,dx[4],dc[4],gx[4][3],gc[4][3];
};
namespace pbe_detail {
REMAG_PBE_HD inline double square(double x) {return x*x;}
REMAG_PBE_HD inline double cube(double x) {return x*x*x;}
REMAG_PBE_HD inline double fourth(double x) {return square(square(x));}
REMAG_PBE_HD inline double norm(const double* v) {
    return sqrt(square(v[0])+square(v[1])+square(v[2]));
}
// Two-spin PW92C, as called by PBEformXC even for the scalar functional.
REMAG_PBE_HD inline void pw92(const double* d,double& ec,double* vc) {
    const double pi=4*atan(1.),one=1.+1.e-12,third=1./3.;
    const double a[3]={.031091,.015545,.016887};
    const double alpha[3]={.21370,.20548,.11125};
    const double beta[3][4]={{7.5957,3.5876,1.6382,.49294},
                            {14.1189,6.1977,3.3662,.62517},
                            {10.357,3.6231,.88026,.49671}};
    const double total=fmax(1.e-12,d[0]+d[1]);
    const double z=(d[0]-d[1])/total,rs=pow(3/(4*pi*total),third);
    const double dr=-rs/total/3,dz[2]={(one-z)/total,-(one+z)/total};
    double g[3],dg[3];
    for(int i=0;i<3;i++) {
        const double b=beta[i][0]*sqrt(rs)+beta[i][1]*rs+
                       beta[i][2]*pow(rs,1.5)+beta[i][3]*square(rs);
        const double db=beta[i][0]*.5/sqrt(rs)+beta[i][1]+
                        beta[i][2]*1.5*sqrt(rs)+beta[i][3]*2*rs;
        const double c=1+1/(2*a[i]*b),dc=-(c-1)*db/b;
        g[i]=-2*a[i]*(1+alpha[i]*rs)*log(c);
        dg[i]=-2*a[i]*(alpha[i]*log(c)+(1+alpha[i]*rs)*dc/c);
    }
    const double c=1/(pow(2.,4./3.)-2),fpp=8*c/9;
    const double f=(pow(one+z,4./3.)+pow(one-z,4./3.)-2)*c;
    const double df=(4./3.)*(pow(one+z,third)-pow(one-z,third))*c;
    ec=g[0]-g[2]*f/fpp*(one-fourth(z))+(g[1]-g[0])*f*fourth(z);
    const double decdr=dg[0]-dg[2]*f/fpp*(one-fourth(z))+(dg[1]-dg[0])*f*fourth(z);
    const double decdz=-g[2]/fpp*(df*(one-fourth(z))-f*4*cube(z))+
                      (g[1]-g[0])*(df*fourth(z)+f*4*cube(z));
    for(int s=0;s<2;s++) vc[s]=ec+total*(decdr*dr+decdz*dz[s]);
}

REMAG_PBE_HD inline PbePoint collinear(int spins,const double* density,const double gradient[4][3]) {
    PbePoint out{};
    const double pi=4*atan(1.),third=1./3.,beta=.066725,kappa=.804;
    const double mu=beta*square(pi)/3,gamma=(1-log(2.))/square(pi);
    double d[2],g[2][3],gt[3];
    for(int s=0;s<2;s++) {
        d[s]=spins==1?.5*density[0]:density[s];
        for(int j=0;j<3;j++) g[s][j]=spins==1?.5*gradient[0][j]:gradient[s][j];
    }
    for(int j=0;j<3;j++) gt[j]=spins==1?gradient[0][j]:gradient[0][j]+gradient[1][j];
    const double total=fmax(1.e-12,spins==1?density[0]:density[0]+density[1]);
    const double gm[2]={norm(g[0]),norm(g[1])},gmt=fmax(1.e-12,norm(gt));
    double ecu,vcu[2];pw92(d,ecu,vcu);
    const double kf=pow(3*square(pi)*total,third),ks=sqrt(4*kf/pi);
    const double z=fmin(1.-1.e-12,fmax(-1.+1.e-12,(d[0]-d[1])/total));
    const double phi=.5*(pow(1+z,2./3.)+pow(1-z,2./3.));
    const double t=gmt/(2*phi*ks*total),f1=ecu/gamma/cube(phi),f2=exp(-f1);
    const double a=beta/gamma/(f2-1),f3=square(t)+a*fourth(t);
    const double f4=beta/gamma*f3/(1+a*f3),h=gamma*cube(phi)*log(1+f4);
    out.c=ecu+h;
    const double dkf=third*kf/total,dks=.5*ks*dkf/kf;
    const double dz[2]={1/total-z/total,-1/total-z/total};
    const double dpdz=.5*(2./3.)*(1/pow(1+z,third)-1/pow(1-z,third));
    for(int s=0;s<2;s++) {
        const double dec=(vcu[s]-ecu)/total,dp=dpdz*dz[s];
        const double dt=-t*(dp/phi+dks/ks+1/total);
        const double df1=f1*(dec/ecu-3*dp/phi),df2=-f2*df1;
        const double da=-a*df2/(f2-1),df3=(2*t+4*a*cube(t))*dt+da*fourth(t);
        const double df4=f4*(df3/f3-(da*f3+a*df3)/(1+a*f3));
        double dh=3*h*dp/phi;
        dh=dh+gamma*cube(phi)*df4/(1+f4);
        out.dc[s]=vcu[s]+h+total*dh;
        for(int j=0;j<3;j++) {
            const double dtg=(t/gmt)*gt[j]/gmt,df3g=dtg*(2*t+4*a*cube(t));
            const double df4g=f4*df3g*(1/f3-a/(1+a*f3));
            out.gc[s][j]=total*(gamma*cube(phi)*df4g/(1+f4));
        }
    }
    double fx=0;
    for(int s=0;s<2;s++) {
        const double ds=fmax(1.e-12,2*d[s]),gms=fmax(1.e-12,2*gm[s]);
        const double kfs=pow(3*square(pi)*ds,third),ss=gms/(2*kfs*ds);
        const double ff1=1+mu*square(ss)/kappa,f=1+kappa-kappa/ff1;
        // Literal nonrelativistic scalar EXCHNG arithmetic.
        const double a0=pow(4/(9*pi),third),alpha=2*third;
        const double rs=pow(3/(4*pi*ds),third),vx=-(3*alpha/(2*pi*a0*rs)),ex=3*vx/4;
        fx=fx+ds*ex*f;
        const double dk=third*kfs/ds,dsd=ss*(-dk/kfs-1/ds);
        const double df1d=2*(ff1-1)*dsd/ss,dfd=kappa*df1d/square(ff1);
        out.dx[s]=vx*f+ds*ex*dfd;
        for(int j=0;j<3;j++) {
            const double gs=2*g[s][j],dsg=(ss/gms)*gs/gms;
            const double df1g=2*mu*ss*dsg/kappa,dfg=kappa*df1g/square(ff1);
            out.gx[s][j]=ds*ex*dfg;
        }
    }
    out.x=.5*fx/total;
    return out;
}
} // namespace pbe_detail

// Caller validates spins (1/2/4) and skips points with total density <1e-15.
// Gradient layout is [spin][Cartesian component], matching Fortran GD(3,nSpin).
REMAG_PBE_HD inline PbePoint pbe_point(int spins,const double* density,const double gradient[4][3]) {
    using namespace pbe_detail;
    double d[2],g[4][3]{};
    if(spins!=4) {
        for(int s=0;s<spins;s++) {
            d[s]=fmax(density[s],0.);
            for(int j=0;j<3;j++) g[s][j]=gradient[s][j];
        }
        return collinear(spins,d,g);
    }
    const double total=density[0]+density[1],z=density[0]-density[1];
    const double xy=2*sqrt(square(density[2])+square(density[3]));
    const double pol=sqrt(square(z)+square(xy)),theta=pol>1.e-12?atan2(xy,z):0.;
    const double c2=cos(theta/2),s2=sin(theta/2),st=sin(theta),ct=cos(theta);
    const double phi=atan2(-density[3],density[2]),cp=cos(phi),sp=sin(phi);
    d[0]=.5*(total+pol);d[1]=.5*(total-pol);
    for(int j=0;j<3;j++) {
        g[0][j]=gradient[0][j]*square(c2)+gradient[1][j]*square(s2)+
                2*c2*s2*(gradient[2][j]*cp-gradient[3][j]*sp);
        g[1][j]=gradient[0][j]*square(s2)+gradient[1][j]*square(c2)-
                2*c2*s2*(gradient[2][j]*cp-gradient[3][j]*sp);
    }
    auto base=collinear(2,d,g);PbePoint out{};out.x=base.x;out.c=base.c;
    for(int kind=0;kind<2;kind++) {
        const double* v=kind?base.dc:base.dx;double* dest=kind?out.dc:out.dx;
        const double (*gv)[3]=kind?base.gc:base.gx;double (*gd)[3]=kind?out.gc:out.gx;
        const double vp=(v[0]-v[1])*ct;
        dest[0]=.5*(v[0]+v[1]+vp);dest[1]=.5*(v[0]+v[1]-vp);
        dest[2]=.5*(v[0]-v[1])*st*cp;dest[3]=-.5*(v[0]-v[1])*st*sp;
        for(int j=0;j<3;j++) {
            gd[0][j]=gv[0][j]*square(c2)+gv[1][j]*square(s2);
            gd[1][j]=gv[0][j]*square(s2)+gv[1][j]*square(c2);
            gd[2][j]=.5*(gv[0][j]-gv[1][j])*st*cp;
            gd[3][j]=-.5*(gv[0][j]-gv[1][j])*st*sp;
        }
    }
    return out;
}
} // namespace remag
#undef REMAG_PBE_HD
