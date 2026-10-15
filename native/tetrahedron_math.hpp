// SPDX-License-Identifier: GPL-3.0-or-later
// Shared host/device arithmetic for tetrahedron occupations of one band in
// one tetrahedron whose band energy is linear between its four corners.
// Corner energies e[0..3] are sorted ascending. Weights are in units of the
// tetrahedron's share of the zone: a full tetrahedron gives 1/4 per corner.
// Linear weights and the correction (Bloechl, Jepsen and Andersen, PRB 49,
// 16223 (1994), appendix) dw_i = D(ef)/40 sum_j (e_j - e_i).
#pragma once
#ifdef __CUDACC__
#define REMAG_TETRA_HD __host__ __device__
#else
#define REMAG_TETRA_HD
#endif
namespace remag {
// Occupied fraction of the tetrahedron below ef (sum of the linear weights).
REMAG_TETRA_HD inline double tetra_count(const double* e,double ef) {
    if(ef<=e[0]) return 0.;
    if(ef>=e[3]) return 1.;
    const double x1=ef-e[0];
    if(ef<e[1]) return x1*x1*x1/((e[1]-e[0])*(e[2]-e[0])*(e[3]-e[0]));
    if(ef<e[2]) {
        const double x2=ef-e[1],e31=e[2]-e[0],e41=e[3]-e[0],e32=e[2]-e[1],e42=e[3]-e[1];
        // Volume fractions of the three pieces of the occupied polyhedron.
        const double c1=x1*x1/(e41*e31),c2=x1*x2*(e[2]-ef)/(e41*e32*e31),c3=x2*x2*(e[3]-ef)/(e42*e32*e41);
        return c1+c2+c3;
    }
    const double x4=e[3]-ef;
    return 1.-x4*x4*x4/((e[3]-e[0])*(e[3]-e[1])*(e[3]-e[2]));
}
// Corner weights w[0..3] at ef; bloechl adds the correction above.
REMAG_TETRA_HD inline void tetra_weights(const double* e,double ef,int bloechl,double* w) {
    w[0]=w[1]=w[2]=w[3]=0.;
    if(ef<=e[0]) return;
    if(ef>=e[3]) {w[0]=w[1]=w[2]=w[3]=.25;return;}
    double dos;
    const double e21=e[1]-e[0],e31=e[2]-e[0],e41=e[3]-e[0];
    if(ef<e[1]) {
        const double x=ef-e[0],c=.25*x*x*x/(e21*e31*e41);
        w[0]=c*(4.-x*(1./e21+1./e31+1./e41));w[1]=c*x/e21;w[2]=c*x/e31;w[3]=c*x/e41;
        dos=3.*x*x/(e21*e31*e41);
    } else if(ef<e[2]) {
        const double x1=ef-e[0],x2=ef-e[1],y3=e[2]-ef,y4=e[3]-ef,e32=e[2]-e[1],e42=e[3]-e[1];
        const double c1=.25*x1*x1/(e41*e31),c2=.25*x1*x2*y3/(e41*e32*e31),c3=.25*x2*x2*y4/(e42*e32*e41);
        w[0]=c1+(c1+c2)*y3/e31+(c1+c2+c3)*y4/e41;
        w[1]=c1+c2+c3+(c2+c3)*y3/e32+c3*y4/e42;
        w[2]=(c1+c2)*x1/e31+(c2+c3)*x2/e32;
        w[3]=(c1+c2+c3)*x1/e41+c3*x2/e42;
        dos=3./(e31*e41)*(e21+2.*x2-(e31+e42)*x2*x2/(e32*e42));
    } else {
        const double x=e[3]-ef,e42=e[3]-e[1],e43=e[3]-e[2],c=.25*x*x*x/(e41*e42*e43);
        w[0]=.25-c*x/e41;w[1]=.25-c*x/e42;w[2]=.25-c*x/e43;
        w[3]=.25-c*(4.-x*(1./e41+1./e42+1./e43));
        dos=3.*x*x/(e41*e42*e43);
    }
    if(bloechl) {
        const double s=e[0]+e[1]+e[2]+e[3];
        for(int i=0;i<4;i++) w[i]+=dos/40.*(s-4.*e[i]);
    }
}
}
