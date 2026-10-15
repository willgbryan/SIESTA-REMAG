// SPDX-License-Identifier: GPL-3.0-or-later
// Shared host/device arithmetic for site spin constraints in pinned SIESTA's
// sparse spinor convention (diag3k.F, compute_energies.F90, mulliken.F):
//   uu = H1+iH5, dd = H2+iH6, ud = H3-iH4, du = H7+iH8 (D likewise),
//   Ebs = sum_c H_c D_c (SOC) or H1D1+H2D2+2(H3D3+H4D4) (NC),
//   Mulliken M = sum (D3+D7, D4+D8, D1-D2) S (SOC) or (2D3, 2D4, D1-D2) S (NC).
// The constraint adds W_s (b_s.sigma) per site s, with the Mulliken weight of
// Cuadrado et al. (2018): S_mn when both orbitals are on s, S_mn/2 when one is.
// Per sparse entry that is f.sigma S_mn with f = (b_row+b_col)/2, so
// Ebs gains exactly sum_s b_s.M_s for a Hermitian D.
#pragma once
#include <cstddef>
#ifdef __CUDACC__
#define REMAG_SPIN_CONSTRAINT_HD __host__ __device__
#else
#define REMAG_SPIN_CONSTRAINT_HD
#endif
namespace remag {
// Add f.sigma times overlap s to entry i of an (n,spins) column-major H.
// Components 5 and 6 (imaginary diagonal) are unchanged.
REMAG_SPIN_CONSTRAINT_HD inline void spin_constraint_add(int spins,size_t n,size_t i,double s,
        const double* row_field,const double* col_field,double* h) {
    const double fx=.5*(row_field[0]+col_field[0])*s,fy=.5*(row_field[1]+col_field[1])*s,
                 fz=.5*(row_field[2]+col_field[2])*s;
    h[i]+=fz;h[i+n]-=fz;h[i+2*n]+=fx;h[i+3*n]+=fy;
    if(spins==8){h[i+6*n]+=fx;h[i+7*n]+=fy;}
}
// Mulliken spin moment contribution of entry i to its row orbital.
REMAG_SPIN_CONSTRAINT_HD inline void spin_constraint_moment(int spins,size_t n,size_t i,double s,
        const double* d,double* m) {
    if(spins==8){m[0]+=(d[i+2*n]+d[i+6*n])*s;m[1]+=(d[i+3*n]+d[i+7*n])*s;}
    else{m[0]+=2.*d[i+2*n]*s;m[1]+=2.*d[i+3*n]*s;}
    m[2]+=(d[i]-d[i+n])*s;
}
// Site moment: orbital moments of the site summed in orbital order.
REMAG_SPIN_CONSTRAINT_HD inline void spin_constraint_site_moment(int site,int rows,const int* orbital_site,
        const double* orbital_moments,double* m) {
    m[0]=m[1]=m[2]=0.;
    for(int r=0;r<rows;r++) if(orbital_site[r]==site) {
        m[0]+=orbital_moments[3*r];m[1]+=orbital_moments[3*r+1];m[2]+=orbital_moments[3*r+2];
    }
}
// Augmented-Lagrangian field b + rho M_perp: the derivative of b.M + rho/2 |M_perp|^2
// with M_perp = M - (M.t)t, for target unit vector t.
REMAG_SPIN_CONSTRAINT_HD inline void spin_constraint_penalized_field(const double* b,const double* t,double rho,
        const double* m,double* f) {
    const double parallel=m[0]*t[0]+m[1]*t[1]+m[2]*t[2];
    for(int k=0;k<3;k++) f[k]=b[k]+rho*(m[k]-parallel*t[k]);
}
}
