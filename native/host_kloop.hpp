// SPDX-License-Identifier: GPL-3.0-or-later
// Threaded CPU k-point loop for SIESTA's spin-orbit diagonalization (diag3k.F):
// the same eigenvalues, density and energy-density matrices, with
//   - k points shared among threads in fixed contiguous blocks and summed in
//     block order, so results are deterministic for a given thread count;
//   - Bloch phases factored as e^{-ik.x_ij} = e^{-ik.p_j} e^{+ik.x_i} over the
//     positions of supercell and unit-cell orbitals, one sincos per orbital
//     instead of one per matrix element;
//   - the density built as one complex matrix product per k point instead of
//     a rank-one update per state;
//   - a choice of LAPACK eigensolver;
//   - eigenvectors kept from the eigenvalue pass when they fit in the memory
//     budget, so each k point is assembled and solved once instead of twice.
// Dense matrices use diag3k's spinor order: row (js, juo) -> js + 2*juo.
#pragma once
#include <cstddef>

extern "C" {
// Geometry for this call. xorb(3,nuo): positions of the unit-cell orbitals
// (their atoms); xij(3,maxnh) as SIESTA. solver: 1 divide and conquer
// (zhegvd), 2 MRRR (zheevr after a Cholesky reduction, which can stop at the
// occupied states), 3 QR (zhegv). Returns 0, or a negative status.
int remag_hostk_setup(int nuo, int no, int maxnh, const int* numh, const int* listhptr, const int* listh,
                      const int* indxuo, const double* xij, const double* xorb, int nk, const double* kpoint,
                      int threads, double vector_memory_mb, int solver);
// Eigenvalues eo(maxe,nk), the lowest neig of 2*nuo, for H(maxnh,8), S(maxnh).
int remag_hostk_spectrum(const double* h, const double* s, int neig, int maxe, double* eo);
// Dnew(maxnd,8), Enew(maxnd,4) from occupations qo(maxe,nk) (k weight times
// occupation) and eo, over states with |q| > occtol. Overwrites Dnew, Enew.
int remag_hostk_density(const double* h, const double* s, int neig, int maxe, const double* eo, const double* qo,
                        double occtol, int maxnd, const int* numd, const int* listdptr, const int* listd,
                        double* dnew, double* enew);
// Seconds spent in each stage since start: assembly, eigensolve, density, total.
void remag_hostk_timings(double* seconds);
}
