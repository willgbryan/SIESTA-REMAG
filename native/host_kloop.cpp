// SPDX-License-Identifier: GPL-3.0-or-later
// Threaded CPU k-point loop for SIESTA's spin-orbit diagonalization; see host_kloop.hpp.
#include "host_kloop.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using cd = std::complex<double>;

extern "C" {
void zhegvd_(const int* itype, const char* jobz, const char* uplo, const int* n, cd* a, const int* lda, cd* b,
             const int* ldb, double* w, cd* work, const int* lwork, double* rwork, const int* lrwork, int* iwork,
             const int* liwork, int* info);
void zhegv_(const int* itype, const char* jobz, const char* uplo, const int* n, cd* a, const int* lda, cd* b,
            const int* ldb, double* w, cd* work, const int* lwork, double* rwork, int* info);
void zpotrf_(const char* uplo, const int* n, cd* a, const int* lda, int* info);
void zhegst_(const int* itype, const char* uplo, const int* n, cd* a, const int* lda, const cd* b, const int* ldb,
             int* info);
void zheevr_(const char* jobz, const char* range, const char* uplo, const int* n, cd* a, const int* lda,
             const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w,
             cd* z, const int* ldz, int* isuppz, cd* work, const int* lwork, double* rwork, const int* lrwork,
             int* iwork, const int* liwork, int* info);
void ztrsm_(const char* side, const char* uplo, const char* ta, const char* diag, const int* m, const int* n,
            const cd* alpha, const cd* a, const int* lda, cd* b, const int* ldb);
void zgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k, const cd* alpha, const cd* a,
            const int* lda, const cd* b, const int* ldb, const cd* beta, cd* c, const int* ldc);
// OpenBLAS thread control, when the BLAS provides it.
int openblas_get_num_threads(void) __attribute__((weak));
void openblas_set_num_threads(int) __attribute__((weak));
}

namespace {
using clock_type = std::chrono::steady_clock;
double since(clock_type::time_point t) {return std::chrono::duration<double>(clock_type::now()-t).count();}

// Eigensolvers: divide and conquer (zhegvd, every vector), MRRR (Cholesky,
// zhegst, zheevr for the lowest states only, ztrsm) and QR (zhegv).
enum Solver {solver_divide = 1, solver_mrrr = 2, solver_qr = 3};

struct Geometry {
    int nuo=0, no=0, maxnh=0, nk=0, threads=1, solver=solver_divide;
    const int *numh=nullptr, *listhptr=nullptr, *listh=nullptr, *indxuo=nullptr;
    const double *xij=nullptr, *kpoint=nullptr;
    std::vector<double> xorb;                 // 3*nuo
    std::vector<double> psup;                 // 3*no: supercell orbital positions
    std::vector<char> known;                  // psup defined for this supercell orbital
    bool factored=true;                       // phases factor through orbital positions
    double vector_budget=0.;                  // bytes
    std::vector<cd> vectors;                  // kept eigenvectors: nk * n * neig
    int kept_neig=0;
    double t_assembly=0., t_solve=0., t_density=0., t_total=0.;
} g;

int n2() {return 2*g.nuo;}

// Workspace of one thread.
struct Work {
    std::vector<cd> h, s, z, m, me, psiq, phase_s, phase_u, work;
    std::vector<double> w, rwork;
    std::vector<int> iwork, isuppz;
    const cd* psi = nullptr;                  // eigenvectors of the last solve, n x neig
    double t_assembly=0., t_solve=0., t_density=0.;
    void reserve(bool vectors) {
        const int n = n2();
        h.assign(size_t(n)*n, 0.); s.assign(size_t(n)*n, 0.);
        w.resize(n);
        int lwork = 0, lrwork = 0, liwork = 0;
        if (g.solver == solver_divide) {
            // zhegvd optimal sizes (LAPACK documentation).
            lwork = vectors ? 2*n + n*n : n + 1; lrwork = vectors ? 1 + 5*n + 2*n*n : n; liwork = vectors ? 3 + 5*n : 1;
        } else if (g.solver == solver_mrrr) {
            // zheevr: 2n minimum, (nb+1)n optimal; 64 covers the usual blocking.
            lwork = 65*n; lrwork = 24*n; liwork = 10*n;
            isuppz.resize(2*size_t(n));
            if (vectors) z.resize(size_t(n)*n);
        } else {
            lwork = 65*n; lrwork = std::max(1, 3*n-2); liwork = 1;
        }
        work.resize(lwork); rwork.resize(lrwork); iwork.resize(liwork);
        phase_s.resize(g.no); phase_u.resize(g.nuo);
    }
};

void phases(int ik, Work& t) {
    const double* k = g.kpoint + 3*size_t(ik);
    for (int i = 0; i < g.nuo; i++) {
        const double x = k[0]*g.xorb[3*i] + k[1]*g.xorb[3*i+1] + k[2]*g.xorb[3*i+2];
        t.phase_u[i] = cd(std::cos(x), std::sin(x));            // e^{+ik.x_i}
    }
    for (int j = 0; j < g.no; j++) {
        if (!g.known[j]) continue;
        const double x = k[0]*g.psup[3*j] + k[1]*g.psup[3*j+1] + k[2]*g.psup[3*j+2];
        t.phase_s[j] = cd(std::cos(x), -std::sin(x));           // e^{-ik.p_j}
    }
}

// e^{-ik.x_ij} for sparse entry ind of row iuo with column jo (0-based).
inline cd phase(int ik, const Work& t, int iuo, int jo, size_t ind) {
    if (g.factored && g.known[jo]) return t.phase_s[jo]*t.phase_u[iuo];
    const double* k = g.kpoint + 3*size_t(ik);
    const double* x = g.xij + 3*ind;
    const double a = k[0]*x[0] + k[1]*x[1] + k[2]*x[2];
    return cd(std::cos(a), -std::sin(a));
}

// H(k), S(k) in diag3k's layout: Haux(js,juo,is,iuo) at row js+2*juo, column is+2*iuo.
void assemble(int ik, const double* h, const double* s, Work& t) {
    auto start = clock_type::now();
    const int n = n2();
    std::fill(t.h.begin(), t.h.end(), cd(0.)); std::fill(t.s.begin(), t.s.end(), cd(0.));
    if (g.factored) phases(ik, t);
    const size_t mh = g.maxnh;
    for (int iuo = 0; iuo < g.nuo; iuo++) {
        const size_t c0 = size_t(2*iuo)*n, c1 = size_t(2*iuo+1)*n;
        for (int j = 0; j < g.numh[iuo]; j++) {
            const size_t ind = size_t(g.listhptr[iuo]) + j;
            const int jo = g.listh[ind]-1, juo = g.indxuo[jo]-1;
            const cd ph = phase(ik, t, iuo, jo, ind);
            const size_t r0 = 2*size_t(juo), r1 = r0+1;
            const cd sv = s[ind]*ph;
            t.s[r0+c0] += sv; t.s[r1+c1] += sv;
            t.h[r0+c0] += cd(h[ind], h[ind+4*mh])*ph;          // uu = H1 + i H5
            t.h[r1+c1] += cd(h[ind+mh], h[ind+5*mh])*ph;       // dd = H2 + i H6
            t.h[r0+c1] += cd(h[ind+2*mh], -h[ind+3*mh])*ph;    // ud = H3 - i H4
            t.h[r1+c0] += cd(h[ind+6*mh], h[ind+7*mh])*ph;     // du = H7 + i H8
        }
    }
    t.t_assembly += since(start);
}

// Lowest `wanted` eigenpairs of H(k) psi = e S(k) psi: values in t.w, and with
// vectors, t.psi. Divide and conquer and QR find every pair regardless.
int solve(Work& t, bool vectors, int wanted) {
    auto start = clock_type::now();
    const int n = n2(), one = 1, lwork = int(t.work.size()), lrwork = int(t.rwork.size()),
              liwork = int(t.iwork.size());
    int info = 0;
    const char* jobz = vectors ? "V" : "N";
    if (g.solver == solver_divide) {
        zhegvd_(&one, jobz, "U", &n, t.h.data(), &n, t.s.data(), &n, t.w.data(), t.work.data(), &lwork,
                t.rwork.data(), &lrwork, t.iwork.data(), &liwork, &info);
        t.psi = t.h.data();
    } else if (g.solver == solver_qr) {
        zhegv_(&one, jobz, "U", &n, t.h.data(), &n, t.s.data(), &n, t.w.data(), t.work.data(), &lwork, t.rwork.data(),
               &info);
        t.psi = t.h.data();
    } else {
        // S = U^H U; C = U^-H H U^-1; C y = e y; psi = U^-1 y.
        zpotrf_("U", &n, t.s.data(), &n, &info);
        if (info) return 1000+info;
        zhegst_(&one, "U", &n, t.h.data(), &n, t.s.data(), &n, &info);
        if (info) return info;
        const int il = 1, iu = std::max(1, std::min(wanted, n));
        const double zero = 0.;
        int found = 0;
        cd* z = vectors ? t.z.data() : t.h.data();
        zheevr_(jobz, iu == n ? "A" : "I", "U", &n, t.h.data(), &n, &zero, &zero, &il, &iu, &zero, &found, t.w.data(),
                z, &n, t.isuppz.data(), t.work.data(), &lwork, t.rwork.data(), &lrwork, t.iwork.data(), &liwork, &info);
        if (!info && found < iu) info = 2000;
        if (!info && vectors) {
            const cd alpha(1.);
            ztrsm_("L", "U", "N", "N", &n, &found, &alpha, t.s.data(), &n, z, &n);
        }
        t.psi = z;
    }
    t.t_solve += since(start);
    return info;
}

// Run f(ik, work) over all k points in fixed contiguous blocks, one per thread.
template <class F> int for_k(bool vectors, F f) {
    const int threads = std::max(1, std::min(g.threads, g.nk));
    std::vector<Work> work(threads);
    std::vector<int> status(threads, 0);
    int blas = 0;
    if (openblas_get_num_threads && openblas_set_num_threads && threads > 1) {
        blas = openblas_get_num_threads(); openblas_set_num_threads(1);
    }
    auto run = [&](int t) {
        try {
            work[t].reserve(vectors);
            const int a = int(size_t(g.nk)*t/threads), b = int(size_t(g.nk)*(t+1)/threads);
            for (int ik = a; ik < b && !status[t]; ik++) status[t] = f(ik, work[t], t);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "REMAG host k loop: %s\n", e.what()); status[t] = -501;
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; t++) pool.emplace_back(run, t);
    run(0);
    for (auto& p : pool) p.join();
    if (blas) openblas_set_num_threads(blas);
    for (auto& w : work) {g.t_assembly += w.t_assembly; g.t_solve += w.t_solve; g.t_density += w.t_density;}
    for (int s : status) if (s) return s;
    return 0;
}
}

extern "C" int remag_hostk_setup(int nuo, int no, int maxnh, const int* numh, const int* listhptr, const int* listh,
                                 const int* indxuo, const double* xij, const double* xorb, int nk,
                                 const double* kpoint, int threads, double vector_memory_mb, int solver) {
    if (nuo < 1 || no < nuo || maxnh < 1 || nk < 1 || threads < 1 || !numh || !listhptr || !listh || !indxuo || !xij ||
        !xorb || !kpoint || solver < solver_divide || solver > solver_qr) return -502;
    g.nuo = nuo; g.no = no; g.maxnh = maxnh; g.nk = nk; g.threads = threads; g.solver = solver;
    g.numh = numh; g.listhptr = listhptr; g.listh = listh; g.indxuo = indxuo; g.xij = xij; g.kpoint = kpoint;
    g.xorb.assign(xorb, xorb+3*size_t(nuo));
    g.psup.assign(3*size_t(no), 0.); g.known.assign(no, 0); g.factored = true;
    // p_j = x_ij + x_i must be the same for every entry of supercell orbital j.
    for (int iuo = 0; iuo < nuo && g.factored; iuo++)
        for (int j = 0; j < numh[iuo]; j++) {
            const size_t ind = size_t(listhptr[iuo]) + j;
            const int jo = listh[ind]-1;
            if (jo < 0 || jo >= no || indxuo[jo] < 1 || indxuo[jo] > nuo) return -503;
            double p[3];
            for (int x = 0; x < 3; x++) p[x] = xij[3*ind+x] + xorb[3*iuo+x];
            if (!g.known[jo]) {for (int x = 0; x < 3; x++) g.psup[3*jo+x] = p[x]; g.known[jo] = 1; continue;}
            for (int x = 0; x < 3; x++)
                if (std::fabs(g.psup[3*jo+x]-p[x]) > 1.e-9*(1.+std::fabs(p[x]))) {g.factored = false; break;}
        }
    g.vector_budget = vector_memory_mb*1048576.;
    return 0;
}

extern "C" int remag_hostk_spectrum(const double* h, const double* s, int neig, int maxe, double* eo) {
    auto start = clock_type::now();
    const int n = n2();
    if (!h || !s || !eo || neig < 1 || neig > n || maxe < neig) return -504;
    const double bytes = double(g.nk)*n*neig*sizeof(cd);
    const bool keep = bytes <= g.vector_budget;
    g.kept_neig = 0;
    if (keep) g.vectors.assign(size_t(g.nk)*n*neig, cd(0.)); else std::vector<cd>().swap(g.vectors);
    int status = for_k(keep, [&](int ik, Work& t, int) {
        assemble(ik, h, s, t);
        const int info = solve(t, keep, neig);
        if (info) return -510-std::abs(info)%100;
        std::copy_n(t.w.data(), neig, eo + size_t(maxe)*ik);
        if (keep) std::copy_n(t.psi, size_t(n)*neig, g.vectors.data() + size_t(ik)*n*neig);
        return 0;
    });
    if (!status && keep) g.kept_neig = neig;
    g.t_total += since(start);
    return status;
}

extern "C" int remag_hostk_density(const double* h, const double* s, int neig, int maxe, const double* eo,
                                   const double* qo, double occtol, int maxnd, const int* numd, const int* listdptr,
                                   const int* listd, double* dnew, double* enew) {
    auto start = clock_type::now();
    const int n = n2();
    if (!h || !s || !eo || !qo || !numd || !listdptr || !listd || !dnew || !enew || neig < 1 || neig > n ||
        maxnd < 1) return -505;
    const bool kept = g.kept_neig == neig;
    const int threads = std::max(1, std::min(g.threads, g.nk));
    // Per-thread accumulators, summed in thread order at the end.
    std::vector<std::vector<double>> dacc(threads), eacc(threads);
    int status = for_k(!kept, [&](int ik, Work& t, int thread) {
        const double* q = qo + size_t(maxe)*ik;
        const double* e = eo + size_t(maxe)*ik;
        int m = 0;
        for (int i = neig-1; i >= 0; i--) if (std::fabs(q[i]) > occtol) {m = i+1; break;}
        if (dacc[thread].empty()) {dacc[thread].assign(size_t(maxnd)*8, 0.); eacc[thread].assign(size_t(maxnd)*4, 0.);}
        if (m == 0) return 0;
        const cd* psi;
        if (kept) {
            psi = g.vectors.data() + size_t(ik)*n*neig;
            if (g.factored) phases(ik, t);
        } else {
            assemble(ik, h, s, t);
            // Only the occupied states are needed now.
            const int info = solve(t, true, m);
            if (info) return -520-std::abs(info)%100;
            psi = t.psi;
        }
        auto t0 = clock_type::now();
        // M = Psi diag(q) Psi^H and ME = Psi diag(q e) Psi^H over the m occupied states.
        t.psiq.resize(size_t(n)*m); t.m.resize(size_t(n)*n); t.me.resize(size_t(n)*n);
        const cd one(1.), zero(0.);
        for (int i = 0; i < m; i++) for (int r = 0; r < n; r++) t.psiq[r+size_t(n)*i] = q[i]*psi[r+size_t(n)*i];
        zgemm_("N", "C", &n, &n, &m, &one, t.psiq.data(), &n, psi, &n, &zero, t.m.data(), &n);
        for (int i = 0; i < m; i++) for (int r = 0; r < n; r++) t.psiq[r+size_t(n)*i] *= e[i];
        zgemm_("N", "C", &n, &n, &m, &one, t.psiq.data(), &n, psi, &n, &zero, t.me.data(), &n);
        // Dk(a,juo,b,iuo) = sum q c_a(iuo) c_b(juo)^* = M[(a,iuo),(b,juo)].
        double* D = dacc[thread].data(); double* E = eacc[thread].data();
        const size_t md = maxnd;
        for (int iuo = 0; iuo < g.nuo; iuo++)
            for (int j = 0; j < numd[iuo]; j++) {
                const size_t ind = size_t(listdptr[iuo]) + j;
                const int jo = listd[ind]-1, juo = g.indxuo[jo]-1;
                const cd ph = phase(ik, t, iuo, jo, ind);
                auto at = [&](const std::vector<cd>& x, int a, int b) {
                    return x[(2*size_t(iuo)+a) + size_t(n)*(2*size_t(juo)+b)]*ph;
                };
                const cd d11 = at(t.m, 0, 0), d22 = at(t.m, 1, 1), d12 = at(t.m, 0, 1), d21 = at(t.m, 1, 0);
                D[ind] += d11.real(); D[ind+md] += d22.real(); D[ind+2*md] += d12.real();
                D[ind+3*md] -= d12.imag(); D[ind+4*md] += d11.imag(); D[ind+5*md] += d22.imag();
                D[ind+6*md] += d21.real(); D[ind+7*md] += d21.imag();
                const cd e11 = at(t.me, 0, 0), e22 = at(t.me, 1, 1), e12 = at(t.me, 0, 1);
                E[ind] += e11.real(); E[ind+md] += e22.real(); E[ind+2*md] += e12.real(); E[ind+3*md] -= e12.imag();
            }
        t.t_density += since(t0);
        return 0;
    });
    if (!status) {
        std::fill_n(dnew, size_t(maxnd)*8, 0.); std::fill_n(enew, size_t(maxnd)*4, 0.);
        for (int t = 0; t < threads; t++) {
            if (dacc[t].empty()) continue;
            for (size_t i = 0; i < size_t(maxnd)*8; i++) dnew[i] += dacc[t][i];
            for (size_t i = 0; i < size_t(maxnd)*4; i++) enew[i] += eacc[t][i];
        }
    }
    std::vector<cd>().swap(g.vectors); g.kept_neig = 0;
    g.t_total += since(start);
    return status;
}

extern "C" void remag_hostk_timings(double* seconds) {
    seconds[0] = g.t_assembly; seconds[1] = g.t_solve; seconds[2] = g.t_density; seconds[3] = g.t_total;
}
