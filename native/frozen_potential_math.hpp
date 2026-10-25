// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define REMAG_MAE_HD __host__ __device__
#else
#define REMAG_MAE_HD
#endif

namespace remag::mae {
// Shared host/device arithmetic for the frozen-potential executor. It does not
// obtain a converged reference, solve eigenproblems or certify the force theorem.
REMAG_MAE_HD inline bool finite(double x) {
    return x == x && x != HUGE_VAL && x != -HUGE_VAL;
}

struct SpinRotation {
    double delta[9]{}; // R-I, row-major; active U = w I - i (x,y,z).sigma
};

// Normalize after scaling so otherwise valid large/small quaternions cannot
// overflow/underflow the norm. Failure leaves the destination unchanged.
REMAG_MAE_HD inline bool spin_rotation(const double q[4], SpinRotation& out) {
    double scale = 0.;
    for (int i=0;i<4;i++) {
        if (!finite(q[i])) return false;
        scale = fmax(scale, fabs(q[i]));
    }
    if (scale == 0.) return false;
    double a[4], norm = 0.;
    for (int i=0;i<4;i++) { a[i]=q[i]/scale; norm+=a[i]*a[i]; }
    norm=sqrt(norm);
    const double w=a[0]/norm, x=a[1]/norm, y=a[2]/norm, z=a[3]/norm;
    out={{-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
           2*(x*y+w*z), -2*(x*x+z*z), 2*(y*z-w*x),
           2*(x*z-w*y), 2*(y*z+w*x), -2*(x*x+y*y)}};
    return true;
}

// Literal component additions in pinned setup_hamiltonian.F, full offsite SOC.
// off contains four complex values as re,im pairs, ordered uu,dd,ud,du.
REMAG_MAE_HD inline void offsite_soc(const double off[8], double soc[8]) {
    const double values[8]={off[0],off[2],off[4],off[5],off[1],off[3],off[6],-off[7]};
    for (int i=0;i<8;i++) soc[i]=values[i];
}

// Each real-space orbital-pair entry need not itself be Hermitian. Preserve
// the complete complex spin block, including imaginary diagonal components.
// SIESTA assembly interprets it as [[h0+i*h4,h2-i*h3],[h6+i*h7,h1+i*h5]].
// Compute H' = H + (R-I) B.sigma, B from H-Hsoc. This leaves both the large
// spin-independent part and lattice SOC fixed, and makes identity exact.
// The caller must check the return value before consuming any result. In-place
// output is supported; inputs are unchanged on nonfinite-data failure.
REMAG_MAE_HD inline bool rotate_exchange(const SpinRotation& r, const double h[8],
                                        const double soc[8], double out[8]) {
    for (int i=0;i<8;i++) if (!finite(h[i]) || !finite(soc[i])) return false;
    for (double x:r.delta) if (!finite(x)) return false;
    const double b[2][3]={
        { .5*((h[2]+h[6])-(soc[2]+soc[6])),
          .5*((h[3]+h[7])-(soc[3]+soc[7])),
          .5*((h[0]-h[1])-(soc[0]-soc[1])) },
        { .5*((h[7]-h[3])-(soc[7]-soc[3])),
          .5*((h[2]-h[6])-(soc[2]-soc[6])),
          .5*((h[4]-h[5])-(soc[4]-soc[5])) }
    };
    double d[2][3]{};
    for (int c=0;c<2;c++) for (int i=0;i<3;i++)
        for (int j=0;j<3;j++) d[c][i]+=r.delta[3*i+j]*b[c][j];
    const double v[8]={h[0]+d[0][2], h[1]-d[0][2],
        h[2]+d[0][0]+d[1][1], h[3]-d[1][0]+d[0][1],
        h[4]+d[1][2], h[5]-d[1][2],
        h[6]+d[0][0]-d[1][1], h[7]+d[1][0]+d[0][1]};
    for (double x:v) if (!finite(x)) return false;
    for (int i=0;i<8;i++) out[i]=v[i];
    return true;
}

struct CompensatedSum {
    double sum=0., correction=0.;
    REMAG_MAE_HD void add(double x) {
        const double next=sum+x;
        correction+=fabs(sum)>=fabs(x) ? (sum-next)+x : (x-next)+sum;
        sum=next;
    }
    REMAG_MAE_HD void merge(const CompensatedSum& other) {
        const double value=other.sum, residual=other.correction;
        add(value); add(residual);
    }
    REMAG_MAE_HD double value() const { return sum+correction; }
};

// The same FD entropy floor as pinned fermid.F / occupations.cu. Weights and
// electron counts are explicit; full spinors have degeneracy one.
REMAG_MAE_HD inline double fd_entropy(double p) {
    const double occupied=fmax(p,1.e-15), empty=fmax(1.-p,1.e-15);
    return -occupied*log(occupied)-empty*log(empty);
}

struct BandDifference {
    double band_energy=0., band_free_energy=0., canonical_free_energy=0.;
    double entropy=0., charge_a=0., charge_b=0., charge_difference=0.;
    double charge_error_a=0., charge_error_b=0.;
    bool finite_result=false, electron_count_passed=false;
};

// Pair identical k/band layouts and weights at a common electronic temperature.
// q_a/q_b already include the k weight, as in fermi_device. Subtract states
// before summing, rather than two extensive band energies. All energies,
// chemical potentials, the common origin and k_B*T use one consistent unit.
// No zero-temperature extrapolation or full-SCF total energy is implied.
struct BandPairSum {
    double origin;
    CompensatedSum centered, entropy, charge_a, charge_b, charge_difference;
    bool valid=true;
    unsigned long long states=0;
    REMAG_MAE_HD explicit BandPairSum(double energy_origin):origin(energy_origin) {
        valid=finite(origin);
    }
    REMAG_MAE_HD bool add(double ea,double eb,double qa,double qb,double weight) {
        if (!valid || !finite(ea) || !finite(eb) || !finite(qa) || !finite(qb) ||
            !finite(weight) || weight<=0. || qa<0. || qb<0. || qa>weight || qb>weight) {
            valid=false; return false;
        }
        centered.add(qa*(eb-ea)+(qb-qa)*(eb-origin));
        entropy.add(weight*(fd_entropy(qb/weight)-fd_entropy(qa/weight)));
        charge_a.add(qa); charge_b.add(qb); charge_difference.add(qb-qa); ++states;
        return true;
    }
    REMAG_MAE_HD bool merge(const BandPairSum& other) {
        if (this==&other || !valid || !other.valid || origin!=other.origin) {
            valid=false; return false;
        }
        centered.merge(other.centered); entropy.merge(other.entropy);
        charge_a.merge(other.charge_a); charge_b.merge(other.charge_b);
        charge_difference.merge(other.charge_difference); states+=other.states;
        return true;
    }
    REMAG_MAE_HD BandDifference finish(double temperature,double mu_a,double mu_b,
                                      double electrons,double charge_tolerance=1.e-10) const {
        BandDifference out;
        if (!valid || !states || !finite(temperature) || temperature<0. ||
            !finite(mu_a) || !finite(mu_b) || !finite(electrons) || electrons<0. ||
            !finite(charge_tolerance) || charge_tolerance<0.) return out;
        out.charge_a=charge_a.value(); out.charge_b=charge_b.value();
        out.charge_difference=charge_difference.value(); out.entropy=entropy.value();
        out.charge_error_a=out.charge_a-electrons; out.charge_error_b=out.charge_b-electrons;
        out.band_energy=centered.value()+origin*out.charge_difference;
        out.band_free_energy=out.band_energy-temperature*out.entropy;
        // Omega_b + mu_b*N - (Omega_a + mu_a*N), evaluated at the SAME target N.
        // Preserve the raw band sum above; do not silently replace it with this
        // residual-charge correction or confuse the two thermodynamic scopes.
        out.canonical_free_energy=centered.value()-temperature*out.entropy+
            (origin-mu_b)*out.charge_error_b-(origin-mu_a)*out.charge_error_a;
        out.finite_result=finite(out.band_energy)&&finite(out.band_free_energy)&&
            finite(out.canonical_free_energy)&&finite(out.entropy)&&
            finite(out.charge_a)&&finite(out.charge_b)&&finite(out.charge_difference);
        out.electron_count_passed=out.finite_result&&fabs(out.charge_error_a)<=charge_tolerance&&
            fabs(out.charge_error_b)<=charge_tolerance;
        return out;
    }
};
}
#undef REMAG_MAE_HD
