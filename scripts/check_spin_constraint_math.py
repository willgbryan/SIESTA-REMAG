#!/usr/bin/env python3
"""Validate site spin-constraint arithmetic and the host outer search on the CPU only.

Exact dyadic fixtures compare the shared packing/moment header with an
operator built from the Mulliken site weights of Cuadrado et al. (2018) in
pinned SIESTA's spinor storage, and check Tr(rho H_c) = sum_s b_s.M_s.
A self-consistent noncollinear p-orbital model with spin-orbit coupling
drives the production host ABI (spin_constraint_state.cpp) through every
optimizer mode and checks the energy-torque identity. No GPU or SIESTA
integration is claimed here.
"""
import argparse
import ctypes as ct
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'native' if (ROOT/'native').is_dir() else Path('/opt/remag/native')
WRAPPER = r'''#include "spin_constraint_math.hpp"
#include "spin_constraint_optimizer.hpp"
#include <vector>
extern "C" void pack(int spins,int rows,size_t capacity,const int* num,const int* ptr,const int* cols,
                     const int* mapping,const double* s,const double* field,double* h) {
 for(int r=0;r<rows;r++) for(int j=0;j<num[r];j++) {
  const size_t i=size_t(ptr[r])+j;const int c=mapping?mapping[cols[i]-1]-1:(cols[i]-1)%rows;
  remag::spin_constraint_add(spins,capacity,i,s[i],field+3*size_t(r),field+3*size_t(c),h);
 }
}
extern "C" void moments(int spins,int rows,size_t capacity,const int* num,const int* ptr,const double* s,
                        const double* d,double* m) {
 for(int r=0;r<rows;r++) for(int j=0;j<num[r];j++)
  remag::spin_constraint_moment(spins,capacity,size_t(ptr[r])+j,s[size_t(ptr[r])+j],d,m+3*size_t(r));
}
extern "C" void penalize(int rows,const int* site,const double* moments,const double* fields,const double* targets,
                         double rho,double* out) {
 for(int r=0;r<rows;r++) {
  if(site[r]<0){out[3*r]=out[3*r+1]=out[3*r+2]=0.;continue;}
  double m[3];remag::spin_constraint_site_moment(site[r],rows,site,moments,m);
  remag::spin_constraint_penalized_field(fields+3*site[r],targets+3*site[r],rho,m,out+3*size_t(r));
 }
}
extern "C" void* optimizer_new() {return new remag::SpinConstraintOptimizer();}
extern "C" void optimizer_free(void* o) {delete static_cast<remag::SpinConstraintOptimizer*>(o);}
extern "C" int optimizer_configure(void* o,int sites,const int* options,const double* v,const double* angles) {
 remag::SpinConstraintSettings s;s.sites=sites;s.direction=options[0];s.line=options[1];s.criteria=options[2];
 s.maximum_evaluations=options[3];s.initial_field=v[0];s.energy_tolerance=v[1];s.slope_ratio=v[2];
 s.angle_tolerance=v[3];s.torque_tolerance=v[4];s.minimum_moment=v[5];s.maximum_field=v[6];
 s.coarse_angle_tolerance=v[7];s.coarse_torque_tolerance=v[8];s.penalty=v[9];
 std::vector<remag::SpinConstraintSite> f;
 for(int i=0;i<sites;i++) f.push_back(remag::spin_constraint_site(angles[2*i],angles[2*i+1]));
 return static_cast<remag::SpinConstraintOptimizer*>(o)->configure(s,f.data());
}
extern "C" void optimizer_fields(void* o,double* b) {static_cast<remag::SpinConstraintOptimizer*>(o)->fields(b);}
extern "C" int optimizer_observe(void* o,double value,const double* m,int* done) {
 return static_cast<remag::SpinConstraintOptimizer*>(o)->observe(value,m,done);
}
extern "C" double optimizer_penalty(void* o) {return static_cast<remag::SpinConstraintOptimizer*>(o)->penalty();}
extern "C" int optimizer_coarse(void* o) {return static_cast<remag::SpinConstraintOptimizer*>(o)->coarse();}
extern "C" void optimizer_measure(void* o,double* v) {
 auto* p=static_cast<remag::SpinConstraintOptimizer*>(o);const auto& m=p->measure();
 v[0]=m.angle;v[1]=m.torque;v[2]=m.field;v[3]=p->evaluations();v[4]=p->lines();
}
extern "C" void frame(double theta,double phi,double* out) {
 const auto f=remag::spin_constraint_site(theta,phi);
 for(int k=0;k<3;k++){out[k]=f.target[k];out[3+k]=f.e1[k];out[6+k]=f.e2[k];}
}
'''
# Only type names are needed: the host owner declares, never calls, device code.
CUDA_STUB = 'typedef int cudaError_t;\ntypedef struct CUstream_st* cudaStream_t;\n'
SOURCES = ['spin_constraint_math.hpp', 'spin_constraint_optimizer.hpp', 'spin_constraint_state.cpp',
           'spin_constraint.hpp', 'execution.cpp', 'execution.hpp', 'hamiltonian.hpp']
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
I32 = np.int32


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(directory):
    directory = Path(directory)
    (directory/'stub').mkdir(exist_ok=True)
    (directory/'stub'/'cuda_runtime_api.h').write_text(CUDA_STUB)
    (directory/'wrapper.cpp').write_text(WRAPPER)
    flags = ['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
             '-I'+str(NATIVE), '-I'+str(directory/'stub')]
    if platform.system() == 'Darwin': flags.append('-Wl,-undefined,dynamic_lookup')
    library = directory/'libspin_constraint_check.so'
    subprocess.run(flags+[str(directory/'wrapper.cpp'), str(NATIVE/'spin_constraint_state.cpp'),
                          str(NATIVE/'execution.cpp'), '-o', str(library)], check=True)
    return library


def load(library):
    lib = ct.CDLL(str(library))
    p, i, d = ct.c_void_p, ct.c_int, ct.c_double
    signatures = {
        'pack': ([i, i, ct.c_size_t, p, p, p, p, p, p, p], None),
        'moments': ([i, i, ct.c_size_t, p, p, p, p, p], None),
        'optimizer_new': ([], p), 'optimizer_free': ([p], None),
        'optimizer_configure': ([p, i, p, p, p], i), 'optimizer_fields': ([p, p], None),
        'optimizer_observe': ([p, d, p, p], i), 'optimizer_measure': ([p, p], None), 'frame': ([d, d, p], None),
        'optimizer_coarse': ([p], i), 'penalize': ([i, p, p, p, p, d, p], None), 'optimizer_penalty': ([p], d),
        'remag_spin_constraint_penalty': ([], d),
        'remag_spin_constraint_configure': ([i, p, i, p, p, p], i), 'remag_spin_constraint_begin': ([i], i),
        'remag_spin_constraint_fields': ([i, p], i),
        'remag_spin_constraint_host_add': ([i, i, i, p, p, p, p, p, p, p], i),
        'remag_spin_constraint_host_moments': ([i, i, i, p, p, p, p], i),
        'remag_spin_constraint_observe': ([d, i, p, p, p], i), 'remag_spin_constraint_measure': ([p], i),
        'remag_spin_constraint_retain_geometry': ([], i), 'remag_spin_constraint_coarse': ([], i)}
    for name, (args, result) in signatures.items():
        f = getattr(lib, name); f.argtypes = args; f.restype = result
    return lib


def ptr(a): return None if a is None else a.ctypes.data


# ---------------------------------------------------------------- sparse layout
class Sparse:
    """Rows of unit-cell orbitals; columns (nu, r) are supercell orbitals nu+n*(r+1)+1.

    The mirror of entry (io, nu, r) is (nu, io, -r); SIESTA's sparse patterns
    are symmetric in this sense, and Hermitian matrices pair the two blocks."""
    def __init__(self, n, rows):
        self.n = n; self.entries = [(io, nu, r) for io, row in enumerate(rows) for nu, r in row]
        self.num = np.array([len(row) for row in rows], I32)
        self.ptr = np.concatenate([[0], np.cumsum(self.num)[:-1]]).astype(I32)
        self.cols = np.array([nu+n*(r+1)+1 for _, nu, r in self.entries], I32)
        self.mapping = np.array([j % n+1 for j in range(3*n)], I32)
        where = {e: k for k, e in enumerate(self.entries)}
        self.mirror = np.array([where[(nu, io, -r)] for io, nu, r in self.entries])
        self.capacity = len(self.entries)


def random_pattern(rng, n, images, density=.45):
    pairs = {(i, i, 0) for i in range(n)}
    for i in range(n):
        for j in range(n):
            for r in ((-1, 0, 1) if images else (0,)):
                if (i, j, r) != (j, i, -r) and rng.random() < density:
                    pairs |= {(i, j, r), (j, i, -r)}
    rows = [[] for _ in range(n)]
    for i, j, r in sorted(pairs): rows[i].append((j, r))
    for row in rows: rng.shuffle(row)
    return Sparse(n, rows)


def dyadic(rng, shape, scale=256):
    return rng.integers(-scale, scale+1, size=shape)/64.


def spinor_blocks(rng, sp, noncollinear):
    """rho[(io,s),(col,s')] per entry: Hermitian pairing with the mirror entry."""
    blocks = np.zeros((sp.capacity, 2, 2), complex)
    for k, m in enumerate(sp.mirror):
        if k > m: continue
        x = dyadic(rng, (2, 2))+1j*dyadic(rng, (2, 2))
        if k == m or noncollinear: x = np.triu(x)+np.triu(x, 1).conj().T; x[0, 0] = x[0, 0].real; x[1, 1] = x[1, 1].real
        blocks[k] = x; blocks[m] = x.conj().T
    return blocks


def pack_density(blocks, spins):
    """SIESTA diag3g/diag2g storage of rho blocks, (spins, capacity) = Fortran (capacity, spins)."""
    a, c, f, b = blocks[:, 0, 0], blocks[:, 0, 1], blocks[:, 1, 0], blocks[:, 1, 1]
    if spins == 8: return np.array([a.real, b.real, c.real, -c.imag, a.imag, b.imag, f.real, f.imag])
    return np.array([a.real, b.real, c.real, -c.imag])


def unpack_hamiltonian(h, spins):
    """Operator blocks H[(col,s),(io,s')] of SIESTA diag3g/diag2g assembly."""
    o = np.zeros((h.shape[1], 2, 2), complex)
    o[:, 0, 0] = h[0]+1j*(h[4] if spins == 8 else 0); o[:, 1, 1] = h[1]+1j*(h[5] if spins == 8 else 0)
    o[:, 0, 1] = h[2]-1j*h[3]; o[:, 1, 0] = (h[6]+1j*h[7]) if spins == 8 else h[2]+1j*h[3]
    return o


# ---------------------------------------------------------------- exact checks
def F(x): return Fraction(float(x))


def exact_fixture(lib, rng, spins, images, label):
    n = int(rng.integers(7, 12)); sp = random_pattern(rng, n, images)
    atoms = np.sort(rng.integers(0, 5, size=n)); atoms[:5] = np.arange(5); atoms.sort()
    site_of_atom = {0: 0, 1: 1, 2: 1, 3: 2}            # atom 4 is unconstrained; site 1 is a group
    orbital_site = np.array([site_of_atom.get(int(a), -1) for a in atoms], I32)
    s = np.zeros(sp.capacity)
    for k, m in enumerate(sp.mirror):
        if k <= m: s[k] = s[m] = dyadic(rng, ())
    blocks = spinor_blocks(rng, sp, spins == 4)
    d = np.ascontiguousarray(pack_density(blocks, spins))
    b = dyadic(rng, (3, 3))
    field = np.ascontiguousarray(np.array([b[t] if t >= 0 else np.zeros(3) for t in orbital_site]).ravel())
    h = np.zeros((spins, sp.capacity)); m = np.zeros(3*n)
    mapping = sp.mapping if images else None
    lib.pack(spins, n, sp.capacity, ptr(sp.num), ptr(sp.ptr), ptr(sp.cols), ptr(mapping), ptr(s), ptr(field), ptr(h))
    lib.moments(spins, n, sp.capacity, ptr(sp.num), ptr(sp.ptr), ptr(s), ptr(d), ptr(m))
    # Oracle operator from the paper's weights: S if both orbitals on a site,
    # S/2 if one, per site b.sigma; then SIESTA storage of that operator.
    expected = [[Fraction(0)]*sp.capacity for _ in range(spins)]
    energy_operator = [Fraction(0), Fraction(0)]
    site_moment = [[Fraction(0)]*3 for _ in range(3)]
    for k, (io, nu, r) in enumerate(sp.entries):
        o = [[(Fraction(0), Fraction(0)) for _ in range(2)] for _ in range(2)]
        for t in range(3):
            w = F(s[k])*(Fraction(int(orbital_site[io] == t))+Fraction(int(orbital_site[nu] == t)))/2
            bx, by, bz = (F(v) for v in b[t])
            add = [[(w*bz, Fraction(0)), (w*bx, -w*by)], [(w*bx, w*by), (-w*bz, Fraction(0))]]
            o = [[(o[i][j][0]+add[i][j][0], o[i][j][1]+add[i][j][1]) for j in range(2)] for i in range(2)]
        stored = [o[0][0][0], o[1][1][0], o[0][1][0], -o[0][1][1]]
        if spins == 8: stored += [o[0][0][1], o[1][1][1], o[1][0][0], o[1][0][1]]
        for c in range(spins): expected[c][k] = stored[c]
        rho = [[(F(blocks[k, i, j].real), F(blocks[k, i, j].imag)) for j in range(2)] for i in range(2)]
        for i in range(2):                      # Tr(rho H): rho[(io,s),(col,s')] H[(col,s'),(io,s)]
            for j in range(2):
                (ar, ai), (br, bi) = rho[i][j], o[j][i]
                energy_operator[0] += ar*br-ai*bi; energy_operator[1] += ar*bi+ai*br
        if orbital_site[io] >= 0:
            for alpha, sigma in enumerate(PAULI):
                trace = sum(F((blocks[k, i, j]*sigma[j, i]).real) for i in range(2) for j in range(2))
                site_moment[orbital_site[io]][alpha] += F(s[k])*trace
    packed_ok = all(F(h[c, k]) == expected[c][k] for c in range(spins) for k in range(sp.capacity))
    weights = [1, 1, 1, 1, 1, 1, 1, 1] if spins == 8 else [1, 1, 2, 2]
    energy_packed = sum(weights[c]*F(h[c, k])*F(d[c, k]) for c in range(spins) for k in range(sp.capacity))
    energy_moments = sum(F(b[t, a])*site_moment[t][a] for t in range(3) for a in range(3))
    candidate_sites = [[sum(F(m[3*r+a]) for r in range(n) if orbital_site[r] == t) for a in range(3)] for t in range(3)]
    moments_ok = candidate_sites == site_moment
    # Sensitivity control: an anti-Hermitian defect in one mirror pair must
    # break the identity, so the exact equality above is not vacuous.
    k = next(k for k, mm in enumerate(sp.mirror) if k != mm and orbital_site[sp.entries[k][0]] >= 0 and s[k] != 0)
    broken = blocks.copy(); broken[k, 0, 1] += .25
    dbroken = pack_density(broken, spins)
    energy_broken = sum(weights[c]*F(h[c, kk])*F(dbroken[c, kk]) for c in range(spins) for kk in range(sp.capacity))
    passed = packed_ok and moments_ok and energy_operator[1] == 0 and \
        energy_packed == energy_operator[0] == energy_moments and energy_broken != energy_moments
    return dict(fixture=label, spins=spins, images=images, orbitals=n, entries=sp.capacity,
                exact_packing=packed_ok, exact_moments=moments_ok,
                trace_identity=bool(energy_packed == energy_operator[0] == energy_moments),
                imaginary_trace_zero=energy_operator[1] == 0, defect_detected=energy_broken != energy_moments,
                energy=float(energy_moments), passed=bool(passed))


def exact_penalty(lib, rng):
    """b + rho M_perp per orbital, exactly, against Fractions (dyadic inputs)."""
    rows = 13; site = np.array([0, 0, 1, -1, 1, 2, 2, 2, -1, 0, 1, 2, -1], I32)
    moments = dyadic(rng, 3*rows); fields = dyadic(rng, 9); rho = .375
    targets = np.array([[0., 0., 1.], [1., 0., 0.], [0., .6, .8]]).ravel()   # dyadic-free but exact unit vectors
    out = np.zeros(3*rows); lib.penalize(rows, ptr(site), ptr(moments), ptr(fields), ptr(targets), rho, ptr(out))
    ok = True
    for r in range(rows):
        s = site[r]
        if s < 0: ok &= bool(np.all(out[3*r:3*r+3] == 0)); continue
        m = [sum(F(moments[3*q+k]) for q in range(rows) if site[q] == s) for k in range(3)]
        tt = [F(x) for x in targets[3*s:3*s+3]]; parallel = sum(a*b for a, b in zip(m, tt))
        expected = [F(fields[3*s+k])+F(rho)*(m[k]-parallel*tt[k]) for k in range(3)]
        # 0.6 and 0.8 are inexact in binary: compare those to rounding, the rest exactly.
        if s == 2: ok &= bool(np.allclose(out[3*r:3*r+3], [float(e) for e in expected], rtol=0, atol=1.e-15))
        else: ok &= [F(x) for x in out[3*r:3*r+3]] == expected
    return dict(case='penalized_field', passed=ok)


# ---------------------------------------------------------------- analytic dual
SETTINGS = dict(initial_field=.02, energy_tolerance=1.e-12, slope_ratio=.1, angle_tolerance=np.deg2rad(1.e-4),
                torque_tolerance=1.e-6, minimum_moment=1.e-3, maximum_field=10.)


def values(settings):
    keys = ['initial_field', 'energy_tolerance', 'slope_ratio', 'angle_tolerance', 'torque_tolerance',
            'minimum_moment', 'maximum_field', 'coarse_angle_tolerance', 'coarse_torque_tolerance', 'penalty']
    return np.array([settings.get(k, 0.) for k in keys])


def frames(lib, angles):
    out = []
    for theta, phi in angles:
        v = np.zeros(9); lib.frame(theta, phi, ptr(v)); out.append(v.reshape(3, 3))
    return out


def quadratic_dual(lib, rng, direction, line, criteria=1, override=None, perturb=None, maximum=1000):
    """L(x) = c - (x-x*)^T A (x-x*)/2: concave, gradient -A(x-x*) as transverse moments."""
    sites = 3; angles = [(.3, .2), (1.7, 2.9), (2.6, -1.)]
    q = np.linalg.qr(rng.normal(size=(6, 6)))[0]; a = (q*np.geomspace(4., 1., 6))@q.T
    target = rng.normal(size=6)*.3; f = frames(lib, angles)
    settings = dict(SETTINGS, **(override or {}))
    o = lib.optimizer_new()
    try:
        status = lib.optimizer_configure(o, sites, ptr(np.array([direction, line, criteria, maximum], I32)),
                                         ptr(values(settings)), ptr(np.array(angles).ravel()))
        if status: return dict(status=status)
        tiers = []
        for evaluation in range(1, maximum+2):
            b = np.zeros(9); lib.optimizer_fields(o, ptr(b)); b = b.reshape(3, 3)
            x = np.array([[b[s]@f[s][1], b[s]@f[s][2]] for s in range(3)]).ravel()
            g = -a@(x-target); value = 1.-.5*(x-target)@a@(x-target)
            rho = lib.optimizer_penalty(o)
            if rho > 0:   # the penalty inside the SCF: response (1 + rho chi)^-1 with chi = a
                g = np.linalg.solve(np.eye(6)+rho*a, g)
            m = np.array([1.5*f[s][0]+g[2*s]*f[s][1]+g[2*s+1]*f[s][2] for s in range(3)]).ravel()
            coarse = lib.optimizer_coarse(o); tiers.append(coarse)
            if perturb: value, m = perturb(evaluation, value, m, coarse)
            done = ct.c_int(-1); status = lib.optimizer_observe(o, value, ptr(m), ct.byref(done))
            if status or done.value == 1: break
        v = np.zeros(5); lib.optimizer_measure(o, ptr(v))
        return dict(status=status, converged=done.value == 1, evaluations=int(v[3]), lines=int(v[4]),
                    coarse_evaluations=int(sum(tiers)), fine_evaluations=int(len(tiers)-sum(tiers)),
                    max_multiplier_error=float(np.max(np.abs(x-target))), max_torque=float(v[1]))
    finally:
        lib.optimizer_free(o)


# ---------------------------------------------------------------- toy SCF model
class Model:
    """Equilateral triangle of p-orbital atoms: crystal field along z,
    Slater-Koster hopping and overlap, on-site xi L.S, and the mean-field
    interaction sum_a Uq/2 (q_a-q0)^2 - U/4 |m_a|^2 on Mulliken charges and
    spin moments, with Fermi-Dirac occupations at fixed charge. Without self
    consistency a fixed exchange field -Delta sigma_z replaces the interaction:
    the Mermin functional is then strictly convex in rho, every trial field has
    one solution and the dual is exactly concave."""
    def __init__(self, self_consistent=True):
        self.atoms = 3; self.n = 9; self.electrons = 12; self.kT = .004; self.U = .3; self.Uq = .6; self.xi = .03
        self.self_consistent = self_consistent; self.exchange = .15
        positions = np.array([[0., 0., 0.], [2.2, 0., 0.], [1.1, 1.1*np.sqrt(3.), 0.]])
        self.atom = np.repeat(np.arange(3), 3)
        h = np.zeros((9, 9)); s = np.eye(9)
        for i in range(3):
            h[3*i+2, 3*i+2] = .06                                     # crystal field on pz
            for j in range(3):
                if i == j: continue
                u = positions[j]-positions[i]; distance = np.linalg.norm(u); u /= distance
                sigma, pi = -.08*np.exp(-(distance-2.)), .025*np.exp(-(distance-2.))
                block = (sigma-pi)*np.outer(u, u)+pi*np.eye(3)
                h[3*i:3*i+3, 3*j:3*j+3] = block; s[3*i:3*i+3, 3*j:3*j+3] = -.4*block
        l = np.zeros((3, 3, 3), complex)                              # (L_k)_ij = -i eps_kij
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    if len({i, j, k}) == 3: l[k, i, j] = -1j*np.sign((i-k)*(j-k)*(j-i))
        self.h0 = np.kron(h, np.eye(2)).astype(complex)
        for i in range(3):
            self.h0[6*i:6*i+6, 6*i:6*i+6] += self.xi*sum(np.kron(l[k], PAULI[k]/2) for k in range(3))
        self.s = s; self.s2 = np.kron(s, np.eye(2))
        self.w = [.5*(np.diag((self.atom == a).astype(float))@s+s@np.diag((self.atom == a).astype(float))) for a in range(3)]
        self.sparse = Sparse(9, [[(j, 0) for j in range(9)] for _ in range(9)])
        self.overlap = np.array([s[i, j] for i, j, _ in self.sparse.entries])

    def operator_blocks(self, dense):
        """H[(col,s),(io,s')] for each sparse entry of a dense (2n,2n) operator."""
        return np.array([dense[2*j:2*j+2, 2*i:2*i+2] for i, j, _ in self.sparse.entries])

    def store(self, dense):
        o = self.operator_blocks(dense); a, c, f, b = o[:, 0, 0], o[:, 0, 1], o[:, 1, 0], o[:, 1, 1]
        return np.ascontiguousarray(np.array([a.real, b.real, c.real, -c.imag, a.imag, b.imag, f.real, f.imag]))

    def dense(self, stored):
        o = unpack_hamiltonian(stored, 8); h = np.zeros((18, 18), complex)
        for k, (i, j, _) in enumerate(self.sparse.entries): h[2*j:2*j+2, 2*i:2*i+2] += o[k]
        return h

    def populations(self, rho):
        """Mulliken (q, mx, my, mz) per atom."""
        return np.array([[np.real(np.trace(rho@np.kron(self.w[a], sigma))) for sigma in [np.eye(2)]+PAULI]
                         for a in range(3)])

    def solve(self, h):
        e, c = eigh(h, self.s2)
        def occupation(mu): return 1./(1.+np.exp(np.clip((e-mu)/self.kT, -700, 700)))
        lo, hi = e[0]-1., e[-1]+1.
        for _ in range(200):
            mu = .5*(lo+hi)
            if occupation(mu).sum() > self.electrons: hi = mu
            else: lo = mu
        f = occupation(mu); g = f[(f > 0) & (f < 1)]
        return (c*f)@c.conj().T, -np.sum(g*np.log(g)+(1-g)*np.log(1-g))

    def interaction(self, p):
        q0 = self.electrons/3
        return sum(np.kron(self.w[a], self.Uq*(p[a, 0]-q0)*np.eye(2)-.5*self.U*sum(p[a, k+1]*PAULI[k] for k in range(3)))
                   for a in range(3))

    def scf(self, add_constraint, populations, tolerance=1.e-12):
        """Self-consistent Mulliken (q, m) per atom, continued from the previous
        trial. Plain mixing stalls along the soft moment-rotation mode, so a few
        damped steps are followed by damped Newton with a difference Jacobian."""
        def evaluate(x):
            stored = self.store(self.h0+self.interaction(x.reshape(3, 4))); add_constraint(stored)
            rho, entropy = self.solve(self.dense(stored))
            return rho, entropy, self.populations(rho).ravel()-x
        if not self.self_consistent:
            fixed = self.h0-self.exchange*sum(np.kron(self.w[a], PAULI[2]) for a in range(3))
            stored = self.store(fixed); add_constraint(stored)
            rho, entropy = self.solve(self.dense(stored))
            return rho, self.populations(rho), np.real(np.trace(rho@fixed))-self.kT*entropy
        x = populations.ravel().copy()
        for _ in range(20): x = x+.3*evaluate(x)[2]
        for _ in range(400):
            rho, entropy, r = evaluate(x)
            if np.max(np.abs(r)) < tolerance: break
            h = 1.e-7; jacobian = np.array([(evaluate(x+h*e)[2]-r)/h for e in np.eye(x.size)]).T
            step = -np.linalg.lstsq(jacobian, r, rcond=1.e-12)[0]
            step *= min(1., 1.5/np.max(np.abs(step)))
            for _ in range(40):
                if np.linalg.norm(evaluate(x+step)[2]) < np.linalg.norm(r): break
                step *= .5
            x = x+step
        else: raise RuntimeError('toy SCF did not converge')
        p = (x+r).reshape(3, 4); q0 = self.electrons/3
        free = np.real(np.trace(rho@self.h0))+np.sum(.5*self.Uq*(p[:, 0]-q0)**2-.25*self.U*np.sum(p[:, 1:]**2, axis=1))-self.kT*entropy
        return rho, p, free

    def density(self, rho):
        blocks = np.array([rho[2*i:2*i+2, 2*j:2*j+2] for i, j, _ in self.sparse.entries])
        return np.ascontiguousarray(pack_density(blocks, 8))


def initial(moments, electrons):
    return np.hstack([np.full((3, 1), electrons/3), moments])


SITES = [(np.deg2rad(35.), np.deg2rad(20.), [0]), (np.deg2rad(80.), np.deg2rad(110.), [1, 2])]


def density_scf(model, add, rho, mixing=.2, history=8, tolerance=1.e-11, maximum=3000):
    """SIESTA-style SCF on the density: H[rho] (+ the constraint, which may
    depend on rho) -> rho_out, Anderson-mixed. Returns rho, iterations."""
    xs, fs = [], []
    for iteration in range(1, maximum+1):
        if model.self_consistent: h = model.h0+model.interaction(model.populations(rho))
        else: h = model.h0-model.exchange*sum(np.kron(model.w[a], PAULI[2]) for a in range(3))
        stored = model.store(h); add(stored, rho)
        out, entropy = model.solve(model.dense(stored)); r = out-rho
        if np.max(np.abs(r)) < tolerance: return out, entropy, iteration
        xs.append(rho.ravel().copy()); fs.append(r.ravel().copy()); xs, fs = xs[-history:], fs[-history:]
        x = rho.ravel()+mixing*r.ravel()
        if len(fs) > 1:
            df = np.diff(fs, axis=0); dx = np.diff(xs, axis=0)
            gamma = np.linalg.lstsq(df.T, r.ravel(), rcond=1.e-12)[0]
            x = x-(dx.T+mixing*df.T)@gamma
        rho = x.reshape(rho.shape); rho = .5*(rho+rho.conj().T)
    return None, None, maximum


def free_energy(model, rho, entropy):
    if model.self_consistent:
        p = model.populations(rho); q0 = model.electrons/3
        return np.real(np.trace(rho@model.h0))+np.sum(.5*model.Uq*(p[:, 0]-q0)**2-.25*model.U*np.sum(p[:, 1:]**2, axis=1))-model.kT*entropy
    fixed = model.h0-model.exchange*sum(np.kron(model.w[a], PAULI[2]) for a in range(3))
    return np.real(np.trace(rho@fixed))-model.kT*entropy


def worker(library, mode, sites, criterion, tolerance, coarse=None, penalty=0., self_consistent=False, maximum_scf=3000):
    """One production host-ABI search in a fresh process (the owner is a singleton).
    The constraint is applied by host_add from the density that built each H."""
    lib = load(library); model = Model(self_consistent=self_consistent); sp = model.sparse
    orbital_site = np.full(9, -1, I32)
    for t, (_, _, atoms) in enumerate(sites):
        for a in atoms: orbital_site[model.atom == a] = t
    angles = np.array([[th, ph] for th, ph, _ in sites]).ravel()
    direction, line = mode
    settings = dict(SETTINGS, initial_field=1.e-4, energy_tolerance=1.e-10, angle_tolerance=np.deg2rad(tolerance),
                    torque_tolerance=tolerance, maximum_field=2., penalty=penalty)
    if coarse: settings.update(coarse_angle_tolerance=np.deg2rad(coarse['angle_deg']), coarse_torque_tolerance=coarse['torque'])
    noise = np.random.default_rng(5); tiers = []; scf_iterations = []
    assert lib.remag_spin_constraint_configure(len(sites), ptr(angles), 9, ptr(orbital_site),
        ptr(np.array([direction, line, criterion, 400, 0], I32)), ptr(values(settings))) == 0
    moments = np.zeros((3, 3))
    for t, (th, ph, atoms) in enumerate(sites):      # DM.InitSpin along the targets
        for a in atoms: moments[a] = 2.*np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    # Initial density: one SCF-free solve with exchange fields along the targets.
    guide = model.h0+sum(np.kron(model.w[a], -.25*sum(moments[a, k]*PAULI[k] for k in range(3))) for a in range(3))
    rho, _ = model.solve(guide); history = []; status = 0; done = ct.c_int(-1); step = 0
    for step in range(1, 401):
        assert lib.remag_spin_constraint_begin(step) == 0; rho_step = lib.remag_spin_constraint_penalty()
        def add(stored, density):
            d = model.density(density)
            assert lib.remag_spin_constraint_host_add(9, sp.capacity, 8, ptr(sp.num), ptr(sp.ptr), ptr(sp.cols),
                                                      ptr(sp.mapping), ptr(model.overlap), ptr(d), ptr(stored)) == 0
        out, entropy, iterations = density_scf(model, add, rho, maximum=maximum_scf); scf_iterations.append(iterations)
        if out is None:
            return dict(status=-1, converged=False, steps=step, scf_failed=True, scf_iterations=scf_iterations)
        rho = out; free = free_energy(model, rho, entropy)
        d = model.density(rho); tiers.append(lib.remag_spin_constraint_coarse())
        if tiers[-1]:   # stand-in for a cheaper SCF: a relative density error
            d = np.ascontiguousarray(d*(1.+coarse['noise']*noise.normal(size=d.shape))); free += coarse['noise']*noise.normal()
        assert lib.remag_spin_constraint_host_moments(9, sp.capacity, 8, ptr(sp.num), ptr(sp.ptr),
                                                      ptr(model.overlap), ptr(d)) == 0
        fields = np.zeros(3*len(sites)); assert lib.remag_spin_constraint_fields(fields.size, ptr(fields)) == 0
        site_moments = np.zeros(3*len(sites)); value = ct.c_double(); done = ct.c_int(-1)
        status = lib.remag_spin_constraint_observe(free, site_moments.size, ptr(site_moments), ct.byref(value), ct.byref(done))
        moments = model.populations(rho)[:, 1:]
        expected = np.array([sum(moments[a] for a in atoms) for _, _, atoms in sites]).ravel()
        perp = 0.
        for s, (th, ph, _) in enumerate(sites):
            tv = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]); m = site_moments[3*s:3*s+3]
            perp += .5*rho_step*(m@m-(m@tv)**2)
        history.append(dict(step=step, free=free, lagrangian=value.value,
                            moment_error=float(np.max(np.abs(site_moments-expected))) if not tiers[-1] else 0.,
                            lagrangian_error=abs(value.value-(free+fields@site_moments+perp))))
        if status or done.value == 1: break
    measure = np.zeros(5); lib.remag_spin_constraint_measure(ptr(measure))
    return dict(status=status, converged=done.value == 1, steps=step, free_energy=free, fields=fields.tolist(),
                coarse_steps=int(sum(tiers)), fine_steps=int(len(tiers)-sum(tiers)), scf_iterations=scf_iterations,
                moments=site_moments.tolist(), max_angle_deg=float(np.rad2deg(measure[0])), max_torque=float(measure[1]),
                lines=int(measure[4]), max_moment_error=max(h['moment_error'] for h in history),
                max_lagrangian_error=max(h['lagrangian_error'] for h in history),
                lagrangian=[h['lagrangian'] for h in history])


def spawn(library, directory, **job):
    result = subprocess.run([sys.executable, __file__, '--worker', json.dumps(job), '--library', str(library)],
                            cwd=directory, capture_output=True, text=True, timeout=1800)
    if result.returncode: raise RuntimeError(result.stderr)
    return json.loads(result.stdout.splitlines()[-1])


def envelope(model, rng):
    """Theory check on the toy: L is concave in the multipliers and dL/dx = M_perp."""
    sites = SITES; base = model.scf(lambda h: None, initial(np.tile([0., 0., 2.], (3, 1)), model.electrons))
    def lagrangian(x):
        def add(stored):
            field = np.zeros((9, 3))
            for t, (th, ph, atoms) in enumerate(sites):
                f = np.array([[np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)], [-np.sin(ph), np.cos(ph), 0.]])
                for a in atoms: field[model.atom == a] = x[2*t:2*t+2]@f
            stored += model.store(sum(np.kron(model.w[a], sum(field[3*a, k]*PAULI[k] for k in range(3)))
                                      for a in range(3)))
        rho, p, free = model.scf(add, base[1]); m = p[:, 1:]
        total = 0.; gradient = []
        for t, (th, ph, atoms) in enumerate(sites):
            f = np.array([[np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)], [-np.sin(ph), np.cos(ph), 0.]])
            mt = sum(m[a] for a in atoms); total += x[2*t:2*t+2]@(f@mt); gradient += list(f@mt)
        return free+total, np.array(gradient)
    x = rng.normal(size=4)*.03; step = 1.e-4
    value, gradient = lagrangian(x)
    def central(h, e): return (lagrangian(x+h*e)[0]-lagrangian(x-h*e)[0])/(2*h)
    numeric = np.array([(4*central(step/2, e)-central(step, e))/3 for e in np.eye(4)])   # Richardson
    hessian = np.array([(lagrangian(x+step*e)[1]-lagrangian(x-step*e)[1])/(2*step) for e in np.eye(4)])
    eigenvalues = np.linalg.eigvalsh(.5*(hessian+hessian.T))
    return dict(gradient_error=float(np.max(np.abs(numeric-gradient))), gradient_scale=float(np.max(np.abs(gradient))),
                hessian_eigenvalues=eigenvalues.tolist(), hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
                passed=bool(np.max(np.abs(numeric-gradient)) <= 1.e-6*max(1., np.max(np.abs(gradient))) and eigenvalues.max() < 0))


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = dict(gpu_validation=None, integrated_into_siesta=False, script_sha256=sha(__file__),
                  sources={name: sha(NATIVE/name) for name in SOURCES})
    with tempfile.TemporaryDirectory(prefix='remag-spin-constraint-') as tmp:
        library = build(tmp); lib = load(library); rng = np.random.default_rng(20261013)
        report['exact'] = [exact_fixture(lib, rng, spins, images, f'{"soc" if spins == 8 else "nc"}_{"images" if images else "folded"}_{i}')
                           for spins in (8, 4) for images in (False, True) for i in range(3)]
        report['exact'].append(exact_penalty(lib, rng))
        modes = {(0, 0): 'steepest_golden', (0, 1): 'steepest_secant', (1, 0): 'bfgs_golden', (1, 1): 'bfgs_secant',
                 (2, 1): 'multipliers'}
        quadratic = []
        for (direction, line), name in modes.items():
            r = quadratic_dual(lib, np.random.default_rng(7), direction, line, criteria=2,
                               override=dict(penalty=.4) if direction == 2 else None)
            r.update(mode=name, passed=r['status'] == 0 and r['converged'] and r['max_multiplier_error'] < 1.e-5)
            quadratic.append(r)
        # Coarse tier: noisy observations (a cheaper SCF) until 1e-2 torque,
        # then exact ones; the answer must not depend on the coarse noise.
        noise = np.random.default_rng(11)
        for (direction, line), name in modes.items():
            if line == 0: continue                      # rejected: see coarse_with_golden
            r = quadratic_dual(lib, np.random.default_rng(7), direction, line, criteria=2,
                               override=dict(coarse_angle_tolerance=np.pi, coarse_torque_tolerance=1.e-2,
                                             penalty=.4 if direction == 2 else 0.),
                               perturb=lambda i, v, m, c: (v+(1.e-6*noise.normal() if c else 0.), m+(1.e-4*noise.normal(size=m.size) if c else 0.)))
            r.update(mode=name+'_coarse', passed=r['status'] == 0 and r['converged'] and r['max_multiplier_error'] < 1.e-5
                     and r['coarse_evaluations'] > 0 and r['fine_evaluations'] > 0)
            quadratic.append(r)
        failures = []
        frames_targets = np.array([f[0] for f in frames(lib, [(.3, .2), (1.7, 2.9), (2.6, -1.)])])
        def expect(name, code, **kw):
            r = quadratic_dual(lib, np.random.default_rng(7), 1, 1, criteria=2, **kw)
            failures.append(dict(case=name, expected=code, status=r['status'], passed=r['status'] == code))
        expect('invalid_settings', 1, override=dict(slope_ratio=1.5))
        failures.append(dict(case='multipliers_without_penalty', expected=1,
                             status=quadratic_dual(lib, np.random.default_rng(7), 2, 1, criteria=2)['status']))
        failures[-1]['passed'] = failures[-1]['status'] == 1
        expect('coarse_with_golden', 1, override=dict(coarse_angle_tolerance=np.pi, coarse_torque_tolerance=1.e-2))
        failures[-1]['status'] = quadratic_dual(lib, np.random.default_rng(7), 0, 0, criteria=2,
            override=dict(coarse_angle_tolerance=np.pi, coarse_torque_tolerance=1.e-2))['status']
        failures[-1]['passed'] = failures[-1]['status'] == 1
        expect('coarse_tighter_than_final', 1, override=dict(coarse_angle_tolerance=1.e-9, coarse_torque_tolerance=1.e-9))
        expect('nonfinite_energy', 2, perturb=lambda i, v, m, c: (np.nan if i == 3 else v, m))
        # Intermediate overshoots are valid trials; a search ending there is not.
        expect('collapsed_moment', 3, perturb=lambda i, v, m, c: (v, m*1.e-6))
        expect('antiparallel_moment', 4, perturb=lambda i, v, m, c: (v, m-3.*np.repeat(frames_targets, 1, axis=0).ravel()))
        overshoot = quadratic_dual(lib, np.random.default_rng(7), 1, 1, criteria=2,
                                   perturb=lambda i, v, m, c: (v, m-3.*frames_targets.ravel() if i == 2 else m))
        failures.append(dict(case='intermediate_antiparallel_trial', expected=0, status=overshoot['status'],
                             passed=overshoot['status'] == 0 and overshoot['converged']))
        expect('field_limit', 5, override=dict(maximum_field=.05, initial_field=.04))
        expect('evaluation_limit', 6, maximum=4)
        report['analytic_dual'] = quadratic; report['failure_codes'] = failures
        model = Model(); report['theory'] = envelope(model, np.random.default_rng(3))
        cases = dict(noncollinear=[[a, b, c] for a, b, c in SITES],
                     cluster=[[np.deg2rad(60.), np.deg2rad(30.), [0, 1, 2]]])
        searches = []
        for case, sites in cases.items():
            group = []
            for (direction, line), name in modes.items():
                r = spawn(library, tmp, mode=[direction, line], sites=sites, criterion=3, tolerance=1.e-3,
                          penalty=.02 if direction == 2 else 0.)
                r.update(case=case, mode=name); group.append(r)
            # L is second order in the multiplier error; F is first order in M_perp.
            finals = [r['lagrangian'][-1] for r in group]; fields = np.array([r['fields'] for r in group])
            for r in group:
                r['passed'] = bool(r['status'] == 0 and r['converged'] and r['max_angle_deg'] <= 1.e-3 and
                                   r['max_moment_error'] < 1.e-12 and r['max_lagrangian_error'] < 1.e-12 and
                                   abs(r['lagrangian'][-1]-np.mean(finals)) < 1.e-9 and
                                   np.max(np.abs(np.array(r['fields'])-fields.mean(axis=0))) < 1.e-4)
            searches += group
        for r in [r for r in searches if r['mode'] in ('bfgs_secant', 'steepest_secant')]:
            sites = cases[r['case']]; plain = r
            c = spawn(library, tmp, mode=[1, 1] if r['mode'] == 'bfgs_secant' else [0, 1], sites=sites, criterion=3,
                      tolerance=1.e-3, coarse=dict(angle_deg=1., torque=3.e-2, noise=1.e-4))
            c.update(case=r['case'], mode=r['mode']+'_coarse',
                     passed=bool(c['status'] == 0 and c['converged'] and c['coarse_steps'] > 0 and c['fine_steps'] > 0 and
                                 c['max_angle_deg'] <= 1.e-3 and abs(c['lagrangian'][-1]-plain['lagrangian'][-1]) < 1.e-7))
            searches.append(c)
        # The failure seen in SIESTA: a soft coherent rotation (the anisotropy case)
        # on the self-consistent model. The paper's zero-field first trial must
        # relax the moment through the soft mode; the penalty keeps it stiff.
        soft = [[np.deg2rad(60.), np.deg2rad(30.), [0, 1, 2]]]
        paper = spawn(library, tmp, mode=[0, 0], sites=soft, criterion=3, tolerance=1.e-3, self_consistent=True, maximum_scf=1500)
        fixed = spawn(library, tmp, mode=[2, 1], sites=soft, criterion=3, tolerance=1.e-3, penalty=.02,
                      self_consistent=True, maximum_scf=1500)
        report['soft_mode'] = dict(paper_first_trial_scf_iterations=paper['scf_iterations'][0],
            paper_converged=paper['converged'], multipliers=dict((k, fixed[k]) for k in
            ('status', 'converged', 'steps', 'scf_iterations', 'max_angle_deg', 'fields', 'lagrangian')),
            passed=bool(fixed['converged'] and max(fixed['scf_iterations']) < 1500 and fixed['max_angle_deg'] <= 1.e-3))
        report['host_abi_searches'] = searches
        # Energy-torque identity: dF*/dtheta_0 = -(b_0.e_theta)(M_0.t_0) for the
        # first site; the cluster case is the coherent rotation of an MAE sweep.
        delta = np.deg2rad(.25); torque = []
        for case, sites in cases.items():
            th, ph, atoms = sites[0]; runs = {}
            for shift in (-1, 0, 1):
                shifted = [[th+shift*delta, ph, atoms]]+sites[1:]
                runs[shift] = spawn(library, tmp, mode=[1, 1], sites=shifted, criterion=2, tolerance=1.e-10)
            if case == 'cluster':      # the same identity under the default method, self-consistently
                for shift in (-1, 0, 1):
                    shifted = [[th+shift*delta, ph, atoms]]
                    runs[('scf', shift)] = spawn(library, tmp, mode=[2, 1], sites=shifted, criterion=2, tolerance=1.e-9,
                                                 penalty=.02, self_consistent=True)
            frame = frames(lib, [(th, ph)])[0]
            b0 = np.array(runs[0]['fields'][:3]); m0 = np.array(runs[0]['moments'][:3])
            predicted = -(b0@frame[1])*(m0@frame[0]); numeric = (runs[1]['free_energy']-runs[-1]['free_energy'])/(2*delta)
            torque.append(dict(case=case, predicted=predicted, finite_difference=numeric, delta_deg=.25,
                               relative_error=abs(predicted-numeric)/abs(numeric),
                               passed=bool(all(runs[s]['converged'] for s in (-1, 0, 1)) and
                                           abs(predicted-numeric) <= 1.e-4*abs(numeric))))
            if ('scf', 0) in runs:
                b0 = np.array(runs[('scf', 0)]['fields'][:3]); m0 = np.array(runs[('scf', 0)]['moments'][:3])
                predicted = -(b0@frame[1])*(m0@frame[0])
                numeric = (runs[('scf', 1)]['free_energy']-runs[('scf', -1)]['free_energy'])/(2*delta)
                torque.append(dict(case='cluster_self_consistent_multipliers', predicted=predicted, finite_difference=numeric,
                                   delta_deg=.25, relative_error=abs(predicted-numeric)/abs(numeric),
                                   passed=bool(all(runs[('scf', s)]['converged'] for s in (-1, 0, 1)) and
                                               abs(predicted-numeric) <= 1.e-3*abs(numeric))))
        report['energy_torque'] = torque
    groups = [report['exact'], report['analytic_dual'], report['failure_codes'], report['host_abi_searches']]
    report['passed'] = bool(all(r['passed'] for g in groups+[report['energy_torque']] for r in g) and
                            report['theory']['passed'] and report['soft_mode']['passed'])
    (output/'check.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker'); parser.add_argument('--library', type=Path)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker)
        print(json.dumps(worker(args.library, job['mode'], job['sites'], job['criterion'], job['tolerance'], job.get('coarse'),
                                job.get('penalty', 0.), job.get('self_consistent', False), job.get('maximum_scf', 3000))))
        sys.exit(0)
    if not args.output: parser.error('--output is required')
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ('passed',)}))
    sys.exit(0 if result['passed'] else 1)
