#!/usr/bin/env python3
"""Check REMAG's threaded CPU k loop (native/host_kloop.cpp) against diag3k on random problems.

Each problem is a spin-orbit tight-binding model in SIESTA's sparse format:
nuo orbitals in a unit cell, supercell images, random Hermitian 2x2 spinor
blocks packed into the eight H components of diag3k.F, and a nonorthogonal
overlap. The reference is a line-by-line numpy port of diag3k: per-element
Bloch phases, a generalized eigensolver, and rank-one density updates. The C++
loop must give the same eigenvalues and density and energy-density matrices for
several thread counts, with eigenvectors kept or recomputed, with the phase
factorization disabled, and with each of the three eigensolvers.
"""
import argparse
import ctypes as ct
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'native' if (ROOT/'native').is_dir() else Path('/opt/remag/native')
D = ct.POINTER(ct.c_double)
I = ct.POINTER(ct.c_int)


def build(directory):
    out = directory/'libhostk.so'
    flags = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', '-pthread', f'-I{NATIVE}',
             str(NATIVE/'host_kloop.cpp'), '-o', str(out)]
    if platform.system() == 'Darwin':
        flags += ['-framework', 'Accelerate', '-Wl,-U,_openblas_get_num_threads', '-Wl,-U,_openblas_set_num_threads']
    else:
        flags += ['-llapack', '-lblas']
    subprocess.run(flags, check=True)
    lib = ct.CDLL(str(out))
    for name in ('remag_hostk_setup', 'remag_hostk_spectrum', 'remag_hostk_density'): getattr(lib, name).restype = ct.c_int
    return lib


def p(a, kind=D): return a.ctypes.data_as(kind)


def model(rng, nuo, cells=1, cutoff=1.9):
    """Sparse SOC model. Returns a dict of SIESTA-format arrays (Fortran order) and the geometry."""
    cell = np.eye(3)*3.0 + rng.normal(scale=.2, size=(3, 3))
    xorb = rng.uniform(0, 1, size=(nuo, 3))@cell
    shifts = [np.array(r) for r in itertools.product(range(-cells, cells+1), repeat=3)]
    images = {tuple(r): n for n, r in enumerate(shifts)}
    blocks, overlap = {}, {}
    for i in range(nuo):
        for j in range(nuo):
            for r in shifts:
                key = (j, tuple(r), i); mirror = (i, tuple(-r), j)
                if key in blocks: continue
                x = xorb[j] + r@cell - xorb[i]
                if np.linalg.norm(x) > cutoff*3 and not (i == j and not r.any()): continue
                decay = np.exp(-np.linalg.norm(x)/3)
                b = (rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))*decay
                o = .05*rng.normal()*decay
                if key == mirror:
                    b = (b + b.conj().T)/2 + 3*np.eye(2)*(i == j and not r.any()); o = 1.
                blocks[key] = b; overlap[key] = o
                blocks[mirror] = b.conj().T; overlap[mirror] = o
    rows = [[] for _ in range(nuo)]
    for (j, r, i), b in blocks.items(): rows[i].append((j, r, b, overlap[(j, r, i)]))
    numh = np.array([len(r) for r in rows], dtype=np.int32)
    listhptr = np.concatenate([[0], np.cumsum(numh)[:-1]]).astype(np.int32)
    maxnh = int(numh.sum())
    listh = np.zeros(maxnh, np.int32); H = np.zeros((maxnh, 8), order='F'); S = np.zeros(maxnh)
    xij = np.zeros((maxnh, 3))
    no = nuo*len(shifts)
    for i, row in enumerate(rows):
        for n, (j, r, b, o) in enumerate(row):
            ind = listhptr[i] + n
            jo = j + nuo*images[r]
            listh[ind] = jo + 1
            H[ind] = [b[0, 0].real, b[1, 1].real, b[0, 1].real, -b[0, 1].imag, b[0, 0].imag, b[1, 1].imag,
                      b[1, 0].real, b[1, 0].imag]
            S[ind] = o
            xij[ind] = xorb[j] + np.array(r)@cell - xorb[i]
    indxuo = np.array([jo % nuo + 1 for jo in range(no)], dtype=np.int32)
    return dict(nuo=nuo, no=no, maxnh=maxnh, numh=numh, listhptr=listhptr, listh=listh, H=H, S=S, xij=xij,
                indxuo=indxuo, xorb=xorb, cell=cell)


def reference(m, kpoints, qfun, occtol=1e-12):
    """diag3k.F in numpy: eigenvalues, then Dnew(maxnd,8), Enew(maxnd,4) for occupations qfun(eo)."""
    nuo, n = m['nuo'], 2*m['nuo']
    eo, vecs = [], []
    for k in kpoints:
        h = np.zeros((n, n), complex); s = np.zeros((n, n), complex)
        for iuo in range(nuo):
            for j in range(m['numh'][iuo]):
                ind = m['listhptr'][iuo] + j
                juo = m['indxuo'][m['listh'][ind]-1]-1
                ph = np.exp(-1j*(k@m['xij'][ind])); Hc = m['H'][ind]
                r0, r1, c0, c1 = 2*juo, 2*juo+1, 2*iuo, 2*iuo+1
                s[r0, c0] += m['S'][ind]*ph; s[r1, c1] += m['S'][ind]*ph
                h[r0, c0] += (Hc[0]+1j*Hc[4])*ph; h[r1, c1] += (Hc[1]+1j*Hc[5])*ph
                h[r0, c1] += (Hc[2]-1j*Hc[3])*ph; h[r1, c0] += (Hc[6]+1j*Hc[7])*ph
        w, v = eigh(h, s)
        eo.append(w); vecs.append(v)
    eo = np.array(eo); qo = qfun(eo)
    maxnd = m['maxnh']
    Dn = np.zeros((maxnd, 8)); En = np.zeros((maxnd, 4))
    for ik, k in enumerate(kpoints):
        Dk = np.zeros((2, nuo, 2, nuo), complex); Ek = np.zeros_like(Dk)
        for ie in range(n):
            q = qo[ik, ie]
            if abs(q) <= occtol: continue
            c = vecs[ik][:, ie].reshape(nuo, 2).T            # c[s, orbital]
            ee = q*eo[ik, ie]
            for a in range(2):
                for b in range(2):
                    cicj = np.outer(c[a], c[b].conj())       # [iuo, juo]
                    Dk[a, :, b, :] += q*cicj.T; Ek[a, :, b, :] += ee*cicj.T
        for iuo in range(nuo):
            for j in range(m['numh'][iuo]):
                ind = m['listhptr'][iuo] + j
                juo = m['indxuo'][m['listh'][ind]-1]-1
                ph = np.exp(-1j*(k@m['xij'][ind]))
                d11, d22, d12, d21 = (Dk[0, juo, 0, iuo]*ph, Dk[1, juo, 1, iuo]*ph, Dk[0, juo, 1, iuo]*ph,
                                      Dk[1, juo, 0, iuo]*ph)
                Dn[ind] += [d11.real, d22.real, d12.real, -d12.imag, d11.imag, d22.imag, d21.real, d21.imag]
                e11, e22, e12 = Ek[0, juo, 0, iuo]*ph, Ek[1, juo, 1, iuo]*ph, Ek[0, juo, 1, iuo]*ph
                En[ind] += [e11.real, e22.real, e12.real, -e12.imag]
    return eo, qo, Dn, En


def run(lib, m, kpoints, qo_ref, threads, budget_mb, xorb=None, solver=1):
    nk, n = len(kpoints), 2*m['nuo']
    xo = np.ascontiguousarray(m['xorb'] if xorb is None else xorb)
    xij = np.ascontiguousarray(m['xij']); kp = np.ascontiguousarray(kpoints)
    status = lib.remag_hostk_setup(m['nuo'], m['no'], m['maxnh'], p(m['numh'], I), p(m['listhptr'], I),
                                   p(m['listh'], I), p(m['indxuo'], I), p(xij), p(xo), nk, p(kp), threads,
                                   ct.c_double(budget_mb), solver)
    assert status == 0, status
    H = np.asfortranarray(m['H']); S = m['S']
    eo = np.zeros((nk, n))
    status = lib.remag_hostk_spectrum(p(H), p(S), n, n, p(eo))
    assert status == 0, status
    qo = np.ascontiguousarray(qo_ref)
    Dn = np.zeros((8, m['maxnh'])); En = np.zeros((4, m['maxnh']))
    status = lib.remag_hostk_density(p(H), p(S), n, n, p(eo), p(qo), ct.c_double(1e-12), m['maxnh'], p(m['numh'], I),
                                     p(m['listhptr'], I), p(m['listh'], I), p(Dn), p(En))
    assert status == 0, status
    return eo, Dn.T, En.T


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    rng = np.random.default_rng(20261015)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(Path(tmp))
        for case, (nuo, nk) in enumerate([(3, 5), (5, 7), (8, 4)]):
            m = model(rng, nuo)
            kpoints = rng.uniform(-.5, .5, size=(nk, 3))@(2*np.pi*np.linalg.inv(m['cell']).T)
            # Smooth occupations, some negative (as tetrahedron weights can be), k weight 1/nk.
            mu = 0.
            qfun = lambda eo: (1/(1+np.exp((eo-mu)/.5)) - .02*np.sin(eo))/len(kpoints)
            eo_r, qo_r, Dr, Er = reference(m, kpoints, qfun)
            for threads, budget, factor, solver in ((1, 1e4, True, 1), (3, 1e4, True, 1), (4, 0., True, 1),
                                                    (2, 1e4, False, 1), (3, 1e4, True, 2), (2, 0., True, 2),
                                                    (3, 1e4, True, 3), (2, 0., True, 3)):
                # Inconsistent positions: the factorization check must switch it off.
                xo = None if factor else m['xorb'] + rng.normal(scale=.3, size=m['xorb'].shape)
                eo, Dn, En = run(lib, m, kpoints, qo_r, threads, budget, xo, solver)
                results.append(dict(case=case, nuo=nuo, nk=nk, threads=threads, kept=budget > 0, factored=factor,
                                    solver={1: 'divide', 2: 'mrrr', 3: 'qr'}[solver],
                                    eigenvalue_error=float(np.abs(eo-eo_r).max()),
                                    density_error=float(np.abs(Dn-Dr).max()), energy_density_error=float(np.abs(En-Er).max()),
                                    density_scale=float(np.abs(Dr).max())))
    passed = all(r['eigenvalue_error'] < 1e-11 and r['density_error'] < 1e-11*max(1., r['density_scale'])
                 and r['energy_density_error'] < 1e-10 for r in results)
    report = dict(cases=results, passed=bool(passed))
    text = json.dumps(report, indent=1)
    if a.output: a.output.mkdir(parents=True, exist_ok=True); (a.output/'check.json').write_text(text+'\n')
    print(text)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
