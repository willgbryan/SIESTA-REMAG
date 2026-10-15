#!/usr/bin/env python3
"""Check REMAG's tetrahedron occupations (native/tetrahedra.cpp) on the CPU.

Single-tetrahedron weights are compared with exact integrals over the occupied
part of the tetrahedron, found by clipping it with the plane E = ef. Grid
weights are compared with an independent numpy implementation, written from
Quantum ESPRESSO's ktetra.f90 (opt_tetra_init, opt_tetra_weights), on random
band structures. A tight-binding band shows each method's convergence with
the k mesh. Retained pw.x runs (benchmarks/tetrahedra-qe) give eigenvalues on
full grids together with QE's own Fermi energies and occupations; the same
eigenvalues must reproduce them here.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'native' if (ROOT/'native').is_dir() else Path('/opt/remag/native')
QE_RECORDS = ROOT/'benchmarks/tetrahedra-qe/reference'
WRAPPER = r'''#include "tetrahedron_math.hpp"
#include "tetrahedra.hpp"
extern "C" void corner(const double* e,double ef,int bloechl,double* w) {remag::tetra_weights(e,ef,bloechl,w);}
extern "C" double count(const double* e,double ef) {return remag::tetra_count(e,ef);}
// QE-compatible settings (forced diagonal, no degeneracy averaging, given ef) for the checks only.
extern "C" int grid_occupations(int method,const int* grid,const double* displacement,const double* cell,int nk,
    const double* k,const double* w,int trs,int diagonal,int nspinor,int maxspn,int ne,const double* e,double charge,
    double* occupations,double* ef,int average,int given) {
 remag::TetraGrid g;int status=remag::tetra_grid(method,grid,displacement,cell,nk,k,w,trs,g,diagonal);
 if(status) return status;
 return remag::tetra_occupations(g,method,nspinor,maxspn,ne,ne,e,charge,occupations,ef,average!=0,given!=0);
}
'''
P = ct.POINTER(ct.c_double)


def build(directory):
    (directory/'wrapper.cpp').write_text(WRAPPER)
    library = directory/'libtetra.so'
    subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                    '-pthread', f'-I{NATIVE}', str(directory/'wrapper.cpp'), str(NATIVE/'tetrahedra.cpp'), '-o', str(library)],
                   check=True)
    lib = ct.CDLL(str(library))
    lib.corner.argtypes = [P, ct.c_double, ct.c_int, P]
    lib.count.restype = ct.c_double; lib.count.argtypes = [P, ct.c_double]
    lib.remag_tetra_occupations.restype = ct.c_int
    return lib


def d(a): return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(P)


def qe_mode(lib, method, grid, cell, k, w, energies, charge, nspinor, diagonal=-1, average=True, fermi=None):
    """Occupations with the check-only settings; fermi given skips the Fermi search."""
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    nk, maxspn, ne = energies.shape
    out = np.zeros_like(energies); ef = ct.c_double(0. if fermi is None else fermi)
    keep = [np.ascontiguousarray(x, dtype=np.float64) for x in (np.zeros(3), cell, k, w)]
    status = lib.grid_occupations(method, (ct.c_int*3)(*grid), d(keep[0]), d(keep[1]), nk, d(keep[2]), d(keep[3]), 0,
                                  diagonal, nspinor, maxspn, ne, d(energies), ct.c_double(charge), out.ctypes.data_as(P),
                                  ct.byref(ef), int(average), int(fermi is not None))
    return status, out, ef.value


def occupations(lib, method, grid, cell, k, w, energies, charge, nspinor=2, trs=False, displacement=(0., 0., 0.),
                threads=1):
    """energies: (nk, maxspn, ne). Returns status, occupations of the same shape, ef."""
    energies = np.ascontiguousarray(energies, dtype=np.float64)
    nk, maxspn, ne = energies.shape
    out = np.zeros_like(energies); ef = ct.c_double(); tetrahedra = ct.c_int()
    keep = [np.ascontiguousarray(x, dtype=np.float64) for x in (displacement, cell, k, w)]
    status = lib.remag_tetra_occupations(method, (ct.c_int*3)(*grid), d(keep[0]), d(keep[1]), nk, d(keep[2]), d(keep[3]),
                                         int(trs), nspinor, maxspn, ne, ne, d(energies), ct.c_double(charge),
                                         out.ctypes.data_as(P), ct.byref(ef), ct.byref(tetrahedra), threads)
    return status, out, ef.value


# Exact single-tetrahedron integrals. The occupied region {sum l_i e_i < ef}
# of the barycentric simplex is clipped exactly; each piece is a tetrahedron
# over which the linear l_i integrates to volume times its centroid value.
def clip(e, ef):
    corners = np.eye(4)
    below = [i for i in range(4) if e[i] < ef]
    if not below: return []
    if len(below) == 4: return [corners]
    above = [i for i in range(4) if i not in below]
    cut = {(i, j): corners[i]+(ef-e[i])/(e[j]-e[i])*(corners[j]-corners[i]) for i in below for j in above}
    if len(below) == 1:
        i = below[0]; return [np.array([corners[i]]+[cut[i, j] for j in above])]
    if len(below) == 3:
        j = above[0]; small = np.array([corners[j]]+[cut[i, j] for i in below])
        return [corners, small]  # whole simplex minus the unoccupied corner (sign below)
    (i, k), (j, l) = below, above
    prism = [corners[i], corners[k], cut[i, j], cut[i, l], cut[k, j], cut[k, l]]
    a, b, c1, c2, c3, c4 = prism  # triangles (a, c1, c2) and (b, c3, c4), matched in that order
    return [np.array([a, c1, c2, c4]), np.array([a, c1, c3, c4]), np.array([a, b, c3, c4])]


def exact_weights(e, ef):
    pieces = clip(e, ef)
    def integral(piece):
        x = piece[:, :3]-piece[0, :3]
        return abs(np.linalg.det(x[1:4]))*piece.mean(axis=0)
    if len(pieces) == 2 and sum(np.asarray(e) < ef) == 3:
        return integral(pieces[0])-integral(pieces[1])
    return sum((integral(p) for p in pieces), np.zeros(4))


def check_corners(lib, rng):
    worst = dict(linear=0., count=0., bloechl=0.)
    for trial in range(3000):
        e = np.sort(rng.uniform(-1., 1., 4))
        if trial % 7 == 0: e[1] = e[0]          # degenerate corners
        if trial % 11 == 0: e[3] = e[2]
        ef = rng.uniform(-1.2, 1.2)
        w, wb = np.zeros(4), np.zeros(4)
        lib.corner(d(e), ef, 0, w.ctypes.data_as(P)); lib.corner(d(e), ef, 1, wb.ctypes.data_as(P))
        exact = exact_weights(e, ef)
        worst['linear'] = max(worst['linear'], np.abs(w-exact).max())
        worst['count'] = max(worst['count'], abs(lib.count(d(e), ef)-exact.sum()))
        h = 1.e-6
        dos = (exact_weights(e, ef+h).sum()-exact_weights(e, ef-h).sum())/(2*h)
        correction = dos/40.*(e.sum()-4.*e)
        if min(abs(ef-x) for x in e) > 1.e-4:
            worst['bloechl'] = max(worst['bloechl'], np.abs(wb-w-correction).max())
    return dict(worst, passed=bool(worst['linear'] < 1.e-12 and worst['count'] < 1.e-12 and worst['bloechl'] < 1.e-7))


# Model crystal: hcp cell (Bohr) and nearest-neighbour tight-binding bands.
A, C = 4.7376, 7.6893
CELL = np.array([[A, 0., 0.], [-A/2, A*np.sqrt(3)/2, 0.], [0., 0., C]])  # rows a_j


def mesh(grid, displacement=(0., 0., 0.)):
    """Cartesian k of the full grid, first index fastest, and the fractional coordinates."""
    n1, n2, n3 = grid
    f = np.array([[(i1+displacement[0])/n1, (i2+displacement[1])/n2, (i3+displacement[2])/n3]
                  for i3 in range(n3) for i2 in range(n2) for i1 in range(n1)])
    return f@(2*np.pi*np.linalg.inv(CELL).T), f


def model_bands(k, spin_split=0.):
    x = k@CELL.T
    s = -np.cos(x[:, 0])-np.cos(x[:, 1])-np.cos(x[:, 0]+x[:, 1])-.5*np.cos(x[:, 2])
    b = np.stack([s, .6*s+2.5+.4*np.cos(x[:, 2]), 9.+.1*s], axis=-1)
    if not spin_split: return b[:, None, :]
    return np.stack([b-spin_split, b+spin_split], axis=1)


def reduce_trs(k, f, grid):
    """SIESTA-style list: one of each (k, -k) pair, weight doubled."""
    index = {}; kept, weights = [], []
    for i, x in enumerate(f):
        key = tuple(np.rint(x*grid).astype(int) % grid); minus = tuple(np.rint(-x*grid).astype(int) % grid)
        if minus in index: weights[index[minus]] += 1./len(f); continue
        index[key] = len(kept); kept.append(i); weights.append(1./len(f))
    return np.array(kept), np.array(weights)


def check_grids(lib):
    """Normalization, k-list reduction, shifted grids, spin layouts and refusals."""
    out = {}; grid = (8, 8, 5)
    k, f = mesh(grid); w = np.full(len(k), 1./len(k)); e = model_bands(k)
    for method in (1, 2, 3):
        status, occ, ef = occupations(lib, method, grid, CELL, k, w, e, 1.3)
        kept, wr = reduce_trs(k, f, np.array(grid))
        status_r, occ_r, ef_r = occupations(lib, method, grid, CELL, k[kept], wr, e[kept], 1.3, trs=True)
        folded = np.zeros_like(occ_r)
        position = {tuple(np.rint(x*grid).astype(int) % grid): j for j, x in enumerate(f[kept])}
        for i, x in enumerate(f):
            key = tuple(np.rint(x*grid).astype(int) % grid); minus = tuple(np.rint(-x*grid).astype(int) % grid)
            folded[position.get(key, position.get(minus))] += occ[i]
        out[f'method{method}'] = dict(status=status, charge_error=abs(occ.sum()-1.3), trs_status=status_r,
                                      trs_fermi_difference=abs(ef-ef_r), trs_occupation_difference=np.abs(folded-occ_r).max())
    # Half-shifted grid, two spin channels with one Fermi level, spinless factor 2.
    ks, _ = mesh(grid, (.5, .5, .5)); es = model_bands(ks, spin_split=.3)
    status, occ, ef = occupations(lib, 3, grid, CELL, ks, w, es, 2.1, displacement=(.5, .5, .5))
    out['shifted_two_spin'] = dict(status=status, charge_error=abs(occ.sum()-2.1), up_minus_down=occ[:, 0].sum()-occ[:, 1].sum())
    status, occ, ef = occupations(lib, 3, grid, CELL, k, w, e, 2.6, nspinor=1)
    status1, occ1, ef1 = occupations(lib, 3, grid, CELL, k, w, e, 1.3, nspinor=2)
    out['spinless'] = dict(status=status, fermi_difference=abs(ef-ef1), occupation_ratio=float(occ.sum()/occ1.sum()))
    refusals = dict(
        off_grid=occupations(lib, 1, grid, CELL, k+np.array([1.e-3, 0., 0.]), w, e, 1.3)[0],
        incomplete=occupations(lib, 1, grid, CELL, k[:-1], w[:-1], e[:-1], 1.3)[0],
        weights=occupations(lib, 1, grid, CELL, k, w*1.01, e, 1.3)[0],
        top_band=occupations(lib, 1, grid, CELL, k, w, e[:, :, :1], .99)[0],
        too_few=occupations(lib, 1, grid, CELL, k, w, e[:, :, :1], 1.5)[0])
    out['refusals'] = refusals
    # Threads split the tetrahedra into blocks summed in order; the results may
    # differ from one thread only by rounding.
    threads = {}
    for method in (1, 2, 3):
        status, occ, ef = occupations(lib, method, grid, CELL, k, w, e, 1.3)
        for n in (2, 3, 7):
            status_n, occ_n, ef_n = occupations(lib, method, grid, CELL, k, w, e, 1.3, threads=n)
            threads[f'method{method}_threads{n}'] = dict(status=status_n, fermi_difference=abs(ef_n-ef),
                                                         occupation_difference=float(np.abs(occ_n-occ).max()))
    out['threads'] = threads
    expected = dict(off_grid=-402, incomplete=-403, weights=-405, top_band=-407, too_few=-406)
    out['passed'] = bool(all(v['status'] == 0 and v['trs_status'] == 0 and v['charge_error'] < 1.e-10
                             and v['trs_fermi_difference'] == 0. and v['trs_occupation_difference'] < 1.e-15
                             for key, v in out.items() if key.startswith('method'))
                         and out['shifted_two_spin']['status'] == 0 and out['shifted_two_spin']['charge_error'] < 1.e-10
                         and out['spinless']['status'] == 0 and out['spinless']['fermi_difference'] < 1.e-12
                         and abs(out['spinless']['occupation_ratio']-2.) < 1.e-12 and refusals == expected
                         and all(v['status'] == 0 and v['fermi_difference'] < 1.e-12 and v['occupation_difference'] < 1.e-14
                                 for v in threads.values()))
    return out


def check_convergence(lib):
    """Band energy and Fermi level of the model against a dense optimized-tetrahedron reference."""
    def run(method, n):
        grid = (n, n, max(2, round(n*0.6)))
        k, _ = mesh(grid); w = np.full(len(k), 1./len(k)); e = model_bands(k)
        status, occ, ef = occupations(lib, method, grid, CELL, k, w, e, 1.3)
        return status, float((occ*e).sum()), ef
    reference = run(3, 60)
    rows = []
    for n in (6, 9, 12, 18, 24):
        row = dict(n=n)
        for method, name in ((1, 'linear'), (2, 'bloechl'), (3, 'optimized')):
            status, band, ef = run(method, n)
            row[name] = dict(status=status, band_energy_error=band-reference[1], fermi_error=ef-reference[2])
        rows.append(row)
    last = rows[-1]
    passed = all(r[m]['status'] == 0 for r in rows for m in ('linear', 'bloechl', 'optimized')) and \
        abs(last['optimized']['band_energy_error']) < abs(last['linear']['band_energy_error'])
    return dict(reference_mesh='60x60x36 optimized', reference_band_energy=reference[1], rows=rows, passed=bool(passed))


# Quantum ESPRESSO 7.3.1, PW/src/tetra.f90, transcribed into numpy. Grid
# points are numbered with the first index fastest, as in REMAG; QE's own
# order (third index fastest) does not change any weight.
WLSM = np.array([
    [1440, 0, 30, 0, -38, 7, 17, -28, -56, 9, -46, 9, -38, -28, 17, 7, -18, -18, 12, -18],
    [0, 1440, 0, 30, -28, -38, 7, 17, 9, -56, 9, -46, 7, -38, -28, 17, -18, -18, -18, 12],
    [30, 0, 1440, 0, 17, -28, -38, 7, -46, 9, -56, 9, 17, 7, -38, -28, 12, -18, -18, -18],
    [0, 30, 0, 1440, 7, 17, -28, -38, 9, -46, 9, -56, -28, 17, 7, -38, -18, 12, -18, -18]])/1260.


def qe_opt_tetra(cell, grid, method):
    """opt_tetra_init: six tetrahedra on the shortest diagonal, 20 points each (4 for linear)."""
    bg = 2*np.pi*np.linalg.inv(cell).T
    bvec2 = bg/np.array(grid)[:, None]
    bvec3 = np.array([-bvec2[0]+bvec2[1]+bvec2[2], bvec2[0]-bvec2[1]+bvec2[2], bvec2[0]+bvec2[1]-bvec2[2], bvec2.sum(0)])
    ii = int(np.argmin((bvec3**2).sum(1)))
    ivvec0 = np.zeros(4, int); divvec = np.eye(4, dtype=int)
    ivvec0[ii] = 1; divvec[ii, ii] = -1
    ivvec = []
    for i1 in range(3):
        for i2 in range(3):
            if i2 == i1: continue
            for i3 in range(3):
                if i3 in (i1, i2): continue
                v = [ivvec0[:3]]; v.append(v[0]+divvec[:3, i1]); v.append(v[1]+divvec[:3, i2]); v.append(v[2]+divvec[:3, i3])
                v += [2*v[0]-v[1], 2*v[1]-v[2], 2*v[2]-v[3], 2*v[3]-v[0], 2*v[0]-v[2], 2*v[1]-v[3], 2*v[2]-v[0],
                      2*v[3]-v[1], 2*v[0]-v[3], 2*v[1]-v[0], 2*v[2]-v[1], 2*v[3]-v[2], v[3]-v[0]+v[1], v[0]-v[1]+v[2],
                      v[1]-v[2]+v[3], v[2]-v[3]+v[0]]
                ivvec.append(np.array(v[:20 if method == 3 else 4]))
    return tetra_points(ivvec, grid)


def qe_bloechl_tetra(grid):
    """tetra_init: six tetrahedra sharing the fixed edge n3-n6 of each subcell."""
    n = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (1, 1, 0), 5: (0, 0, 1), 6: (1, 0, 1), 7: (0, 1, 1), 8: (1, 1, 1)}
    sets = [(1, 2, 3, 6), (2, 3, 4, 6), (1, 3, 5, 6), (3, 4, 6, 8), (3, 6, 7, 8), (3, 5, 6, 7)]
    return tetra_points([np.array([n[c] for c in t]) for t in sets], grid)


def tetra_points(offsets, grid):
    g = np.array(grid); out = []
    for i3 in range(g[2]):
        for i2 in range(g[1]):
            for i1 in range(g[0]):
                for v in offsets:
                    p = (np.array([i1, i2, i3])+v) % g
                    out.append(p[:, 0]+g[0]*(p[:, 1]+g[1]*p[:, 2]))
    return np.array(out).reshape(-1, len(offsets[0]))


def qe_opt_weights(et, ef, tetra, method):
    """opt_tetra_weights_only for one spin: et (nk, nbnd); weights without the spin factor."""
    nntetra = tetra.shape[1]; wlsm = WLSM if method == 3 else np.eye(4)
    wg = np.zeros_like(et); ntetra = len(tetra)
    for nt in range(ntetra):
        for ibnd in range(et.shape[1]):
            e = wlsm[:, :nntetra]@et[tetra[nt], ibnd]
            itetra = np.argsort(e, kind='stable'); e = e[itetra]
            a = np.zeros((4, 4))
            for i2 in range(4):
                for i in range(4):
                    a[i2, i] = 0. if abs(e[i]-e[i2]) < 1.e-12 else (ef-e[i])/(e[i2]-e[i])
            if e[0] <= ef < e[1]:
                c = a[1, 0]*a[2, 0]*a[3, 0]*.25
                w0 = c*np.array([1.+a[0, 1]+a[0, 2]+a[0, 3], a[1, 0], a[2, 0], a[3, 0]])
            elif e[1] <= ef < e[2]:
                c1 = a[3, 0]*a[2, 0]*.25; c2 = a[3, 0]*a[2, 1]*a[0, 2]*.25; c3 = a[3, 1]*a[2, 1]*a[0, 3]*.25
                w0 = np.array([c1+(c1+c2)*a[0, 2]+(c1+c2+c3)*a[0, 3], c1+c2+c3+(c2+c3)*a[1, 2]+c3*a[1, 3],
                               (c1+c2)*a[2, 0]+(c2+c3)*a[2, 1], (c1+c2+c3)*a[3, 0]+c3*a[3, 1]])
            elif e[2] <= ef < e[3]:
                c = a[0, 3]*a[1, 3]*a[2, 3]
                w0 = .25*np.array([1.-c*a[0, 3], 1.-c*a[1, 3], 1.-c*a[2, 3], 1.-c*(1.+a[3, 0]+a[3, 1]+a[3, 2])])
            elif e[3] <= ef:
                w0 = np.full(4, .25)
            else:
                w0 = np.zeros(4)
            w0 = w0/ntetra
            for i2 in range(nntetra):
                wg[tetra[nt, i2], ibnd] += wlsm[itetra, i2]@w0
    return wg


def qe_bloechl_weights(et, ef, tetra):
    """tetra_weights_only for one spin, without the spin factor."""
    wg = np.zeros_like(et); ntetra = len(tetra)
    for nt in range(ntetra):
        for ibnd in range(et.shape[1]):
            etetra = et[tetra[nt], ibnd]; order = np.argsort(etetra, kind='stable')
            e1, e2, e3, e4 = etetra[order]; kp = tetra[nt][order]; s = e1+e2+e3+e4
            if ef >= e4:
                w = np.full(4, .25/ntetra); dosef = 0.
            elif ef >= e3:
                c4 = .25/ntetra*(e4-ef)**3/(e4-e1)/(e4-e2)/(e4-e3)
                dosef = 3./ntetra*(e4-ef)**2/(e4-e1)/(e4-e2)/(e4-e3)
                w = .25/ntetra-c4*(e4-ef)/np.array([e4-e1, e4-e2, e4-e3, 1.])
                w[3] = .25/ntetra-c4*(4.-(e4-ef)*(1./(e4-e1)+1./(e4-e2)+1./(e4-e3)))
            elif ef >= e2:
                c1 = .25/ntetra*(ef-e1)**2/(e4-e1)/(e3-e1)
                c2 = .25/ntetra*(ef-e1)*(ef-e2)*(e3-ef)/(e4-e1)/(e3-e2)/(e3-e1)
                c3 = .25/ntetra*(ef-e2)**2*(e4-ef)/(e4-e2)/(e3-e2)/(e4-e1)
                dosef = 1./ntetra/(e3-e1)/(e4-e1)*(3.*(e2-e1)+6.*(ef-e2)-3.*(e3-e1+e4-e2)*(ef-e2)**2/(e3-e2)/(e4-e2))
                w = np.array([c1+(c1+c2)*(e3-ef)/(e3-e1)+(c1+c2+c3)*(e4-ef)/(e4-e1),
                              c1+c2+c3+(c2+c3)*(e3-ef)/(e3-e2)+c3*(e4-ef)/(e4-e2),
                              (c1+c2)*(ef-e1)/(e3-e1)+(c2+c3)*(ef-e2)/(e3-e2),
                              (c1+c2+c3)*(ef-e1)/(e4-e1)+c3*(ef-e2)/(e4-e2)])
            elif ef >= e1:
                c4 = .25/ntetra*(ef-e1)**3/(e2-e1)/(e3-e1)/(e4-e1)
                dosef = 3./ntetra*(ef-e1)**2/(e2-e1)/(e3-e1)/(e4-e1)
                w = np.array([c4*(4.-(ef-e1)*(1./(e2-e1)+1./(e3-e1)+1./(e4-e1))),
                              c4*(ef-e1)/(e2-e1), c4*(ef-e1)/(e3-e1), c4*(ef-e1)/(e4-e1)])
            else:
                continue
            w = w+dosef*(s-4.*np.array([e1, e2, e3, e4]))/40.
            for i in range(4): wg[kp[i], ibnd] += w[i]
    return wg


def qe_average(wg, et):
    """The degenerate-state averaging at the end of opt_tetra_weights_only."""
    wg = wg.copy()
    for ik in range(et.shape[0]):
        for ibnd in range(et.shape[1]):
            wg1 = wg[ik, ibnd]
            for jbnd in range(ibnd+1, et.shape[1]):
                if abs(et[ik, ibnd]-et[ik, jbnd]) < 1.e-6: wg1 += wg[ik, jbnd]
                else:
                    wg[ik, ibnd:jbnd] = wg1/(jbnd-ibnd); break
    return wg


def qe_weights(et, ef, cell, grid, method):
    """All QE weights for et (nk_full, nspin, nbnd) at ef, without the nspin=1 factor of 2."""
    tetra = qe_bloechl_tetra(grid) if method == 2 else qe_opt_tetra(cell, grid, method)
    out = np.zeros_like(et)
    for spin in range(et.shape[1]):
        w = qe_bloechl_weights(et[:, spin], ef, tetra) if method == 2 else qe_opt_weights(et[:, spin], ef, tetra, method)
        out[:, spin] = w if method == 2 else qe_average(w, et[:, spin])
    return out


def check_against_qe_code(lib, rng):
    """REMAG in its QE-compatible settings against the numpy transcription, at a common ef."""
    results = []
    for case, (grid, nspin, ne) in enumerate([((4, 4, 3), 1, 3), ((3, 5, 2), 2, 3), ((6, 6, 4), 1, 3)]):
        k, _ = mesh(grid); w = np.full(len(k), 1./len(k))
        e = model_bands(k, spin_split=.2 if nspin == 2 else 0.)[:, :, :ne]
        e = e+1.e-3*rng.standard_normal(e.shape)   # break the model's symmetry
        e[:, :, 1] = np.where(np.arange(len(k))[:, None] % 3 == 0, e[:, :, 0], e[:, :, 1])  # exact degeneracies
        e.sort(axis=2)
        for method in (1, 2, 3):
            ef = float(np.median(e[:, :, 0]))
            reference = qe_weights(e, ef, CELL, grid, method)
            status, occ, _ = qe_mode(lib, method, grid, CELL, k, w, e, 1., 2, diagonal=1 if method == 2 else -1,
                                     average=method != 2, fermi=ef)
            results.append(dict(grid=grid, spins=nspin, method=method, status=status,
                                max_difference=float(np.abs(occ-reference).max())))
    return dict(cases=results, passed=bool(all(r['status'] == 0 and r['max_difference'] < 1.e-15 for r in results)))


def check_qe_records(lib, path=QE_RECORDS/'qe-tetrahedra.json'):
    """Finished pw.x runs: their eigenvalues must give QE's Fermi energy and occupations here.

    pw.x writes occupations as fractions of a full state (wg/wk); REMAG returns
    k weight times occupation, so its values are divided by the k weight and
    the spinless factor before comparing. Energies are converted to Ry, the
    unit of QE's 1e-6 degeneracy test and of SIESTA.
    """
    if not path.is_file(): return dict(skipped=f'{path} not found', passed=True)
    method_number = dict(linear=1, bloechl=2, optimized=3)
    results = []
    for r in json.loads(path.read_text())['runs']:
        method = method_number[r['method']]; grid = r['kmesh']
        cell = np.array(r['cell_bohr']); k = np.array(r['k_cartesian_per_bohr'])
        e = 2.*np.array(r['eigenvalues_ha']); q = np.array(r['occupations'])
        spins = 2 if r['lsda'] else 1; nspinor = 2 if (r['lsda'] or r['noncolin']) else 1
        e = e.reshape(len(k), spins, -1); q = q.reshape(len(k), spins, -1)
        w = np.full(len(k), 1./len(k)); full = 2. if nspinor == 1 else 1.
        qe_fermi = 2.*r['fermi_energy_ha']
        settings = dict(diagonal=1, average=False) if method == 2 else {}
        status, at_qe, _ = qe_mode(lib, method, grid, cell, k, w, e, r['nelec'], nspinor, fermi=qe_fermi, **settings)
        status_own, own, fermi = qe_mode(lib, method, grid, cell, k, w, e, r['nelec'], nspinor, **settings)
        scale = 1./(w[:, None, None]*full)
        results.append(dict(system=r['system'], method=r['method'], grid=grid, k_points=len(k), bands=e.shape[2],
                            status=status, status_own_fermi=status_own,
                            fermi_difference_ry=fermi-qe_fermi,
                            max_occupation_difference_at_qe_fermi=float(np.abs(at_qe*scale-q).max()),
                            max_occupation_difference_own_fermi=float(np.abs(own*scale-q).max()),
                            electrons_from_qe_occupations=float((q*full).sum()/len(k))))
    passed = all(x['status'] == 0 and x['status_own_fermi'] == 0 and abs(x['fermi_difference_ry']) < 1.e-8
                 and x['max_occupation_difference_at_qe_fermi'] < 1.e-10 and x['max_occupation_difference_own_fermi'] < 1.e-6
                 for x in results)
    return dict(source=path.name, runs=results, passed=bool(passed and results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(20261014)
    with tempfile.TemporaryDirectory() as tmp:
        lib = build(Path(tmp))
        report = dict(corners=check_corners(lib, rng), grids=check_grids(lib), convergence=check_convergence(lib),
                      qe_code=check_against_qe_code(lib, rng), qe_runs=check_qe_records(lib))
    report['passed'] = all(v['passed'] for v in report.values() if isinstance(v, dict))
    text = json.dumps(report, indent=1)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True); (args.output/'check.json').write_text(text+'\n')
    print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
