#!/usr/bin/env python3
"""Write REMAG-SIESTA and pw.x inputs for Mn3Sn with 120-degree magnetic order.

Mn3Sn is hexagonal (P6_3/mmc, D0_19): Mn on 6h (x, 2x, 1/4) with x = 0.8388,
Sn on 2c (1/3, 2/3, 1/4), a = 5.665 A, c = 4.531 A (Kuroda et al., Nat. Mater.
16, 1090 (2017)). The six Mn form three pairs related by inversion; each pair
shares one in-plane moment direction. A Mn site whose position, seen from the
centre of its kagome triangle, points at angle alpha carries a moment at
phi = -alpha + phi0 in the inverse-triangular state (Gamma5, negative vector
chirality, the observed order) and phi = +alpha + phi0 in the triangular state
with positive chirality (Gamma3).

    python make_inputs.py <root> --code siesta --psml-dir DIR --order gamma5 --u 1.0 --j 0.5
    python make_inputs.py <root> --code qe --pseudo-dir DIR --order gamma3 --u 0

REMAG-SIESTA runs hold every Mn at its target direction with
REMAG.SpinConstraint, one site per atom, unless --unconstrained. pw.x runs are
unconstrained: both orders keep their magnetic symmetry, which pw.x uses.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

A, C, X = 5.665, 4.531, 0.8388
BOHR = 0.529177210903
KB_RY = 8.617333262e-5/13.605693122994
MN = [(X, 2*X, .25), (-2*X, -X, .25), (X, -X, .25), (-X, -2*X, .75), (2*X, X, .75), (-X, X, .75)]
SN = [(1/3, 2/3, .25), (2/3, 1/3, .75)]
PAIR = [0, 1, 2, 0, 1, 2]         # inversion partners share a moment


def cart(f):
    return (f[0]*A - f[1]*A/2, f[1]*A*math.sqrt(3)/2)


def angles(order, phi0):
    """In-plane moment angle (degrees) of each of the six Mn."""
    # Positions of the three z=1/4 sites relative to their triangle's centre,
    # using the triangle nearest the origin: fractional coordinates folded to (-1/2, 1/2].
    fold = [tuple(((v+.5) % 1)-.5 for v in MN[i][:2]) for i in range(3)]
    xy = [cart(f) for f in fold]
    cx, cy = sum(p[0] for p in xy)/3, sum(p[1] for p in xy)/3
    alpha = [math.degrees(math.atan2(p[1]-cy, p[0]-cx)) for p in xy]
    sign = {'gamma5': -1., 'gamma3': 1.}[order]
    phi = [(sign*a+phi0) % 360. for a in alpha]
    return [phi[PAIR[i]] for i in range(6)]


def siesta(order, phi0, kmesh, temperature, u, j, constrained, shift, mesh, nc_gga, dm_tol='1.e-6', h_tol='1.e-5'):
    phi = angles(order, phi0)
    lines = ["SystemName REMAG Mn3Sn", "SystemLabel remag", "NumberOfAtoms 8", "NumberOfSpecies 2",
             "%block ChemicalSpeciesLabel", "1 25 Mn", "2 50 Sn", "%endblock ChemicalSpeciesLabel",
             "LatticeConstant 1.0 Ang", "%block LatticeVectors",
             f"{A:.6f} 0.0 0.0", f"{-A/2:.6f} {A*math.sqrt(3)/2:.6f} 0.0", f"0.0 0.0 {C:.6f}",
             "%endblock LatticeVectors", "AtomicCoordinatesFormat Fractional",
             "%block AtomicCoordinatesAndAtomicSpecies"]
    lines += [f"{p[0] % 1:.6f} {p[1] % 1:.6f} {p[2]:.6f} 1" for p in MN]
    lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 2" for p in SN]
    lines += ["%endblock AtomicCoordinatesAndAtomicSpecies", "%block kgrid_Monkhorst_Pack"]
    lines += [" ".join(str(kmesh[i] if i == k else 0) for k in range(3))+" 0.0" for i in range(3)]
    lines += ["%endblock kgrid_Monkhorst_Pack", "Spin spin-orbit+offsite", "%block DM.InitSpin"]
    lines += [f"{i+1} 3.0 90.0000 {phi[i]:.4f}" for i in range(6)]
    lines += ["%endblock DM.InitSpin", "PAO.BasisSize TZP", f"PAO.EnergyShift {shift} Ry", f"MeshCutoff {mesh} Ry",
              "XC.functional GGA", "XC.authors PBE"]
    if nc_gga: lines += [f"REMAG.XC.NoncollinearGGA {nc_gga}"]
    if u > 0 or j > 0:
        # Liechtenstein DFT+U on Mn 3d (SIESTA's only scheme with spin-orbit);
        # projectors from the default method 2 radii.
        lines += ["%block DFTU.Proj", "Mn 1", "n=3 2", f"{u:.4f} {j:.4f}", "0.000 0.000", "%endblock DFTU.Proj"]
    lines += [f"ElectronicTemperature {temperature} K", "OccupationFunction FD", "SolutionMethod diagon",
              "Diag.Algorithm divide-and-conquer", "MaxSCFIterations 600", "SCF.MustConverge true",
              f"DM.Tolerance {dm_tol}", f"SCF.H.Tolerance {h_tol} eV", "DM.MixingWeight 0.02", "DM.NumberPulay 8",
              "SCF.Mix.Spin all", "DM.UseSaveDM false", "WriteMullikenPop 1", "Write.DM false",
              "Write.DM.End.Of.Cycle false", "SaveHS false", "Timing.Split.SCF.Steps false",
              "UseTreeTimer true", "UseWallTime true"]
    if constrained:
        lines += ["%block REMAG.SpinConstraint"] + [f"90.0000 {phi[i]:.4f} {i+1}" for i in range(6)]
        lines += ["%endblock REMAG.SpinConstraint", "REMAG.SpinConstraint.Criterion both",
                  "REMAG.SpinConstraint.AngleTolerance 0.01", "REMAG.SpinConstraint.TorqueTolerance 1.e-3",
                  "REMAG.SpinConstraint.MaxEvaluations 60"]
    return "\n".join(lines)+"\n"


def qe(order, phi0, kmesh, temperature, u, j, ecut, pseudo_dir):
    phi = angles(order, phi0)
    hub = ""
    if u > 0 or j > 0:
        # J > 0 selects pw.x's Liechtenstein scheme, as in SIESTA.
        hub = "HUBBARD ortho-atomic\n"+"".join(f"U Mn{s}-3d {u}\nJ Mn{s}-3d {max(j, 1.e-12)}\n" for s in (1, 2, 3))
    species = "".join(f"Mn{s} 54.938044 Mn.upf\n" for s in (1, 2, 3))+"Sn 118.71 Sn.upf\n"
    mag = "".join(f"  starting_magnetization({s}) = 0.3\n  angle1({s}) = 90.0\n  angle2({s}) = {phi[s-1]:.4f}\n"
                  for s in (1, 2, 3))
    pos = "".join(f"Mn{PAIR[i]+1} {p[0] % 1:.6f} {p[1] % 1:.6f} {p[2]:.6f}\n" for i, p in enumerate(MN))
    pos += "".join(f"Sn {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in SN)
    return f"""&control
  calculation = 'scf'
  prefix = 'mn3sn'
  pseudo_dir = '{pseudo_dir}'
  outdir = './tmp'
  verbosity = 'high'
/
&system
  ibrav = 4
  celldm(1) = {A/BOHR:.8f}
  celldm(3) = {C/A:.8f}
  nat = 8
  ntyp = 4
  ecutwfc = {ecut}
  ecutrho = {4*ecut}
  occupations = 'smearing'
  smearing = 'fd'
  degauss = {KB_RY*temperature:.10e}
  noncolin = .true.
  lspinorb = .true.
{mag}/
&electrons
  conv_thr = 1.0d-7
  mixing_beta = 0.1
  electron_maxstep = 400
/
ATOMIC_SPECIES
{species}ATOMIC_POSITIONS crystal
{pos}K_POINTS automatic
{kmesh[0]} {kmesh[1]} {kmesh[2]} 0 0 0
{hub}"""


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path); p.add_argument('--code', choices=['siesta', 'qe'], required=True)
    p.add_argument('--order', choices=['gamma5', 'gamma3'], default='gamma5')
    p.add_argument('--phi0', type=float, default=0.); p.add_argument('--kmesh', default='8,8,9')
    p.add_argument('--temperature', type=float, default=1000.); p.add_argument('--u', type=float, default=0.)
    p.add_argument('--j', type=float, default=0.); p.add_argument('--unconstrained', action='store_true')
    p.add_argument('--shift', type=float, default=0.01); p.add_argument('--mesh', type=int, default=600)
    p.add_argument('--nc-gga', default='local-frame'); p.add_argument('--ecut', type=float, default=96.)
    p.add_argument('--psml-dir', type=Path); p.add_argument('--pseudo-dir', default='/root/pseudo/mn3sn')
    # The local-frame GGA leaves dH near 1e-3 eV in Gamma5, where |m| vanishes between sites.
    p.add_argument('--dm-tolerance', default='1.e-6'); p.add_argument('--h-tolerance', default='1.e-5')
    a = p.parse_args()
    kmesh = [int(x) for x in a.kmesh.split(',')]
    d = a.root; d.mkdir(parents=True, exist_ok=False)
    manifest = dict(code=a.code, order=a.order, phi0_deg=a.phi0, kmesh=kmesh, temperature_k=a.temperature,
                    u_ev=a.u, j_ev=a.j, mn_phi_deg=angles(a.order, a.phi0), a=A, c=C, x=X)
    if a.code == 'siesta':
        text = siesta(a.order, a.phi0, kmesh, a.temperature, a.u, a.j, not a.unconstrained, a.shift, a.mesh,
                      a.nc_gga, a.dm_tolerance, a.h_tolerance)
        (d/'input.fdf').write_text(text)
        for el in ('Mn', 'Sn'):
            src = a.psml_dir/f'{el}.psml'; (d/f'{el}.psml').write_bytes(src.read_bytes())
            manifest[f'{el.lower()}_psml_sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
        manifest.update(constrained=not a.unconstrained, energy_shift_ry=a.shift, mesh_cutoff_ry=a.mesh,
                        nc_gga=a.nc_gga, dm_tolerance=float(a.dm_tolerance), h_tolerance_ev=float(a.h_tolerance))
    else:
        (d/'pw.in').write_text(qe(a.order, a.phi0, kmesh, a.temperature, a.u, a.j, a.ecut, a.pseudo_dir))
        manifest.update(ecutwfc_ry=a.ecut)
    (d/'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('prepared', d, ' '.join(f'{x:.1f}' for x in manifest['mn_phi_deg']))
