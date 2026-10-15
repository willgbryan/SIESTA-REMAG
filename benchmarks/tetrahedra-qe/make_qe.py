#!/usr/bin/env python3
"""Write pw.x inputs whose tetrahedron occupations REMAG must reproduce.

Each run uses one of pw.x's tetrahedron methods on a full Gamma-centred grid
(nosym, noinv), so that every grid point appears in the output with its
eigenvalues, its occupations and the Fermi energy. Systems, with PseudoDojo
standard pseudopotentials (nc-sr-04_pw for LDA, nc-fr-04_pbe for spin-orbit):

    al        fcc Al, no spin, LDA (PW92)
    co        hcp Co, collinear spin, LDA (PW92)
    co-soc    hcp Co, noncollinear spin with spin-orbit, PBE

Co-fr.upf is the fully relativistic PBE Co.upf renamed.

Usage:

    python make_qe.py <root> --pseudo-dir /path/to/pseudo
"""
import argparse
import json
from pathlib import Path

BOHR = 0.529177210903
METHODS = {'bloechl': 'tetrahedra', 'linear': 'tetrahedra_lin', 'optimized': 'tetrahedra_opt'}
SYSTEMS = {
    'al': dict(kmesh=[8, 8, 8], ecut=40., pseudo='Al.upf', mass=26.9815385,
               cell="  ibrav = 2\n  celldm(1) = 7.6000\n  nat = 1\n",
               spin="", positions="Al 0.0 0.0 0.0\n", species='Al'),
    'co': dict(kmesh=[6, 6, 4], ecut=90., pseudo='Co.upf', mass=58.933194,
               cell=f"  ibrav = 4\n  celldm(1) = {2.507/BOHR:.8f}\n  celldm(3) = {4.069/2.507:.8f}\n  nat = 2\n",
               spin="  nspin = 2\n  starting_magnetization(1) = 0.4\n",
               positions="Co 0.333333333333 0.666666666667 0.25\nCo 0.666666666667 0.333333333333 0.75\n",
               species='Co'),
    'co-soc': dict(kmesh=[6, 6, 4], ecut=90., pseudo='Co-fr.upf', mass=58.933194,
                   cell=f"  ibrav = 4\n  celldm(1) = {2.507/BOHR:.8f}\n  celldm(3) = {4.069/2.507:.8f}\n  nat = 2\n",
                   spin="  noncolin = .true.\n  lspinorb = .true.\n  starting_magnetization(1) = 0.4\n"
                        "  angle1(1) = 0.0\n  angle2(1) = 0.0\n",
                   positions="Co 0.333333333333 0.666666666667 0.25\nCo 0.666666666667 0.333333333333 0.75\n",
                   species='Co'),
}


def deck(system, method, pseudo_dir):
    s = SYSTEMS[system]; k = s['kmesh']
    return f"""&control
  calculation = 'scf'
  prefix = 'run'
  pseudo_dir = '{pseudo_dir}'
  outdir = './tmp'
  verbosity = 'high'
/
&system
{s['cell']}  ntyp = 1
  ecutwfc = {s['ecut']}
  ecutrho = {4*s['ecut']}
  occupations = '{METHODS[method]}'
  nosym = .true.
  noinv = .true.
{s['spin']}/
&electrons
  conv_thr = 1.0d-8
  mixing_beta = 0.3
  electron_maxstep = 200
/
ATOMIC_SPECIES
{s['species']} {s['mass']} {s['pseudo']}
ATOMIC_POSITIONS crystal
{s['positions']}K_POINTS automatic
{k[0]} {k[1]} {k[2]} 0 0 0
"""


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path); p.add_argument('--pseudo-dir', default='/root/pseudo')
    p.add_argument('--systems', default=','.join(SYSTEMS)); p.add_argument('--methods', default=','.join(METHODS))
    a = p.parse_args()
    for system in a.systems.split(','):
        for method in a.methods.split(','):
            d = a.root/system/method; d.mkdir(parents=True, exist_ok=False)
            (d/'pw.in').write_text(deck(system, method, a.pseudo_dir))
            (d/'manifest.json').write_text(json.dumps(dict(system=system, method=method, kmesh=SYSTEMS[system]['kmesh'],
                                                           ecutwfc_ry=SYSTEMS[system]['ecut']), indent=1))
    print('prepared', a.root)
