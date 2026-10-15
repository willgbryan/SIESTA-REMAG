#!/usr/bin/env python3
"""Write pw.x inputs that match make_siesta.py: cell, positions, Gamma-centred
k mesh, Fermi-Dirac smearing at the same temperature, PseudoDojo fully
relativistic PBE Co. constrained_magnetization 'atomic direction' adds
lambda (Ry) times the squared error in each atom's polar angle cosine; pw.x
leaves that penalty out of the printed total energy. Usage:

    python make_qe.py <root> --pseudo-dir /path/with/Co.upf --kmesh 12,12,7 --lambdas 25
"""
import argparse, json
from pathlib import Path
A, C = 2.507, 4.069
BOHR = 0.529177210903
KB_RY = 8.617333262e-5/13.605693122994       # Boltzmann constant, Ry per K


def deck(theta, kmesh, ecut, temperature, lam, pseudo_dir, calculation='scf', prefix='co', phi=0.,
         occupation='smearing', dft=None):
    """occupation 'smearing' is Fermi-Dirac at temperature; otherwise a pw.x tetrahedron method such as
    'tetrahedra_opt'. dft overrides the pseudopotential's functional, e.g. 'pw' for LDA (PW92)."""
    constrained = f"  constrained_magnetization = 'atomic direction'\n  lambda = {lam}\n" if lam > 0 else ""
    occupations = ("  occupations = 'smearing'\n  smearing = 'fd'\n"
                   f"  degauss = {KB_RY*temperature:.10e}\n") if occupation == 'smearing' else \
        f"  occupations = '{occupation}'\n"
    if dft: occupations += f"  input_dft = '{dft}'\n"
    # Tetrahedron weights are only piecewise smooth; pw.x's SCF stalls near
    # 1e-9 Ry with them, so those runs stop at 1e-8 Ry (0.14 micro-eV).
    conv = '1.0d-10' if occupation == 'smearing' else '1.0d-8'
    return f"""&control
  calculation = '{calculation}'
  prefix = '{prefix}'
  pseudo_dir = '{pseudo_dir}'
  outdir = './tmp'
  verbosity = 'high'
/
&system
  ibrav = 4
  celldm(1) = {A/BOHR:.8f}
  celldm(3) = {C/A:.8f}
  nat = 2
  ntyp = 1
  ecutwfc = {ecut}
  ecutrho = {4*ecut}
{occupations}  noncolin = .true.
  lspinorb = .true.
  starting_magnetization(1) = 0.4
  angle1(1) = {theta}
  angle2(1) = {phi}
{constrained}/
&electrons
  conv_thr = {conv}
  mixing_beta = 0.2
  electron_maxstep = 300
/
ATOMIC_SPECIES
Co 58.933194 Co.upf
ATOMIC_POSITIONS crystal
Co 0.333333333333 0.666666666667 0.25
Co 0.666666666667 0.333333333333 0.75
K_POINTS automatic
{kmesh[0]} {kmesh[1]} {kmesh[2]} 0 0 0
"""


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path); p.add_argument('--pseudo-dir', default='/root/pseudo')
    p.add_argument('--thetas', default='0,90'); p.add_argument('--kmesh', default='6,6,3')
    p.add_argument('--ecut', type=float, default=100.); p.add_argument('--temperature', type=float, default=500.)
    p.add_argument('--lambdas', default='10'); p.add_argument('--occupation', default='smearing')
    p.add_argument('--dft', default=None)
    a = p.parse_args(); kmesh = [int(x) for x in a.kmesh.split(',')]
    for lam in [float(x) for x in a.lambdas.split(',')]:
        for theta in [float(t) for t in a.thetas.split(',')]:
            d = a.root/f"lambda{lam:g}"/f"theta{theta:06.2f}"; d.mkdir(parents=True, exist_ok=False)
            (d/'pw.in').write_text(deck(theta, kmesh, a.ecut, a.temperature, lam, a.pseudo_dir,
                                       occupation=a.occupation, dft=a.dft))
            (d/'manifest.json').write_text(json.dumps(dict(theta_deg=theta, kmesh=kmesh, ecutwfc_ry=a.ecut,
                temperature_k=a.temperature, lambda_ry=lam, occupation=a.occupation, dft=a.dft, a=A, c=C), indent=1))
    print('prepared', a.root)
