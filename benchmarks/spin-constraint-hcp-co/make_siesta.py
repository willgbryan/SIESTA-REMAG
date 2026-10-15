#!/usr/bin/env python3
"""Write REMAG-SIESTA inputs for constrained E(theta) of hcp Co.

Both atoms form one constrained site, so every run is a coherent rotation of
the whole magnetization to polar angle theta (from the c axis) at azimuth phi
(from a). Each run directory gets input.fdf, the pseudopotential and
manifest.json. Usage:

    python make_siesta.py <root> --psml Co.psml --kmesh 36,36,20 --thetas 0,45,90
"""
import argparse, hashlib, json, math, shutil
from pathlib import Path

A, C = 2.507, 4.069        # Angstrom, room-temperature hcp Co


XC = {'PBE': ["XC.functional GGA", "XC.authors PBE"], 'PW92': ["XC.functional LDA", "XC.authors PW92"]}
SOC = {'onsite': "Spin spin-orbit", 'offsite': "Spin spin-orbit+offsite", 'none': "Spin non-collinear"}
# Explicit bases beyond PAO.BasisSize: semicore 3s3p doubled, or close to quadruple zeta
# with two 4p polarization shells and a 4f shell polarizing 3d. Radii follow PAO.EnergyShift.
PAO = {'semicore-dz': ["%block PAO.Basis", "Co 4", "n=3 0 2", "0. 0.", "n=3 1 2", "0. 0.",
                       "n=4 0 3 P 1", "0. 0. 0.", "n=3 2 3", "0. 0. 0.", "%endblock PAO.Basis"],
       'rich': ["%block PAO.Basis", "Co 4", "n=3 0 3", "0. 0. 0.", "n=3 1 3", "0. 0. 0.",
                "n=4 0 4 P 2", "0. 0. 0. 0.", "n=3 2 4 P 1", "0. 0. 0. 0.", "%endblock PAO.Basis"]}


def deck(theta, phi, kmesh, basis, mesh, temperature, shift, constrained, tolerance, penalty,
         soc='onsite', occupation='FD', xc='PBE', nc_gga=None, h_tolerance='1.e-6', pao_block=None,
         mc_polar=None):
    """occupation is FD (Fermi-Dirac at temperature) or a tetrahedron method, e.g. Tetrahedra.Optimized.
    nc_gga 'local-frame' evaluates the noncollinear GGA as QE does (REMAG.XC.NoncollinearGGA); None keeps upstream's."""
    lines = ["SystemName REMAG hcp Co constrained", "SystemLabel remag", "NumberOfAtoms 2", "NumberOfSpecies 1",
             "%block ChemicalSpeciesLabel", "1 27 Co", "%endblock ChemicalSpeciesLabel",
             "LatticeConstant 1.0 Ang", "%block LatticeVectors",
             f"{A:.6f} 0.0 0.0", f"{-A/2:.6f} {A*math.sqrt(3)/2:.6f} 0.0", f"0.0 0.0 {C:.6f}",
             "%endblock LatticeVectors", "AtomicCoordinatesFormat Fractional",
             "%block AtomicCoordinatesAndAtomicSpecies",
             "0.333333333333 0.666666666667 0.25 1", "0.666666666667 0.333333333333 0.75 1",
             "%endblock AtomicCoordinatesAndAtomicSpecies", "%block kgrid_Monkhorst_Pack"]
    lines += [" ".join(str(kmesh[i] if i == j else 0) for j in range(3))+" 0.0" for i in range(3)]
    lines += ["%endblock kgrid_Monkhorst_Pack", SOC[soc], "%block DM.InitSpin",
              f"1 1.6 {theta:.4f} {phi:.4f}", f"2 1.6 {theta:.4f} {phi:.4f}", "%endblock DM.InitSpin",
              *(PAO[pao_block] if pao_block else [f"PAO.BasisSize {basis}"]),
              f"PAO.EnergyShift {shift} Ry", f"MeshCutoff {mesh} Ry",
              *XC[xc], *([f"REMAG.XC.NoncollinearGGA {nc_gga}"] if nc_gga else []),
              *([f"REMAG.XC.Multicollinear.Polar {mc_polar}"] if mc_polar else []),
              *([f"ElectronicTemperature {temperature} K", "OccupationFunction FD"] if occupation == 'FD'
                else [f"OccupationFunction {occupation}"]),
              "SolutionMethod diagon",
              "Diag.Algorithm divide-and-conquer",
              "MaxSCFIterations 400", "SCF.MustConverge true", "DM.Tolerance 1.e-7", f"SCF.H.Tolerance {h_tolerance} eV",
              "DM.MixingWeight 0.05", "DM.NumberPulay 8", "SCF.Mix.Spin all", "DM.UseSaveDM false",
              "WriteMullikenPop 1", "Write.DM false", "Write.DM.End.Of.Cycle false", "SaveHS false",
              "Timing.Split.SCF.Steps false", "UseTreeTimer true", "UseWallTime true"]
    if constrained:
        lines += ["%block REMAG.SpinConstraint", f"{theta:.4f} {phi:.4f} 1 2", "%endblock REMAG.SpinConstraint",
                  "REMAG.SpinConstraint.Criterion both", f"REMAG.SpinConstraint.AngleTolerance {tolerance}",
                  "REMAG.SpinConstraint.TorqueTolerance 1.e-3", "REMAG.SpinConstraint.MaxEvaluations 60"]
        if penalty is not None: lines += [f"REMAG.SpinConstraint.Penalty {penalty} Ry"]
    return "\n".join(lines)+"\n"


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path); p.add_argument('--psml', type=Path, required=True)
    p.add_argument('--thetas', default='0,45,90'); p.add_argument('--phi', type=float, default=0.)
    p.add_argument('--kmesh', default='36,36,20'); p.add_argument('--basis', default='TZP'); p.add_argument('--mesh', type=int, default=600)
    p.add_argument('--temperature', type=float, default=500.); p.add_argument('--shift', type=float, default=0.02)
    p.add_argument('--unconstrained', action='store_true'); p.add_argument('--tolerance', type=float, default=0.01)
    p.add_argument('--penalty', type=float); p.add_argument('--soc', choices=list(SOC), default='onsite')
    p.add_argument('--occupation', default='FD'); p.add_argument('--xc', choices=list(XC), default='PBE')
    p.add_argument('--nc-gga', choices=['legacy', 'local-frame', 'multicollinear'])
    p.add_argument('--mc-polar', type=int, help='polar nodes per hemisphere for multicollinear')
    # local-frame's sign convention leaves dH flickering near 5e-6 eV.
    p.add_argument('--h-tolerance', default='1.e-6'); p.add_argument('--pao-block', choices=list(PAO))
    a = p.parse_args()
    kmesh = [int(x) for x in a.kmesh.split(',')]
    for theta in [float(t) for t in a.thetas.split(',')]:
        d = a.root/f"theta{theta:06.2f}"; d.mkdir(parents=True, exist_ok=False)
        (d/'input.fdf').write_text(deck(theta, a.phi, kmesh, a.basis, a.mesh, a.temperature, a.shift,
                                        not a.unconstrained, a.tolerance, a.penalty, a.soc, a.occupation, a.xc, a.nc_gga,
                                        a.h_tolerance, a.pao_block, a.mc_polar))
        shutil.copy2(a.psml, d/'Co.psml')
        (d/'manifest.json').write_text(json.dumps(dict(theta_deg=theta, phi_deg=a.phi, kmesh=kmesh, basis=a.basis,
            mesh_cutoff_ry=a.mesh, temperature_k=a.temperature, energy_shift_ry=a.shift, constrained=not a.unconstrained,
            penalty=a.penalty, soc=a.soc, occupation=a.occupation, xc=a.xc, nc_gga=a.nc_gga, h_tolerance_ev=float(a.h_tolerance), pao_block=a.pao_block, mc_polar=a.mc_polar, psml_sha256=hashlib.sha256(a.psml.read_bytes()).hexdigest(), a=A, c=C), indent=1))
    print("prepared", a.root)
