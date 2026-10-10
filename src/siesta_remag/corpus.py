"""Small reproducible corpus. Pseudos come from the pinned upstream tests.

These are numerical regression workloads, not converged materials predictions.
Mesh, basis and k sampling need separate physical convergence studies.
"""
import hashlib
import itertools
import json
import re
import shutil
from pathlib import Path

CASES = {
    "fe2_collinear": ("fe", (1, 1, 1), "polarized", (3, 3, 3), 0),
    "fe2_forces": ("fe", (1, 1, 1), "polarized", (3, 3, 3), 0),
    "fe2_relax": ("fe", (1, 1, 1), "polarized", (3, 3, 3), 0),
    "fe16_collinear": ("fe", (2, 2, 2), "polarized", (2, 2, 2), 0),
    "fe54_collinear": ("fe", (3, 3, 3), "polarized", (1, 1, 1), 0),
    "fe128_collinear": ("fe", (4, 4, 4), "polarized", (1, 1, 1), 0),
    "fe2_noncollinear": ("fe", (1, 1, 1), "non-collinear", (3, 3, 3), 30),
    "fe16_noncollinear": ("fe", (2, 2, 2), "non-collinear", (2, 2, 2), 30),
    "fe2_noncollinear_gamma": ("fe", (1, 1, 1), "non-collinear", (1, 1, 1), 30),
    "fept2_soc_gamma_z": ("fept", (1, 1, 1), "spin-orbit", (1, 1, 1), 0),
    "fept2_soc_gamma_x": ("fept", (1, 1, 1), "spin-orbit", (1, 1, 1), 90),
    "fept2_soc_z": ("fept", (1, 1, 1), "spin-orbit", (3, 3, 3), 0),
    "fept2_soc_x": ("fept", (1, 1, 1), "spin-orbit", (3, 3, 3), 90),
    "fept16_soc_z": ("fept", (2, 2, 2), "spin-orbit", (2, 2, 2), 0),
    "fept16_soc_x": ("fept", (2, 2, 2), "spin-orbit", (2, 2, 2), 90),
    "fept128_soc_z": ("fept", (4, 4, 4), "spin-orbit", (2, 2, 2), 0),
    "fept128_soc_x": ("fept", (4, 4, 4), "spin-orbit", (2, 2, 2), 90),
}

# Nonparallel seeds with nonzero azimuth exercise both transverse components.
# These are unconstrained initial states, not an assertion of a stable texture.
CANTED_CASES = {f"fept{size}_soc_canted": base for size, base in
                ((2, "fept2_soc_z"), (16, "fept16_soc_z"), (128, "fept128_soc_z"))}
CASES.update({name: CASES[base] for name, base in CANTED_CASES.items()})
CANTED_ANGLES = ((35., 45.), (70., 125.), (60., 210.), (45., 300.))

# Exercise requested energy-density exports in all supported spin layouts.
ENERGY_EXPORT_CASES = {
    base+'__'+output: (base, output in ('tsde', 'both'), output in ('netcdf', 'both'))
    for base in ('fe2_forces', 'fe2_noncollinear_gamma', 'fept2_soc_gamma_z')
    for output in ('tsde', 'netcdf', 'both')
}
CASES.update({name: CASES[base] for name, (base, _, _) in ENERGY_EXPORT_CASES.items()})

# These trajectories have converged pristine CPU preflights, including
# nontrivial depth-three coefficients in all three magnetic spin layouts.
HISTORY_CASES = dict(fe2_history_relax='fe2_relax',
    fe2_noncollinear_history_relax='fe2_noncollinear', fept2_soc_history_relax='fept2_soc_z')
CASES.update({name: CASES[base] for name, base in HISTORY_CASES.items()})
NO_DENSITY_EXPORT_CASES = {base+'__nodm': base for base in
    ('fe2_forces', 'fe2_noncollinear_gamma', 'fept2_soc_gamma_z', *HISTORY_CASES)}
CASES.update({name: CASES[base] for name, base in NO_DENSITY_EXPORT_CASES.items()})
NO_MATRIX_EXPORT_CASES = {base+'__nomatrices': base for base in
    ('fe2_forces', 'fe2_noncollinear_gamma', 'fept2_soc_gamma_z', *HISTORY_CASES)}
CASES.update({name: CASES[base] for name, base in NO_MATRIX_EXPORT_CASES.items()})

IO_POLICIES = ("end-of-cycle", "checkpoint")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(name: str, upstream: Path, target: Path, *, io_policy="end-of-cycle", split_scf_timings=True, initial_populations=False) -> Path:
    if io_policy not in IO_POLICIES:
        raise ValueError(f"Unknown I/O policy: {io_policy}")
    material, repeat, spin, kmesh, theta = CASES[name]
    requested_name=NO_DENSITY_EXPORT_CASES.get(name,NO_MATRIX_EXPORT_CASES.get(name,name))
    history=requested_name in HISTORY_CASES
    base_name, tsde, netcdf = ENERGY_EXPORT_CASES.get(requested_name, (requested_name, False, False))
    base_name=HISTORY_CASES.get(base_name,base_name)
    if material == "fe":
        lattice = (2.87, 2.87, 2.87)
        basis = [(0., 0., 0., 1), (.5, .5, .5, 1)]
        species = [(26, "Fe_nc")]
    else:
        # Tetragonal L10 primitive cell in the same convention as upstream FePt tests.
        lattice = (2.73, 2.73, 3.79)
        basis = [(0., 0., 0., 1), (.5, .5, .5, 2)]
        species = [(26, "Fe_fept_SOC"), (78, "Pt_fept_SOC")]
    atoms = [((i+x)*lattice[0], (j+y)*lattice[1], (k+z)*lattice[2], s)
             for i, j, k in itertools.product(*(range(n) for n in repeat))
             for x, y, z, s in basis]
    if base_name in ("fe2_forces", "fe2_relax") or history or name in CANTED_CASES:
        x, y, z, s = atoms[-1]
        atoms[-1] = (x+.04, y+.02, z, s)
    target.mkdir(parents=True, exist_ok=False)
    pseudo_hashes = {}
    for _, label in species:
        pseudo = upstream / "Tests/Pseudos" / f"{label}.psf"
        shutil.copy2(pseudo, target)
        pseudo_hashes[pseudo.name] = sha256(pseudo)
    lines = [f"SystemName REMAG regression {name}", "SystemLabel remag",
             f"NumberOfAtoms {len(atoms)}", f"NumberOfSpecies {len(species)}",
             "%block ChemicalSpeciesLabel"]
    lines += [f"{i} {z} {label}" for i, (z, label) in enumerate(species, 1)]
    lines += ["%endblock ChemicalSpeciesLabel", "LatticeConstant 1.0 Ang",
              "%block LatticeVectors"]
    lines += [" ".join(str(lattice[i]*repeat[i] if i == j else 0.) for j in range(3)) for i in range(3)]
    lines += ["%endblock LatticeVectors", "AtomicCoordinatesFormat NotScaledCartesianAng",
              "%block AtomicCoordinatesAndAtomicSpecies"]
    lines += [f"{x:.12f} {y:.12f} {z:.12f} {s}" for x, y, z, s in atoms]
    lines += ["%endblock AtomicCoordinatesAndAtomicSpecies", "%block kgrid_Monkhorst_Pack"]
    lines += [" ".join(str(kmesh[i] if i == j else 0) for j in range(3)) + " 0.0" for i in range(3)]
    lines += ["%endblock kgrid_Monkhorst_Pack", f"Spin {spin}", "%block DM.InitSpin"]
    if name in CANTED_CASES:
        initial_spins = [dict(atom=i, magnitude_muB=2.5 if s == 1 else .3,
                             theta_deg=CANTED_ANGLES[(i-1) % len(CANTED_ANGLES)][0],
                             phi_deg=CANTED_ANGLES[(i-1) % len(CANTED_ANGLES)][1])
                         for i, (*_, s) in enumerate(atoms, 1)]
        lines += [f"{a['atom']} {a['magnitude_muB']} {a['theta_deg']:.1f} {a['phi_deg']:.1f}"
                  for a in initial_spins]
    else:
        lines += [f"{i} {2.5 if s == 1 else .3} {theta:.1f} 0." for i, (*_, s) in enumerate(atoms, 1)]
    mesh_cutoff=400 if history and material=='fept' else 100
    lines += ["%endblock DM.InitSpin", "PAO.BasisSize SZP", f"MeshCutoff {mesh_cutoff} Ry",
              "XC.functional GGA", "XC.authors PBE", f"ElectronicTemperature {1000 if material == 'fept' else 300} K",
              "SolutionMethod diagon", "Diag.Algorithm divide-and-conquer",
              f"MaxSCFIterations {1000 if name in CANTED_CASES else 200}", "SCF.MustConverge true",
              f"DM.Tolerance {1.e-7 if len(atoms) >= 30 else 1.e-6}",
              f"SCF.H.Tolerance {1.e-6 if len(atoms) >= 30 else 1.e-5} eV",
              "DM.MixingWeight 0.05", f"DM.NumberPulay {8 if material == 'fept' else 6}",
              "SCF.Mix.Spin all", "DM.UseSaveDM false", "WriteForces true",
              "WriteMullikenPop 1", "SaveRho true", "SaveDeltaRho true", f"SaveHS {'false' if name in NO_MATRIX_EXPORT_CASES else 'true'}",
              "UseTreeTimer true", "UseParallelTimer false", "UseWallTime true",
              f"Timing.Split.SCF.Steps {'true' if split_scf_timings else 'false'}"]
    # Keep the final restart/analysis artifact without serializing the DM on
    # every SCF iteration. Both reference and candidate receive this policy.
    lines += [f"Write.DM {'true' if io_policy == 'checkpoint' else 'false'}",
              f"Write.DM.End.Of.Cycle {'false' if name in NO_DENSITY_EXPORT_CASES or name in NO_MATRIX_EXPORT_CASES else 'true'}",
              f"TS.DE.Save {'true' if tsde else 'false'}", f"CDF.Save {'true' if netcdf else 'false'}"]
    if initial_populations:
        lines += ['Charge.Mulliken init+geometry+end', 'Charge.Mulliken.Format 1', 'Charge.Spin init+scf+geometry+end']
    if base_name == "fe2_relax" or history:
        lines += ["MD.TypeOfRun CG", f"MD.NumCGsteps {40 if history else 20}", "MD.MaxForceTol 0.01 eV/Ang"]
    if history:
        lines += ['DM.AllowExtrapolation true', 'DM.History.Depth 3']
        if material=='fept':lines += ['MD.MaxCGDispl 0.02 Ang']
    content = "\n".join(lines) + "\n"
    (target / "input.fdf").write_text(content)
    # SystemName is descriptive and the spin initialization is the swept variable.
    family = re.sub(r"%block DM.InitSpin.*?%endblock DM.InitSpin", "", content, flags=re.S)
    family = re.sub(r"^SystemName.*\n", "", family)
    manifest = dict(name=name, atoms=len(atoms), spin=spin, kmesh=kmesh,
                    repeat=repeat, theta_deg=theta, pseudopotentials=pseudo_hashes,
                    energy_density_outputs=dict(tsde=tsde, netcdf=netcdf),
                    initial_population_output=bool(initial_populations),
                    input_sha256=sha256(target / "input.fdf"),
                    purpose="numerical-regression", orientation_constrained=False, io_policy=io_policy, split_scf_timings=split_scf_timings,
                    orientation_family_sha256=hashlib.sha256(family.encode()).hexdigest())
    if base_name == "fe2_relax" or history:
        manifest["relaxation_force_tolerance_ev_ang"] = .01
    manifest['density_matrix_output']=name not in NO_DENSITY_EXPORT_CASES and name not in NO_MATRIX_EXPORT_CASES
    manifest['hamiltonian_output']=name not in NO_MATRIX_EXPORT_CASES
    if history:
        manifest.update(density_history_depth=3,allow_dm_extrapolation=True,mesh_cutoff_ry=mesh_cutoff)
    if name in CANTED_CASES:
        manifest.update(theta_deg=None, initial_spins=initial_spins,
                        magnetic_initialization="nonparallel-unconstrained",
                        stable_canted_state_verified=False)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target
