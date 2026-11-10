"""Generate a small, self-contained PBE/PSML case; no arbitrary FDF overlays."""
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from ..corpus import sha256
from .common import digest, save

GPA_PER_EV_A3 = 160.21766208
MU0_MUB_PER_A3 = 11.654064


def validate_atoms(atoms, settings):
    if not 0 < len(atoms) <= settings.max_atoms or not all(atoms.pbc):
        raise ValueError('A fully periodic structure within max_atoms is required')
    if atoms.constraints:
        raise ValueError('Remove input constraints: discovery relaxes positions and the full cell')
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all() or atoms.cell.volume < 1:
        raise ValueError('Nonfinite or singular geometry')
    if not np.isfinite(atoms.get_initial_magnetic_moments()).all():
        raise ValueError('Nonfinite initial magnetic moments')


def geometry(atoms):
    return dict(lattice_A=atoms.cell.array.tolist(), species=atoms.get_chemical_symbols(),
                fractional_coordinates=atoms.get_scaled_positions(wrap=False).tolist())


def kmesh(atoms, settings):
    reciprocal = 2 * np.pi * np.linalg.inv(atoms.cell.array).T
    mesh = np.maximum(1, np.ceil(np.linalg.norm(reciprocal, axis=1) / settings.k_spacing_inv_ang)).astype(int)
    if int(np.prod(mesh)) > settings.max_kpoints:
        raise ValueError('Requested reciprocal spacing exceeds max_kpoints; review explicitly, no silent coarsening')
    return mesh.tolist()


def axes(atoms):
    z = atoms.cell.array[2] / np.linalg.norm(atoms.cell.array[2])
    x = atoms.cell.array[0] - np.dot(atoms.cell.array[0], z) * z
    x /= np.linalg.norm(x)
    return {'z': z.tolist(), 'x': x.tolist(), 'y': np.cross(z, x).tolist()}


def pseudos_for(atoms, directory):
    """Only fully relativistic PBE PSML; caller supplies licensed/cached files."""
    result = {}
    for symbol in dict.fromkeys(atoms.get_chemical_symbols()):
        path = Path(directory) / (symbol + '.psml')
        tree = ET.parse(path)
        spec = next((x for x in tree.iter() if x.tag.split('}')[-1] == 'pseudo-atom-spec'), None)
        if spec is None or spec.get('atomic-label') != symbol or spec.get('relativity') != 'dirac':
            raise ValueError(f'{symbol}: fully relativistic PSML required')
        functionals = {int(x.get('id')) for x in spec.iter() if x.tag.split('}')[-1] == 'functional'}
        if functionals != {101, 130}:
            raise ValueError(f'{symbol}: expected PBE exchange and correlation')
        result[path.name] = sha256(path)
    return result


def prepare_case(atoms, pseudos, destination, settings, *, mesh=None, direction=None,
                 spins=None, relax=False, restart=False):
    validate_atoms(atoms, settings)
    destination = Path(destination)
    hashes = pseudos_for(atoms, pseudos)
    mesh = mesh or kmesh(atoms, settings)
    symbols = atoms.get_chemical_symbols()
    order = list(dict.fromkeys(symbols))
    if spins is None:
        # Preserve an explicitly supplied collinear branch; otherwise a cheap FM seed.
        spins = atoms.get_initial_magnetic_moments() if atoms.has('initial_magmoms') else [
            {'Fe': 3., 'Co': 2., 'Ni': 1., 'Mn': 3., 'Cr': 2.}.get(s, 0.) for s in symbols]
    spins = np.asarray(spins, dtype=float)
    if spins.shape != (len(atoms),) or not np.isfinite(spins).all():
        raise ValueError('One finite collinear starting moment per atom is required')
    axis = np.array(direction if direction is not None else [0., 0., 1.], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError('Invalid spin axis')
    axis /= np.linalg.norm(axis)
    theta, phi = math.degrees(math.acos(np.clip(axis[2], -1, 1))), math.degrees(math.atan2(axis[1], axis[0]))
    lines = ['SystemName REMAG discovery screen', 'SystemLabel remag', f'NumberOfAtoms {len(atoms)}',
             f'NumberOfSpecies {len(order)}', '%block ChemicalSpeciesLabel']
    lines += [f'{i} {atoms.numbers[symbols.index(s)]} {s}' for i, s in enumerate(order, 1)]
    lines += ['%endblock ChemicalSpeciesLabel', 'LatticeConstant 1 Ang', '%block LatticeVectors']
    lines += [' '.join(f'{v:.15g}' for v in row) for row in atoms.cell.array]
    lines += ['%endblock LatticeVectors', 'AtomicCoordinatesFormat Fractional', '%block AtomicCoordinatesAndAtomicSpecies']
    lines += [' '.join(f'{v:.15g}' for v in row) + f' {order.index(s)+1}' for row, s in zip(atoms.get_scaled_positions(wrap=False), symbols)]
    lines += ['%endblock AtomicCoordinatesAndAtomicSpecies', '%block kgrid_Monkhorst_Pack']
    lines += [' '.join(str(mesh[i] if i == j else 0) for j in range(3)) + ' 0.0' for i in range(3)]
    lines += ['%endblock kgrid_Monkhorst_Pack', 'Spin ' + ('spin-orbit' if direction is not None else 'polarized'), '%block DM.InitSpin']
    for i, m in enumerate(spins, 1):
        # FDF distinguishes integers from reals here. Only the atom index is an
        # integer; scalar mode ignores angles, so preserve its signed moment.
        if direction is None:
            lines.append(f'{i} {m:.12f}')
        else:
            lines.append(f'{i} {abs(m):.12f} {theta if m >= 0 else 180-theta:.12f} {phi if m >= 0 else phi+180:.12f}')
    lines += ['%endblock DM.InitSpin', 'PAO.BasisSize DZP', 'PAO.EnergyShift 0.02 Ry', 'PAO.SplitNorm 0.25',
              'PAO.Polarization.NonPerturbative true', f'MeshCutoff {settings.mesh_ry} Ry', 'XC.functional GGA', 'XC.authors PBE',
              f'ElectronicTemperature {settings.smearing_k} K', 'SolutionMethod diagon', 'Diag.Algorithm divide-and-conquer',
              f'MaxSCFIterations {settings.max_scf}', 'SCF.MustConverge true', f'DM.Tolerance {settings.dm_tolerance}',
              f'SCF.DM.Tolerance {settings.dm_tolerance}', f'SCF.H.Tolerance {settings.h_tolerance_ev} eV',
              'SCF.DM.Converge true', 'SCF.H.Converge true', 'SCF.Mix Hamiltonian',
              f'DM.MixingWeight {settings.mixing_weight}', f'DM.NumberPulay {settings.mixing_history}', 'SCF.Mix.Spin all',
              f'DM.UseSaveDM {str(restart).lower()}', 'Write.DM false', 'Write.DM.End.Of.Cycle true',
              'WriteForces true', 'WriteMullikenPop 1', 'SaveHS true', 'SaveRho false', 'SaveDeltaRho false',
              'UseTreeTimer true', 'UseParallelTimer false', 'UseWallTime true', 'Timing.Split.SCF.Steps false']
    if relax:
        # Native Broyden uses components. This guarantees the advertised vector bound.
        lines += ['MD.TypeOfRun Broyden', 'MD.VariableCell true', f'MD.NumCGsteps {settings.dft_steps}',
                  f'MD.MaxForceTol {settings.dft_fmax_ev_ang / math.sqrt(3):.12g} eV/Ang',
                  f'MD.MaxStressTol {settings.dft_stress_gpa} GPa', 'MD.MaxCGDispl 0.1 Ang']
    else:
        lines += ['MD.NumCGsteps 0']
    if direction is not None:
        lines += ['WriteOrbMom true']
    destination.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        shutil.copyfile(Path(pseudos) / name, destination / name)
    (destination / 'input.fdf').write_text('\n'.join(lines) + '\n')
    manifest = dict(name=destination.name, atoms=len(atoms), spin='spin-orbit' if direction is not None else 'polarized',
                    input_sha256=sha256(destination / 'input.fdf'), pseudopotentials=hashes, geometry=geometry(atoms),
                    kmesh=mesh, direction=axis.tolist() if direction is not None else None,
                    settings=settings.to_dict(), screening=True, density_matrix_output=True, hamiltonian_output=True,
                    relaxation=relax, orientation_constrained=False,
                    geometry_sha256=digest(geometry(atoms)))
    save(destination / 'manifest.json', manifest)
    return destination
