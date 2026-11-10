"""Scientific handoff checks independent of process exit codes."""
import math
from pathlib import Path
import re

import numpy as np

from ..corpus import sha256
from .common import load
from .inputs import GPA_PER_EV_A3, MU0_MUB_PER_A3


def native_review(folder, result, settings, *, relaxation=False, axis=None):
    folder = Path(folder)
    text = (folder/'stdout.log').read_text(errors='replace')
    stderr = (folder/'stderr.log').read_text(errors='replace')
    obs = result['observables']
    grids = re.findall(r'InitMesh:\s+MESH\s*=\s*(\d+)\s+x\s+(\d+)\s+x\s+(\d+)', text)
    points = re.findall(r'k-grid:\s+Number of k-points\s*=\s*(\d+)', text)
    errors = list(result.get('parse_errors', []))
    if not grids or not points:
        errors.append('Missing native mesh/k-point evidence')
    if not result['valid'] or result['returncode'] != 0 or result['timed_out']:
        errors.append('Native calculation did not complete successfully')
    if 'SCF_NOT_CONV' in text + stderr or 'SCF cycle converged' not in text or 'Job completed' not in text:
        errors.append('Missing native SCF completion or explicit SCF failure')
    expected_spin = 8 if axis is not None else 2
    if obs.get('spin_components') != expected_spin:
        errors.append('Unexpected spin layout')
    modes = re.findall(r'^redata:\s+Spin configuration\s*=\s*(\S+)', text, re.M)
    expected_mode = 'spin-orbit+offsite' if axis is not None else 'collinear'
    if not modes or modes[-1] != expected_mode:
        errors.append('Native spin configuration differs from the requested model')
    if axis is not None and 'Attempting to read DM from file... Succeeded...' not in text:
        errors.append('Native SOC did not confirm reading its prepared density seed')
    try:
        n = load(folder/'manifest.json')['atoms']
        forces = np.asarray(obs['forces_ev_ang'], float)
        stress = np.asarray(obs['stress_ev_ang3'], float)
        moments = np.asarray(obs['atom_moments_muB'], float)
        net = np.asarray(obs['total_moment_muB'], float)
        if forces.shape != (n, 3) or stress.shape != (3, 3) or moments.shape != (n, 3) or net.shape != (3,):
            raise ValueError('Missing force/stress/site-moment arrays')
        values = np.r_[forces.ravel(), stress.ravel(), moments.ravel(), net,
                       obs['total_energy_ev'], obs['free_energy_ev']]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite native observables')
        fmax = float(np.linalg.norm(forces, axis=1).max())
        smax = float(abs(stress).max() * GPA_PER_EV_A3)
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc)); fmax = smax = None
    logs = list(folder.glob('fdf*.log'))
    effective_hashes = {}
    try:
        if len(logs) != 1:
            raise ValueError('Expected one native effective-options file')
        options = {}; inside = False
        for line in logs[0].read_text().splitlines():
            parts = line.split('#', 1)[0].split()
            if not parts:
                continue
            if parts[0].lower() == '%block': inside = True
            elif parts[0].lower() == '%endblock': inside = False
            elif not inside: options.setdefault(re.sub(r'[-_.]', '', parts[0]).lower(), []).append(parts[1:])
        for key, expected in [('dmtolerance', settings.dm_tolerance), ('scfdmtolerance', settings.dm_tolerance), ('scfhtolerance', settings.h_tolerance_ev)]:
            if not options.get(key): raise ValueError('Missing native option ' + key)
            for parts in options[key]:
                value = float(parts[0].replace('D', 'E'))
                if key == 'scfhtolerance':
                    unit = parts[1].lower()
                    if unit not in ('ev', 'ry'): raise ValueError('Unsupported Hamiltonian tolerance units')
                    if unit == 'ry': value *= 13.605693122994
                if not math.isclose(value, expected, rel_tol=1e-9):
                    raise ValueError('Native tolerance mismatch: ' + key)
        for key in ('scfdmconverge', 'scfhconverge', 'scfmustconverge'):
            if not options.get(key) or any(p[0].lower() not in ('t', 'true', '.true.') for p in options[key]):
                raise ValueError('Required native convergence option disabled: ' + key)
        effective_hashes[logs[0].name] = sha256(logs[0])
    except (ValueError, IndexError) as exc:
        errors.append(str(exc))
    optimizer_passed = bool(re.search(r'^outcoor:\s+Relaxed atomic coordinates', text, re.M)) if relaxation else None
    if relaxation and (not optimizer_passed or fmax is None or fmax > settings.dft_fmax_ev_ang or smax > settings.dft_stress_gpa):
        errors.append('DFT relaxation has not passed native completion and force/stress targets')
    angle = None
    if axis is not None:
        try:
            net = np.asarray(obs['total_moment_muB'], float)
            if np.linalg.norm(net) <= 1e-8: raise ValueError('Zero net moment has no accepted axis')
            angle = math.degrees(math.acos(np.clip(abs(np.dot(net, axis)) / np.linalg.norm(net), 0, 1)))
            if angle > settings.axis_tolerance_deg:
                errors.append('Final moment left the requested axis')
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    artifacts = {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and
                 (p.name in ('input.fdf', 'result.json', 'stdout.log', 'stderr.log') or p.suffix in ('.DM', '.HSX', '.STRUCT_OUT', '.XV', '.psml'))}
    return dict(accepted=not errors, errors=errors, solver_seconds=result['wall_seconds'],
                scf_iterations=obs['scf_iterations'], timed_out=result['timed_out'],
                native_optimizer_converged=optimizer_passed, max_force_ev_ang=fmax,
                native_grid=list(map(int, grids[-1])) if grids else None,
                native_kpoints=int(points[-1]) if points else None,
                max_stress_gpa=smax, axis_deviation_deg=angle, effective_options_sha256=effective_hashes,
                artifact_sha256=artifacts)


def evaluated_structure(folder, expected_numbers):
    """Only call after native optimizer completion; trial XV is never a handoff."""
    from ase import Atoms
    lines = (Path(folder)/'remag.STRUCT_OUT').read_text().splitlines()
    cell = np.array([[float(x) for x in line.split()] for line in lines[:3]])
    count = int(lines[3]); rows = [line.split() for line in lines[4:]]
    if len(rows) != count or count != len(expected_numbers):
        raise ValueError('STRUCT_OUT atom count differs')
    numbers = [int(row[1]) for row in rows]
    if numbers != list(expected_numbers):
        raise ValueError('STRUCT_OUT species/order changed')
    frac = np.array([[float(x) for x in row[2:5]] for row in rows])
    if not np.isfinite(frac).all() or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0:
        raise ValueError('Invalid STRUCT_OUT geometry')
    return Atoms(numbers=numbers, scaled_positions=frac, cell=cell, pbc=True)


def branch_check(current, reference, axis, reference_axis, settings):
    a = np.asarray(current['atom_moments_muB']) @ np.asarray(axis)
    b = np.asarray(reference['atom_moments_muB']) @ np.asarray(reference_axis)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid site moments for branch comparison')
    rms = float(np.sqrt(np.mean((a-b)**2)))
    refnet = np.linalg.norm(reference['total_moment_muB'])
    if refnet <= 1e-8: raise ValueError('Reference has no net magnetic moment')
    change = float(abs(np.linalg.norm(current['total_moment_muB'])-refnet) / refnet)
    flips = bool(np.any((abs(b) > .5) & (a*b < 0)))
    return dict(passed=rms < settings.branch_rms_mub and change < settings.branch_net_fraction and not flips,
                site_rms_mub=rms, net_fractional_change=change, robust_site_flip=flips)


def directional_result(rows, volume):
    e = {k: (r['total_energy_ev']-rows['z']['total_energy_ev'])*160217.66208/volume for k, r in rows.items()}
    f = {k: (r['free_energy_ev']-rows['z']['free_energy_ev'])*160217.66208/volume for k, r in rows.items()}
    axial = all(v > 0 for d in (e, f) for k, v in d.items() if k != 'z')
    transverse = all(v < 0 for d in (e, f) for k, v in d.items() if k != 'z')
    return dict(delta_E_MJ_m3=e, delta_F_MJ_m3=f,
                Js_T=float(np.linalg.norm(rows['z']['total_moment_muB']) * MU0_MUB_PER_A3 / volume),
                signal='axial' if axial else 'transverse' if transverse else 'mixed_or_unresolved',
                axis_convention='z parallel to relaxed c; x perpendicular projection of a; y=z cross x',
                K1=None, K2=None, kappa=None, precision_qualified=False, physical_anisotropy_qualified=False,
                magnetic_ground_state_qualified=False, synthesizability_qualified=False)
