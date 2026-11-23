"""Frozen-potential angular screening and ingestion of existing CUDA evidence."""
import json
from pathlib import Path
import time
import numpy as np
from ..angular_energy import unit_directions, fit_angular_energy
from .backend import MftPlan
from .operators import RY_EV, rotation_between, rotated_direction
from .state import file_hash
from .electronic import band_descriptors,orbital_groups

LIMITATIONS = [
    'Frozen-potential band free energies are not self-consistent total-energy differences.',
    'Results depend on the reference potential, k mesh, basis and electronic smearing.',
    'Site/orbital Mulliken populations are representation-dependent diagnostics.',
    'No physical convergence or screening-throughput gain is implied by successful execution.',
]


def directions_array(reference, directions):
    # No crystallographic symmetry is invented when it has not been supplied.
    if isinstance(directions, str):
        if directions != 'cartesian':
            raise ValueError('Supply symmetry-adapted directions explicitly; automatic symmetry discovery is unavailable')
        directions = np.eye(3)
    points = [unit_directions([reference])[0]]
    for n in unit_directions(directions):
        if not any(np.linalg.norm(n-p) < 1.e-12 for p in points):
            points.append(n)
    return np.asarray(points)


def angular_summary(points, *, volume_angstrom3=None, model=None):
    directions = [p['direction'] for p in points]
    free = np.array([p['delta_free_ev'] for p in points])
    if not np.isfinite(free).all() or not len(free):
        raise ValueError('Finite angular free energies required')
    lowest = int(np.argmin(free))
    result = dict(lowest_sampled_direction=directions[lowest], lowest_sampled_index=lowest,
                  sampled_barrier_ev=float(np.max(free)-np.min(free)),
                  classification='insufficient_angular_sampling', angular_model=None,
                  anisotropy_density_mj_m3=None)
    if volume_angstrom3 is not None:
        if not np.isfinite(volume_angstrom3) or volume_angstrom3 <= 0:
            raise ValueError('Positive evaluated cell volume required')
        result['anisotropy_density_mj_m3'] = result['sampled_barrier_ev']*160217.6634/volume_angstrom3
    if model:
        fit = fit_angular_energy(directions, free, model, energy_kind='frozen_band_free', energy_unit='eV/cell')
        result['angular_model'] = fit
        if model == 'axial4':
            a, b = fit['coefficients']['A2'], fit['coefficients']['A4']
            candidates = [0., 1.]
            if b > 0 and 0 < -a/(2*b) < 1:
                candidates.append(-a/(2*b))
            t = min(candidates, key=lambda x: a*x+b*x*x)
            result['classification'] = ('axial_model_easy_axis' if t == 0 else
                                        'axial_model_easy_plane' if t == 1 else 'axial_model_easy_cone')
            result['angular_model']['K1_ev_cell'] = a
            result['angular_model']['K2_ev_cell'] = b
    return result


def _point(index, direction, values):
    spin = np.asarray(values['spin'])
    norm = float(np.linalg.norm(spin))
    angle = None if norm < 1.e-8 else float(np.degrees(np.arctan2(
        np.linalg.norm(np.cross(spin/norm, direction)), np.dot(spin/norm, direction))))
    return dict(index=index, direction=list(direction), spin_muB=spin.tolist(),
                spin_direction_error_degrees=angle, chemical_potential_ev=values['chemical_potential']*RY_EV,
                electron_count=values['charge'], delta_band_ev=values['delta_band']*RY_EV,
                delta_free_ev=values['delta_free']*RY_EV,
                delta_canonical_free_ev=values['delta_canonical_free']*RY_EV,
                torque_ev_per_radian=(np.asarray(values['torque'])*RY_EV).tolist(),
                max_residual=values['max_residual'], max_orthogonality_error=values['max_orthogonality_error'],
                seconds=values['seconds'])


def _entropy(p):
    a, b = np.maximum(p, 1.e-15), np.maximum(1-p, 1.e-15)
    return -a*np.log(a)-b*np.log(b)


def calculate_mft(state, directions='cartesian', *, angular_model=None, output=None,
                  library_path=None, check_soc_free=True, memory_limit_bytes=2*1024**3,
                  fermi_window_ev=.25, descriptor_count=8, retain_orbital_weights=False):
    """Full frozen-H diagonalization, fixed N, FP64 CUDA, no CPU solver.

    The reference is always included first. Explicit directions are Cartesian
    in the crystal frame. A two-direction sweep yields an energy difference,
    never a uniquely identified K1. Optional NPZ retains k/band evidence.
    """
    if fermi_window_ev <= 0 or not np.isfinite(fermi_window_ev) or descriptor_count < 1:
        raise ValueError('Positive descriptor window and count required')
    started = time.perf_counter()
    m = state.metadata
    axes = directions_array(m['reference_direction'], directions)
    nk, n, _ = state.shape
    quaternions = [rotation_between(axes[0], a) for a in axes]
    points, descriptors, spectra, occupations, populations = [], [], [], [], []
    checks = dict(generator_spectrum_max_abs_ev=None, no_soc_rotation_max_abs_ev=None)
    root = None
    if output is not None:
        root = Path(output);root.mkdir(parents=True, exist_ok=False)
    with MftPlan(state, library_path=library_path, memory_limit_bytes=memory_limit_bytes) as plan:
        if state.reference_spectrum is not None:
            _, generator_e, _, _ = plan.solve([1., 0., 0., 0.], soc=m['reference_has_soc'], reference=True)
            checks['generator_spectrum_max_abs_ev'] = float(np.max(abs(generator_e-state.reference_spectrum))*RY_EV)
            if checks['generator_spectrum_max_abs_ev'] > 1.e-7:
                raise RuntimeError('Loaded generator does not reconstruct the original SCF spectrum')
        if check_soc_free:
            _, e0, _, _ = plan.solve([1., 0., 0., 0.], soc=False, reference=True)
            q = quaternions[1] if len(quaternions) > 1 else [2**-.5, 2**-.5, 0., 0.]
            _, e1, _, _ = plan.solve(q, soc=False)
            checks['no_soc_rotation_max_abs_ev'] = float(np.max(abs(e1-e0))*RY_EV)
            if checks['no_soc_rotation_max_abs_ev'] > 1.e-7:
                raise RuntimeError('SOC-free global spin rotation changes the spectrum')
        for i, (axis, q) in enumerate(zip(axes, quaternions, strict=True)):
            values, e, occ, pop = plan.solve(q, reference=i == 0)
            points.append(_point(i, axis, values))
            if i == 0:
                ref_e, ref_occ, ref_mu = e, occ, values['chemical_potential']
            w = state.weights[:, None]
            # Same centered paired estimator as native, retaining k localization.
            k_delta = np.sum(ref_occ*(e-ref_e)+(occ-ref_occ)*e-
                m['temperature_ry']*w*(_entropy(occ/w)-_entropy(ref_occ/w)), axis=1)*RY_EV
            hot = np.argsort(abs(k_delta))[::-1][:descriptor_count]
            near = abs(e-values['chemical_potential'])*RY_EV <= fermi_window_ev
            candidates = np.argwhere(near)
            if len(candidates):
                shifts = abs((e-ref_e)[near])
                candidates = candidates[np.argsort(shifts)[::-1][:descriptor_count]]
            states = [dict(k=int(k), band=int(b), energy_relative_to_mu_ev=float((e[k,b]-values['chemical_potential'])*RY_EV),
                           angular_eigenvalue_shift_ev=float((e[k,b]-ref_e[k,b])*RY_EV),
                           named_orbital_weights={name:float(pop[k,b,ids].sum()) for name,ids in orbital_groups(m).items()},
                           orbital_weights=pop[k,b].tolist() if m.get('orbitals') is not None else None)
                      for k,b in candidates]
            descriptors.append(dict(direction_index=i,
                k_anisotropy_hotspots=[dict(k=int(k), coordinate=state.kpoints[k].tolist(),
                    weighted_delta_free_ev=float(k_delta[k])) for k in hot], near_fermi_states=states,
                near_fermi_window_ev=fermi_window_ev,
                orbital_response=band_descriptors(m,e,values['chemical_potential'],state.weights,pop,width_ev=fermi_window_ev),
                band_matching='energy order; crossings can exchange band identity',
                orbital_population=(np.einsum('kb,kbo->o', occ, pop).tolist() if m.get('orbitals') else None)))
            if root:
                spectra.append(e);occupations.append(occ)
                if retain_orbital_weights:populations.append(pop)
        workspace = values['workspace_bytes']
    result = dict(method='full_frozen_hamiltonian', status='screening', engine='remag_mft_sweep_cuda_fp64',
        energy_normalization='per_evaluated_cell', approximation='magnetic_force_theorem',
        reference_direction=axes[0].tolist(), points=points, electronic_descriptors=descriptors,
        checks=checks, limitations=list(LIMITATIONS), raw_evidence=[],
        unsupported_observables=['directional_orbital_moments', 'site_resolved_torque'],
        numerical_sensitivity=None, physical_accuracy_qualified=False,
        performance=dict(total_wall_seconds=time.perf_counter()-started, workspace_bytes=workspace,
                         per_orientation_seconds=[p['seconds'] for p in points],
                         repeated_directional_scf_speedup=None))
    result.update(angular_summary(points, volume_angstrom3=m.get('volume_angstrom3'), model=angular_model))
    if root:
        arrays=dict(energies_ry=np.asarray(spectra),occupations=np.asarray(occupations),kpoints=state.kpoints,weights=state.weights)
        if retain_orbital_weights:arrays['mulliken_orbital_weights']=np.asarray(populations)
        np.savez(root/'mft-spectra.npz', **arrays)
        result['raw_orbital_weights_retained']=bool(retain_orbital_weights)
        result['raw_evidence'].append(dict(path=str((root/'mft-spectra.npz').resolve()), sha256=file_hash(root/'mft-spectra.npz')))
        (root/'mft.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def read_mft_evidence(path, *, reference_direction, volume_angstrom3=None, angular_model=None):
    """Expose an already executed complete native sweep without a new solve.

    A physical bias does not erase useful screening evidence. Failed solves or
    incomplete k/direction coverage still cannot produce a magnetic result.
    """
    path = Path(path)
    if path.is_dir():
        path = path/'remag-frozen-potential.json'
    report = json.loads(path.read_text())
    if report.get('schema') != 1 or report.get('status') != 'complete' or report.get('energy_unit') != 'Ry':
        raise ValueError('A complete native frozen-potential report in Ry is required')
    reference_soc=report.get('reference_soc_scale',1)
    if (report.get('reference')!='Final SCF generator Hamiltonian' or type(reference_soc) is not int
            or reference_soc not in (0,1) or report.get('target_soc_scale',1)!=1
            or report.get('lattice_operator')!=('full_offsite' if reference_soc else 'spin_traceless_offsite')
            or type(report.get('kpoints')) is not int or report['kpoints']<1
            or type(report.get('bands')) is not int or report['bands']<2 or report['bands']%2):
        raise ValueError('Unsupported or missing native generator/operator provenance')
    error = report.get('generator_spectrum_max_abs_error_ry')
    if not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error*RY_EV <= 1.e-7:
        raise ValueError('Native generator spectrum identity failed or is missing')
    if report.get('reference_soc_scale', 1) == 0:
        control = report.get('reference_control') or {}
        if (control.get('completed') is not True or control.get('status') != 0 or
                control.get('eigensystem_jobs') != report.get('kpoints')):
            raise ValueError('SOC-free generator identity solve is missing or failed')
    ref = unit_directions([reference_direction])[0]
    points = []
    for i, p in enumerate(report.get('points', [])):
        if (p.get('index') != i or p.get('completed') is not True or p.get('status') != 0 or
                p.get('solver_info') != 0 or p.get('electron_count_passed') is not True or
                p.get('eigensystem_jobs') != report.get('kpoints')):
            raise ValueError('Failed or incomplete native frozen point')
        for key in ('band_difference_ry', 'band_free_difference_ry', 'canonical_free_difference_ry', 'fermi_ry', 'charge'):
            if not isinstance(p.get(key), (int, float)) or not np.isfinite(p[key]):
                raise ValueError(f'Invalid frozen output {key}')
        for key in ('charge_error_reference', 'charge_error_current'):
            if not isinstance(p.get(key), (int, float)) or not np.isfinite(p[key]) or abs(p[key]) > 1.e-10:
                raise ValueError('Frozen point does not preserve electron count')
        axis = rotated_direction(ref, p['quaternion'])
        if i == 0 and np.linalg.norm(axis-ref) > 1.e-12:
            raise ValueError('Point zero is not the reference direction')
        spin = p.get('spin_muB')
        if spin is not None and (np.asarray(spin).shape != (3,) or not np.isfinite(spin).all()):
            raise ValueError('Invalid native spin expectation')
        point = dict(index=i, direction=axis.tolist(), spin_muB=spin,
            chemical_potential_ev=p['fermi_ry']*RY_EV, electron_count=p['charge'],
            delta_band_ev=p['band_difference_ry']*RY_EV, delta_free_ev=p['band_free_difference_ry']*RY_EV,
            delta_canonical_free_ev=p['canonical_free_difference_ry']*RY_EV,
            torque_ev_per_radian=None, max_residual=None, max_orthogonality_error=None, seconds=p.get('seconds'))
        points.append(point)
    if not points:
        raise ValueError('No executed angular points')
    result = dict(method='full_frozen_hamiltonian', status='screening', engine='existing_remag_cuda_probe',
        energy_normalization='per_evaluated_cell', approximation='magnetic_force_theorem',
        reference_direction=ref.tolist(), points=points, electronic_descriptors=None,
        checks={'generator_spectrum_max_abs_ev': error*RY_EV}, limitations=list(LIMITATIONS)+[
            'Imported report retains its original diagnostics; absent residuals and descriptors are unavailable.',
            'Existing FePt/FeCo evidence shows material reference-dependent bias relative to directional SCF.'],
        raw_evidence=[dict(path=str(path.resolve()), sha256=file_hash(path))],
        unsupported_observables=['electronic_torque', 'orbital_moments', 'orbital_response'],
        numerical_sensitivity=None, physical_accuracy_qualified=False,
        performance=dict(total_wall_seconds=None, per_orientation_seconds=[p['seconds'] for p in points],
                         repeated_directional_scf_speedup=None))
    result.update(angular_summary(points, volume_angstrom3=volume_angstrom3, model=angular_model))
    return result
