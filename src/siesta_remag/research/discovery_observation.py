"""Saved discovery results in the research observation protocol.

The exact authored export is never replaced by relaxed floating coordinates.
The raw collection is immutable; this reproducible view lives beside it and
copies collected stage bytes for the existing electronic inspection tools.
No service calls, native execution, exact-source authoring or reservations.
"""
from pathlib import Path
import shutil

import numpy as np

from ..corpus import sha256
from ..discovery.common import load, save
from ..discovery.inputs import axes, MU0_MUB_PER_A3
from ..discovery.review import evaluated_structure
from .relaxation_execution import review_relaxation


FORMAT = 'REMAG-DISCOVERY-OBSERVATION/1'
VIEW = 'research-observation'
STAGES = {'dft-relax': 'scalar', 'soc-z': 'soc-z', 'soc-x': 'soc-x', 'soc-y': 'soc-y'}


def _observation(attempt):
    """Reproduce a deterministic view from a stopped, reviewed attempt."""
    attempt = Path(attempt).resolve()
    from .relaxation_recovery import verified_execution, NAME as RECOVERY
    execution = verified_execution(attempt)
    if (execution.get('format') != 'REMAG-RELAXATION-EXECUTION/1' or
            execution.get('status') not in ('measured', 'held_native_review') or
            execution.get('allocation_stopped') is not True or
            execution.get('app_lifecycle', {}).get('state', '').lower() != 'stopped' or
            execution.get('app_id') != load(attempt/'receipt.json').get('app_id')):
        raise ValueError('Import requires a reviewed discovery attempt with recorded authoritative STOPPED state')
    request = load(attempt/'campaign/query/request.json')
    runtime = load(attempt/'runtime.json')
    authorization = load(attempt/'authorization.json')
    if authorization['config']['runtime_sha256'] != sha256(attempt/'runtime.json'):
        raise ValueError('Saved runtime differs from its authorization')
    review = review_relaxation(attempt, request, runtime, require_current_code=False)
    if (execution['source_sha256'] != review['source_binding']['source_sha256'] or
            execution['request_sha256'] != sha256(attempt/'campaign/query/request.json') or
            (execution['status'] == 'measured') != review['healthy']):
        raise ValueError('Saved execution outcome does not reproduce')
    # Do not trust a separately edited summary even when raw evidence is intact.
    for key in ('healthy', 'native_status', 'report_sha256', 'source_binding', 'observations', 'screen', 'relaxation'):
        if execution.get('review', {}).get(key) != review[key]:
            raise ValueError('Saved execution review differs: '+key)
    raw = load(attempt/'collected/query/report.json')
    source = load(attempt/'campaign/structure.json')
    stages = [dict(s, stage=STAGES[s['stage']], native_stage=s['stage']) for s in raw['stages']]
    observations = {STAGES[k]: v for k, v in review['observations'].items()}
    geometry = None; metrics = {}; anisotropy = None
    if stages[0]['accepted']:
        atoms = evaluated_structure(attempt/'collected/query/dft-relax', source['geometry']['atomic_numbers'])
        geometry = dict(format='GPAC-PHYSICAL-GEOMETRY/1', units='angstrom',
            cell=atoms.cell.array.tolist(), fractional=atoms.get_scaled_positions(wrap=False).tolist(),
            atomic_numbers=atoms.numbers.tolist(), site_ids=source['geometry']['site_ids'])
        metrics = dict(volume_a3=float(atoms.get_volume()),
            Js_T=float(np.linalg.norm(observations['scalar']['total_moment_muB'])*MU0_MUB_PER_A3/atoms.get_volume()),
            max_force_ev_ang=stages[0]['max_force_ev_ang'], max_stress_gpa=stages[0]['max_stress_gpa'])
        if review['healthy'] and raw['status'] == 'screened':
            anisotropy = dict(review['screen'],
                mae_ev_cell=observations['soc-x']['total_energy_ev']-observations['soc-z']['total_energy_ev'],
                transverse_split_ev_cell=observations['soc-y']['total_energy_ev']-observations['soc-x']['total_energy_ev'],
                convention='E[x] - E[z]; positive favors z; axes refer to the relaxed cell',
                uncertainty_ev_cell=None, axes_cartesian=axes(atoms))
    manifest = load(attempt/'collected/archive-manifest.json')
    copies = {}
    for relative, expected in manifest['files'].items():
        parts = Path(relative).parts
        if len(parts) == 3 and parts[0] == 'query' and parts[1] in STAGES:
            copies[STAGES[parts[1]]+'/'+parts[2]] = (relative, expected)
    binding_files = ['execution.json', 'authorization.json', 'runtime.json', 'receipt.json',
        'preflight.json', 'campaign/execution.json', 'campaign/structure.json',
        'collected/report.json', 'collected/query/report.json', 'collected/archive-manifest.json']
    if 'review_recovery_sha256' in execution: binding_files.append(RECOVERY)
    result = dict(format=FORMAT, status=execution['status'], native_status=raw['status'],
        source_sha256=review['source_binding']['source_sha256'], structure=source,
        evaluated_geometry=geometry, geometry_origin='accepted native variable-cell relaxation' if geometry is not None else 'no accepted relaxed geometry',
        request=dict(source_sha256=review['source_binding']['source_sha256'], fixed_geometry=False,
            settings=request['settings'], kmesh=raw['prepared_bundle']['kmesh'], query=dict(kind='anisotropy'),
            basis_size='DZP', basis_energy_shift_ry=.02, pseudopotentials=request['pseudopotentials']),
        stages=stages, observations=observations, anisotropy=anisotropy, relaxation_metrics=metrics,
        relaxation_feedback=review['feedback'], physical_conditions=dict(positions='relaxed', cell='fully_variable', external_pressure_gpa=0, smearing_k=request['settings']['smearing_k']),
        executable_sha256=runtime['executable_sha256'], environment=runtime.get('environment', {}),
        source_code=raw['source_code'], artifacts={k: v[1] for k, v in copies.items()},
        provenance=dict(attempt_files={k: sha256(attempt/k) for k in binding_files},
            native_stage_mapping=STAGES, producer_code='captured launch-policy hashes; current validators replayed'),
        wall_seconds=review['native_wall_seconds'], usable_result_seconds=execution['usable_result_seconds'],
        latency_scope=('Native wall_seconds excludes overhead; usable_result_seconds includes request through saved-only recovery'
                       if 'review_recovery_sha256' in execution else
                       'Native wall_seconds excludes overhead; usable_result_seconds includes request through original review'),
        physical_accuracy_qualified=False, physical_stability_qualified=False, synthesizability_qualified=False,
        scope='Measured geometry and magnetic screen linked to the unchanged exact source. No thermodynamic, phonon or synthesis qualification.')
    return result, copies


def import_relaxation_result(attempt):
    """Create once; repeated reads verify existing bytes without replacing them."""
    attempt = Path(attempt).resolve(); output = attempt/VIEW
    if output.exists():
        result = verified_observation(output/'report.json')
    else:
        result, copies = _observation(attempt)  # Validate before creating anything.
        output.mkdir()
        for target, (relative, expected) in copies.items():
            destination = output/target; destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(attempt/'collected'/relative, destination)
            if sha256(destination) != expected:
                raise ValueError('Collected stage changed during import: '+relative)
        save(output/'report.json', result)
        result = verified_observation(output/'report.json')
    return dict(format='REMAG-DISCOVERY-IMPORT/1', report=str(output/'report.json'),
        report_sha256=sha256(output/'report.json'), source_sha256=result['source_sha256'],
        status=result['status'], native_status=result['native_status'], solver_executed=False,
        physical_stability_qualified=False)


def verified_observation(path):
    path = Path(path).resolve()
    if path.name != 'report.json' or path.parent.name != VIEW:
        raise ValueError('Discovery observation must remain beside its original attempt collection')
    report = load(path)
    expected, _ = _observation(path.parent.parent)
    if report != expected:
        raise ValueError('Discovery observation does not reproduce from its saved native evidence')
    for name, checksum in report['artifacts'].items():
        artifact = path.parent/name
        if not artifact.resolve().is_relative_to(path.parent) or sha256(artifact) != checksum:
            raise ValueError('Discovery observation artifact changed: '+name)
    return dict(report, verification=dict(mode='replayed discovery observation', omitted_artifacts=[],
        collection_manifest_sha256=report['provenance']['attempt_files']['collected/archive-manifest.json'],
        scope='Original collection and copied stage bytes verified; native/input/branch/geometry checks reproduced; no solver run'))


def metric_observation(report, stage, observable):
    """Metrics with distinct units; no missing/failed result is a numeric zero."""
    units = {'Js_T': 'T', 'volume_a3': 'angstrom^3', 'max_force_ev_ang': 'eV/angstrom', 'max_stress_gpa': 'GPa'}
    if stage != 'relaxation' or observable not in units:
        raise ValueError('Select a measured relaxation metric')
    accepted = report['status'] == 'measured' and observable in report['relaxation_metrics']
    return dict(observable=stage+'.'+observable, unit=units[observable],
        value=report['relaxation_metrics'].get(observable) if accepted else None,
        status='measured' if accepted else 'failed')
