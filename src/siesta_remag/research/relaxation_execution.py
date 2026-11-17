"""Offline preparation checks and saved-only review for controlled discovery.

The shared research executor owns all allocation effects. This module cannot
launch work. A completed negative magnetic screen is a usable observation,
not a claim of stability or synthesis feasibility.
"""
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from ..corpus import sha256
from ..discovery.collection import verify_archive
from ..discovery.common import digest, load
from ..discovery.config import Settings
from ..discovery.inputs import axes, pseudos_for, prepare_case, starting_moments, validate_atoms, MU0_MUB_PER_A3
from ..discovery.pipeline import source_identity, verify_bundle
from ..discovery.review import branch_check, directional_result, evaluated_structure, native_review
from .preflight import check_execution_hold, check_mesh_capacity, validate_runtime, preflight_input_audits


def preflight_relaxation(bundle, runtime):
    """No service calls: validate the portable preparation before reservation."""
    from ase.io import read
    bundle = Path(bundle)
    receipt = verify_bundle(bundle)
    request = load(bundle/'request.json')
    settings = Settings.from_dict(request['settings'])
    original, prepared = [read(bundle/name) for name in ('input.extxyz', 'orb/relaxed.extxyz')]
    for atoms in (original, prepared):
        validate_atoms(atoms, settings)
    if original.numbers.tolist() != prepared.numbers.tolist():
        raise ValueError('Preparation changed species or site order')
    moments = starting_moments(original)
    if not np.array_equal(moments, starting_moments(prepared)):
        raise ValueError('Preparation changed the declared magnetic starting branch')
    if pseudos_for(prepared, bundle/'pseudos') != request['pseudopotentials']:
        raise ValueError('Prepared pseudopotential identity/species differs')
    capabilities = validate_runtime(runtime)
    check_execution_hold(runtime)
    # Exercise the actual generated FDF against the same input-policy audits
    # used by the runner. A valid mesh alone cannot admit an unsupported mode.
    with TemporaryDirectory() as scratch:
        generated = prepare_case(prepared, bundle/'pseudos', Path(scratch)/'dft-relax',
                                 settings, mesh=receipt['kmesh'], relax=True, diagnostics=True)
        audits = preflight_input_audits((generated/'input.fdf').read_text(), runtime.get('environment', {}))
    result = dict(format='REMAG-RELAXATION-PREFLIGHT/1',
        request_sha256=sha256(bundle/'request.json'), prepared_sha256=sha256(bundle/'prepared.json'),
        runtime_sha256=digest(runtime), kmesh=receipt['kmesh'],
        device_scf_enabled=runtime.get('environment', {}).get('REMAG_DEVICE_SCF') == '1',
        magnetic_start=dict(moments_muB=moments.tolist(),
                            provenance='explicit' if original.has('initial_magmoms') else 'element_default'),
        physical_conditions=dict(positions='relaxed', cell='fully_variable', external_pressure_gpa=0,
                                 smearing_k=settings.smearing_k),
        paid_actions=0, input_audits=audits, physical_stability_qualified=False,
        scope='Prepared identity, species, mesh and runtime capability; native convergence and stability remain untested.')
    if result['device_scf_enabled']:
        result.update(check_mesh_capacity({'kmesh': receipt['kmesh']}, capabilities['device_scf']))
    else:
        result.update(mesh_capacity_passed=None, reason='Host-controlled path; device-cycle limit does not apply')
    return result


def verify_relaxation_source(structure, bundle):
    """Bind an exact authored source to the original, pre-relaxation geometry."""
    from ase.io import read
    from gpac.physical import verify_export
    verified = verify_export(load(structure))
    atoms = read(Path(bundle)/'input.extxyz')
    g = verified['geometry']
    if (atoms.numbers.tolist() != g['atomic_numbers'] or
            not np.allclose(atoms.cell.array, g['cell'], atol=1e-8, rtol=0) or
            not np.allclose(atoms.get_scaled_positions(wrap=False), g['fractional'], atol=1e-8, rtol=0)):
        raise ValueError('Authored source differs from discovery input geometry or site order')
    return dict(source_sha256=verified['source']['sha256'], export_sha256=verified['sha256'],
                structure_file_sha256=sha256(structure))


def review_relaxation(attempt, request, runtime, *, require_current_code=True):
    """Reproduce native/branch checks and exact-source feedback from collection."""
    from .relaxation import inspect_relaxation
    from .experiments import experiment_code
    from ase.io import read
    attempt = Path(attempt)
    root = attempt/'collected'
    verify_archive(root)
    policy = load(attempt/'campaign/execution.json')
    top = load(root/'report.json')
    report_path = root/'query/report.json'
    report = load(report_path)
    bundle = attempt/'campaign/query'
    receipt = verify_bundle(bundle)
    binding = verify_relaxation_source(attempt/'campaign/structure.json', bundle)
    # A new execution must match this checkout. Saved-only ingestion instead
    # binds the captured producer hashes to the original launch policy. It
    # still reruns the native, input, runtime and geometry checks below.
    captured_code = report.get('source_code')
    policy_code = policy.get('expected_source_code')
    if (not isinstance(captured_code, dict) or not captured_code or not isinstance(policy_code, dict) or
            any(not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                for k, v in policy_code.items()) or
            captured_code != {k: v for k, v in policy_code.items() if not k.startswith('research/')} or
            (require_current_code and (policy_code != experiment_code() or captured_code != source_identity()))):
        raise ValueError('Collected relaxation producer code differs from its launch policy')
    diagnostics = policy.get('discovery_diagnostics', False)
    if type(diagnostics) is not bool or report.get('diagnostics', False) != diagnostics:
        raise ValueError('Discovery diagnostics differ from launch policy')
    invocation_fields = dict(bundle=receipt['request_sha256'], files=receipt['files'],
        executable=runtime['executable_sha256'], runtime_env=runtime.get('environment', {}),
        threads=runtime.get('threads', 1), code=captured_code)
    if diagnostics: invocation_fields['diagnostics'] = True
    expected_invocation = digest(invocation_fields)
    if (policy.get('operation') != 'relaxation' or policy.get('source_binding') != binding or
            top.get('policy') != policy or top.get('invocation_sha256') != digest(policy) or
            report.get('prepared_bundle') != receipt or load(bundle/'request.json') != request or
            report.get('input_sha256') != request['input_sha256'] or
            report.get('invocation_sha256') != expected_invocation or
            report.get('settings') != request['settings'] or
            report.get('runtime_env') != runtime.get('environment', {}) or
            report.get('executable_sha256') != runtime['executable_sha256']):
        raise ValueError('Collected relaxation source/input/executable/runtime differs')
    expected_runtime = dict(executable_sha256=runtime['executable_sha256'],
                            capabilities=runtime.get('capabilities'), actual_files_verified=True)
    if top.get('runtime_verification') != expected_runtime:
        raise ValueError('Collected relaxation runtime verification is missing or differs')
    if top.get('experiments') != [dict(id='query', status=report['status'], report_sha256=sha256(report_path))]:
        raise ValueError('Collected relaxation report binding differs')
    for name, expected in report['accepted_artifacts'].items():
        path = (report_path.parent/name).resolve()
        if not path.is_relative_to(report_path.parent.resolve()) or sha256(path) != expected:
            raise ValueError('Accepted relaxation artifact changed: '+name)
    settings = Settings.from_dict(request['settings'])
    names = [row['stage'] for row in report['stages']]
    ordered = ['dft-relax', 'soc-z', 'soc-x', 'soc-y']
    if names != ordered[:len(names)] or not names:
        raise ValueError('Unexpected discovery stage sequence')
    observations, rows = {}, {}
    relaxed = None
    for saved in report['stages']:
        name = saved['stage']; folder = report_path.parent/name
        manifest = load(folder/'manifest.json')
        if manifest['settings'] != request['settings'] or manifest['kmesh'] != receipt['kmesh']:
            raise ValueError('Native stage settings or mesh differs')
        result = load(folder/'result.json')
        axis = None if name == 'dft-relax' else axes(relaxed)[name[4:]]
        # Recreate the generated native input from the source-bound preparation,
        # not from the collected input's own self-declared manifest.
        with TemporaryDirectory(prefix='relaxation-input-review-') as temp:
            expected = prepare_case(read(bundle/'orb/relaxed.extxyz') if axis is None else relaxed,
                bundle/'pseudos', Path(temp)/name, settings, mesh=receipt['kmesh'],
                relax=axis is None, direction=axis, restart=axis is not None,
                diagnostics=diagnostics,
                spins=None if axis is None else np.asarray(observations['dft-relax']['atom_moments_muB'])[:, 2])
            if ((expected/'input.fdf').read_bytes() != (folder/'input.fdf').read_bytes() or
                    load(expected/'manifest.json') != manifest):
                raise ValueError('Native input does not reproduce from the source-bound preparation')
        review = native_review(folder, result, settings, relaxation=name == 'dft-relax', axis=axis)
        if any(saved.get(k) != v for k, v in review.items()):
            raise ValueError('Saved native review does not reproduce')
        observations[name] = result['observables']
        if not review['accepted']:
            if name != names[-1]: raise ValueError('Execution continued after native failure')
            continue
        if name == 'dft-relax':
            relaxed = evaluated_structure(folder, load(attempt/'campaign/structure.json')['geometry']['atomic_numbers'])
        else:
            actual = evaluated_structure(folder, relaxed.numbers)
            if (not np.allclose(actual.cell.array, relaxed.cell.array, atol=2e-8, rtol=0) or
                    not np.allclose(actual.positions, relaxed.positions, atol=2e-7, rtol=0)):
                raise ValueError('SOC geometry differs from accepted relaxation')
            branch = branch_check(result['observables'], rows.get('z', observations['dft-relax']),
                                  axis, axes(relaxed)['z'] if rows else [0, 0, 1], settings)
            if saved.get('branch') != branch: raise ValueError('Saved magnetic branch does not reproduce')
            rows[name[4:]] = result['observables']
    healthy = (top['status'] == 'measured' and report['status'] in ('screened', 'screened_low_moment')
               and all(s['accepted'] and s.get('branch', {}).get('passed', True) for s in report['stages']))
    if healthy:
        js = np.linalg.norm(observations['dft-relax']['total_moment_muB'])*MU0_MUB_PER_A3/relaxed.get_volume()
        if report['status'] == 'screened_low_moment':
            if names != ['dft-relax'] or js >= settings.minimum_js_t or report['screen'] is not None:
                raise ValueError('Low-moment screen does not reproduce')
        elif (names != ordered or js < settings.minimum_js_t or
              report['screen'] != directional_result(rows, relaxed.get_volume()) or
              len({(tuple(s['native_grid']), s['native_kpoints']) for s in report['stages'][1:]}) != 1):
            raise ValueError('Directional screen does not reproduce')
    feedback = inspect_relaxation(load(attempt/'campaign/structure.json'), bundle, report_path) if healthy else None
    return dict(healthy=healthy, native_status=report['status'], report='collected/query/report.json',
        report_sha256=sha256(report_path), collection='collected/archive-manifest.json', spectra=[],
        observations=observations, screen=report.get('screen'), relaxation=report.get('relaxation'),
        source_binding=binding, feedback=feedback, physical_stability_qualified=False,
        native_wall_seconds=report.get('whole_pipeline_dft_seconds'), remote_timing=top.get('timing'),
        verification=dict(native_reviews_reproduced=True, accepted_artifact_hashes_verified=True),
        runtime_verification=top['runtime_verification'])
