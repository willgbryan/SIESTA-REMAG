"""Offline preparation checks and saved-only review for controlled discovery.

The shared research executor owns all allocation effects. This module cannot
launch work. A completed negative magnetic screen is a usable observation,
not a claim of stability or synthesis feasibility.
"""
from pathlib import Path
import math
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


def verify_native_reproduction(saved, replayed, settings):
    """Compare the derived axis angle in cosine space; keep all gates exact.

    Near alignment, acos magnifies one-ULP dot-product differences. Comparing
    degrees at a fixed relative tolerance would therefore be ill-conditioned.
    Only this derived angle permits eight binary64 epsilons in cosine space;
    a threshold crossing, changed acceptance/error list or other field fails.
    Original native results, hashes and measured angle are retained verbatim.
    """
    key = 'axis_deviation_deg'
    if any(k not in saved or saved[k] != v for k, v in replayed.items() if k != key):
        raise ValueError('Saved native review does not reproduce')
    if key not in saved:
        raise ValueError('Saved native review has no axis deviation')
    a, b = saved[key], replayed[key]
    if a is None or b is None:
        if a is not b: raise ValueError('Saved native axis presence differs')
        return dict(exact=True, axis_cosine_delta=None, axis_cosine_roundoff_bound=None,
                    native_thresholds_unchanged=True)
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 90 for x in (a, b)):
        raise ValueError('Invalid saved native axis deviation')
    for row in (saved, replayed):
        if ((row[key] > settings.axis_tolerance_deg) !=
                ('Final moment left the requested axis' in row['errors'])):
            raise ValueError('Saved native axis decision contradicts its metric')
    delta = abs(math.cos(math.radians(a)) - math.cos(math.radians(b)))
    bound = float(8 * np.finfo(float).eps)
    if delta > bound:
        raise ValueError('Saved native axis exceeds dot-product roundoff')
    return dict(exact=a == b, axis_cosine_delta=delta, axis_cosine_roundoff_bound=bound,
                native_thresholds_unchanged=True)


def _possible_net_changes(current, reference):
    """Reproduce 3-vector length reductions from the recorded binary64 inputs."""
    from itertools import permutations

    def lengths(row):
        vector = np.asarray(row.get('total_moment_muB', []), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError('Saved magnetic branch needs finite net vectors')
        x, y, z = map(float, vector)
        candidates = {math.hypot(x, y, z), float(np.linalg.norm(vector))}
        for a, b, c in permutations((x, y, z)):
            candidates.add(math.sqrt((a*a + b*b) + c*c))
        return candidates

    return {abs(a-b)/b for a in lengths(current) for b in lengths(reference) if b > 1e-8}


def verify_branch_reproduction(saved, replayed, current, reference, settings):
    """Allow binary64 projection roundoff, never a changed magnetic gate.

    The RMS subtracts order-one projected moments. BLAS/CPU differences can
    change that small residual at the input moments' precision, rather than at
    the residual's much smaller ULP. Keep the original bytes and compare only
    the derived RMS within eight epsilons of the input moment scale. A differing
    net fractional change must reproduce exactly from the saved 3-vectors. Both
    saved and replayed pass/fail decisions must satisfy the unchanged limits.
    """
    keys = {'passed', 'site_rms_mub', 'net_fractional_change', 'robust_site_flip'}
    if not isinstance(saved, dict) or set(saved) != keys or set(replayed) != keys:
        raise ValueError('Saved magnetic branch does not reproduce')
    for result in (saved, replayed):
        if (any(type(result[k]) is not bool for k in ('passed', 'robust_site_flip')) or
                any(type(result[k]) not in (int, float) or not np.isfinite(result[k]) or result[k] < 0
                    for k in ('site_rms_mub', 'net_fractional_change'))):
            raise ValueError('Invalid saved magnetic branch metrics')
        passed = (result['site_rms_mub'] < settings.branch_rms_mub and
                  result['net_fractional_change'] < settings.branch_net_fraction and
                  not result['robust_site_flip'])
        if result['passed'] != passed:
            raise ValueError('Saved magnetic branch decision contradicts its metrics')
    net = 'net_fractional_change'
    if saved[net] != replayed[net]:
        values = _possible_net_changes(current, reference)
        if (saved[net] not in values or replayed[net] not in values or
                (saved[net] < settings.branch_net_fraction) != (replayed[net] < settings.branch_net_fraction)):
            raise ValueError('Saved magnetic branch does not reproduce')
    exact = {'passed', 'robust_site_flip'}
    if any(saved[k] != replayed[k] for k in exact):
        raise ValueError('Saved magnetic branch does not reproduce')
    moments = [np.asarray(row['atom_moments_muB'], dtype=float) for row in (current, reference)]
    if any(m.ndim != 2 or m.shape[1] != 3 or not m.size or not np.isfinite(m).all() for m in moments):
        raise ValueError('Invalid moments for magnetic branch reproduction')
    scale = max(1., *(float(np.linalg.norm(m, axis=1).max()) for m in moments))
    bound = float(8 * np.finfo(float).eps * scale)
    delta = abs(saved['site_rms_mub'] - replayed['site_rms_mub'])
    if delta > bound:
        raise ValueError('Saved magnetic branch exceeds projection roundoff')
    return dict(exact=saved == replayed, site_rms_delta_mub=delta,
                projection_roundoff_bound_mub=bound, magnetic_thresholds_unchanged=True)


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
    from .runtime_policy import verify_saved
    try: verify_saved(runtime, top.get('runtime_verification'))
    except ValueError as exc:
        raise ValueError('Collected relaxation runtime verification is missing or differs') from exc
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
    observations, rows, input_reproduction, branch_reproduction, native_reproduction = {}, {}, {}, {}, {}
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
            from .input_reproduction import verify_prepared_input
            input_reproduction[name] = verify_prepared_input(expected, folder)
        review = native_review(folder, result, settings, relaxation=name == 'dft-relax', axis=axis)
        native_reproduction[name] = verify_native_reproduction(saved, review, settings)
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
            reference = rows.get('z', observations['dft-relax'])
            branch = branch_check(result['observables'], reference,
                                  axis, axes(relaxed)['z'] if rows else [0, 0, 1], settings)
            branch_reproduction[name] = verify_branch_reproduction(
                saved.get('branch'), branch, result['observables'], reference, settings)
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
        verification=dict(native_reviews_reproduced=True, accepted_artifact_hashes_verified=True,
                          input_reproduction=input_reproduction, branch_reproduction=branch_reproduction,
                          native_reproduction=native_reproduction),
        runtime_verification=top['runtime_verification'])
