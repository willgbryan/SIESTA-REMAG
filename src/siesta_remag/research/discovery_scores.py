"""Saved candidate cohorts and imported GPU allocation measurements.

These reports quantify a declared finite validation protocol, not physical
accuracy, chemical stability, cryptographic independence or provider billing.
"""
from datetime import datetime, timedelta
import json
import math
import re

from ..corpus import sha256
from ..discovery.common import load, digest
from .convergence import VARIABLES, assess, numerical_configuration

ARMS = ('scalar', 'physics')


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value)-set(required)-set(optional):
        raise ValueError('Missing or unknown discovery evaluation fields')


def _time(value):
    if not isinstance(value, str): raise ValueError('Require an ISO timestamp with timezone')
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.utcoffset() is None: raise ValueError('Require an ISO timestamp with timezone')
    return t


def report_interval(report):
    """Missing legacy timing remains unknown, never substituted by a timeout."""
    start, seconds = report.get('started_at'), report.get('wall_seconds')
    if start is None or seconds is None: return None
    _time(start)
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid measured report duration')
    return dict(started_at=start, wall_seconds=seconds)


def validate_policy(value):
    """Freeze comparable scoring rules before either policy arm runs."""
    from gpac.exact import validate_proposal_policy
    _fields(value, {'assessment_policy', 'excluded_findings', 'controls'},
        {'candidate_equivalence','stability','robustness'})
    p = json.loads(json.dumps(value, allow_nan=False))
    p['assessment_policy'] = validate_proposal_policy(p['assessment_policy'])
    if 'max_stretch_deviation' in p['assessment_policy']:
        raise ValueError('Cohort assessment has no common reference cell; omit reference-strain rule')
    codes = p['excluded_findings']
    allowed = {'short_distance', 'coordination_range', 'same_species_isolation', 'cell_complexity',
               'orbit_complexity', 'translational_redundancy', 'family_composition', 'family_authored_hall'}
    if not isinstance(codes, list) or any(not isinstance(c, str) or c not in allowed for c in codes) or len(set(codes)) != len(codes):
        raise ValueError('Declare distinct supported assessment finding codes')
    controls = p['controls']
    if not isinstance(controls, list) or not 1 <= len(controls) <= len(VARIABLES):
        raise ValueError('Declare 1..6 finite numerical controls')
    seen = set()
    for c in controls:
        _fields(c, {'varied', 'tolerance_ev', 'successive_changes', 'transverse_tolerance_ev'})
        if c['varied'] not in VARIABLES or c['varied'] in seen: raise ValueError('Control variables must be distinct')
        seen.add(c['varied'])
        if type(c['successive_changes']) is not int or not 1 <= c['successive_changes'] <= 10:
            raise ValueError('Require 1..10 successive changes')
        for k in ('tolerance_ev', 'transverse_tolerance_ev'):
            if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
                raise ValueError('Require positive finite validation tolerances')
    from .candidate_metrics import validate_metrics
    validate_metrics(p)
    return p


def _endpoint(root, value, source, export):
    from .study import _path
    from .tools import _verified_report
    _fields(value, {'report'}, {'collection'})
    path = _path(root, value['report'])
    r = _verified_report(path, _path(root, value['collection']) if 'collection' in value else None)
    if r['source_sha256'] != source or r['structure'] != export:
        raise ValueError('Candidate validation source or geometry differs')
    healthy = (r['status'] == 'measured' and r['request'].get('fixed_geometry') is True
        and [s['stage'] for s in r['stages']] == ['scalar', 'soc-z', 'soc-x', 'soc-y']
        and all(s['accepted'] and s.get('branch', {}).get('passed', True) for s in r['stages']))
    return r, dict(report_sha256=sha256(path), report=str(path.relative_to(root)), interval=report_interval(r),
                   native_status=r['status'], accepted=healthy, verification=r['verification'])


def score_candidates(root, registration, arms, manifest):
    """Score all successful author_structure sources in the frozen arm history.

    Unknown outcomes remain pending; no caller-supplied `validated` boolean is
    accepted. Controls must end at the scored report. Their branch, source and
    runtime checks are reproduced from saved reports.
    """
    from .study import _path
    from .series import _branches
    from ..discovery.config import Settings
    from gpac.physical import verify_export
    from gpac.proposal import assess as assess_geometry
    policy = registration['definition']['evaluation'].get('discovery')
    if policy is None: raise ValueError('Discovery criteria were not registered before the sessions')
    policy = validate_policy(policy)
    _fields(manifest, {'format', 'arms'}, {'reference_phases'})
    if manifest['format'] != 'REMAG-STUDY-CANDIDATE-EVIDENCE/1' or set(manifest['arms']) != set(ARMS):
        raise ValueError('Expected candidate evidence for both arms')
    if 'reference_phases' in manifest and 'stability' not in policy:
        raise ValueError('Reference stability criteria were not registered')
    results = {}; expected_reports = {a:set() for a in ARMS}; windows = {}
    phases=None
    if 'stability' in policy:
        from .candidate_metrics import load_reference_phases
        phases=load_reference_phases(root,policy['stability'],manifest.get('reference_phases',{}))
        for arm in ARMS:expected_reports[arm].update(phases[2])
        windows.update(phases[3])
    for arm in ARMS:
        where = _path(root, registration['definition']['arms'][arm]['workspace'])
        inventory = arms[arm].get('authored_candidates')
        if inventory is None: raise ValueError('Candidate scoring requires audited terminal session histories')
        candidates = {}
        for c in inventory: candidates.setdefault(c['source_sha256'], []).append(c)
        supplied = manifest['arms'][arm]
        if not isinstance(supplied, dict) or set(supplied)-set(candidates):
            raise ValueError('Validation source was not authored in the frozen arm history')
        rows = []
        for source, authorship in sorted(candidates.items()):
            export = verify_export(load(_path(where, authorship[0]['structure_path'])))
            geometry = assess_geometry(export['source']['program'], policy['assessment_policy'])
            finding_codes = sorted({f['code'] for f in geometry['findings']})
            plausible = not set(finding_codes) & set(policy['excluded_findings'])
            row = dict(source_sha256=source, authoring_steps=[a['step'] for a in authorship],
                status='pending', plausible_under_declared_geometry_policy=plausible,
                assessment=geometry, controls={}, validated_under_declared_controls=False,
                properties=None, thermodynamic_stability=None, magnetic_ground_state=None,
                structural_robustness=None, physical_accuracy_qualified=False)
            rows.append(row)
            if 'candidate_equivalence' in policy:
                from gpac.candidate_identity import candidate_key
                row['candidate_equivalence']=candidate_key(export['source']['program'],length_unit='angstrom')
            if source not in supplied: continue
            sample = supplied[source]; _fields(sample, {'report', 'controls'}, {'collection','robustness'})
            if 'robustness' in sample and 'robustness' not in policy:
                raise ValueError('Structural robustness criteria were not registered')
            if not isinstance(sample['controls'], dict) or set(sample['controls'])-{c['varied'] for c in policy['controls']}:
                raise ValueError('Unregistered candidate validation control')
            r, bound = _endpoint(root, {k:v for k,v in sample.items() if k in ('report','collection')}, source, export)
            expected_reports[arm].add(bound['report_sha256']); row['validation_report'] = bound
            windows[bound['report_sha256']] = bound['interval']
            # Distinct timestamps/hashes are an auditable post-session condition,
            # not proof against copied artifacts or hidden external knowledge.
            after_session = (None if r.get('started_at') is None else
                _time(r['started_at']) >= _time(arms[arm]['finished_at']))
            row['post_session_run_recorded'] = after_session
            from .candidate_metrics import properties,assess_stability,assess_robustness
            if 'robustness' in policy:
                metric, hashes, times=assess_robustness(root,r,policy['robustness'],sample.get('robustness',{}),baseline_accepted=bound['accepted'])
                row['structural_robustness']=metric;expected_reports[arm].update(hashes);windows.update(times)
            # Retain every supplied refinement in cost coverage even when the
            # candidate's final calculation failed. A failed result earns no
            # validation credit, but its preceding calculations still cost time.
            control_reports = {}
            for c in policy['controls']:
                name = c['varied']; endpoints = sample['controls'].get(name)
                if endpoints is None: continue
                if not isinstance(endpoints, list) or not 2 <= len(endpoints) <= 16:
                    raise ValueError('Each numerical control needs 2..16 ordered saved reports')
                reports = []; bindings = []
                for ep in endpoints:
                    rr, bb = _endpoint(root, ep, source, export); reports.append(rr); bindings.append(bb)
                    expected_reports[arm].add(bb['report_sha256'])
                    windows[bb['report_sha256']] = bb['interval']
                if bindings[-1]['report_sha256'] != bound['report_sha256']:
                    raise ValueError('Each validation ladder must end at the scored candidate report')
                if len({b['report_sha256'] for b in bindings}) != len(bindings):
                    raise ValueError('Repeated reports are not independent refinements')
                control_reports[name] = (reports, bindings)
            if not bound['accepted']:
                row.update(status='failed_native'); continue
            props=properties(r);n=props['site_count']
            row['properties'] = props
            if 'stability' in policy:
                metric, hashes, times=assess_stability(root,r,policy['stability'],manifest.get('reference_phases',{}),prepared=phases)
                row['thermodynamic_stability']=metric;expected_reports[arm].update(hashes);windows.update(times)
            row['comparison_configuration'] = {k:v for k,v in numerical_configuration(r).items() if k!='source_sha256'}
            identity = list(range(n)); checks = []
            for c in policy['controls']:
                name = c['varied']; endpoints = sample['controls'].get(name)
                if endpoints is None:
                    row['controls'][name] = dict(status='pending'); checks.append(False); continue
                reports, bindings = control_reports[name]
                if not all(b['accepted'] for b in bindings):
                    row['controls'][name] = dict(status='failed_native', reports=bindings); checks.append(False); continue
                stability = assess(reports, **c)
                branches = [_branches(rr, r, identity, identity, ['scalar','soc-z','soc-x','soc-y'],
                    Settings.from_dict(r['request']['settings'])) for rr in reports]
                passed = stability['observed_stability_passed'] and all(v['passed'] for b in branches for v in b.values())
                # Retain actual file hashes even though the reused assessor was
                # invoked with verified dictionaries rather than unverified paths.
                row['controls'][name] = dict(status='passed' if passed else 'failed', reports=bindings,
                    numerical_stability=stability, magnetic_branches=branches)
                checks.append(passed)
            row['validated_under_declared_controls'] = after_session is True and all(checks)
            row['status'] = 'validated' if row['validated_under_declared_controls'] else 'unvalidated'
        equivalence_groups={}
        if 'candidate_equivalence' in policy:
            for row in rows:
                equivalence_groups.setdefault(row['candidate_equivalence']['key_sha256'],[]).append(row['source_sha256'])
        screened=[r for r in rows if r['validated_under_declared_controls'] and r['plausible_under_declared_geometry_policy']
            and ('stability' not in policy or (r['thermodynamic_stability'] or {}).get('observed_static_screen_passed') is True)
            and ('robustness' not in policy or (r['structural_robustness'] or {}).get('observed_robustness_passed') is True)]
        qualified = [r for r in rows if r['validated_under_declared_controls']]
        eligible = [r for r in qualified if r['plausible_under_declared_geometry_policy']]
        best = {}
        for label in ('E','F'):
            key = label+'_anisotropy_MJ_m3'
            ordered = sorted(eligible, key=lambda r: (-r['properties'][key], r['source_sha256']))
            # Different numerical models/runtimes are separate comparisons.
            groups = {}
            for r in ordered:
                key = digest(r['comparison_configuration'])
                group = groups.setdefault(key, dict(configuration=r['comparison_configuration'], candidates=[]))
                group['candidates'].append(dict(source_sha256=r['source_sha256'], **r['properties']))
            best[label] = groups
        results[arm] = dict(rows=rows, authored_attempts=len(inventory), distinct_exact_sources=len(rows),
            failed_authoring_attempts=arms[arm].get('failed_authoring_attempts'),
            geometrically_plausible_exact_sources=sum(r['plausible_under_declared_geometry_policy'] for r in rows),
            validated_exact_sources=len(qualified), validated_plausible_exact_sources=len(eligible),
            pending_validation=sum(r['status']=='pending' for r in rows),
            validated_density_rankings=best, crystallographically_distinct_count=None,
            candidate_equivalence_policy=policy.get('candidate_equivalence'),candidate_equivalence_groups=equivalence_groups,
            distinct_under_declared_policy=len(equivalence_groups) if 'candidate_equivalence' in policy else None,
            plausible_groups_under_declared_policy=len({r['candidate_equivalence']['key_sha256'] for r in rows if r['plausible_under_declared_geometry_policy']}) if 'candidate_equivalence' in policy else None,
            screened_exact_sources=[r['source_sha256'] for r in screened],
            screened_groups_under_declared_policy=len({r['candidate_equivalence']['key_sha256'] for r in screened}) if 'candidate_equivalence' in policy else None,
            validation_controls=policy['controls'])
    return dict(format='REMAG-STUDY-CANDIDATE-SCORES/1', arms=results, evidence_sha256=digest(manifest),
        required_usage_reports={a:sorted(v) for a,v in expected_reports.items()},
        required_usage_intervals=windows,
        scope='All successful persisted author_structure outputs, grouped by exact source hash; imports/repeated sources are not novel discoveries. Rankings use energy density, preserve signed E/F separately and require all declared finite controls plus a recorded post-session native run. Not fitted K1, unique-crystal recognition, provider authentication or physical/chemical stability qualification.')


def score_usage(root, arms, manifest, *, required_reports=None, report_windows=None):
    """Read normalized allocation rows, bound to explicitly imported raw files.

    Source bytes are retained and hashed, but the importer cannot authenticate
    a cloud provider or prove that an operator supplied every outside expense.
    Each input record selects exactly one row from a JSON source file by index.
    Shared acquisition/validation time is charged one half to each arm.
    """
    from .study import _path
    _fields(manifest, {'format', 'records'})
    if manifest['format'] != 'REMAG-STUDY-GPU-USAGE/1' or not isinstance(manifest['records'], list) or len(manifest['records']) > 4096:
        raise ValueError('Expected a bounded allocation-usage manifest')
    required_reports = required_reports or {a:[] for a in ARMS}
    report_windows = report_windows or {}
    seen = set(); imported = set(); rows = []; covered = {a:set() for a in ARMS}; apps = {a:set() for a in ARMS}
    totals = {a:{} for a in ARMS}; intervals = {}
    for item in manifest['records']:
        _fields(item, {'arm', 'phase', 'source', 'source_sha256', 'row'})
        arm = item['arm']; phase = item['phase']
        if arm not in (*ARMS, 'shared') or phase not in ('initial_data', 'search', 'validation', 'grading'):
            raise ValueError('Unknown allocation arm or phase')
        if arm=='shared' and phase=='search': raise ValueError('Policy search allocations cannot be shared')
        path = _path(root, item['source'])
        if sha256(path) != item['source_sha256']: raise ValueError('Usage source hash differs')
        source = load(path)
        _fields(source, {'format', 'provider', 'allocations'})
        if source['format'] != 'REMAG-ALLOCATION-MEASUREMENTS/1' or not isinstance(source['provider'], str) or not source['provider']:
            raise ValueError('Identify the allocation measurement provider')
        if not isinstance(source['allocations'], list) or not 0 < len(source['allocations']) <= 4096:
            raise ValueError('Invalid measured allocation rows')
        i = item['row']
        if type(i) is not int or not 0 <= i < len(source['allocations']): raise ValueError('Unknown allocation row')
        row = source['allocations'][i]
        _fields(row, {'allocation_id', 'app_id', 'gpu_type', 'gpu_count', 'started_at', 'stopped_at', 'report_sha256s'})
        for k in ('allocation_id','app_id','gpu_type'):
            if not isinstance(row[k], str) or not row[k]: raise ValueError('Allocation identities must be nonempty')
        identity = (source['provider'], row['allocation_id'])
        if identity in seen or (item['source_sha256'], i) in imported: raise ValueError('Duplicate measured allocation')
        seen.add(identity); imported.add((item['source_sha256'], i))
        if type(row['gpu_count']) is not int or not 1 <= row['gpu_count'] <= 1024: raise ValueError('Positive integer GPU count required')
        start, end = _time(row['started_at']), _time(row['stopped_at'])
        if end <= start: raise ValueError('Allocation must have a positive terminal interval')
        # One allocation may be represented by disjoint hardware segments. An
        # overlapping segment for the same app/device type double-counts time.
        key = (source['provider'], row['app_id'], row['gpu_type'])
        if any(start < b and a < end for a,b in intervals.get(key, [])):
            raise ValueError('Overlapping allocation intervals for one app and GPU type')
        intervals.setdefault(key, []).append((start, end))
        hashes = row['report_sha256s']
        if not isinstance(hashes, list) or len(hashes)>512 or any(not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h) for h in hashes) or len(set(hashes))!=len(hashes):
            raise ValueError('Declare distinct measured report hashes')
        hours = (end-start).total_seconds()*row['gpu_count']/3600
        charged = ARMS if arm=='shared' else (arm,)
        for a in charged:
            totals[a][row['gpu_type']] = totals[a].get(row['gpu_type'], 0) + hours/len(charged)
            covered[a].update(hashes)
            if phase=='search': apps[a].add(row['app_id'])
        rows.append(dict(arm=arm, phase=phase, provider=source['provider'], **row,
            gpu_hours=hours, source=str(path.relative_to(root)), source_sha256=sha256(path), source_row=i))
    result = {}
    for arm in ARMS:
        expected_apps = set(arms[arm].get('native_app_ids', []))
        expected_reports = set(required_reports.get(arm, [])) | set(arms[arm].get('native_report_sha256s', []))
        missing_apps = sorted(expected_apps-apps[arm]); missing_reports = sorted(expected_reports-covered[arm])
        missing_times = []
        for h in sorted(expected_reports & covered[arm]):
            window = report_windows.get(h)
            if window is None: missing_times.append(h); continue
            begin = _time(window['started_at']); finish = begin + timedelta(seconds=window['wall_seconds'])
            by_app = {}
            for row in rows:
                if row['arm'] in (arm,'shared') and h in row['report_sha256s']:
                    by_app.setdefault((row['provider'],row['app_id']), []).append((_time(row['started_at']),_time(row['stopped_at'])))
            contained = False
            for segments in by_app.values():
                end = begin
                for a,b in sorted(segments):
                    if a <= end and b > end: end = b
                contained |= end >= finish and any(a<=begin<b for a,b in segments)
            if not contained: raise ValueError('Allocation intervals do not contain the bound native report')
        complete = (arms[arm].get('session_verified', False) and arms[arm].get('unresolved_attempts') == 0
            and not missing_apps and not missing_reports
            and not missing_times
            and len(expected_apps) == arms[arm].get('native_submissions'))
        result[arm] = dict(imported_gpu_hours_by_type=totals[arm], imported_gpu_hours=sum(totals[arm].values()),
            actual_gpu_hours=sum(totals[arm].values()) if complete else None, known_evidence_coverage_complete=complete,
            missing_app_ids=missing_apps, missing_report_sha256s=missing_reports,
            missing_report_time_bindings=missing_times, actual_billed_usd=None)
    return dict(format='REMAG-STUDY-GPU-USAGE-SCORES/1', arms=result, records=rows, evidence_sha256=digest(manifest),
        shared_allocation_fraction_per_arm=.5, provider_authenticated=False, outside_usage_completeness_verified=False,
        same_imported_gpu_time_by_type=(all(r['known_evidence_coverage_complete'] for r in result.values()) and totals['scalar']==totals['physics']),
        scope='Imported allocation count × terminal duration, including idle time inside that interval; retained by GPU type. Shared initial/validation/grading acquisition is split equally. Known apps/reports checked; no inference from reservations, native wall time or invoices. Provider authenticity and unreported outside usage require external evidence.')
