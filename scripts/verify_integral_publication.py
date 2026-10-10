#!/usr/bin/env python3
"""Reconstruct terminal publication arithmetic, failure guards and copy totals.

Does not import the GPU checker's validation. Empty integral geometries test
the common owner; nonempty kinetic/nonlocal/overlap physics is checked by the
separate component and material audits. This is explicit copy accounting only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RECOVERY = '''missing_hartree early_grid early_nonlocal early_collect early_publish
unrecorded_kinetic changed_initial_force retained_pair nonlocal_unmarked_reuse nonlocal_retained_generator grid_before_hartree duplicate_hartree
reset_before_collection changed_publication_input duplicate_collect
reset_before_publication duplicate_publish'''.split()
SCENARIOS = {'normal', 'recovery', 'failed_status', 'failed_nonfinite', 'stale'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    folder = root/'integral-publication-check'
    report, meta = read(folder/'check.json'), read(root/'metadata.json')
    errors, checks, seen = [], [], set()
    if report['passed'] is not True or report['checker_sha256'] != meta.get('integral_publication_check_sha256'):
        errors.append('Failed check or checker binding')
    for row in report['checks']:
        name, spins, core = row['scenario'], row['spins'], row['partial_core']
        key = (name, spins, core)
        issues = []
        if key in seen or name not in SCENARIOS or spins not in (1, 2, 4, 8) or core not in (0, 1):
            issues.append('Unexpected or duplicate fixture')
        seen.add(key)
        files = {k: folder/row[k+'_file'] for k in ('report', 'residency', 'log')}
        if any(sha(p) != row[k+'_sha256'] for k, p in files.items()):
            issues.append('Raw evidence binding differs')
        p, raw = read(files['report']), read(files['residency'])
        failed = name not in ('normal', 'recovery')
        count = 2 if name == 'recovery' else 1
        rejected = RECOVERY*2 if name == 'recovery' else (
            'failed_collect retry_collect failed_publish failed_reset replace_failed_state'.split() if failed else [])
        if (p['scenario'], p['spins'], p['partial_core'], p['rejections']) != (name, spins, core, rejected):
            issues.append('Missing lifecycle/recovery coverage')
        # Contract density and potential with the upstream SOC hermitization,
        # and apply each atomic/grid term separately in Fortran tensor order.
        rho = [.4, .3, .16 if spins == 8 else .2, .02 if spins == 8 else -.1]
        potential = [1.2, 1.8, .2, .3]
        df = .14*(sum(rho[i]*potential[i]*(2 if i > 1 else 1) for i in range(min(spins, 4)))-.18)
        neutral = .08
        partial = .1*((1.2 if spins == 1 else 1.5)-.6) if core else 0.
        change = [partial*a+neutral*b+df*c for a, b, c in zip((.2, -.3, .4), (-.2, .5, -.1), (.3, .2, -.4))]
        expected = [0.]*10+[initial+c for initial, c in zip((.1, -.2, .3), change)]
        expected += [.01*(3*i+j+1)+.001*(3*i+j+1)+dx*change[i]/2
                     for i in range(3) for j, dx in enumerate((.4, .2, -.3))]
        if len(p['cycles']) != count:
            issues.append('Missing geometry')
        error = 0.
        for g, cycle in enumerate(p['cycles']):
            wanted = dict(epoch=g+1, published=not failed, failed=failed, captured_nodes=6+core,
                          successful_export_bytes=0 if failed else 180)
            if any(cycle[k] != v for k, v in wanted.items()):
                issues.append('Owner completion/capture differs')
            actual = np.array(cycle['output'], dtype=float)
            if failed:
                if actual.size:
                    issues.append('Rejected owner leaked output')
            elif actual.shape != (22,) or not np.isfinite(actual).all():
                issues.append('Incomplete/nonfinite publication')
            else:
                error = max(error, float(np.max(np.abs(actual-expected))))
        if error >= 1e-12 or abs(error-row['max_absolute_error']) >= 1e-15:
            issues.append('Retained analytic energy/force/stress mismatch')
        def counter(h=0, d=0, n=0):
            return dict(host_to_device_bytes=h, device_to_host_bytes=d, copy_calls=n)
        totals = {k: counter() for k in ('grid_flow_geometry', 'final_force_output')}
        for interval in raw['intervals']:
            for boundary, traffic in interval['boundaries'].items():
                if boundary not in totals or interval['phase'] != 'post_scf':
                    issues.append('Unexpected transfer or phase')
                    continue
                for k, v in traffic.items():
                    totals[boundary][k] += v
        # Status failure may export only status; nonfinite output is rejected
        # after the vector copy. Retries may not repeat either export.
        downloads = 0 if name == 'stale' else 4 if name == 'failed_status' else 180*count
        copies = 0 if name == 'stale' else 1 if name == 'failed_status' else 2*count
        wanted = dict(grid_flow_geometry=counter(180*count, 0, 2*count), final_force_output=counter(0, downloads, copies))
        if totals != wanted:
            issues.append('Shape-derived transfer/retry accounting differs')
        if (raw['schema_version'] != 2 or raw['phase_markers_seen'] is not True or
                raw['geometry_epochs'] != (2 if name == 'stale' else count) or raw['strict_residency_verified'] is not False or
                any(raw[k] is not None for k in ('library_transfers', 'unified_memory_migrations', 'file_io'))):
            issues.append('Accounting coverage/scope differs')
        if row['native_return'] != 0 or row['passed'] is not True or row['errors']:
            issues.append('Native worker/check failed')
        checks.append(dict(scenario=name, spins=spins, partial_core=core, passed=not issues, errors=issues,
                           geometries=count, rejections=len(rejected), max_absolute_error=error, transfers=totals,
                           evidence_sha256={str(p.relative_to(root)): sha(p) for p in files.values()}))
    if len(checks) != 40 or seen != {(n, s, c) for n in SCENARIOS for s in (1, 2, 4, 8) for c in (0, 1)}:
        errors.append('Incomplete spin/core/scenario coverage')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                verifier_sha256=sha(Path(__file__)), report_sha256=sha(folder/'check.json'),
                metadata_sha256=sha(root/'metadata.json'), independent_cuda_trace=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'integral-publication-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'integral-publication-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Integral publication audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
