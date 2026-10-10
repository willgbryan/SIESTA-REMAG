#!/usr/bin/env python3
"""Audit retained eigenvector configuration, scientific controls and timing.

Configured solve counts are not independent kernel execution counters. Each
candidate repetition shares one CPU reference and one cache-disabled control;
end-to-end ratios include SCF trajectory differences, setup and final outputs.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import tarfile

from siesta_remag.compare import compare
from siesta_remag.output import read_eigenvalues
from siesta_remag.vector_cache import record_errors


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    meta, ledger = read(root/'metadata.json'), read(root/'ledger.json')
    errors, hashes, checks, groups = [], {}, [], []
    for name in ('source-verification.json', 'component-evidence-verification.json', 'scientific-recheck.json',
                 'device-cycle-verification.json', 'terminal-export-verification.json', 'integral-force-material-verification.json'):
        if not (root/name).is_file() or read(root/name).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
        if (root/name).is_file():
            hashes[name] = sha(root/name)
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for fn in (record_errors, compare, read_eigenvalues):
            path = Path(inspect.getsourcefile(fn))
            if hashlib.sha256(archive.extractfile('src/siesta_remag/'+path.name).read()).hexdigest() != sha(path):
                raise ValueError('Loaded cache/scientific validator differs from executed archive')
    roles = ['candidate']+['candidate-repeat'+str(i) for i in range(2, meta['candidate_repeats']+1)]
    wanted = {(case, role) for case in meta['cases'] for role in roles+['candidate-without-vector-cache']}
    recorded = [tuple(Path(row.get('candidate_path', row['benchmark']+'/candidate')).parts) for row in ledger]
    if set(recorded) != wanted or len(recorded) != len(wanted) or not all(row['passed'] for row in ledger):
        errors.append('Incomplete/failed full cache/control corpus')
    if not meta.get('vector_cache') or not meta.get('vector_cache_ablation') or not meta.get('device_scf'):
        errors.append('Required cache/control execution modes missing')
    for case in meta['cases']:
        reference = read(root/case/'reference/result.json')
        control_dir = root/case/'candidate-without-vector-cache'
        control = read(control_dir/'result.json')
        records, direct, local = [], [], []
        for role in roles+['candidate-without-vector-cache']:
            folder = root/case/role
            result = read(folder/'result.json')
            native = read(folder/'remag-device-scf.json')
            enabled = role != 'candidate-without-vector-cache'
            issues = record_errors(native['cycles'], read_eigenvalues(folder/'remag.EIG'), result['benchmark'], enabled)
            if result['runtime_settings'].get('REMAG_DEVICE_VECTOR_CACHE') != str(int(enabled)) or not result['valid']:
                issues.append('Cache mode or execution validity differs')
            if native['cycles'] != result['telemetry']['gpu_device_cycle']['cycles']:
                issues.append('Native configured vector bank differs from result')
            if enabled:
                settings = dict(result['runtime_settings']); settings['REMAG_DEVICE_VECTOR_CACHE'] = '0'
                if settings != control['runtime_settings'] or result['executable'] != control['executable']:
                    issues.append('Cache control differs in other runtime settings or executable')
                comparison = compare(control, result, reference_dir=control_dir, candidate_dir=folder)
                if not comparison['passed'] or not comparison['performance_comparable']:
                    issues.append('Direct scientific/timing control comparison failed')
                if role == 'candidate' and comparison != read(root/('ablation-vector-cache-'+case+'.json')):
                    issues.append('Recorded direct cache comparison was not reproduced')
                direct.append(comparison)
                records.append(result)
            local.append(dict(case=case, role=role, passed=not issues, errors=issues,
                              configurations=[c['eigensystems'] for c in native['cycles']],
                              evidence_sha256={str(p.relative_to(root)):sha(p) for p in
                                               (folder/'result.json', folder/'remag-device-scf.json', folder/'remag.EIG')}))
        checks.extend(local)
        times = [r['wall_seconds'] for r in records]
        median = statistics.median(times)
        cpu_rows = [row for row in ledger if Path(row.get('candidate_path', row['benchmark']+'/candidate')).parts == (case, 'candidate') or
                    (Path(row.get('candidate_path', row['benchmark']+'/candidate')).parts[0] == case and 'candidate-repeat' in row.get('candidate_path', ''))]
        local_passed = all(c['passed'] for c in local)
        groups.append(dict(case=case, passed=local_passed, candidate_median_seconds=median, candidate_range_seconds=[min(times), max(times)],
                           control_seconds=control['wall_seconds'], cpu_seconds=reference['wall_seconds'],
                           control_over_median=control['wall_seconds']/median if local_passed else None,
                           cpu_over_median=reference['wall_seconds']/median if local_passed and all(row['performance_comparable'] for row in cpu_rows) else None,
                           cpu_iterations=reference['observables']['scf_iterations'], control_iterations=control['observables']['scf_iterations'],
                           candidate_iterations=[r['observables']['scf_iterations'] for r in records],
                           control_device_seconds=control['profile']['device_cycle_seconds'],
                           candidate_device_seconds=[r['profile']['device_cycle_seconds'] for r in records], direct_comparisons=direct))
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors,
                checks=checks, groups=groups, prerequisite_sha256=hashes, verifier_sha256=sha(Path(__file__)),
                independent_kernel_counts=False, strict_residency_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'vector-cache-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'vector-cache-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Vector cache:', len(result['checks']), 'calculations; passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
