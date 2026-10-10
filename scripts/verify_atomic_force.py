#!/usr/bin/env python3
"""Reproduce material core/neutral force shadows and final consumer coverage.

CPU core/neutral results remain authoritative in this diagnostic. This verifies
retained electronic-input consumption, not complete force or SCF residency.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

from siesta_remag.atomic_force import shadow_errors


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    metadata = json.loads((root/'metadata.json').read_text())
    ledger = json.loads((root/'ledger.json').read_text())
    validator = sha(Path(inspect.getsourcefile(shadow_errors)))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        with archive.extractfile('src/siesta_remag/atomic_force.py') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != validator:
                raise ValueError('Loaded shadow validator differs from retained source')
    checks, seen = [], set()
    for path in sorted(root.glob('*/*/result.json')):
        result = json.loads(path.read_text())
        if result.get('runtime_settings', {}).get('REMAG_ATOMIC_FORCE_SHADOW') != '1':
            continue
        folder = path.parent
        case, role = folder.parent.name, folder.name
        seen.add(case)
        faults, rows, grid_rows, hashes = [], [], [], {'result.json': sha(path)}
        for pattern, destination in (('remag-atomic-force-shadow-*.json', rows), ('remag-grid-force-*.json', grid_rows)):
            for raw in sorted(folder.glob(pattern)):
                destination.append(json.loads(raw.read_text()))
                hashes[raw.name] = sha(raw)
        faults.extend(shadow_errors(rows, grid_rows, result['benchmark']['atoms']))
        scientific = [r for r in ledger if r.get('candidate_path', r['benchmark']+'/candidate') == case+'/'+role]
        if not result['valid'] or len(scientific) != 1 or not scientific[0]['passed']:
            faults.append('Upstream scientific comparison failed')
        if scientific and scientific[0].get('performance_comparable') is not False:
            faults.append('Shadow timings were not excluded')
        telemetry = result['telemetry']
        if rows != telemetry.get('atomic_force_shadow') or grid_rows != telemetry.get('grid_force'):
            faults.append('Raw material records differ from telemetry')
        traffic = dict(host_to_device_bytes=sum(r['setup_bytes'] for r in rows),
                       device_to_host_bytes=sum(r['export_bytes'] for r in rows))
        if traffic != telemetry.get('gpu_atomic_grid_force_shadow') or telemetry.get('residency', {}).get('accounting_errors'):
            faults.append('Atomic-force transfer accounting failed')
        checks.append(dict(case=case, role=role, passed=not faults, errors=faults, contractions=len(rows),
            max_force_error_ry_per_bohr=max((r['max_force_error_ry_per_bohr'] for r in rows), default=None),
            max_stress_error_ry_per_bohr3=max((r['max_stress_error_ry_per_bohr3'] for r in rows), default=None),
            traffic=traffic, evidence_sha256=hashes))
    errors = []
    if not metadata.get('atomic_force_shadow') or seen != set(metadata['cases']):
        errors.append('Shadow validation did not cover every requested case')
    # The independent production verifier includes both directional MAE pairs,
    # full ledger completion and each authoritative dfscf publication.
    gate = root/'grid-force-verification.json'
    if not gate.is_file() or not json.loads(gate.read_text())['passed']:
        errors.append('Authoritative dfscf/scientific verification is missing or failed')
    return dict(scope=__doc__, passed=bool(checks) and not errors and all(c['passed'] for c in checks),
                errors=errors, checks=checks, final_force_residency_verified=False,
                ledger_sha256=sha(root/'ledger.json'), metadata_sha256=sha(root/'metadata.json'),
                grid_force_verification_sha256=sha(gate) if gate.is_file() else None,
                implementation_archive_sha256=sha(root/'implementation.tar.gz'),
                validator_sha256=validator, verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'atomic-force-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'atomic-force-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Material core/neutral shadows:', len(result['checks']), 'calculations; passed=', result['passed'])
    if not result['passed']:
        raise SystemExit(1)
