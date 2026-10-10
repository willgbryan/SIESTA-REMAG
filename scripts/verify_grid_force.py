#!/usr/bin/env python3
"""Audit authoritative GPU dfscf publication and its upstream scientific ledger.

The native record establishes execution/ownership, not scientific agreement.
Agreement comes from the unchanged end-to-end comparator, including forces,
stress, relaxation, density and directional MAE. This does not establish full
final-observable residency: derivative staging and legacy exports remain.
"""
import argparse
import hashlib
import inspect
import json
import tarfile
from pathlib import Path

from siesta_remag.grid_force import record_errors


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    ledger = json.loads((root/'ledger.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    validator_hash = sha(Path(inspect.getsourcefile(record_errors)))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        with archive.extractfile('src/siesta_remag/grid_force.py') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != validator_hash:
                raise ValueError('Loaded force validator differs from retained implementation')
    checks, errors, seen = [], [], set()
    for path in sorted(root.glob('*/*/result.json')):
        result = json.loads(path.read_text())
        if result.get('runtime_settings', {}).get('REMAG_GRID_FORCE') != '1':
            continue
        folder = path.parent
        case, role = folder.parent.name, folder.name
        seen.add(case)
        faults, rows, hashes = [], [], {'result.json': sha(path)}
        expected_atomic = bool(metadata.get('atomic_grid_force')) and role != 'candidate-without-atomic-force'
        if (result.get('runtime_settings', {}).get('REMAG_ATOMIC_GRID_FORCE') == '1') != expected_atomic:
            faults.append('Atomic-force runtime differs from requested mode')
        scientific = [v for v in ledger if v.get('candidate_path', v['benchmark']+'/candidate') == case+'/'+role]
        if not result['valid'] or len(scientific) != 1 or not scientific[0]['passed']:
            faults.append('Upstream scientific comparison did not pass')
        for raw in sorted(folder.glob('remag-grid-force-*.json')):
            record = json.loads(raw.read_text())
            rows.append(record)
            hashes[raw.name] = sha(raw)
            faults.extend(record_errors(record, result['benchmark']['atoms']))
            if bool(record.get('atomic_terms_authoritative')) != expected_atomic:
                faults.append('Atomic-force record differs from requested mode')
        telemetry = result['telemetry']
        cycles = telemetry.get('gpu_device_cycle', {}).get('cycles', [])
        if not rows or len(rows) != len(cycles) or any(a['epoch'] >= b['epoch'] for a, b in zip(rows, rows[1:])):
            faults.append('Missing/repeated geometry coverage')
        if rows != telemetry.get('grid_force'):
            faults.append('Raw force records differ from telemetry')
        traffic = dict(host_to_device_bytes=sum(r['setup_bytes'] for r in rows),
                       device_to_host_bytes=sum(r['export_bytes'] for r in rows))
        if traffic != telemetry.get('gpu_grid_force') or telemetry.get('residency', {}).get('accounting_errors'):
            faults.append('Force transfer accounting failed')
        checks.append(dict(case=case, role=role, passed=not faults, errors=faults,
                           geometries=len(rows), traffic=traffic, evidence_sha256=hashes))
    if not metadata.get('grid_force') or seen != set(metadata['cases']):
        errors.append('Production forces did not cover every requested case')
    if not ledger or any(not row['passed'] for row in ledger):
        errors.append('Scientific ledger contains an invalid calculation')
    mae = {p.name: dict(passed=json.loads(p.read_text())['passed'], sha256=sha(p)) for p in root.glob('mae-*.json')}
    for family, label in (('fept2_soc', 'fept2'), ('fept16_soc', 'fept16'), ('fept2_soc_gamma', 'fept2-gamma')):
        if {family+'_x', family+'_z'} <= set(metadata['cases']):
            roles = [''] + (['-candidate-without-grid-force'] if metadata.get('grid_force_ablation') else [])
            roles += ['-candidate-without-atomic-force'] if metadata.get('atomic_force_ablation') else []
            roles += ['-candidate-without-final-grid'] if metadata.get('final_grid_ablation') else []
            roles += ['-candidate-without-atomic-basis'] if metadata.get('atomic_basis_ablation') else []
            roles += ['-candidate-without-orbital-basis'] if metadata.get('orbital_basis_ablation') else []
            roles += ['-candidate-without-final-density'] if metadata.get('final_density_ablation') else []
            roles += ['-candidate-repeat'+str(i) for i in range(2, metadata.get('candidate_repeats', 1)+1)]
            if any('mae-'+label+role+'.json' not in mae for role in roles):
                errors.append('Missing directional MAE comparison for '+family)
    if any(not r['passed'] for r in mae.values()):
        errors.append('Directional MAE comparison failed')
    ablations = {}
    for kind in ('grid', 'atomic'):
        group = {p.name: dict(passed=json.loads(p.read_text())['passed'], sha256=sha(p))
                 for p in root.glob('ablation-'+kind+'-force-*.json')}
        ablations.update(group)
        expected_files = {'ablation-'+kind+'-force-'+case+'.json' for case in metadata['cases']}
        if metadata.get(kind+'_force_ablation') and (set(group) != expected_files or any(not a['passed'] for a in group.values())):
            errors.append('Same-allocation '+kind+' force comparison is missing or failed')
    return dict(scope=__doc__, passed=bool(checks) and not errors and all(c['passed'] for c in checks),
                errors=errors, checks=checks, mae=mae, ablations=ablations,
                ledger_sha256=sha(root/'ledger.json'), metadata_sha256=sha(root/'metadata.json'),
                implementation_archive_sha256=sha(root/'implementation.tar.gz'),
                validator_sha256=validator_hash,
                verifier_sha256=sha(Path(__file__)), final_force_residency_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'grid-force-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'grid-force-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Authoritative grid forces:', len(result['checks']), 'calculations; passed=', result['passed'])
    if not result['passed']:
        raise SystemExit(1)
