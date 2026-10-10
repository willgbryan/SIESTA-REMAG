#!/usr/bin/env python3
"""Audit production GPU initialization, explicit transfers and terminal diagnostics.

This does not qualify GPU density history, CPU geometry work, hidden library
transfers or complete memory/I/O residency. First-use D still stages from the host; retained history is audited separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    metadata = json.loads((root/'metadata.json').read_text())
    assert metadata['gpu_initialization'] is True
    initial_owner = 'initial_state.py' in metadata.get('harness_sha256', {})
    if initial_owner:
        assert json.loads((root/'initial-state-verification.json').read_text())['passed'] is True
    if metadata.get('gpu_history'):
        assert json.loads((root/'density-history-verification.json').read_text())['passed'] is True
    ledger = json.loads((root/'ledger.json').read_text())
    checks = []
    for row in ledger:
        folder = root/row.get('candidate_path', row['benchmark']+'/candidate')
        def read(name):
            return json.loads((folder/name).read_text())
        result, cycles, grid, phases = (read(n) for n in
            ('result.json', 'remag-device-scf.json', 'remag-grid.json', 'remag-residency.json'))
        errors = []
        setting=result['runtime_settings'].get('REMAG_GPU_INITIALIZATION')
        # The first integrated batch omitted this key from runner.py's saved
        # environment whitelist. Preserve those records. The batch request and
        # each native record independently identify the actual enabled owner.
        if (not row['passed'] or not result['valid'] or
                result['runtime_settings'].get('REMAG_DEVICE_SCF') != '1' or setting not in (None,'1')):
            errors.append('Successful GPU initialization material result is missing')
        rows = cycles['cycles']
        if not rows or grid.get('topology_only_prepares') != len(rows):
            errors.append('Geometry-only grid preparations do not cover every cycle')
        initial = [p for p in phases['intervals'] if p['phase'] == 'initialization']
        totals = dict(host_to_device_bytes=0, device_to_host_bytes=0, copy_calls=0)
        for interval in initial:
            for name, counts in interval['boundaries'].items():
                if name in ('sparse_hamiltonian_input', 'grid_density_input', 'grid_potential_input', 'grid_sparse_input', 'grid_output') and any(counts.values()):
                    errors.append('Host electronic startup boundary remains: '+name)
                for key, value in counts.items():
                    totals[key] += value
        for c in rows:
            d = c.get('initialization', {})
            if d.get('device') is not True or d.get('status') != 0 or d.get('result_bytes') != 32:
                errors.append('Missing successful initial density result')
                continue
            charge, scale = d['charge'], d['scale']
            if not all(math.isfinite(v) and v != 0 for v in (charge, scale)) or d['mix_first'] not in (0, 1) or d['note'] not in (0, 1):
                errors.append('Invalid initial normalization diagnostics')
            elif d['mix_first'] and abs(charge - charge*scale) > 1e-4 + 1e-12:
                errors.append('First-step mixing was enabled despite charge mismatch')
        checks.append(dict(case=row['benchmark'], passed=not errors, errors=errors,
            environment_flag_recorded=setting is not None,
            mode_provenance='batch request plus native per-geometry initialization records',
            geometries=len(rows), iterations=sum(c['iterations'] for c in rows),
            initialization_explicit=totals,
            terminal_initial_diagnostic_bytes=32*len(rows),
            host_density_staging_remains=True,initial_population_host_consumers_remain=not initial_owner,
            evidence_sha256={str((folder/n).relative_to(root)):sha(folder/n) for n in
                ('result.json','remag-device-scf.json','remag-grid.json','remag-residency.json','stdout.log')}))
    return dict(scope=__doc__,passed=bool(checks) and all(c['passed'] for c in checks),checks=checks,
        strict_residency_verified=False,production_history_integrated=bool(metadata.get('gpu_history')),
        verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    report = verify(args.run)
    (args.run/'initialization-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'initialization-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GPU initialization:',len(report['checks']),'passed=',report['passed'])
    if not report['passed']:
        print([c for c in report['checks'] if not c['passed']])
        raise SystemExit(1)
