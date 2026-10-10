#!/usr/bin/env python3
"""Audit compact terminal matrix exports against raw copy counts and shapes.

H generated from final D, Dold and Eold have no post-SCF consumers in the
supported integral route. H that generated D is published directly into H;
D still serves legacy analysis/history. E is exported only for requested files.
This is explicit-copy evidence, not complete CUDA/memory observation.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

from siesta_remag.terminal_exports import record_errors, requested_energy_exports, requested_density_exports, requested_hamiltonian_exports


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    errors, checks, hashes = [], [], {}
    metadata=read(root/'metadata.json')
    for name in ('source-verification.json', 'component-evidence-verification.json',
                 'scientific-recheck.json', 'device-cycle-verification.json', 'hamiltonian-output-verification.json'):
        path = root/name
        if not path.is_file() or read(path).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():
            hashes[name] = sha(path)
    # This audit covers cycle-publication exports, including the absence of
    # deferred spectra there. Final analysis consumes the retained spectrum
    # after accepted forces: verify_final_spectrum runs after the force audit,
    # which depends on this one. Performance requires all three audits.
    if metadata.get('gpu_history'):
        path=root/'density-history-verification.json'
        if not path.is_file() or read(path).get('passed') is not True:
            errors.append('Missing/failed production history audit')
        if path.is_file():hashes[path.name]=sha(path)
    module = Path(inspect.getsourcefile(record_errors))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        archived = archive.extractfile('src/siesta_remag/terminal_exports.py').read()
    if hashlib.sha256(archived).hexdigest() != sha(module):
        raise ValueError('Terminal export validator differs from executed source')
    for path in sorted(root.glob('*/*/result.json')):
        result = read(path)
        if result.get('runtime_settings', {}).get('REMAG_INTEGRAL_FORCE') != '1':
            continue
        folder = path.parent
        native, grid, raw = (read(folder/name) for name in ('remag-device-scf.json', 'remag-final-grid.json', 'remag-residency.json'))
        cycles, final = native['cycles'], grid['cycles']
        reasons = requested_energy_exports((folder/'input.fdf').read_text(), True)
        expected_initialization=bool(metadata.get('gpu_initialization',False)) and result['runtime_settings'].get('REMAG_DEVICE_SCF')=='1'
        h_reasons=requested_hamiltonian_exports((folder/'input.fdf').read_text(),bool(metadata.get('gate_h_export')))
        density_reasons=requested_density_exports((folder/'input.fdf').read_text(),bool(metadata.get('gpu_history')))
        issues = record_errors(cycles, final, True, reasons,expected_initialization,density_reasons,bool(metadata.get("final_spectra")),h_reasons)
        setting=result['runtime_settings'].get('REMAG_GPU_INITIALIZATION')
        if setting is not None and (setting=='1')!=expected_initialization:
            issues.append('Recorded initialization setting differs from batch request')
        if native.get('schema') != 3 or not result['valid']:
            issues.append('Invalid result or missing terminal matrix schema')
        aggregate = dict(host_to_device_bytes=0, device_to_host_bytes=0, copy_calls=0)
        for interval in raw['intervals']:
            traffic = interval['boundaries'].get('sparse_state_export')
            if traffic:
                if interval['phase'] != 'post_scf':
                    issues.append('Cycle export outside its terminal boundary')
                for key, value in traffic.items():
                    aggregate[key] += value
        # Two diagnostics, each exported matrix, legacy E/Q when requested,
        # and optional final-force H. Deferred spectra have a separate audit.
        copy_calls = sum(4-2*int(c['terminal'].get('deferred_spectrum',False))+int(c.get('initialization',{}).get('device',False))+sum(bool(v) for v in c['terminal']['matrix_bytes'])+
                         bool(g['hamiltonian_cycle_export_bytes']) for c, g in zip(cycles, final))
        wanted = dict(host_to_device_bytes=0, device_to_host_bytes=sum(c['export_bytes'] for c in cycles), copy_calls=copy_calls)
        if aggregate != wanted:
            issues.append('Native matrix roles do not reconcile with actual explicit cycle copies')
        removed = []
        for c in cycles:
            t = c['terminal']
            h, e = 8*t['capacity']*t['spins'], 8*t['capacity']*t['energy_spins']
            removed.append(dict(epoch=c['epoch'], geometry=c['geometry'], h_from_final_density=h,
                                previous_density=h, previous_energy_density=e if t['old_energy_host'] else 0,
                                unrequested_energy_density=e if not reasons else 0,
                                unrequested_density=h if not density_reasons else 0,
                                unrequested_hamiltonian=h if not h_reasons else 0))
        checks.append(dict(case=folder.parent.name, role=folder.name, passed=not issues, errors=issues,
                           energy_export_reasons=reasons,density_export_reasons=density_reasons,hamiltonian_export_reasons=h_reasons,
                           cycles=len(cycles), actual_explicit_exports=aggregate, eliminated_legacy_matrix_bytes=removed,
                           evidence_sha256={str(p.relative_to(root)): sha(p) for p in
                                            (path, folder/'input.fdf', folder/'remag-device-scf.json', folder/'remag-final-grid.json', folder/'remag-residency.json')}))
    ledger = read(root/'ledger.json')
    recorded_paths = {tuple(Path(r.get('candidate_path', r['benchmark']+'/candidate')).parts) for r in ledger}
    if len(checks) != len(ledger) or {(c['case'], c['role']) for c in checks} != recorded_paths:
        errors.append('Terminal audit does not cover the complete recorded corpus')
    return dict(scope=__doc__, passed=bool(checks) and not errors and all(c['passed'] for c in checks),
                errors=errors, checks=checks, prerequisite_sha256=hashes,
                validator_sha256=sha(module), verifier_sha256=sha(Path(__file__)), independent_cuda_trace=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'terminal-export-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'terminal-export-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Terminal matrix exports:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
