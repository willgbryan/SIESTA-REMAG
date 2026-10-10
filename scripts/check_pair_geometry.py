#!/usr/bin/env python3
"""Check immutable pre-SCF geometry, two consumers, recovery and stale leases."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENARIOS = ('normal', 'empty', 'recovery', 'stale_begin', 'stale_active', 'stale_staged')
PREPARE = '''missing_bind missing_finish bad_volume bad_span bad_column prepare_in_scf
duplicate_prepare early_bind early_begin query_row query_column query_atom
query_registry query_xyz query_in_scf duplicate_finish query_after_finish
bind_without_epoch duplicate_bind overlap_first'''.split()
CONSUME = '''changed_shape changed_num changed_ptr changed_list changed_volume
duplicate_begin incomplete_stage changed_query extra_query changed_pairs
changed_xyz changed_count duplicate_stage wrong_complete duplicate_complete'''.split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(folder, scenario):
    p, owner, raw = (json.loads((folder/k).read_text()) for k in ('probe.json', 'remag-pair-geometry.json', 'remag-residency.json'))
    held, empty, ng = scenario.startswith('stale_'), scenario == 'empty', 2 if scenario == 'recovery' else 1
    rejections = (PREPARE+CONSUME*2)*2 if scenario == 'recovery' else []
    if held:
        rejections = {'stale_begin': ['stale_validate'], 'stale_active': ['stale_stage', 'stale_complete'],
                      'stale_staged': ['stale_complete']}[scenario]+['held_prepare']
    outputs = []
    for g in range(ng):
        snapshot = [] if empty else [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1]+[v+.125*g for v in (.1, -.2, .3, -.1, .2, -.3)]
        snapshot += [3, 2, 1, 5, 0, 1, 1, 2, 0, 1, 2, 0]
        outputs += [snapshot]*(1 if scenario == 'stale_staged' else 0 if held else 2)
    errors = []
    if p != dict(scenario=scenario, geometries=ng, held=held, rejections=rejections, snapshots=outputs):
        errors.append('Snapshot or rejection/recovery coverage differs')
    wanted = [dict(epoch=g+1, atoms=2, rows=2, columns=2, capacity=8, queries=0 if empty else 2,
                   volume=2.3, setup_bytes=48 if empty else 144, device_allocation_bytes=72 if empty else 144,
                   consumer_mask=0 if held else 5, complete=not held) for g in range(ng)]
    if owner.get('complete') != (not held) or owner.get('cycles') != wanted:
        errors.append('Owner lifecycle or shape-derived allocation differs')
    total = dict(host_to_device_bytes=0, device_to_host_bytes=0, copy_calls=0)
    for interval in raw['intervals']:
        for name, traffic in interval['boundaries'].items():
            if name != 'pair_geometry' or interval['phase'] != 'initialization':
                errors.append('Owner copy outside initial geometry staging')
            for k, v in traffic.items():
                total[k] += v
    if total != dict(host_to_device_bytes=(48 if empty else 144)*ng, device_to_host_bytes=0, copy_calls=(3 if empty else 8)*ng):
        errors.append('Owner transfer accounting differs')
    if raw['geometry_epochs'] != (2 if held else ng) or raw['phase_markers_seen'] is not True:
        errors.append('Epoch/phase coverage differs')
    return errors


def run(executable, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    for scenario in SCENARIOS:
        folder = output.parent/scenario
        folder.mkdir()
        paths = dict(report=folder/'probe.json', owner=folder/'remag-pair-geometry.json',
                     residency=folder/'remag-residency.json', log=folder/'worker.log')
        with paths['log'].open('w') as log:
            code = subprocess.run([str(executable.resolve()), scenario, str(paths['report'])], cwd=folder,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=30).returncode
        errors = ['Native probe failed or omitted evidence'] if code or any(not p.is_file() for p in paths.values()) else validate(folder, scenario)
        row = dict(scenario=scenario, native_return=code, passed=not errors, errors=errors)
        for kind, path in paths.items():
            row[kind+'_file'] = str(path.relative_to(output.parent))
            row[kind+'_sha256'] = sha(path) if path.is_file() else None
        checks.append(row)
    result = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks,
                  checker_sha256=sha(Path(__file__)), native_executable_sha256=sha(executable),
                  independent_cuda_trace=False, probe_snapshot_exports_in_owner_accounting=False)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('Pair geometry:', len(checks), 'passed=', result['passed'], flush=True)
    if not result['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable, args.output)
