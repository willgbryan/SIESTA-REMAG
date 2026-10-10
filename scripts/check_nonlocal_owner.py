#!/usr/bin/env python3
"""Exercise compact geometry recovery and terminal retained-D ownership failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENARIOS = ('normal', 'captured', 'recovery', 'empty', 'failed_reference', 'failed_status', 'stale_setup', 'stale_staged')
SPINS = ((1, 0), (2, 0), (4, 0), (8, 0), (8, 1))
RECOVERY = ('missing_end', 'missing_bind', 'bad_mode', 'bad_volume', 'REMAG_NONLOCAL_FORCE_SHADOW', 'REMAG_PAIR_FORCE_SHADOW',
            'REMAG_TWO_CENTER_SHADOW', 'bad_span', 'overlapping_rows', 'bad_sparse_column', 'prepare_in_scf',
            'duplicate_prepare', 'early_bind', 'early_center', 'table', 'registry_a', 'registry_b', 'null_xyz', 'nonfinite_xyz', 'observe_in_scf')
RECOVERY += tuple('initial_'+s for s in ('row', 'column', 'atom', 'offset', 'angular_low', 'angular_high', 'epsilon', 'coverage'))
RECOVERY += ('unfinished_center', 'bind_without_epoch_advance', 'duplicate_initial_end', 'center_after_staging', 'duplicate_bind',
             'missing_retained', 'missing_kinetic', 'changed_volume', 'changed_consumer', 'changed_split', 'changed_shape', 'changed_num',
             'changed_ptr', 'changed_list', 'grid_fingerprint', 'nonfinite_initial', 'final_begin_in_scf', 'duplicate_final_begin',
             'early_final_end', 'early_pending', 'changed_registry', 'changed_xyz', 'extra_query')
RECOVERY += tuple('final_'+s for s in ('row', 'column', 'atom', 'offset', 'angular_low', 'angular_high', 'epsilon', 'coverage'))
RECOVERY += ('changed_local_row', 'changed_center', 'extra_center', 'nonfinite_reference', 'stage_without_reference', 'duplicate_reference',
             'center_after_reference', 'wrong_pending', 'stage_pairs', 'stage_xyz', 'stage_count', 'stage_consumer', 'apply_before_stage',
             'finish_before_stage', 'duplicate_stage', 'finish_before_apply', 'wrong_stream', 'missing_status', 'apply_in_scf',
             'duplicate_apply', 'duplicate_finish', 'missing_final_begin')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(executable, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    for spins, split in SPINS:
        for scenario in SCENARIOS:
            name = f'{scenario}-spins{spins}-split{split}'
            folder = output.parent/name
            folder.mkdir()
            paths = dict(report=folder/'probe.json', owner=folder/'remag-nonlocal-force.json',
                         residency=folder/'remag-residency.json', log=folder/'worker.log')
            with paths['log'].open('w') as log:
                code = subprocess.run([str(executable.resolve()), scenario, str(spins), str(split), str(paths['report'])],
                                      cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=45).returncode
            issues = []
            if code != 0 or any(not p.is_file() for p in paths.values()):
                issues.append('Native probe failed or omitted evidence')
            else:
                report, owner = (json.loads(paths[k].read_text()) for k in ('report', 'owner'))
                held = scenario.startswith(('failed_', 'stale_'))
                geometries = 2 if scenario == 'recovery' else 1
                rejected = list(RECOVERY)*2 if scenario == 'recovery' else []
                if held:
                    rejected = ['failed_completion', 'retry_completion'] if scenario.startswith('failed_') else ['stale_begin'] if scenario == 'stale_setup' else ['stale_apply', 'stale_finish']
                    rejected += ['held_grid', 'held_overlap', 'held_reset', 'held_prepare']
                if (report['scenario'], report['spins'], report['split'], report['held'], report['rejections']) != (scenario, spins, split, held, rejected):
                    issues.append('Probe scenario or rejection coverage differs')
                if owner['complete'] != (not held) or len(owner['cycles']) != geometries or len(report['cycles']) != geometries:
                    issues.append('Owner lifecycle coverage differs')
                if any(c['retained'] != c['density'] for c in report['cycles']): issues.append('Retained D identity differs')
            row = dict(scenario=scenario, spins=spins, split=split, native_return=code, passed=not issues, errors=issues)
            for kind, path in paths.items():
                row[kind+'_file'] = str(path.relative_to(output.parent))
                row[kind+'_sha256'] = sha(path) if path.is_file() else None
            checks.append(row)
    result = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks,
                  checker_sha256=sha(Path(__file__)), native_executable_sha256=sha(executable))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('Nonlocal owner:', len(checks), 'passed=', result['passed'], flush=True)
    if not result['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable, args.output)
