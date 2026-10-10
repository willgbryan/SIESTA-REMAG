#!/usr/bin/env python3
"""Check terminal publication of a shared GPU force/energy/stress accumulator.

Empty integral geometries isolate ownership; real core/neutral/dfscf kernels
and retained Hartree stress supply nonzero results. Nonempty integral physics
is tested separately. Explicit owner counters exclude probe fixture transfers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

SCENARIOS = ('normal', 'recovery', 'failed_status', 'failed_nonfinite', 'stale')
RECOVERY = '''missing_hartree early_grid early_nonlocal early_collect early_publish
unrecorded_kinetic changed_initial_force retained_pair nonlocal_unmarked_reuse nonlocal_retained_generator grid_before_hartree duplicate_hartree
reset_before_collection changed_publication_input duplicate_collect
reset_before_publication duplicate_publish'''.split()
FAILED = 'failed_collect retry_collect failed_publish failed_reset replace_failed_state'.split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(folder, scenario, spins, core):
    probe, raw = (json.loads((folder/name).read_text()) for name in ('probe.json', 'remag-residency.json'))
    failures = scenario not in ('normal', 'recovery')
    geometries = 2 if scenario == 'recovery' else 1
    rejections = RECOVERY*2 if scenario == 'recovery' else FAILED if failures else []
    errors = []
    if (probe['scenario'], probe['spins'], probe['partial_core'], probe['rejections']) != (scenario, spins, core, rejections):
        errors.append('Scenario/recovery/rejection coverage differs')
    v = np.array([1.2, 1.8, .2, .3])[:min(spins, 4)]
    density = np.array([.4, .3, .2, -.1])[:min(spins, 4)]
    if spins == 8:
        density[2:] = [.16, .02]
    weights = v.copy()
    weights[2:] *= 2
    increment = .08*np.array([-.2, .5, -.1])
    increment += .14*(np.dot(density, weights)-.18)*np.array([.3, .2, -.4])
    if core:
        increment += .1*(v[:min(spins, 2)].mean()-.6)*np.array([.2, -.3, .4])
    expected = np.r_[np.zeros(10), np.array([.1, -.2, .3])+increment,
                     np.arange(1, 10)*.011+np.outer([.4, .2, -.3], increment).ravel(order='F')/2]
    if len(probe['cycles']) != geometries:
        errors.append('Missing geometry')
    max_error = 0.
    for i, row in enumerate(probe['cycles']):
        wanted = dict(epoch=i+1, published=not failures, failed=failures,
                      captured_nodes=6+core, successful_export_bytes=0 if failures else 180)
        if any(row.get(k) != value for k, value in wanted.items()):
            errors.append('Completion/capture/publication differs')
        actual = np.array(row['output'])
        if failures:
            if len(actual):
                errors.append('Failed owner published output')
        elif actual.shape != (22,) or not np.isfinite(actual).all():
            errors.append('Missing/nonfinite force publication')
        else:
            max_error = max(max_error, float(np.max(np.abs(actual-expected))))
    if max_error >= 1e-12:
        errors.append('Analytic energy/force/stress mismatch')
    def counter(h=0, d=0, n=0):
        return dict(host_to_device_bytes=h, device_to_host_bytes=d, copy_calls=n)
    totals = {name: counter() for name in ('grid_flow_geometry', 'final_force_output')}
    for interval in raw['intervals']:
        for name, traffic in interval['boundaries'].items():
            if name not in totals or interval['phase'] != 'post_scf':
                errors.append('Unexpected owner copy or phase')
                continue
            for k, v in traffic.items():
                totals[name][k] += v
    exported = 0 if scenario == 'stale' else 4 if scenario == 'failed_status' else 180*geometries
    copies = 0 if scenario == 'stale' else 1 if scenario == 'failed_status' else 2*geometries
    if totals != dict(grid_flow_geometry=counter(h=180*geometries, n=2*geometries),
                      final_force_output=counter(d=exported, n=copies)):
        errors.append('Owner staging/export/retry traffic differs')
    if (raw['schema_version'] != 2 or raw['phase_markers_seen'] is not True or
            raw['geometry_epochs'] != (2 if scenario == 'stale' else geometries) or
            raw['strict_residency_verified'] is not False or
            any(raw[k] is not None for k in ('library_transfers', 'unified_memory_migrations', 'file_io'))):
        errors.append('Phase/accounting scope differs')
    return errors, max_error


def run(executable, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    for scenario in SCENARIOS:
        for spins in (1, 2, 4, 8):
            for core in (0, 1):
                folder = output.parent/f'{scenario}_spin{spins}_core{core}'
                folder.mkdir()
                paths = dict(report=folder/'probe.json', residency=folder/'remag-residency.json', log=folder/'worker.log')
                with paths['log'].open('w') as log:
                    code = subprocess.run([str(executable.resolve()), scenario, str(spins), str(core), str(paths['report'])],
                                          cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=30).returncode
                errors, error = (['Native probe failed or omitted evidence'], None) if code or any(not p.is_file() for p in paths.values()) else validate(folder, scenario, spins, core)
                row = dict(scenario=scenario, spins=spins, partial_core=core, native_return=code,
                           passed=not errors, errors=errors, max_absolute_error=error)
                for kind, path in paths.items():
                    row[kind+'_file'] = str(path.relative_to(output.parent))
                    row[kind+'_sha256'] = sha(path) if path.is_file() else None
                checks.append(row)
    result = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks,
                  checker_sha256=sha(Path(__file__)), native_executable_sha256=sha(executable),
                  independent_cuda_trace=False, fixture_transfers_in_owner_accounting=False)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('Integral publication:', len(checks), 'passed=', result['passed'], flush=True)
    if not result['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable, args.output)
