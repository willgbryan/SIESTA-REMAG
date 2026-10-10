#!/usr/bin/env python3
"""Independently reconstruct retained GPU population outputs and boundaries.

Checks explicit fixture arrays and graph inventory, not material ownership or
whole-process CUDA activity. Does not import the CUDA checker's validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


FIELDS = ('num', 'ptr', 'lasto', 'overlap', 'density')
FAULTS = {'problem_shape': ('problem', 1), 'negative_offset': ('ptr', 1), 'bad_count': ('num', 1),
          'overlapping_rows': ('ptr', 1), 'tail_span': ('ptr', 1), 'first_atom': ('lasto', 1),
          'last_atom': ('lasto', 1), 'reversed_atoms': ('lasto', 1), 'nonfinite_density': ('density', 2),
          'nonfinite_overlap': ('overlap', 2), 'arithmetic_overflow': ('density/overlap', 2),
          'unused_soc': ('density', 0), 'host_shape': ('config', None)}


def calculate(c, p, d):
    atoms, rows, capacity, spins = map(int, c)
    components = min(spins, 4)
    status = 0 if np.array_equal(p, [capacity, rows, atoms, spins]) else 1
    orbital = np.zeros((rows, components))
    previous = 0
    for row in range(rows):
        start, count = int(d['ptr'][row]), int(d['num'][row])
        if start < previous or start < 0 or start > capacity or count < 0 or count > capacity-start:
            status = max(status, 1)
            continue
        previous = start+count
        selected = d['density'][:components, start:start+count].copy()
        if spins == 8:
            selected[2:] = (selected[2:]+d['density'][6:8, start:start+count])*.5
        values = selected*d['overlap'][start:start+count]
        charge = values.sum(axis=1)
        if not np.isfinite(values).all() or not np.isfinite(charge).all():
            status = 2
        else:
            orbital[row] = charge
    lasto = d['lasto']
    atomic = np.zeros((atoms, components))
    if lasto[0] != 0 or lasto[-1] != rows or np.any(lasto < 0) or np.any(lasto > rows) or np.any(np.diff(lasto) < 0):
        status = max(status, 1)
    else:
        for atom, (begin, end) in enumerate(zip(lasto[:-1], lasto[1:])):
            atomic[atom] = orbital[begin:end].sum(axis=0)
        if not np.isfinite(atomic).all():
            status = 2
    total = orbital.sum(axis=0)
    if not np.isfinite(total).all():
        status = 2
    return status, np.concatenate((orbital.ravel(), atomic.ravel(), total))


def verify(root):
    folder = root/'population-check'
    report, meta = read(folder/'check.json'), read(root/'metadata.json')
    ref = read(folder/report['oracle_file'])
    errors, checks, seen = [], [], set()
    for name in ('source-verification.json', 'component-evidence-verification.json'):
        if not (root/name).is_file() or read(root/name).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
    if report['checker_sha256'] != meta.get('population_check_sha256') or ref['script_sha256'] != meta.get('population_oracle_sha256'):
        errors.append('Checker/oracle source binding differs')
    paths = {k: folder/report[k+'_file'] for k in ('output', 'inputs', 'oracle')}
    if any(sha(p) != report[k+'_sha256'] for k, p in paths.items()) or sha(folder/ref['arrays_file']) != ref['arrays_sha256']:
        errors.append('Raw array/report binding differs')
    original, inputs, output = (np.load(p) for p in (folder/ref['arrays_file'], paths['inputs'], paths['output']))
    expected_scenarios = set()
    wanted_fixtures = {(spin,rows,mode) for spin in (1,2,4,8)
                       for rows in (1,5,257,*((131073,262145) if spin==8 else ()))
                       for mode in ('finite_padding','poisoned_padding','empty_rows')}
    actual_fixtures = {(r['spins'],r['rows'],r['fixture'].split('_',2)[2]) for r in ref['checks']}
    if actual_fixtures != wanted_fixtures or len(ref['checks']) != len(wanted_fixtures):
        errors.append('Missing/duplicate primitive fixtures, including actual second-stride coverage')
    for row in ref['checks']:
        label = row['fixture']
        variants = ['normal']
        if row['spins'] == 8 and row['rows'] == 257 and label.endswith('_finite_padding'):
            variants += list(FAULTS)
        if row['rows'] in (131073,262145) and label.endswith('_finite_padding'):
            variants += ['tail_span']
        for variant in variants:
            for capture in (False, True):
                for replays in ((1, 3) if variant == 'normal' else (1,)):
                    expected_scenarios.add((label, variant, capture, replays))
    for row in report['checks']:
        issues = []
        label, key, variant = row['fixture'], row['inputs_key'], row['variant']
        scenario = (label, variant, row['captured'], row['replays'])
        if scenario in seen or scenario not in expected_scenarios:
            issues.append('Unexpected/duplicate execution')
        seen.add(scenario)
        c, p = inputs[key+'_config'], inputs[key+'_problem']
        d = {k: inputs[key+'_'+k] for k in FIELDS}
        changed = set()
        for name in (*FIELDS, 'config'):
            actual = c if name == 'config' else d[name]
            if not np.array_equal(actual, original[label+'_'+name], equal_nan=True):
                changed.add(name)
        base = original[label+'_config']
        if not np.array_equal(p, base[[2, 1, 0, 3]]):
            changed.add('problem')
        allowed, wanted = ('', 0) if variant == 'normal' else FAULTS[variant]
        if changed != (set(allowed.split('/')) if allowed else set()):
            issues.append('Mutation/unchanged input binding differs')
        actual = output[row['scenario']]
        if variant == 'host_shape':
            if c[1] != 0 or row['native_return'] != -1 or row['stage'] != 0 or row['status'] != -9 or any(row[k] for k in ('allocated', 'staged', 'exported')) or not np.isnan(actual).all():
                issues.append('Invalid host shape did not fail before staging')
            error = None
        else:
            with np.errstate(over='ignore', invalid='ignore'):
                status, expected = calculate(c, p, d)
            if status != wanted or row['status'] != status or row['native_return'] != 0 or row['stage'] != 4:
                issues.append('Independently derived error status differs')
            error = float(np.max(np.abs(actual-expected))) if status == 0 else None
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                issues.append('Incomplete/nonfinite GPU outputs')
            elif status == 0 and not np.all(np.abs(actual-expected) <= 1e-10+1e-12*np.abs(expected)):
                issues.append('Independent population contraction differs')
            if variant in ('normal', 'unused_soc') and not np.all(np.abs(actual-original[label+'_expected']) <= 1e-10+1e-12*np.abs(original[label+'_expected'])):
                issues.append('Original Fortran population/charge comparison failed')
            atoms, rows, capacity, spins = map(int, c)
            staged = 16+4*(2*rows+atoms+1)+8*capacity*(1+spins)
            exported = 8*min(spins, 4)*(rows+atoms+1)+4
            expected_counts = dict(allocated=staged+exported, staged=staged, exported=exported,
                                   poisoned=exported*row['replays'], graph_nodes=4*int(row['captured']), forbidden_nodes=0)
            if any(row[k] != v for k, v in expected_counts.items()):
                issues.append('Shape-derived boundaries/capture differ')
        checks.append(dict(scenario=row['scenario'], passed=not issues, errors=issues, max_absolute_error=error))
    if seen != expected_scenarios or len(checks) != 198 or report.get('passed') is not True:
        errors.append('Incomplete/failed population GPU qualification')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                report_sha256=sha(folder/'check.json'), verifier_sha256=sha(Path(__file__)), independent_cuda_trace=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'population-component-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'population-component-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Population component audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
