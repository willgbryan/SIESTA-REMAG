#!/usr/bin/env python3
"""Independently audit retained SOC projector GPU outputs, inputs and boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_soc_projector_oracle import verify as verify_oracle, sha, FIELDS


def verify(root):
    folder = root/'soc-projector-check'
    path = folder/'check.json'
    report = json.loads(path.read_text())
    oracle_path = folder/report['oracle_file']
    oracle = json.loads(oracle_path.read_text())
    outputs_path = folder/report['output_file']
    mutations_path = folder/report['inputs_file']
    arrays_path = folder/oracle['arrays_file']
    errors, checks = [], []
    for file, expected in ((oracle_path, report['oracle_sha256']), (outputs_path, report['output_sha256']),
                           (mutations_path, report['inputs_sha256']), (arrays_path, report['oracle_arrays_sha256'])):
        if sha(file) != expected:
            errors.append('Content hash differs: '+file.name)
    meta = json.loads((root/'metadata.json').read_text())
    if report['native_library_sha256'] != meta['native_binary_sha256']['/opt/candidate/remag-native/libremag_soc_projector_check.so']:
        errors.append('Native library differs from invocation binary identity')
    audit = verify_oracle(oracle_path, Path('upstream/siesta'))
    (root/'soc-projector-oracle-verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    if not audit['passed']:
        errors.append('Independent pristine oracle audit failed')
    faults = ('epsilon_nan', 'left_nan', 'right_nan', 'gradient_nan', 'overflow', 'sticky')
    host = ('l_zero', 'l_high', 'split', 'negative_count')
    expected_cases, expected_inputs = {}, set()
    with np.load(arrays_path) as reference, np.load(outputs_path) as outputs, np.load(mutations_path) as mutations:
        for fixture in oracle['checks']:
            label = fixture['fixture']
            l, split, n = map(int, reference[label+'_config'])
            variants = ['normal']
            if l == 2 and n == 257:
                variants += ['empty', *faults, *host]
            if n == 131073:
                variants.append('tail_nan')
            pristine = {k: reference[label+'_'+k] for k in FIELDS}
            for variant in variants:
                key = label+'_'+variant
                config = np.array([l, split, n], dtype=np.int32)
                data = {k: v.copy() for k, v in pristine.items()}
                code = initial = 0
                if variant == 'empty':
                    config[2] = 0
                    data = {k: v if k == 'epsilon' else v[:0] for k, v in data.items()}
                elif variant in ('epsilon_nan', 'left_nan', 'right_nan', 'gradient_nan', 'tail_nan'):
                    code = 2
                    field = {'epsilon_nan':'epsilon', 'left_nan':'left', 'right_nan':'right', 'gradient_nan':'left_gradient', 'tail_nan':'left_gradient'}[variant]
                    if variant == 'tail_nan': data[field][-1, -1, -1, -1] = np.nan
                    else: data[field][:] = np.nan
                elif variant == 'overflow':
                    code = 2
                    data['left_gradient'][:] = np.finfo('d').max
                    data['right'][:] = 8.
                elif variant == 'sticky': code = initial = 7
                elif variant == 'l_zero': config[0] = 0
                elif variant == 'l_high': config[0] = 6
                elif variant == 'split': config[1] = 2
                elif variant == 'negative_count': config[2] = -1
                if variant != 'normal':
                    for field, expected in {**data, 'config':config}.items():
                        item = key+'_'+field
                        expected_inputs.add(item)
                        if item not in mutations or mutations[item].dtype != expected.dtype or not np.array_equal(mutations[item], expected, equal_nan=True):
                            errors.append('Executed mutation differs: '+item)
                for captured in ((False,) if variant in host else (False, True)):
                    expected_cases[key+f'_captured{int(captured)}'] = (label, variant, config.copy(), code, initial, captured)
        seen = set()
        for row in report['checks']:
            name = row['scenario']
            if name not in expected_cases or name in seen:
                errors.append('Unexpected or duplicate scenario: '+name)
                continue
            seen.add(name)
            label, variant, config, code, initial, captured = expected_cases[name]
            l, split, n = map(int, config)
            cycles = 3 if captured else 1
            issues = []
            if any(row.get(k) != v for k, v in dict(fixture=label, variant=variant, captured=captured, cycles=cycles, initial_status=initial, expected_status=code).items()):
                issues.append('Scenario metadata differs')
            actual = outputs[name]
            maximum = None
            if variant in host:
                if row['native_return'] != -1 or row['status'] != -9 or any(row[k] for k in ('allocated','staged','exported','poisoned','graph_nodes','forbidden_nodes','stage')) or not np.isnan(actual).all():
                    issues.append('Invalid host configuration was not rejected before execution')
            else:
                staged = 20+80*(2*l+1)*n
                expected = dict(native_return=0, status=code, staged=staged, allocated=staged+264*n+(32 if n == 0 else 0),
                                exported=264*n+4, poisoned=264*n*cycles, graph_nodes=int(captured), forbidden_nodes=0, stage=4)
                if any(row.get(k) != v for k, v in expected.items()):
                    issues.append('Execution/status/shape-derived boundary differs')
                if actual.shape != (n, 33) or not np.isfinite(actual).all():
                    issues.append('Output shape, poison overwrite or finiteness differs')
                elif variant in ('normal', 'empty'):
                    wanted = reference[label+'_expected'][:n]
                    delta = np.abs(actual-wanted)
                    maximum = float(delta.max()) if n else 0.
                    if np.any(delta > 1e-12+1e-12*np.abs(wanted)):
                        issues.append('Retained output differs from pristine operator/gradient')
                elif variant == 'tail_nan':
                    if not np.array_equal(actual[-1], np.zeros(33)):
                        issues.append('Invalid launch-tail query did not write defined zeros')
                elif not np.array_equal(actual, np.zeros_like(actual)):
                    issues.append('Uniform fault/sticky status did not write defined zeros')
            if not row['passed']:
                issues.append('Original scenario failed')
            checks.append(dict(scenario=name, passed=not issues, errors=issues, max_absolute_error=maximum))
        if seen != set(expected_cases) or len(checks) != 168 or set(outputs.files) != seen:
            errors.append('Required output/scenario coverage differs')
        if set(mutations.files) != expected_inputs:
            errors.append('Executed mutation array coverage differs')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                independently_reconstructed_mutation_arrays=len(expected_inputs), report_sha256=sha(path), verifier_sha256=sha(Path(__file__)),
                whole_process_cuda_trace=False, material_integration=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    for filename in ('verify_soc_projector_component.py', 'verify_soc_projector_oracle.py'):
        (args.run/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    (args.run/'soc-projector-component-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('SOC projector component audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
