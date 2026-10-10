#!/usr/bin/env python3
"""Independently audit compact GPU inputs, complete valid outputs and fail-closed faults.

Malformed descriptors have no meaningful partial physics result. For those
scenarios the gate requires rejection, safe completion and finite initialized
outputs, not a particular scheduling-dependent partial sum. Numerical faults
disqualify the whole result; valid/unused-input/sticky/tail cases compare every
energy, force and stress component against the literal upstream reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_nonlocal_compact_oracle import verify as verify_oracle, FIELDS

STRUCTURAL = {'row', 'column', 'atom', 'center', 'center_base', 'center_count', 'block_base', 'block_count',
              'projectors', 'query', 'center_atom', 'block_offset', 'angular_low', 'angular_high',
              'csr_offset', 'csr_count', 'csr_column', 'lookup_missing', 'lookup_first', 'lookup_other_row', 'lookup_padding', 'lookup_high'}
NUMERICAL = {'gradient', 'value', 'xyz', 'density', 'epsilon', 'epsilon_second_nan',
             'arithmetic_overflow', 'force_overflow', 'energy_overflow', 'stress_overflow'}
HOST = {'volume', 'volume_nan', 'spinor', 'soc', 'split', 'capacity', 'components', 'negative_count', 'soc_components'}
UNUSED = {'right_gradient', 'projector_padding', 'xyz_padding', 'density_padding', 'density_components', 'epsilon_second'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mutated(base, variant):
    """Reconstruct the advertised perturbation without importing the checker."""
    a = {k: v.copy() for k, v in base.items()}
    nn, nc, nb, nq, na, nr, no, cap, components, spinor, soc, split, cycles = map(int, a['config'])
    col_mutations = {
        'neighbors': {'row': (0, -2**31), 'column': (1, no+1), 'atom': (2, 0), 'center': (3, nc)},
        'centers': {'center_base': (0, -1), 'center_count': (1, 2**31-1), 'block_base': (2, -1),
                    'block_count': (3, 2**31-1), 'projectors': (4, 2**31-1), 'query': (5, 2**31-1), 'center_atom': (6, 0), 'zero_blocks': (3, 0)},
        'block_indices': {'block_offset': (0, 2**31-1), 'angular_low': (1, -2**31), 'angular_high': (1, 2**31-1)},
    }
    for field, mapping in col_mutations.items():
        if variant in mapping:
            col, value = mapping[variant]
            a[field][:, col] = value
            return a
    flat_mutations = {'csr_offset': ('ptr', -1), 'csr_count': ('num', cap+1), 'csr_column': ('list', no+1), 'lookup_missing': ('lookup', -1)}
    if variant in flat_mutations:
        field, value = flat_mutations[variant]
        a[field].fill(value)
    elif variant.startswith('lookup_'):
        a['lookup'][0, 0] = {'lookup_first': 1, 'lookup_other_row': 26, 'lookup_padding': 0, 'lookup_high': 2**31-1}[variant]
    elif variant in ('gradient', 'value'):
        a['integrals'][:, slice(1, None) if variant == 'gradient' else 0] = np.nan
    elif variant in ('epsilon', 'epsilon_second', 'epsilon_second_nan'):
        a['epsilon'][:, int(variant != 'epsilon')] = np.nan
    elif variant in ('density', 'xyz'): a[variant].fill(np.nan)
    elif variant == 'density_components':
        consumed = {0, 1, 4, 5} if soc and not split else set(range(spinor))
        for spin in range(components):
            if spin not in consumed: a['density'][spin].fill(np.nan)
    elif variant == 'density_padding':
        active = {i for row in range(nr) for i in range(a['ptr'][row], a['ptr'][row]+a['num'][row])}
        a['density'][:, [i for i in range(cap) if i not in active]] = np.nan
    elif variant == 'overwritten_density': a['density'][:, 1] = np.nan
    elif variant in ('right_gradient', 'projector_padding', 'xyz_padding'):
        consumed = set()
        for start, count, b0, bcount, stride, q0, atom in a['centers']:
            for i in range(count):
                q = q0+i*stride
                if variant == 'right_gradient' and a['neighbors'][start+i, 0] == 0:
                    a['integrals'][q:q+stride, 1:] = np.nan
                if variant == 'xyz_padding': consumed.add(q)
                if variant == 'projector_padding':
                    for offset, l in a['block_indices'][b0:b0+bcount]:
                        consumed.update(range(q+offset, q+offset+(4*l+2 if l else 1)))
        if variant != 'right_gradient':
            field = 'xyz' if variant == 'xyz_padding' else 'integrals'
            a[field][[q for q in range(nq) if q not in consumed]] = np.nan
    elif variant in ('tail_excluded', 'tail_row'): a['neighbors'][-1, 0] = 0 if variant == 'tail_excluded' else -1
    elif variant == 'tail_gradient':
        c = a['centers'][a['neighbors'][-1, 3]]
        q = c[5]+(nn-c[0]-1)*c[4]+a['block_indices'][c[2], 0]
        a['integrals'][q, 1] = np.nan
    elif variant.endswith('_overflow'):
        for field in ('density', 'epsilon', 'integrals', 'xyz'): a[field].fill(1.)
        a['neighbors'][:, 2], a['centers'][:, 6] = 1, 2
        maximum = np.finfo(np.float64).max
        if variant == 'arithmetic_overflow': a['integrals'][:, 1:] = maximum/2
        elif variant == 'energy_overflow':
            a['integrals'][:, 0], a['integrals'][:, 1:] = 1e150, 0.
            a['initial'][0] = maximum
        else:
            a['integrals'][:, 1:] = maximum/10000
            a['initial'][slice(12, 15) if variant == 'force_overflow' else slice(9+3*na, None)] = maximum
    elif variant in ('volume', 'volume_nan'): a['volume'][0] = 0. if variant == 'volume' else np.nan
    elif variant in HOST:
        field, value = {'spinor': (9, 0), 'soc': (10, 2), 'split': (11, 2), 'capacity': (7, 0), 'components': (8, 3),
                        'negative_count': (0, -1), 'soc_components': (8, 2)}[variant]
        a['config'][field] = value
    elif variant not in ('normal', 'sticky'): raise ValueError('Unknown mutation: '+variant)
    return a


def expected_variants(row):
    variants = {'normal'}
    if row['components'] == 8 and row['neighbors'] == 19 and row['cycles'] == 1 and row['l'] in (0, 1, 5):
        variants |= STRUCTURAL | (NUMERICAL & {'gradient', 'value', 'xyz', 'density', 'epsilon'}) | UNUSED | HOST | {'overwritten_density', 'zero_blocks', 'sticky'}
        if row['l']:
            variants -= {'density_components', 'epsilon_second'}
            variants.add('epsilon_second_nan')
        if not row['soc']:
            variants.remove('soc_components')
            variants |= {v for v in NUMERICAL if v.endswith('_overflow')}
    if row['neighbors'] == 131073: variants |= {'tail_row', 'tail_gradient', 'tail_excluded'}
    return variants


def verify(path, upstream):
    report = json.loads(path.read_text())
    folder = path.parent
    reference_path = folder/report['oracle_file']
    reference = json.loads(reference_path.read_text())
    oracle_check = verify_oracle(reference_path, upstream)
    errors, checks, seen, binding_checks = [], [], set(), {}
    if not oracle_check['passed']: errors.append('Independent literal reference audit failed')
    for file_key, hash_key in (('output_file', 'output_sha256'), ('inputs_file', 'inputs_sha256'), ('oracle_file', 'oracle_sha256')):
        if sha(folder/report[file_key]) != report[hash_key]: errors.append('Hash differs: '+file_key)
    source_path = folder/reference['arrays_file']
    if sha(source_path) != report['oracle_arrays_sha256']: errors.append('Reference array hash differs')
    with np.load(source_path) as source, np.load(folder/report['inputs_file']) as changes, np.load(folder/report['output_file']) as outputs:
        source_rows = {r['fixture']: r for r in reference['checks']}
        for row in report['checks']:
            label, variant, captured = row['fixture'], row['variant'], row['captured']
            issues = []
            signature = label, variant, captured
            if signature in seen: issues.append('Duplicate scenario')
            seen.add(signature)
            key = label+'_'+variant
            if key not in binding_checks:
                base = {k: source[label+'_'+k] for k in (*FIELDS, 'config', 'volume')}
                wanted = mutated(base, variant)
                binding_issues = []
                bindings = report['input_bindings'][key]
                if set(bindings) != set(wanted): binding_issues.append('Input binding coverage differs')
                for field, expected in wanted.items():
                    binding = bindings[field]
                    recorded = (source if binding['source'] == 'oracle' else changes)[binding['key']]
                    if hashlib.sha256(recorded.tobytes()).hexdigest() != binding['sha256'] or recorded.dtype != expected.dtype or recorded.shape != expected.shape or not np.array_equal(recorded, expected, equal_nan=True):
                        binding_issues.append('Executed input differs: '+field)
                binding_checks[key] = dict(errors=binding_issues, inputs=len(wanted))
            issues.extend(binding_checks[key]['errors'])
            # Reconstruct shapes/status from the scenario, never from its
            # reported expected_status or passed fields.
            base_config = source[label+'_config']
            nn, nc, nb, nq, na, nr, no, cap, components, spinor, soc, split, cycles = map(int, base_config)
            effective_components = 2 if variant == 'soc_components' else components
            staged = 28*nc+16*nn+24*nb+8*nr+4*cap+4*nr*no+56*nq+8*cap*effective_components+8*(3*na+18)+4
            allocated = staged+8*cap+28*(nc == 0)+16*(nn == 0)+24*(nb == 0)+16*(nq == 0)
            actual = outputs[row['scenario']]
            error = None
            numerical = variant not in STRUCTURAL | NUMERICAL | HOST and not (variant == 'overwritten_density' and not soc)
            status = 1 if variant in STRUCTURAL or variant == 'tail_row' else 2 if variant in NUMERICAL or variant == 'tail_gradient' or (variant == 'overwritten_density' and not soc) else 7 if variant == 'sticky' else 0
            if variant in HOST:
                stage = 0 if variant in ('capacity', 'components', 'negative_count') else 3
                if row['native_return'] != -1 or row['stage'] != stage or row['status'] != -9 or row['exported'] != 0 or not np.isnan(actual).all(): issues.append('Invalid host configuration was not rejected')
                wanted_staged, wanted_allocated, poison = (0, 0, 0) if not stage else (staged, allocated, 8*cap)
            else:
                wanted_staged, wanted_allocated, poison = staged, allocated, 8*cap*cycles
                if row['native_return'] != 0 or row['stage'] != 4 or row['status'] != status: issues.append('Execution/status differs')
                if row['graph_nodes'] != 2*captured or row['forbidden_nodes'] != 0: issues.append('Capture contains wrong or forbidden nodes')
                if row['exported'] != 8*(3*na+18)+4: issues.append('Export count differs')
                if numerical:
                    suffix = '_without_last' if variant.startswith('tail_') else '_initial' if variant == 'zero_blocks' else '_expected'
                    expected = source[label+suffix]
                    delta = np.abs(actual-expected)
                    error = float(delta.max())
                    if not np.isfinite(actual).all() or np.any(delta > 1e-10+1e-10*np.abs(expected)): issues.append('Full output differs from pristine reference')
                elif variant in STRUCTURAL and not np.isfinite(actual).all(): issues.append('Invalid topology consumed uninitialized data')
            if actual.shape != (3*na+18,): issues.append('Output shape differs')
            if row['initial_status'] != (7 if variant == 'sticky' else 0): issues.append('Initial status differs')
            if row['staged'] != wanted_staged or row['allocated'] != wanted_allocated or row['poisoned'] != poison: issues.append('Shape-derived storage/staging/poison counts differ')
            checks.append(dict(scenario=row['scenario'], passed=not issues, errors=issues, numerical_comparison=numerical, max_absolute_error=error))
        expected = {(label, v, c) for label, r in source_rows.items() for v in expected_variants(r) for c in ((False,) if v in HOST else (False, True))}
        if seen != expected: errors.append('Scenario coverage differs')
        if set(outputs.files) != {r['scenario'] for r in report['checks']}: errors.append('Output archive coverage differs')
        if set(report['input_bindings']) != set(binding_checks): errors.append('Input binding coverage differs')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                oracle_verification=oracle_check, input_bindings_checked=sum(r['inputs'] for r in binding_checks.values()),
                report_sha256=sha(path), verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', type=Path)
    parser.add_argument('--upstream', type=Path, default=Path('upstream/siesta'))
    args = parser.parse_args()
    result = verify(args.check, args.upstream)
    args.check.with_name('nonlocal-compact-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Compact GPU audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
