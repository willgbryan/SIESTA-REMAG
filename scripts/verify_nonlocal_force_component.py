#!/usr/bin/env python3
"""Audit retained nonlocal GPU contractions with independent sparse gathers."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_nonlocal_force_oracle import verify as verify_oracle
from check_nonlocal_force import FIELDS, sha


def agree(actual, expected):
    finite = np.isfinite(expected)
    return (actual.shape == expected.shape and np.isfinite(actual[finite]).all() and
            np.all(np.abs(actual[finite]-expected[finite]) <= 1e-10+1e-10*np.abs(expected[finite])) and
            np.array_equal(actual[~finite], expected[~finite]))


def evaluate(data, config, volume, initial):
    n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
    out = data['initial'].copy()
    status = initial
    if kind not in (0, 1, 2) or split not in (0, 1) or not np.isfinite(volume) or volume <= 0 or (kind and (components != 8 or spinor != 2)):
        return out, max(status, 1)
    weights = np.zeros((nr, nc, components))
    valid_row = np.ones(nr, bool)
    with np.errstate(invalid='ignore', over='ignore'):
        for row in range(nr):
            offset, count = int(data['ptr'][row]), int(data['num'][row])
            if offset < 0 or offset > cap or count < 0 or count > cap-offset:
                valid_row[row] = False
                continue
            for entry in range(offset, offset+count):
                col = int(data['list'][entry])
                if col < 1 or col > nc:
                    valid_row[row] = False
                    break
                if kind == 0: weights[row, col-1] += data['density'][:components, entry]
                else: weights[row, col-1] = data['density'][:components, entry]
        qr, qc, atoms = data['qr'].astype('int64'), data['qc'].astype('int64'), data['atoms']
        valid = (qr >= 1)&(qr <= nr)&(qc >= 1)&(qc <= nc)&np.all((atoms >= 1)&(atoms <= na), axis=1)
        ri, ci = np.clip(qr-1, 0, nr-1), np.clip(qc-1, 0, nc-1)
        valid &= valid_row[ri]
        if (~valid).any(): status = max(status, 1)
        density = weights[ri, ci]
        diagonal = density[:, :spinor].sum(axis=1)
        scalar, operators = data['scalar'], data['operators']
        energy = np.zeros((n, 9))
        if kind == 0:
            coefficient = diagonal*scalar[:, 0]
            energy[:, 0] = coefficient*scalar[:, 1]*scalar[:, 2]
            force = (2*coefficient*scalar[:, 2])[:, None]*scalar[:, 3:]
        else:
            d11, d22 = density[:, 0]+1j*density[:, 4], density[:, 1]+1j*density[:, 5]
            if kind == 1:
                potential = scalar[:, 0]*scalar[:, 1]*scalar[:, 2]
                if split: energy[:, 0] = diagonal*potential
                else:
                    energy[:, 1], energy[:, 2] = potential*density[:, 0], potential*density[:, 1]
                    energy[:, 5], energy[:, 6] = potential*density[:, 4], potential*density[:, 5]
                force = (2*diagonal*(scalar[:, 0]*scalar[:, 2]))[:, None]*scalar[:, 3:]
            else:
                d12, d21 = density[:, 2]+1j*density[:, 3], density[:, 6]-1j*density[:, 7]
                v = operators[:, 1:5]+1j*operators[:, 5:9]
                g = (operators[:, 9:21]+1j*operators[:, 21:33]).reshape(n, 4, 3)
                contribution = v[:, [0, 3, 2, 1]]*np.column_stack((d11, d22, d21, d12))
                energy[:, 0] = diagonal*operators[:, 0]
                energy[:, 1:5], energy[:, 5:9] = contribution.real, contribution.imag
                force = 2*np.einsum('nka,nk->na', g, np.column_stack((d11, d12, d21, d22))).real
        stress = force[:, :, None]*data['xyz'][:, None, :]/volume
        finite = np.isfinite(diagonal)&np.isfinite(energy).all(axis=1)&np.isfinite(force).all(axis=1)&np.isfinite(stress).all(axis=(1, 2))
        if (valid&~finite).any(): status = max(status, 2)
        selected = valid&finite
        atom_force, out_stress = out[9:9+3*na].reshape(na, 3), out[9+3*na:].reshape(3, 3)
        for _ in range(cycles):
            out[:9] += energy[selected].sum(axis=0)
            np.add.at(atom_force, atoms[selected, 0]-1, -force[selected])
            np.add.at(atom_force, atoms[selected, 1]-1, force[selected])
            out_stress += stress[selected].sum(axis=0)
        if not np.isfinite(out).all(): status = max(status, 2)
    return out, status


def declared(source, label, variant):
    config = source[label+'_config'].copy()
    volume = float(source[label+'_volume'][0])
    data = {k: source[label+'_'+k].copy() for k in FIELDS}
    n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
    initial = 7 if variant == 'sticky' else 0
    if variant == 'volume': volume = 0.
    elif variant == 'volume_nan': volume = float('nan')
    elif variant == 'kind': config[8] = -1
    elif variant == 'split': config[9] = 2
    elif variant == 'row': data['qr'].fill(-2147483648)
    elif variant == 'column': data['qc'].fill(nc+1)
    elif variant == 'atom': data['atoms'][:, 0] = -2147483648
    elif variant == 'center': data['atoms'][:, 1] = na+1
    elif variant == 'offset': data['ptr'].fill(-1)
    elif variant == 'count': data['num'].fill(cap+1)
    elif variant == 'list': data['list'].fill(nc+1)
    elif variant == 'soc_components': config[5] = 2
    elif variant == 'active_gradient':
        if kind == 2: data['operators'][:, 9:] = float('nan')
        else: data['scalar'][:, 3:] = float('nan')
    elif variant == 'active_value': data['operators' if kind == 2 else 'scalar'][:, 0] = float('nan')
    elif variant == 'xyz': data['xyz'].fill(float('inf'))
    elif variant == 'density': data['density'].fill(float('nan'))
    elif variant == 'unused_input': data['scalar' if kind == 2 else 'operators'].fill(float('nan'))
    elif variant == 'unused_density':
        if kind == 0: data['density'][spinor:] = float('nan')
        else: data['density'][[2, 3, 4, 5, 6, 7] if split else [2, 3, 6, 7]] = float('nan')
    elif variant == 'overwritten_density':
        data['density'][0, 1] = float('nan')
        if kind == 0:
            data['qr'].fill(1)
            data['qc'].fill(1)
    elif variant in ('arithmetic_overflow', 'force_overflow', 'energy_overflow', 'imag_energy_overflow', 'stress_overflow'):
        data['qr'].fill(1)
        data['qc'].fill(1)
        data['atoms'][:] = (1, 2)
        for field in ('scalar', 'operators', 'xyz', 'density'): data[field].fill(0.)
        weight = .5 if variant in ('force_overflow', 'stress_overflow') else 1.
        data['density'][0, 1:3] = weight/2 if kind == 0 else weight
        data['scalar'][:, 0] = data['scalar'][:, 2] = 1.
        maximum = np.finfo(np.float64).max
        if variant == 'arithmetic_overflow':
            data['scalar'][:, 3:] = data['operators'][:, 9:12] = maximum
        elif variant == 'force_overflow':
            data['scalar'][:, 3:] = data['operators'][:, 9:12] = maximum/2
        elif variant == 'stress_overflow':
            data['scalar'][:, 3:] = data['operators'][:, 9:12] = 1.
            data['xyz'].fill(maximum/2)
            volume = 1.
        elif kind != 2:
            data['scalar'][:, 0] = maximum/2
            data['scalar'][:, 1] = 1.
        else:
            channel = 5 if variant == 'imag_energy_overflow' else 0 if split else 1
            data['operators'][:, channel] = maximum/2
    elif variant == 'tail_row': data['qr'][-1] = 0
    elif variant == 'tail_gradient': data['operators' if kind == 2 else 'scalar'][-1, 9 if kind == 2 else 3] = float('nan')
    elif variant == 'spinor': config[6] = 0
    elif variant == 'capacity': config[4] = 0
    elif variant == 'components': config[5] = 3
    elif variant == 'negative_count': config[0] = -1
    return config, volume, data, initial


def verify(root):
    folder = root/'nonlocal-force-check'
    path = folder/'check.json'
    report = json.loads(path.read_text())
    reference_path = folder/report['oracle_file']
    oracle = json.loads(reference_path.read_text())
    errors, checks = [], []
    files = ((reference_path, report['oracle_sha256']), (folder/oracle['arrays_file'], report['oracle_arrays_sha256']),
             (folder/report['inputs_file'], report['inputs_sha256']), (folder/report['output_file'], report['output_sha256']))
    for file, expected in files:
        if sha(file) != expected: errors.append('Retained content hash differs: '+file.name)
    meta = json.loads((root/'metadata.json').read_text())
    if report['native_library_sha256'] != meta['native_binary_sha256']['/opt/candidate/remag-native/libremag_nonlocal_force_check.so']:
        errors.append('Native library differs from invocation binary identity')
    audit = verify_oracle(reference_path, Path('upstream/siesta'))
    (root/'nonlocal-force-oracle-verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    if not audit['passed']: errors.append('Independent pristine oracle audit failed')
    host = ('spinor', 'capacity', 'components', 'negative_count')
    faults = ('volume', 'volume_nan', 'kind', 'split', 'row', 'column', 'atom', 'center', 'offset', 'count', 'list',
              'active_gradient', 'active_value', 'xyz', 'density', 'unused_input', 'unused_density', 'overwritten_density',
              'arithmetic_overflow', 'force_overflow', 'energy_overflow', 'stress_overflow', 'sticky', 'tail_row', 'tail_gradient')
    required = set()
    for row in oracle['checks']:
        variants = ['normal']
        if row['components'] == 8 and row['queries'] == 257 and row['cycles'] == 1:
            variants += list(faults)+list(host)
            if row['kind'] != 0: variants.append('soc_components')
            if row['kind'] == 2:
                variants.remove('unused_density')
                variants.append('imag_energy_overflow')
        if row['queries'] == 131073 and row['cycles'] == 1: variants += ['tail_row', 'tail_gradient']
        for variant in variants:
            for captured in ((False,) if variant in host else (False, True)):
                required.add((row['fixture'], variant, captured))
    seen, seen_bindings, verified_mutations = set(), set(), set()
    cache = {}
    with np.load(folder/oracle['arrays_file']) as source, np.load(folder/report['inputs_file']) as mutations, np.load(folder/report['output_file']) as outputs:
        for row in report['checks']:
            label, variant, captured = row['fixture'], row['variant'], row['captured']
            identity = label, variant, captured
            prefix = label+'_'+variant
            name = prefix+f'_captured{int(captured)}'
            issues = []
            if identity not in required or identity in seen:
                errors.append('Unexpected/repeated scenario: '+name)
                continue
            seen.add(identity)
            if prefix not in cache:
                config, volume, data, initial = declared(source, label, variant)
                binding_errors = []
                bindings = report['input_bindings'][prefix]
                if set(bindings) != {*FIELDS, 'config', 'volume'}:
                    binding_errors.append('Missing/extra native input bindings')
                for field, expected in {**data, 'config':config, 'volume':np.array([volume])}.items():
                    binding = bindings[field]
                    if binding['source'] not in ('oracle', 'mutations'):
                        raise ValueError('Unexpected input provenance')
                    retained = source[binding['key']] if binding['source'] == 'oracle' else mutations[binding['key']]
                    if binding['source'] == 'mutations': verified_mutations.add(binding['key'])
                    if retained.dtype != expected.dtype or retained.shape != expected.shape or not np.array_equal(retained, expected, equal_nan=True):
                        binding_errors.append('Executed input differs: '+field)
                    if hashlib.sha256(retained.tobytes()).hexdigest() != binding['sha256']:
                        binding_errors.append('Raw input digest differs: '+field)
                predicted, status = (None, None) if variant in host else evaluate(data, config, volume, initial)
                cache[prefix] = config, initial, predicted, status, binding_errors
                seen_bindings.add(prefix)
            config, initial, predicted, status, binding_errors = cache[prefix]
            issues += binding_errors
            n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
            actual = outputs[name]
            staged = 352*n+8*nr+4*cap+8*cap*components+24*na+164
            maximum = None
            if variant in host:
                stage = 3 if variant == 'spinor' else 0
                if row['native_return'] != -1 or row['stage'] != stage or row['exported'] != 0 or row['status'] != -9 or not np.isnan(actual).all():
                    issues.append('Host configuration rejection differs')
                allocation = staged if stage == 3 else 0
                if row['allocated'] != allocation or row['staged'] != allocation:
                    issues.append('Host rejection allocation/staging differs')
            else:
                if not agree(actual, predicted): issues.append('Independent scalar/SOC contraction differs')
                if variant in ('normal', 'sticky', 'unused_input', 'unused_density') or (variant == 'overwritten_density' and kind != 0):
                    pristine = source[label+'_expected']
                elif variant in ('tail_row', 'tail_gradient'):
                    pristine = source[label+'_without_last']
                else: pristine = None
                if pristine is not None and not agree(actual, pristine):
                    issues.append('Literal Fortran output differs')
                if np.isfinite(predicted).all(): maximum = float(np.max(np.abs(actual-predicted)))
                expected = dict(native_return=0, stage=4, status=status, expected_status=status, graph_nodes=int(captured), forbidden_nodes=0,
                                staged=staged, allocated=staged+(36 if n == 0 else 0), exported=8*(3*na+18)+4)
                if any(row.get(k) != v for k, v in expected.items()): issues.append('Status, graph or shape-derived boundaries differ')
            if row['initial_status'] != initial or row['inputs_key'] != prefix or row['scenario'] != name or not row['passed']:
                issues.append('Scenario identity or final verdict differs')
            checks.append(dict(scenario=name, passed=not issues, errors=issues, max_absolute_error=maximum))
        if seen != required or len(checks) != 398 or set(outputs.files) != {r['scenario'] for r in report['checks']}:
            errors.append('Scenario/output coverage differs')
        if seen_bindings != set(report['input_bindings']) or verified_mutations != set(mutations.files):
            errors.append('Retained input coverage differs')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                reconstructed_input_bindings=sum(len(b) for b in report['input_bindings'].values()), mutated_arrays=len(verified_mutations),
                report_sha256=sha(path), verifier_sha256=sha(Path(__file__)), whole_process_cuda_trace=False, material_integration=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    for name in ('verify_nonlocal_force_component.py', 'verify_nonlocal_force_oracle.py'):
        (args.run/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (args.run/'nonlocal-force-component-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Nonlocal force component audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
