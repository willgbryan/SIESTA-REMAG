#!/usr/bin/env python3
"""Audit sparse nonlocal force fixtures with independent gathers/contractions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from check_nonlocal_force import blocks, FIELDS, sha


def predict(config, volume, data, drop_last=False):
    n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
    weights = np.zeros((nr, nc, components))
    for row in range(nr):
        for entry in range(data['ptr'][row], data['ptr'][row]+data['num'][row]):
            column = data['list'][entry]-1
            if kind == 0:
                weights[row, column] += data['density'][:, entry]
            else:
                weights[row, column] = data['density'][:, entry]
    density = weights[data['qr']-1, data['qc']-1]
    diagonal = density[:, :spinor].sum(axis=1)
    values, operators = data['scalar'], data['operators']
    soc = np.zeros((n, 4), complex)
    if kind == 0:
        scalar = diagonal*values[:, 0]*values[:, 1]*values[:, 2]
        force = (2*diagonal*values[:, 0]*values[:, 2])[:, None]*values[:, 3:]
    else:
        d11 = density[:, 0]+1j*density[:, 4]
        d22 = density[:, 1]+1j*density[:, 5]
        d12 = density[:, 2]+1j*density[:, 3]
        d21 = density[:, 6]-1j*density[:, 7]
        if kind == 1:
            potential = values[:, 0]*values[:, 1]*values[:, 2]
            scalar = diagonal*potential if split else np.zeros(n)
            if not split:
                soc[:, 0], soc[:, 1] = potential*d11, potential*d22
            force = (2*diagonal*(values[:, 0]*values[:, 2]))[:, None]*values[:, 3:]
        else:
            scalar = diagonal*operators[:, 0]
            v = operators[:, 1:5]+1j*operators[:, 5:9]
            g = (operators[:, 9:21]+1j*operators[:, 21:33]).reshape(n, 4, 3)
            soc = v[:, [0, 3, 2, 1]]*np.column_stack((d11, d22, d21, d12))
            force = 2*np.einsum('nka,nk->na', g, np.column_stack((d11, d12, d21, d22))).real
    if drop_last and n:
        scalar[-1] = 0.
        soc[-1] = 0.
        force[-1] = 0.
    result = data['initial'].copy()
    atom_forces = result[9:9+3*na].reshape(na, 3)
    stress = result[9+3*na:].reshape(3, 3)
    for _ in range(cycles):
        result[0] += scalar.sum()
        result[1:5] += soc.real.sum(axis=0)
        result[5:9] += soc.imag.sum(axis=0)
        np.add.at(atom_forces, data['atoms'][:, 0]-1, -force)
        np.add.at(atom_forces, data['atoms'][:, 1]-1, force)
        stress += force.T@data['xyz']/volume
    return result


def verify(path, upstream):
    report = json.loads(path.read_text())
    folder = path.parent
    archive = folder/report['arrays_file']
    wrapper = folder/report['wrapper_file']
    errors, checks, seen = [], [], set()
    snippets = blocks(upstream)
    if sha(upstream/'Src/nlefsm.f') != report['source_sha256'] or hashlib.sha256(''.join(snippets).encode()).hexdigest() != report['literal_blocks_sha256'] or any(s not in wrapper.read_text() for s in snippets):
        errors.append('Literal source differs from pinned upstream')
    if sha(archive) != report['arrays_sha256'] or sha(wrapper) != report['wrapper_sha256']:
        errors.append('Retained content hash differs')
    with np.load(archive) as arrays:
        for row in report['checks']:
            label = row['fixture']
            config = arrays[label+'_config']
            volume = arrays[label+'_volume']
            data = {k: arrays[label+'_'+k] for k in FIELDS}
            raw = config.tobytes()+volume.tobytes()+b''.join(data[k].tobytes() for k in FIELDS)
            issues = []
            n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
            signature = (kind, components, split, n, cycles)
            if signature in seen:
                issues.append('Repeated fixture')
            seen.add(signature)
            if hashlib.sha256(raw).hexdigest() != row['input_sha256']:
                issues.append('Input byte reconstruction differs')
            expected = predict(config, float(volume[0]), data)
            actual = arrays[label+'_expected']
            delta = np.abs(actual-expected)
            if actual.shape != (3*na+18,) or not np.isfinite(actual).all() or np.any(delta > 1e-10+1e-10*np.abs(expected)):
                issues.append('Independent scalar/SOC contraction differs')
            if label+'_without_last' in arrays:
                omitted = predict(config, float(volume[0]), data, True)
                last_delta = np.abs(arrays[label+'_without_last']-omitted)
                if not np.isfinite(last_delta).all() or np.any(last_delta > 1e-10+1e-10*np.abs(omitted)):
                    issues.append('Independent omitted-tail contraction differs')
                delta = np.maximum(delta, last_delta)
            checks.append(dict(fixture=label, passed=not issues, errors=issues, max_absolute_error=float(delta.max())))
    expected = {(0, c, 0, n, r) for c in (1, 2, 4, 8) for n in (0, 1, 257) for r in (1, 3)}
    expected.update((k, 8, s, n, r) for k in (1, 2) for s in (0, 1) for n in (0, 1, 257) for r in (1, 3))
    expected.update((k, 8, int(k != 0), 131073, r) for k in (0, 1, 2) for r in (1, 3))
    if seen != expected or len(checks) != 54:
        errors.append('Fixture coverage differs')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                report_sha256=sha(path), verifier_sha256=sha(Path(__file__)), gpu_validation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--upstream', type=Path, default=Path('upstream/siesta'))
    args = parser.parse_args()
    result = verify(args.reference, args.upstream)
    args.reference.with_name('nonlocal-force-oracle-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Nonlocal-force oracle audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
