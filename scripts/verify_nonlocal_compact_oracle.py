#!/usr/bin/env python3
"""Audit compact references using independent angular matrices and CSR gathers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from check_nonlocal_force import blocks, sha
from check_soc_projector import block
from verify_nonlocal_force_oracle import predict as contract
from verify_soc_projector_oracle import predict as projector

FIELDS = ('centers', 'neighbors', 'block_indices', 'epsilon', 'num', 'ptr', 'list',
          'lookup', 'integrals', 'xyz', 'density', 'initial')


def predict(config, volume, data, drop_last=False):
    nn, nc, nb, nq, na, nr, no, cap, components, spinor, soc, split, cycles = map(int, config)
    present = [set(data['list'][offset:offset+count]) for offset, count in zip(data['ptr'], data['num'])]
    jobs = defaultdict(list)
    for base, count, block_base, block_count, width, query, atom in data['centers']:
        for i in range(base, base+count):
            row, _, left_atom, _ = data['neighbors'][i]
            if not row or (drop_last and i == nn-1): continue
            left = query+(i-base)*width
            for j in range(base, base+count):
                column = data['neighbors'][j, 1]
                if column not in present[row-1]: continue
                right = query+(j-base)*width
                for b in range(block_base, block_base+block_count):
                    offset, l = data['block_indices'][b]
                    key = (int(l), *map(float, data['epsilon'][b]))
                    jobs[key].append((left+offset, right+offset, left, row, column, left_atom, atom))
    result = data['initial'].copy()
    for (l, eps0, eps1), group in jobs.items():
        entries = np.array(group, np.int64)
        for start in range(0, len(entries), 16384):
            a = entries[start:start+16384]
            n = len(a)
            scalar, operators = np.zeros((n, 6)), np.zeros((n, 33))
            if l:
                width = 2*(2*l+1)
                left = data['integrals'][a[:, 0, None]+np.arange(width)]
                right = data['integrals'][a[:, 1, None]+np.arange(width)]
                operators = projector(l, split, dict(epsilon=np.array([eps0, eps1]),
                    left=left[:, :, 0].reshape(n, 2, 2*l+1), right=right[:, :, 0].reshape(n, 2, 2*l+1),
                    left_gradient=left[:, :, 1:].reshape(n, 2, 2*l+1, 3)))
            else:
                scalar[:, 0] = eps0
                scalar[:, 1] = data['integrals'][a[:, 0], 0]
                scalar[:, 2] = data['integrals'][a[:, 1], 0]
                scalar[:, 3:] = data['integrals'][a[:, 0], 1:]
            expanded = dict(qr=a[:, 3], qc=a[:, 4], atoms=a[:, 5:7], num=data['num'], ptr=data['ptr'], list=data['list'],
                            scalar=scalar, operators=operators, xyz=data['xyz'][a[:, 2]], density=data['density'], initial=np.zeros_like(result))
            expanded_config = [n, na, nr, no, cap, components, spinor, cycles, (2 if l else 1) if soc else 0, split]
            result += contract(expanded_config, volume, expanded)
    return result


def verify(path, upstream):
    report = json.loads(path.read_text())
    archive, wrapper = (path.parent/report[k] for k in ('arrays_file', 'wrapper_file'))
    snippets = (*blocks(upstream), block(upstream))
    errors, checks, seen = [], [], set()
    if sha(upstream/'Src/nlefsm.f') != report['source_sha256'] or hashlib.sha256(''.join(snippets).encode()).hexdigest() != report['literal_blocks_sha256'] or any(s not in wrapper.read_text() for s in snippets):
        errors.append('Literal source differs from pinned upstream')
    if sha(archive) != report['arrays_sha256'] or sha(wrapper) != report['wrapper_sha256']:
        errors.append('Retained content hash differs')
    with np.load(archive) as arrays:
        for row in report['checks']:
            label = row['fixture']
            config, volume = arrays[label+'_config'], arrays[label+'_volume']
            data = {k: arrays[label+'_'+k] for k in FIELDS}
            nn, nc, nb, nq, na, nr, no, cap, components, spinor, soc, split, cycles = map(int, config)
            issues = []
            signature = (soc, components, split, row['l'], nn, cycles)
            if signature in seen: issues.append('Repeated fixture')
            seen.add(signature)
            raw = config.tobytes()+volume.tobytes()+b''.join(data[k].tobytes() for k in FIELDS)
            if hashlib.sha256(raw).hexdigest() != row['input_sha256']: issues.append('Input bytes differ')
            lookup = np.full((nr, no), -1, np.int32)
            for r in range(nr):
                for i in range(data['ptr'][r], data['ptr'][r]+data['num'][r]): lookup[r, data['list'][i]-1] = i
            if not np.array_equal(data['lookup'], lookup): issues.append('Lookup differs from CSR last-entry map')
            maximum = 0.
            for omitted, suffix in ((False, '_expected'), (True, '_without_last')):
                expected = predict(config, float(volume[0]), data, omitted)
                actual = arrays[label+suffix]
                delta = np.abs(actual-expected)
                maximum = max(maximum, float(delta.max()))
                if actual.shape != (3*na+18,) or not np.isfinite(actual).all() or np.any(delta > 1e-10+1e-10*np.abs(expected)):
                    issues.append('Independent compact contraction differs: '+suffix)
            checks.append(dict(fixture=label, passed=not issues, errors=issues, max_absolute_error=maximum))
    expected = {(0, c, 0, 0, n, r) for c in (1, 2, 4, 8) for n in (0, 1, 19) for r in (1, 3)}
    expected.update((1, 8, s, l, n, r) for s in (0, 1) for l in range(6) for n in (1, 19) for r in (1, 3))
    expected.add((0, 8, 0, 0, 131073, 1))
    expected.update((1, 8, s, 1, 131073, 1) for s in (0, 1))
    if seen != expected or len(checks) != 75: errors.append('Fixture coverage differs')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
                report_sha256=sha(path), verifier_sha256=sha(Path(__file__)), gpu_validation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--upstream', type=Path, default=Path('upstream/siesta'))
    args = parser.parse_args()
    result = verify(args.reference, args.upstream)
    args.reference.with_name('nonlocal-compact-oracle-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Compact oracle audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
