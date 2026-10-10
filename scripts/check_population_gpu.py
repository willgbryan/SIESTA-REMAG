#!/usr/bin/env python3
"""Compare device populations with pristine Mulliken/dm_charge fixtures.

Captured execution must contain four kernels and no copies or allocation.
Fixture staging, output poisoning and final checking remain outside capture.
This does not qualify production ownership, output formatting or D removal.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np

from check_population import run as oracle, sha


class Problem(ct.Structure):
    _fields_ = [(k, ct.c_int) for k in ('capacity', 'rows', 'atoms', 'spins')]


class Info(ct.Structure):
    _fields_ = [(k, ct.c_size_t) for k in ('allocated', 'staged', 'exported', 'poisoned')]+[
        (k, ct.c_int) for k in ('graph_nodes', 'forbidden_nodes', 'stage')]


FIELDS = ('num', 'ptr', 'lasto', 'overlap', 'density')
FAULTS = ('problem_shape', 'negative_offset', 'bad_count', 'overlapping_rows', 'tail_span',
          'first_atom', 'last_atom', 'reversed_atoms', 'nonfinite_density', 'nonfinite_overlap',
          'arithmetic_overflow', 'unused_soc', 'host_shape')


def run(upstream, build, library, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ref = output.parent/'reference.json'
    oracle(upstream, build, ref)
    manifest = json.loads(ref.read_text())
    data = np.load(ref.with_suffix('.npz'))
    fn = ct.CDLL(str(library.resolve())).remag_check_population
    fn.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(Problem)]+[ct.c_void_p]*6+[
        ct.POINTER(ct.c_int), ct.POINTER(Info)]
    fn.restype = ct.c_int
    checks, outputs, inputs = [], {}, {}
    for row in manifest['checks']:
        label = row['fixture']
        variants = ['normal']
        if row['spins'] == 8 and row['rows'] == 257 and label.endswith('_finite_padding'):
            variants += list(FAULTS)
        if row['rows'] in (131073,262145) and label.endswith('_finite_padding'):
            variants += ['tail_span']
        for variant in variants:
            c = data[label+'_config'].copy()
            atoms, rows, capacity, spins = map(int, c)
            d = {k: data[label+'_'+k].copy(order='C') for k in FIELDS}
            p = Problem(capacity, rows, atoms, spins)
            expected = data[label+'_expected'].copy()
            wanted = 0
            used_row = int(np.flatnonzero(d['num'])[0]) if np.any(d['num']) else None
            used = int(d['ptr'][used_row]) if used_row is not None else None
            if variant == 'problem_shape':
                p.rows += 1
                wanted = 1
            elif variant == 'negative_offset':
                d['ptr'][0] = -1
                wanted = 1
            elif variant == 'bad_count':
                d['num'][used_row] = capacity+1
                wanted = 1
            elif variant == 'overlapping_rows':
                d['ptr'][used_row+1] = used
                wanted = 1
            elif variant == 'tail_span':
                d['ptr'][-1] = capacity
                wanted = 1
            elif variant == 'first_atom':
                d['lasto'][0] = 1
                wanted = 1
            elif variant == 'last_atom':
                d['lasto'][-1] = rows+1
                wanted = 1
            elif variant == 'reversed_atoms':
                d['lasto'][2] = d['lasto'][1]-1
                wanted = 1
            elif variant == 'nonfinite_density':
                d['density'][0, used] = np.nan
                wanted = 2
            elif variant == 'nonfinite_overlap':
                d['overlap'][used] = np.inf
                wanted = 2
            elif variant == 'arithmetic_overflow':
                d['density'][0, used] = np.finfo('d').max
                d['overlap'][used] = 2.
                wanted = 2
            elif variant == 'unused_soc':
                d['density'][4:6] = np.nan
            elif variant == 'host_shape':
                c[1] = 0
            key = label+'_'+variant
            inputs[key+'_config'] = c
            inputs[key+'_problem'] = np.array([p.capacity, p.rows, p.atoms, p.spins], np.int32)
            for name, value in d.items():
                inputs[key+'_'+name] = value
            for captured in (False, True):
                for replays in ((1, 3) if variant == 'normal' else (1,)):
                    name = key+f'_capture{int(captured)}_replays{replays}'
                    out, code, info = np.full(expected.shape, np.nan), ct.c_int(-9), Info()
                    result = fn(c.ctypes.data, int(captured), replays, ct.byref(p),
                                *[d[k].ctypes.data for k in FIELDS], out.ctypes.data, ct.byref(code), ct.byref(info))
                    if variant == 'host_shape':
                        error = None
                        valid = result == -1 and info.stage == 0 and info.allocated == info.staged == info.exported == 0 and code.value == -9 and np.isnan(out).all()
                    else:
                        staged = 16+4*(2*rows+atoms+1)+8*capacity*(spins+1)
                        exported = 8*expected.size+4
                        error = float(np.max(np.abs(out-expected))) if wanted == 0 else None
                        agree = np.isfinite(out).all() and (wanted != 0 or np.all(np.abs(out-expected) <= 1e-10+1e-12*np.abs(expected)))
                        valid = (result == 0 and code.value == wanted and info.stage == 4 and agree and
                                 info.staged == staged and info.allocated == staged+exported and info.exported == exported and
                                 info.poisoned == replays*exported and info.graph_nodes == 4*int(captured) and info.forbidden_nodes == 0)
                    outputs[name] = out
                    checks.append(dict(scenario=name, fixture=label, inputs_key=key, variant=variant, captured=captured,
                                       replays=replays, native_return=result, status=code.value, expected_status=wanted,
                                       max_absolute_error=error, passed=bool(valid), **{k: getattr(info, k) for k, _ in info._fields_}))
    outpath, inpath = output.with_suffix('.npz'), output.parent/'inputs.npz'
    np.savez_compressed(outpath, **outputs)
    np.savez_compressed(inpath, **inputs)
    result = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks,
                  checker_sha256=sha(Path(__file__)), native_library_sha256=sha(library),
                  output_file=outpath.name, output_sha256=sha(outpath), inputs_file=inpath.name, inputs_sha256=sha(inpath),
                  oracle_file=ref.name, oracle_sha256=sha(ref), oracle_arrays_sha256=sha(ref.with_suffix('.npz')))
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('Population CUDA:', len(checks), 'passed=', result['passed'])
    if not result['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'library', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.library, args.output)
