#!/usr/bin/env python3
"""Qualify the device off-site SOC projector against literal pristine Fortran."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
from check_soc_projector import run as oracle, sha, FIELDS

FAULTS = ('epsilon_nan', 'left_nan', 'right_nan', 'gradient_nan', 'overflow', 'sticky')
HOST = ('l_zero', 'l_high', 'split', 'negative_count')


class Info(ct.Structure):
    _fields_ = [(k, ct.c_size_t) for k in ('allocated', 'staged', 'exported', 'poisoned')]+[(k, ct.c_int) for k in ('graph_nodes', 'forbidden_nodes', 'stage')]


def inputs(arrays, label, variant):
    config = arrays[label+'_config'].copy()
    data = {k: arrays[label+'_'+k].copy(order='C') for k in FIELDS}
    expected = arrays[label+'_expected'].copy()
    status = initial = 0
    if variant == 'empty':
        config[2] = 0
        data = {k: v if k == 'epsilon' else v[:0] for k, v in data.items()}
        expected = expected[:0]
    elif variant in FAULTS or variant == 'tail_nan':
        status = 2
        if variant.endswith('_nan'):
            key = dict(epsilon_nan='epsilon', left_nan='left', right_nan='right', gradient_nan='left_gradient', tail_nan='left_gradient')[variant]
            if variant == 'tail_nan':
                data[key][-1, -1, -1, -1] = np.nan
                expected = None  # Sticky fault makes all other query values unusable.
            else:
                data[key][:] = np.nan
        elif variant == 'overflow':
            data['left_gradient'][:] = np.finfo('d').max
            data['right'][:] = 8.
        else:
            status = initial = 7
        if expected is not None:
            expected[:] = 0.
    elif variant in HOST:
        if variant == 'l_zero': config[0] = 0
        elif variant == 'l_high': config[0] = 6
        elif variant == 'split': config[1] = 2
        else: config[2] = -1
    return config, data, expected, status, initial


def run(upstream, build, library, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = output.parent/'reference.json'
    oracle(upstream, build, reference)
    manifest = json.loads(reference.read_text())
    arrays = np.load(reference.with_suffix('.npz'))
    fn = ct.CDLL(str(library.resolve())).remag_check_soc_projector
    fn.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]+[ct.c_void_p]*5+[ct.POINTER(ct.c_int), ct.POINTER(Info)]
    fn.restype = ct.c_int
    checks, retained, executed = [], {}, {}
    for row in manifest['checks']:
        label = row['fixture']
        variants = ['normal']
        if row['l'] == 2 and row['queries'] == 257:
            variants += ['empty', *FAULTS, *HOST]
        if row['queries'] == 131073:
            variants += ['tail_nan']
        for variant in variants:
            config, data, expected, wanted, initial = inputs(arrays, label, variant)
            key = label+'_'+variant
            if variant != 'normal':
                executed[key+'_config'] = config
                for field in FIELDS:
                    executed[key+'_'+field] = data[field]
            for captured in ((False,) if variant in HOST else (False, True)):
                cycles = 3 if captured else 1
                name = key+f'_captured{int(captured)}'
                out = np.full((max(0, int(config[2])), 33), np.nan)
                code, info = ct.c_int(-9), Info()
                result = fn(config.ctypes.data, int(captured), cycles, initial,
                            *[data[k].ctypes.data for k in ('epsilon', 'left', 'right', 'left_gradient')],
                            out.ctypes.data, ct.byref(code), ct.byref(info))
                l, split, n = map(int, config)
                error = None
                if variant in HOST:
                    valid = result == -1 and code.value == -9 and all(getattr(info, k) == 0 for k, _ in info._fields_) and np.isnan(out).all()
                else:
                    staged = 20+80*(2*l+1)*n
                    valid = (result == 0 and code.value == wanted and info.stage == 4 and info.staged == staged and
                             info.allocated == staged+264*n+(32 if n == 0 else 0) and info.exported == 264*n+4 and
                             info.poisoned == 264*n*cycles and info.graph_nodes == int(captured) and info.forbidden_nodes == 0 and np.isfinite(out).all())
                    if expected is not None:
                        error = float(np.max(np.abs(out-expected))) if n else 0.
                        valid = valid and np.all(np.abs(out-expected) <= 1e-12+1e-12*np.abs(expected))
                    else:
                        valid = valid and np.array_equal(out[-1], np.zeros(33))
                retained[name] = out
                checks.append(dict(scenario=name, fixture=label, variant=variant, captured=captured, cycles=cycles,
                                   initial_status=initial, status=code.value, expected_status=wanted, native_return=result,
                                   passed=bool(valid), max_absolute_error=error, **{k: getattr(info, k) for k, _ in info._fields_}))
    values = output.with_suffix('.npz')
    mutations = output.parent/'mutations.npz'
    np.savez_compressed(values, **retained)
    np.savez_compressed(mutations, **executed)
    report = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks,
                  output_file=values.name, output_sha256=sha(values), inputs_file=mutations.name, inputs_sha256=sha(mutations),
                  oracle_file=reference.name, oracle_sha256=sha(reference), oracle_arrays_sha256=sha(reference.with_suffix('.npz')),
                  checker_sha256=sha(Path(__file__)), native_library_sha256=sha(library))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('SOC projector CUDA:', len(checks), 'passed=', report['passed'])
    if not report['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'library', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.library, args.output)
