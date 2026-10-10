#!/usr/bin/env python3
"""Qualify scalar and complex SOC nonlocal GPU contractions against pristine Fortran."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from check_nonlocal_force import run as oracle, sha, FIELDS

FAULTS = ('volume', 'volume_nan', 'kind', 'split', 'row', 'column', 'atom', 'center', 'offset', 'count', 'list',
          'active_gradient', 'active_value', 'xyz', 'density', 'unused_input', 'unused_density', 'overwritten_density',
          'arithmetic_overflow', 'force_overflow', 'energy_overflow', 'stress_overflow', 'sticky', 'tail_row', 'tail_gradient')
HOST = ('spinor', 'capacity', 'components', 'negative_count')


class Info(ct.Structure):
    _fields_ = [(k, ct.c_size_t) for k in ('allocated', 'staged', 'exported')]+[(k, ct.c_int) for k in ('graph_nodes', 'forbidden_nodes', 'stage')]


def variants(row):
    result = ['normal']
    if row['components'] == 8 and row['queries'] == 257 and row['cycles'] == 1:
        result += list(FAULTS)+list(HOST)
        if row['kind']:
            result.append('soc_components')
        if row['kind'] == 2:
            result.remove('unused_density')
            result.append('imag_energy_overflow')
    if row['queries'] == 131073 and row['cycles'] == 1:
        result += ['tail_row', 'tail_gradient']
    return result


def inputs(source, label, variant):
    config = source[label+'_config'].copy()
    volume = float(source[label+'_volume'][0])
    data = {k: source[label+'_'+k].copy() for k in FIELDS}
    expected = source[label+'_expected'].copy()
    n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
    wanted = initial = 0
    if variant in ('volume', 'volume_nan', 'kind', 'split', 'row', 'column', 'atom', 'center', 'offset', 'count', 'list', 'soc_components'):
        wanted = 1
        expected = data['initial'].copy()
        if variant == 'volume': volume = 0.
        elif variant == 'volume_nan': volume = np.nan
        elif variant == 'kind': config[8] = -1
        elif variant == 'split': config[9] = 2
        elif variant == 'row': data['qr'][:] = np.iinfo('i').min
        elif variant == 'column': data['qc'][:] = nc+1
        elif variant == 'atom': data['atoms'][:, 0] = np.iinfo('i').min
        elif variant == 'center': data['atoms'][:, 1] = na+1
        elif variant == 'offset': data['ptr'][:] = -1
        elif variant == 'count': data['num'][:] = cap+1
        elif variant == 'list': data['list'][:] = nc+1
        else: config[5] = 2
    elif variant in ('active_gradient', 'active_value', 'xyz', 'density'):
        wanted = 2
        expected = data['initial'].copy()
        if variant == 'active_gradient':
            data['operators' if kind == 2 else 'scalar'][:, 9 if kind == 2 else 3:] = np.nan
        elif variant == 'active_value':
            data['operators' if kind == 2 else 'scalar'][:, 0] = np.nan
        else: data[variant][:] = np.inf if variant == 'xyz' else np.nan
    elif variant == 'unused_input':
        data['scalar' if kind == 2 else 'operators'][:] = np.nan
    elif variant == 'unused_density':
        selected = list(range(spinor, components)) if kind == 0 else ([2, 3, 4, 5, 6, 7] if split else [2, 3, 6, 7])
        data['density'][selected] = np.nan
    elif variant == 'overwritten_density':
        data['density'][0, 1] = np.nan
        if kind == 0:
            # Only queries for row 1/column 1 are invalidated; preserve a
            # literal reference excluding all other density-dependent terms
            # by directing every query to that duplicate pair.
            data['qr'][:] = data['qc'][:] = 1
            wanted = 2
            expected = data['initial'].copy()
    elif variant in ('arithmetic_overflow', 'force_overflow', 'energy_overflow', 'imag_energy_overflow', 'stress_overflow'):
        data['qr'][:] = data['qc'][:] = 1
        data['atoms'][:] = [1, 2]
        data['scalar'][:] = data['operators'][:] = 0.
        data['xyz'][:] = data['density'][:] = 0.
        # Scalar duplicate entries sum; SOC selects the second one.
        weight = .5 if variant in ('force_overflow', 'stress_overflow') else 1.
        data['density'][0, 1:3] = weight/2 if kind == 0 else weight
        data['scalar'][:, 0] = 1.
        data['scalar'][:, 2] = 1.
        expected = data['initial'].copy()
        wanted = 2
        maximum = np.finfo('d').max
        if variant == 'arithmetic_overflow':
            data['scalar'][:, 3:] = maximum
            data['operators'][:, 9:12] = maximum
        elif variant == 'force_overflow':
            data['scalar'][:, 3:] = maximum/2
            data['operators'][:, 9:12] = maximum/2
            expected[9:12], expected[12:15] = -np.inf, np.inf
        elif variant == 'stress_overflow':
            data['scalar'][:, 3:] = data['operators'][:, 9:12] = 1.
            data['xyz'][:] = maximum/2
            volume = 1.
            expected[9:12] -= n
            expected[12:15] += n
            expected[9+3*na:] = np.inf
        else:
            if kind != 2:
                data['scalar'][:, 0] = maximum/2
                data['scalar'][:, 1] = 1.
            elif variant == 'imag_energy_overflow': data['operators'][:, 5] = maximum/2
            elif split: data['operators'][:, 0] = maximum/2
            else: data['operators'][:, 1] = maximum/2
            index = 5 if variant == 'imag_energy_overflow' else 0 if kind == 0 or split else 1
            expected[index] = np.inf
    elif variant == 'sticky': wanted = initial = 7
    elif variant in ('tail_row', 'tail_gradient'):
        expected = source[label+'_without_last'].copy()
        wanted = 1 if variant == 'tail_row' else 2
        if variant == 'tail_row': data['qr'][-1] = 0
        elif kind == 2: data['operators'][-1, 9] = np.nan
        else: data['scalar'][-1, 3] = np.nan
    elif variant == 'spinor': config[6] = 0
    elif variant == 'capacity': config[4] = 0
    elif variant == 'components': config[5] = 3
    elif variant == 'negative_count': config[0] = -1
    return config, volume, data, expected, wanted, initial


def run(upstream, build, library, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = output.parent/'reference.json'
    oracle(upstream, build, reference)
    manifest = json.loads(reference.read_text())
    source = np.load(reference.with_suffix('.npz'))
    fn = ct.CDLL(str(library.resolve())).remag_check_nonlocal_force
    fn.argtypes = [ct.c_void_p, ct.c_double, ct.c_int, ct.c_int]+[ct.c_void_p]*12+[ct.POINTER(ct.c_int), ct.POINTER(Info)]
    fn.restype = ct.c_int
    checks, outputs, mutations, bindings = [], {}, {}, {}
    for row in manifest['checks']:
        label = row['fixture']
        for variant in variants(row):
            config, volume, data, expected, wanted, initial = inputs(source, label, variant)
            prefix = label+'_'+variant
            bindings[prefix] = {}
            for field, array in {**data, 'config':config, 'volume':np.array([volume])}.items():
                original = source[label+'_'+field]
                unchanged = array.shape == original.shape and array.dtype == original.dtype and np.array_equal(array, original, equal_nan=True)
                key = label+'_'+field if unchanged else prefix+'_'+field
                if not unchanged: mutations[key] = array
                bindings[prefix][field] = dict(source='oracle' if unchanged else 'mutations', key=key, sha256=hashlib.sha256(array.tobytes()).hexdigest())
            for captured in ((False,) if variant in HOST else (False, True)):
                name = prefix+f'_captured{int(captured)}'
                out = np.full(expected.shape, np.nan)
                code, info = ct.c_int(-9), Info()
                result = fn(config.ctypes.data, volume, int(captured), initial, *[data[k].ctypes.data for k in FIELDS], out.ctypes.data, ct.byref(code), ct.byref(info))
                n, na, nr, nc, cap, components, spinor, cycles, kind, split = map(int, config)
                staged = 352*n+8*nr+4*cap+8*cap*components+24*na+164
                error = None
                if variant in HOST:
                    stage = 3 if variant == 'spinor' else 0
                    valid = result == -1 and info.stage == stage and info.exported == 0 and code.value == -9 and np.isnan(out).all()
                    if stage == 0: valid = valid and info.staged == info.allocated == 0
                    else: valid = valid and info.staged == info.allocated == staged
                else:
                    finite = np.isfinite(expected).all()
                    error = float(np.max(np.abs(out-expected))) if finite else None
                    selected = np.isfinite(expected)
                    agree = (np.isfinite(out[selected]).all() and np.all(np.abs(out[selected]-expected[selected]) <= 1e-10+1e-10*np.abs(expected[selected]))
                             and np.array_equal(out[~selected], expected[~selected]))
                    valid = (result == 0 and info.stage == 4 and code.value == wanted and agree and info.graph_nodes == int(captured) and
                             info.forbidden_nodes == 0 and info.staged == staged and info.allocated == staged+(36 if n == 0 else 0) and info.exported == 8*(3*na+18)+4)
                outputs[name] = out
                checks.append(dict(scenario=name, fixture=label, variant=variant, inputs_key=prefix, captured=captured, native_return=result,
                                   status=code.value, expected_status=wanted, initial_status=initial, max_absolute_error=error,
                                   passed=bool(valid), **{k:getattr(info,k) for k,_ in info._fields_}))
    output_arrays, input_arrays = output.with_suffix('.npz'), output.parent/'mutations.npz'
    np.savez_compressed(output_arrays, **outputs)
    np.savez_compressed(input_arrays, **mutations)
    report = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks, input_bindings=bindings,
                  output_file=output_arrays.name, output_sha256=sha(output_arrays), inputs_file=input_arrays.name, inputs_sha256=sha(input_arrays),
                  oracle_file=reference.name, oracle_sha256=sha(reference), oracle_arrays_sha256=sha(reference.with_suffix('.npz')),
                  checker_sha256=sha(Path(__file__)), native_library_sha256=sha(library))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Nonlocal force CUDA:', len(checks), 'passed=', report['passed'])
    if not report['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'library', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.library, args.output)
