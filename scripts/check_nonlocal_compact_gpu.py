#!/usr/bin/env python3
"""Qualify the two-kernel compact nonlocal path and retain every executed input."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from check_nonlocal_compact import run as oracle, sha, FIELDS

TOPOLOGY = ('row', 'column', 'atom', 'center', 'center_base', 'center_count', 'block_base', 'block_count',
            'projectors', 'query', 'center_atom', 'block_offset', 'angular_low', 'angular_high',
            'csr_offset', 'csr_count', 'csr_column', 'lookup_missing', 'lookup_first', 'lookup_other_row', 'lookup_padding', 'lookup_high')
NONFINITE = ('gradient', 'value', 'xyz', 'density', 'epsilon')
UNUSED = ('right_gradient', 'projector_padding', 'xyz_padding', 'density_padding', 'density_components', 'epsilon_second')
HOST = ('volume', 'volume_nan', 'spinor', 'soc', 'split', 'capacity', 'components', 'negative_count', 'soc_components')
OVERFLOW = ('arithmetic_overflow', 'force_overflow', 'energy_overflow', 'stress_overflow')


class Info(ct.Structure):
    _fields_ = [(k, ct.c_size_t) for k in ('allocated', 'staged', 'exported', 'poisoned')]+[(k, ct.c_int) for k in ('graph_nodes', 'forbidden_nodes', 'stage')]


def variants(row):
    result = ['normal']
    if row['components'] == 8 and row['neighbors'] == 19 and row['cycles'] == 1 and row['l'] in (0, 1, 5):
        result += list(TOPOLOGY)+list(NONFINITE)+list(UNUSED)+list(HOST)+['overwritten_density', 'zero_blocks', 'sticky']
        if row['l']:
            result.remove('density_components')
            result.remove('epsilon_second')
            result.append('epsilon_second_nan')
        if not row['soc']:
            result.remove('soc_components')
            result += list(OVERFLOW)
    if row['neighbors'] == 131073:
        result += ['tail_row', 'tail_gradient', 'tail_excluded']
    return result


def inputs(source, label, variant):
    config = source[label+'_config'].copy()
    volume = float(source[label+'_volume'][0])
    data = {k: source[label+'_'+k].copy() for k in FIELDS}
    expected = source[label+'_expected'].copy()
    nn, nc, nb, nq, na, nr, no, cap, components, spinor, soc, split, cycles = map(int, config)
    wanted = initial = 0
    compare = True
    if variant in TOPOLOGY:
        wanted, compare = 1, False
        field, index, value = {
            'row': ('neighbors', (slice(None), 0), -2147483648),
            'column': ('neighbors', (slice(None), 1), no+1),
            'atom': ('neighbors', (slice(None), 2), 0),
            'center': ('neighbors', (slice(None), 3), nc),
            'center_base': ('centers', (slice(None), 0), -1),
            'center_count': ('centers', (slice(None), 1), 2147483647),
            'block_base': ('centers', (slice(None), 2), -1),
            'block_count': ('centers', (slice(None), 3), 2147483647),
            'projectors': ('centers', (slice(None), 4), 2147483647),
            'query': ('centers', (slice(None), 5), 2147483647),
            'center_atom': ('centers', (slice(None), 6), 0),
            'block_offset': ('block_indices', (slice(None), 0), 2147483647),
            'angular_low': ('block_indices', (slice(None), 1), -2147483648),
            'angular_high': ('block_indices', (slice(None), 1), 2147483647),
            'csr_offset': ('ptr', slice(None), -1), 'csr_count': ('num', slice(None), cap+1),
            'csr_column': ('list', slice(None), no+1), 'lookup_missing': ('lookup', slice(None), -1),
            'lookup_first': ('lookup', (0, 0), 1), 'lookup_other_row': ('lookup', (0, 0), 26),
            'lookup_padding': ('lookup', (0, 0), 0), 'lookup_high': ('lookup', (0, 0), 2147483647),
        }[variant]
        data[field][index] = value
    elif variant in NONFINITE or variant == 'epsilon_second_nan':
        wanted, compare = 2, False
        if variant == 'gradient': data['integrals'][:, 1:] = np.nan
        elif variant == 'value': data['integrals'][:, 0] = np.nan
        elif variant == 'epsilon': data['epsilon'][:, 0] = np.nan
        elif variant == 'epsilon_second_nan': data['epsilon'][:, 1] = np.nan
        else: data[variant][:] = np.nan
    elif variant in ('right_gradient', 'projector_padding', 'xyz_padding'):
        used = np.zeros(nq, bool)
        for base, count, block_base, block_count, width, query, _ in data['centers']:
            for i in range(count):
                first = query+i*width
                if variant == 'right_gradient' and data['neighbors'][base+i, 0] == 0:
                    data['integrals'][first:first+width, 1:] = np.nan
                elif variant == 'xyz_padding': used[first] = True
                elif variant == 'projector_padding':
                    for b in range(block_base, block_base+block_count):
                        offset, l = data['block_indices'][b]
                        used[first+offset:first+offset+(2*(2*l+1) if l else 1)] = True
        if variant == 'projector_padding': data['integrals'][~used] = np.nan
        elif variant == 'xyz_padding': data['xyz'][~used] = np.nan
    elif variant == 'density_padding':
        used = np.zeros(cap, bool)
        for start, count in zip(data['ptr'], data['num']): used[start:start+count] = True
        data['density'][:, ~used] = np.nan
    elif variant == 'density_components':
        unused = list(range(spinor, components)) if not soc else list(range(2, 8)) if split else [2, 3, 6, 7]
        data['density'][unused] = np.nan
    elif variant == 'epsilon_second': data['epsilon'][:, 1] = np.nan
    elif variant == 'overwritten_density':
        data['density'][:, 1] = np.nan
        if not soc: wanted, compare = 2, False
    elif variant == 'zero_blocks':
        data['centers'][:, 3] = 0
        expected = data['initial'].copy()
    elif variant == 'sticky': wanted = initial = 7
    elif variant in ('tail_row', 'tail_gradient', 'tail_excluded'):
        expected = source[label+'_without_last'].copy()
        if variant == 'tail_row': data['neighbors'][-1, 0], wanted = -1, 1
        elif variant == 'tail_excluded': data['neighbors'][-1, 0] = 0
        else:
            center = data['centers'][data['neighbors'][-1, 3]]
            first = center[5]+(nn-1-center[0])*center[4]+data['block_indices'][center[2], 0]
            data['integrals'][first, 1], wanted = np.nan, 2
    elif variant in OVERFLOW:
        wanted, compare = 2, False
        data['density'][:] = data['epsilon'][:] = data['integrals'][:] = data['xyz'][:] = 1.
        data['neighbors'][:, 2] = 1
        data['centers'][:, 6] = 2
        maximum = np.finfo('d').max
        if variant == 'arithmetic_overflow': data['integrals'][:, 1:] = maximum/2
        elif variant == 'force_overflow':
            data['integrals'][:, 1:] = maximum/10000
            data['initial'][12:15] = maximum
        elif variant == 'energy_overflow':
            data['integrals'][:, 0] = 1e150
            data['integrals'][:, 1:] = 0.
            data['initial'][0] = maximum
        else:
            data['integrals'][:, 1:] = maximum/10000
            data['initial'][9+3*na:] = maximum
    elif variant in HOST:
        compare = False
        if variant == 'volume': volume = 0.
        elif variant == 'volume_nan': volume = np.nan
        else:
            index, value = {'spinor': (9, 0), 'soc': (10, 2), 'split': (11, 2), 'capacity': (7, 0),
                            'components': (8, 3), 'negative_count': (0, -1), 'soc_components': (8, 2)}[variant]
            config[index] = value
    return config, volume, data, expected, wanted, initial, compare


def run(upstream, build, library, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = output.parent/'reference.json'
    oracle(upstream, build, reference)
    manifest = json.loads(reference.read_text())
    source = np.load(reference.with_suffix('.npz'))
    fn = ct.CDLL(str(library.resolve())).remag_check_nonlocal_compact
    fn.argtypes = [ct.c_void_p, ct.c_double, ct.c_int, ct.c_int]+[ct.c_void_p]*13+[ct.POINTER(ct.c_int), ct.POINTER(Info)]
    fn.restype = ct.c_int
    checks, outputs, mutations, bindings = [], {}, {}, {}
    for row in manifest['checks']:
        label = row['fixture']
        for variant in variants(row):
            config, volume, data, expected, wanted, initial, compare = inputs(source, label, variant)
            prefix = label+'_'+variant
            bindings[prefix] = {}
            for field, array in {**data, 'config': config, 'volume': np.array([volume])}.items():
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
                nn, nc, nb, nq, na, nr, no, cap, components, spinor, soc, split, cycles = map(int, config)
                staged = 28*nc+16*nn+24*nb+8*nr+4*cap+4*nr*no+56*nq+8*cap*components+8*(3*na+18)+4
                allocated = staged+8*cap+(28 if nc == 0 else 0)+(16 if nn == 0 else 0)+(24 if nb == 0 else 0)+(16 if nq == 0 else 0)
                error = None
                if variant in HOST:
                    stage = 0 if variant in ('capacity', 'components', 'negative_count') else 3
                    valid = result == -1 and info.stage == stage and info.exported == 0 and code.value == -9 and np.isnan(out).all()
                    if stage == 0: valid = valid and info.staged == info.allocated == info.poisoned == 0
                    else: valid = valid and info.staged == staged and info.allocated == allocated and info.poisoned == 8*cap
                else:
                    agree = True
                    if compare:
                        delta = np.abs(out-expected)
                        error = float(delta.max())
                        agree = np.isfinite(out).all() and np.all(delta <= 1e-10+1e-10*np.abs(expected))
                    elif variant in TOPOLOGY:
                        agree = np.isfinite(out).all()
                    valid = (result == 0 and info.stage == 4 and code.value == wanted and agree and info.graph_nodes == 2*int(captured) and
                             info.forbidden_nodes == 0 and info.staged == staged and info.allocated == allocated and
                             info.exported == 8*(3*na+18)+4 and info.poisoned == 8*cap*cycles)
                outputs[name] = out
                checks.append(dict(scenario=name, fixture=label, variant=variant, inputs_key=prefix, captured=captured, native_return=result,
                    status=code.value, expected_status=wanted, initial_status=initial, numerical_comparison=compare, max_absolute_error=error,
                    passed=bool(valid), **{k: getattr(info, k) for k, _ in info._fields_}))
    output_arrays, input_arrays = output.with_suffix('.npz'), output.parent/'mutations.npz'
    np.savez_compressed(output_arrays, **outputs)
    np.savez_compressed(input_arrays, **mutations)
    report = dict(scope=__doc__, passed=all(c['passed'] for c in checks), checks=checks, input_bindings=bindings,
        output_file=output_arrays.name, output_sha256=sha(output_arrays), inputs_file=input_arrays.name, inputs_sha256=sha(input_arrays),
        oracle_file=reference.name, oracle_sha256=sha(reference), oracle_arrays_sha256=sha(reference.with_suffix('.npz')),
        checker_sha256=sha(Path(__file__)), native_library_sha256=sha(library))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Compact nonlocal CUDA:', len(checks), 'passed=', report['passed'], flush=True)
    if not report['passed']:
        print([c for c in checks if not c['passed']])
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'library', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.library, args.output)
