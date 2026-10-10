#!/usr/bin/env python3
"""Check isolated core/neutral GPU contractions against pinned arithmetic.

This does not qualify radial evaluation, material point mappings, production
force ownership, or complete final-observable residency.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
from check_atomic_grid_force_formula import run as build_oracle


class Problem(ct.Structure):
    _fields_ = [('dvol', ct.c_double), ('volume', ct.c_double)] + [
        (name, ct.c_int) for name in ('kind', 'diagonal_spins', 'forces', 'stress')]


class Info(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ('allocated', 'staged', 'exported')] + [
        (name, ct.c_int) for name in ('graph_nodes', 'forbidden_nodes', 'stage')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(upstream, build, library, output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    oracle = output.with_name('reference.json')
    build_oracle(upstream, build, oracle)
    source = json.loads(oracle.read_text())
    data = np.load(oracle.with_suffix('.npz'))
    function = ct.CDLL(str(library.resolve())).remag_check_atomic_grid_force
    function.argtypes = [ct.c_int]*6 + [ct.POINTER(Problem)] + [ct.c_void_p]*10 + [ct.POINTER(ct.c_int), ct.POINTER(Info)]
    function.restype = ct.c_int
    checks, retained = [], {}
    names = ('gradient', 'displacement', 'potential', 'auxiliary', 'deformation', 'initial_force', 'initial_stress')

    def evaluate(row, captured, cycles, fault=None):
        label = row['fixture']
        points, atoms = (data[label+'_'+name].copy() for name in ('points', 'atoms'))
        arrays = [np.array(data[label+'_'+name], order='F', copy=True) for name in names]
        p = Problem(row['dvol'], row['volume'], row['kind'], row['nsd'], row['forces'], row['stress'])
        expected_status = 0
        if fault in ('volume', 'dvol', 'kind', 'diagonal_spins', 'forces', 'stress', 'point', 'atom'):
            expected_status = 1
            if fault == 'point':
                points[0] = 0
            elif fault == 'atom':
                atoms[0] = row['atoms']+1
            elif fault in ('volume', 'dvol', 'kind', 'diagonal_spins'):
                setattr(p, fault, 0)
            else:
                setattr(p, fault, 2)
        elif fault:
            expected_status = 2
            index = names.index(fault)
            # Set every field value to NaN so the selected point is covered.
            arrays[index][...] = np.nan
        result = np.full(3*row['atoms']+9, np.nan)
        status, info = ct.c_int(-1), Info()
        code = function(row['points'], row['entries'], row['atoms'], row['nsd'], int(captured), cycles,
            ct.byref(p), points.ctypes.data, atoms.ctypes.data, *[a.ctypes.data for a in arrays],
            result.ctypes.data, ct.byref(status), ct.byref(info))
        name = f'{label}_captured{int(captured)}_cycles{cycles}' + ('_'+fault if fault else '')
        record = dict(fixture=label, captured=captured, cycles=cycles, fault=fault,
            native_return=code, status=status.value, expected_status=expected_status,
            **{key: getattr(info, key) for key, _ in info._fields_})
        valid = (code == 0 and status.value == expected_status and info.stage == 4 and
                 info.forbidden_nodes == 0 and (not captured or info.graph_nodes == 1))
        expected_upload = (8*row['entries'] + 48*row['entries'] + 32 +
                           8*row['points']*(row['nsd']+2) + 8*(3*row['atoms']+9) + 4)
        expected_allocation = expected_upload + (24 if row['entries'] == 0 else 0)
        if (info.staged != expected_upload or info.allocated != expected_allocation or
                info.exported != 8*(3*row['atoms']+9)+4):
            valid = False
        if expected_status == 0:
            initial = np.r_[arrays[-2].ravel(order='F'), arrays[-1].ravel(order='F')]
            expected = initial + cycles*(data[label+'_result']-initial)
            error = float(np.max(np.abs(expected-result)))
            record['max_absolute_error'] = error
            valid = valid and np.isfinite(result).all() and error < 1e-12
        record['passed'] = bool(valid)
        checks.append(record)
        retained[name] = result

    for row in source['checks']:
        for captured in (False, True):
            for cycles in (1, 3):
                evaluate(row, captured, cycles)
    for kind, nsd in ((1, 2), (2, 1)):
        row = next(r for r in source['checks'] if r['kind'] == kind and r['nsd'] == nsd and
                   r['entries'] == 257 and r['forces'] and r['stress'])
        fields = ('potential', 'auxiliary') if kind == 1 else ('deformation',)
        for fault in ('volume', 'dvol', 'kind', 'diagonal_spins', 'forces', 'stress', 'point', 'atom',
                      'gradient', 'displacement', *fields):
            for captured in (False, True):
                evaluate(row, captured, 3, fault)
    artifact = output.with_suffix('.npz')
    np.savez_compressed(artifact, **retained)
    report = dict(scope=__doc__, checks=checks, passed=all(c['passed'] for c in checks),
        oracle_file=oracle.name, oracle_sha256=sha(oracle), oracle_arrays_sha256=sha(oracle.with_suffix('.npz')),
        native_library_sha256=sha(library), checker_sha256=sha(Path(__file__)),
        output_file=artifact.name, output_sha256=sha(artifact))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Atomic grid-force CUDA scenarios:', len(checks), 'passed=', report['passed'])
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'library', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.build, args.library, args.output)
