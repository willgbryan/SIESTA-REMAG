#!/usr/bin/env python3
"""Isolated device orbital values/derivatives against pristine SIESTA all_phi.

Includes compact dfscf geometry reconstruction. This does not qualify actual
material basis packing, production ownership or removal of derivative uploads.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from check_orbital_basis import run as build_oracle, build_reference, fixtures, flatten, pack

ATOL = 1e-10
RTOL = 1e-10
QUERY_FIELDS = ('species', 'orbitals', 'radial_parameters', 'values', 'second', 'query_species', 'query_orbital', 'xyz')
GEOMETRY_FIELDS = ('entry_orbital', 'entry_offset', 'orbital_atom', 'orbital_local', 'atom_species', 'offset', 'subpoint', 'atom_offset')
INDEX_FAULTS = ('species_index', 'orbital_index', 'species_range', 'sample_count', 'sample_offset', 'angular_l', 'angular_m', 'cutoff', 'delta')
VALUE_FAULTS = ('xyz', 'values', 'second', 'radial_range')
GEOMETRY_FAULTS = ('entry_orbital', 'entry_offset', 'orbital_atom', 'orbital_local', 'atom_species', 'offset')


class Info(ct.Structure):
    _fields_ = [(n, ct.c_size_t) for n in ('allocated', 'staged', 'exported')] + [(n, ct.c_int) for n in ('graph_nodes', 'forbidden_nodes', 'stage')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def geometry_fixtures():
    _, species, _, _, _ = list(fixtures())[-1]
    basis = flatten(species)
    atom_species = np.array([1, 2, 1, 2], np.int32)
    orbital_atom, orbital_local = [], []
    for ia, isp in enumerate(atom_species):
        for io in range(len(species[isp-1][1])):
            orbital_atom.append(ia+1);orbital_local.append(io+1)
    no = len(orbital_atom)
    rng = np.random.default_rng(341275)
    offsets = rng.normal(size=(23, 3)) * 1.4
    offsets[0] = 0.;offsets[1] = [0., 0., 1.]
    atom_offset = np.array([[0., 0., 0.], [.1, -.2, .3], [2., -1., 3.], [-2., 1., -3.]])
    entries = 2*no+1
    for nsp in (1, 8):
        subpoint = rng.normal(size=(nsp, 3))*.08;subpoint[0] = 0.
        data = dict(basis, entry_orbital=(np.arange(entries, dtype=np.int32)*17)%no+1,
            entry_offset=(np.arange(entries, dtype=np.int32)*7)%len(offsets)+1,
            orbital_atom=np.array(orbital_atom, np.int32), orbital_local=np.array(orbital_local, np.int32),
            atom_species=atom_species, offset=offsets, subpoint=subpoint, atom_offset=atom_offset)
        # Preserve the original parenthesization: (xdop+xdsp)-dxa.
        io = data['entry_orbital']-1;ia = data['orbital_atom'][io]-1
        xyz = (offsets[data['entry_offset']-1, None, :]+subpoint[None, :, :]-atom_offset[ia, None, :]).reshape(-1, 3)
        qs = np.repeat(atom_species[ia]-1, nsp);qo = np.repeat(data['orbital_local'][io]-1, nsp)
        yield f'geometry_nsp{nsp}', species, data, qs, qo, xyz


def run(upstream, build, library, output):
    output = output.resolve();output.parent.mkdir(parents=True, exist_ok=True)
    oracle = output.with_name('reference.json');build_oracle(upstream, build, oracle)
    reference = json.loads(oracle.read_text());data = np.load(oracle.with_suffix('.npz'))
    lib = ct.CDLL(str(library.resolve()))
    query = lib.remag_check_orbital_basis
    query.argtypes = [ct.c_int]*7+[ct.c_double]+[ct.c_void_p]*9+[ct.POINTER(ct.c_int), ct.POINTER(Info)]
    query.restype = ct.c_int
    geometry = lib.remag_check_orbital_geometry
    geometry.argtypes = [ct.c_int]*9+[ct.c_double]+[ct.c_void_p]*14+[ct.POINTER(ct.c_int), ct.POINTER(Info)]
    geometry.restype = ct.c_int
    checks, retained, geometry_arrays, geometry_checks = [], {}, {}, []

    def save_result(label, kind, captured, cycles, fault, code, status, info, wanted_status, arrays, result, expected, pi=np.pi):
        name = f'{label}_captured{int(captured)}_cycles{cycles}'+('_'+fault if fault else '')
        staged = sum(a.nbytes for a in arrays)+4
        allocated = staged+result.nbytes+sum(a.dtype.itemsize for a in arrays if not a.size)
        valid = (code == 0 and status.value == wanted_status and info.stage == 4 and info.graph_nodes == int(captured)
                 and info.forbidden_nodes == 0 and info.staged == staged and info.allocated == allocated
                 and info.exported == result.nbytes+4)
        # Ordinary execution enters stage 3 before the API validates pi.
        if fault == 'pi':valid = code == -1 and info.stage == 3 and info.exported == 0
        record = dict(scenario=name, fixture=label, kind=kind, captured=captured, cycles=cycles, fault=fault,
            native_return=code, status=status.value, expected_status=wanted_status, pi=float(pi),
            expected_staged=staged, expected_allocated=allocated, expected_exported=result.nbytes+4,
            **{key:getattr(info, key) for key, _ in info._fields_})
        if fault is None or fault == 'sticky':
            error = np.abs(result-expected)
            record['max_absolute_error'] = float(error.max())
            record['max_scaled_error'] = float((error/(ATOL+RTOL*np.abs(expected))).max())
            valid = valid and np.isfinite(result).all() and bool(np.all(error <= ATOL+RTOL*np.abs(expected)))
        record['passed'] = bool(valid);checks.append(record);retained[name] = result
        if not valid:print('FAILED', record, flush=True)

    def evaluate(row, captured, cycles, fault=None):
        label = row['fixture'];arrays = [data[label+'_'+k].copy(order='C') for k in QUERY_FIELDS]
        sp, orb, radial, values, second, qs, qo, xyz = arrays
        initial = 2 if fault == 'sticky' else 0;pi = np.pi;wanted = initial
        if fault in INDEX_FAULTS:
            wanted = 1
            if fault == 'species_index':qs[:] = len(sp)
            elif fault == 'orbital_index':qo[:] = len(orb)
            elif fault == 'species_range':sp[:, 0] = len(orb)
            elif fault == 'sample_count':orb[:, 0] = 1
            elif fault == 'sample_offset':orb[:, 1] = len(values)
            elif fault == 'angular_l':orb[:, 2] = 6
            elif fault == 'angular_m':orb[:, 3] = 6
            elif fault == 'cutoff':radial[:, 0] = np.nan
            elif fault == 'delta':radial[:, 1] = 0.
        elif fault in VALUE_FAULTS:
            wanted = 2
            if fault == 'xyz':xyz[:] = np.nan
            elif fault == 'values':values[:] = np.nan
            elif fault == 'second':second[:] = np.nan
            elif fault == 'radial_range':radial[:, 1] = 1e-6
        elif fault == 'pi':pi = 0.;wanted = -1
        result = np.full((len(qs), 4), np.nan);status = ct.c_int(-1);info = Info()
        code = query(len(sp), len(orb), len(values), len(qs), int(captured), cycles, initial, pi,
            *[a.ctypes.data for a in arrays], result.ctypes.data, ct.byref(status), ct.byref(info))
        save_result(label, 'query', captured, cycles, fault, code, status, info, wanted, arrays, result, data[label+'_expected'], pi)

    for row in reference['checks']:
        for captured in (False, True):
            for cycles in (1, 3):evaluate(row, captured, cycles)
    for fault in (*INDEX_FAULTS, *VALUE_FAULTS, 'sticky'):
        for captured in (False, True):evaluate(reference['checks'][-1], captured, 3, fault)
    evaluate(reference['checks'][-1], False, 1, 'pi')

    with tempfile.TemporaryDirectory(prefix='remag-orbital-geometry-') as temporary:
        tmp = Path(temporary);exe = build_reference(build, tmp)
        for label, species, fields, qs, qo, xyz in geometry_fixtures():
            raw = pack(species, qs, qo, xyz);inp = tmp/'input.bin';out = tmp/'output.bin';inp.write_bytes(raw)
            done = subprocess.run([str(exe), str(inp), str(out)], cwd=tmp, capture_output=True, text=True)
            if done.returncode:raise RuntimeError(done.stdout+done.stderr)
            ref = np.fromfile(out, np.float64);assert ref[0] == np.pi and ref.size == 1+4*len(qs)
            values = ref[1:].reshape(-1, 4)
            expected = np.r_[values[:, 0], values[:, 1:].ravel(), xyz.ravel()]
            assert np.isfinite(expected).all()
            for key, array in dict(fields, query_species=qs, query_orbital=qo, xyz=xyz, expected=expected).items():geometry_arrays[label+'_'+key] = array
            geometry_checks.append(dict(fixture=label, queries=len(qs), input_sha256=hashlib.sha256(raw).hexdigest(),
                entries=len(fields['entry_orbital']), nsp=len(fields['subpoint']), orbitals=len(fields['orbital_atom']),
                atoms=len(fields['atom_species']), offsets=len(fields['offset'])))
            for fault in (None, *GEOMETRY_FAULTS):
                for captured in (False, True):
                    arrays = [fields[k].copy(order='C') for k in (*QUERY_FIELDS[:5], *GEOMETRY_FIELDS)]
                    if fault:
                        arr = arrays[5+GEOMETRY_FIELDS.index(fault)];arr[...] = np.nan if fault == 'offset' else 0
                    nsp = len(fields['subpoint']);n = len(qs)
                    result = np.full(7*n, np.nan);status = ct.c_int(-1);info = Info()
                    code = geometry(len(fields['species']), len(fields['orbitals']), len(fields['values']), len(fields['entry_orbital']), nsp,
                        len(fields['orbital_atom']), len(fields['atom_species']), len(fields['offset']), int(captured), np.pi,
                        *[a.ctypes.data for a in arrays], result.ctypes.data, ct.byref(status), ct.byref(info))
                    wanted = 0 if fault is None else 2 if fault == 'offset' else 1
                    save_result(label, 'geometry', captured, 1, fault, code, status, info, wanted, arrays, result, expected)
    geometry_file = output.with_name('geometry-reference.npz');np.savez_compressed(geometry_file, **geometry_arrays)
    artifact = output.with_suffix('.npz');np.savez_compressed(artifact, **retained)
    report = dict(scope=__doc__, checks=checks, passed=all(c['passed'] for c in checks), atol=ATOL, rtol=RTOL,
        oracle_file=oracle.name, oracle_sha256=sha(oracle), oracle_arrays_sha256=sha(oracle.with_suffix('.npz')),
        geometry_file=geometry_file.name, geometry_sha256=sha(geometry_file), geometry_checks=geometry_checks,
        native_library_sha256=sha(library), checker_sha256=sha(Path(__file__)), output_file=artifact.name, output_sha256=sha(artifact))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Orbital basis CUDA scenarios:', len(checks), 'passed=', report['passed'])
    if not report['passed']:raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'build', 'library', 'output'):parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args();run(args.upstream.resolve(), args.build.resolve(), args.library, args.output)
