#!/usr/bin/env python3
"""Independently audit retained orbital-basis arrays, statuses and coverage.

Reconstructs compact geometry and shape-derived traffic. It verifies stored
pristine-oracle comparisons, not the production force path or CUDA tracing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    folder = root/'orbital-basis-check';path = folder/'check.json';report = json.loads(path.read_text())
    oracle = folder/report['oracle_file'];reference = json.loads(oracle.read_text())
    data = np.load(folder/reference['arrays_file']);geometry = np.load(folder/report['geometry_file']);outputs = np.load(folder/report['output_file'])
    errors, checks = [], []
    required = ['shells_l'+str(i) for i in range(6)]
    fixtures = {r['fixture']:r for r in reference['checks']}
    geo_fixtures = {r['fixture']:r for r in report['geometry_checks']}
    index_faults = ('species_index', 'orbital_index', 'species_range', 'sample_count', 'sample_offset', 'angular_l', 'angular_m', 'cutoff', 'delta')
    value_faults = ('xyz', 'values', 'second', 'radial_range')
    geometry_faults = ('entry_orbital', 'entry_offset', 'orbital_atom', 'orbital_local', 'atom_species', 'offset')
    basis_fields = ('species', 'orbitals', 'radial_parameters', 'values', 'second')
    geometry_fields = ('entry_orbital', 'entry_offset', 'orbital_atom', 'orbital_local', 'atom_species', 'offset', 'subpoint', 'atom_offset')
    query_fields = (*basis_fields, 'query_species', 'query_orbital', 'xyz')
    expected = set()
    for label in required:
        for captured in (False, True):
            for cycles in (1, 3):expected.add((label, captured, cycles, None))
    for fault in (*index_faults, *value_faults, 'sticky'):
        for captured in (False, True):expected.add(('shells_l5', captured, 3, fault))
    expected.add(('shells_l5', False, 1, 'pi'))
    for label in ('geometry_nsp1', 'geometry_nsp8'):
        for fault in (None, *geometry_faults):
            for captured in (False, True):expected.add((label, captured, 1, fault))
    if set(fixtures) != set(required) or set(geo_fixtures) != {'geometry_nsp1', 'geometry_nsp8'}:errors.append('Fixture set differs')
    if report['atol'] != 1e-10 or report['rtol'] != 1e-10:errors.append('Numerical tolerance differs')
    # Rebuild the original binary oracle input from retained tables. Do not
    # regenerate exp() on a different CPU/libm and assume bitwise agreement.
    with tarfile.open(root/'implementation.tar.gz') as archive:
        content = archive.extractfile('scripts/check_orbital_basis.py').read()
    if hashlib.sha256(content).hexdigest() != reference['script_sha256']:errors.append('Archived oracle source hash differs')

    def oracle_input(fields, qs, qo, xyz):
        chunks = [np.array([len(fields['species']), len(qs)], np.int32).tobytes()]
        for start, count in fields['species']:
            radials, radial_indices, magnetic = [], [], []
            for i in range(start, start+count):
                n, offset, l, m = fields['orbitals'][i];cutoff, delta = fields['radial_parameters'][i]
                descriptor = (int(n), int(offset), float(cutoff), float(delta))
                if descriptor not in radials:radials.append(descriptor)
                radial_indices.append(radials.index(descriptor)+1);magnetic.append((l,m))
            chunks.append(np.array([len(radials), count], np.int32).tobytes())
            for n, offset, cutoff, delta in radials:
                chunks.extend((np.array([n], np.int32).tobytes(), np.array([cutoff, delta], np.float64).tobytes(),
                    fields['values'][offset:offset+n].tobytes(), fields['second'][offset:offset+n].tobytes()))
            magnetic = np.array(magnetic, np.int32)
            chunks.extend((magnetic[:, 0].tobytes(), magnetic[:, 1].tobytes(), np.array(radial_indices, np.int32).tobytes()))
        chunks.extend(((qs+1).tobytes(), (qo+1).tobytes(), xyz.tobytes()))
        return b''.join(chunks)

    for label, row in fixtures.items():
        fields = {key:data[label+'_'+key] for key in basis_fields}
        qs, qo, xyz = (data[label+'_'+key] for key in ('query_species','query_orbital','xyz'))
        raw = oracle_input(fields, qs, qo, xyz)
        if hashlib.sha256(raw).hexdigest() != row['input_sha256']:errors.append('Query oracle input hash differs: '+label)
        lmax = int(label[-1]);count = (lmax+1)**2+1
        if fields['species'].tolist() != [[0,count],[count,count]]:errors.append('Species layout differs: '+label)
        expected_lm = [(l,m) for l in range(lmax+1) for m in range(-l,l+1)]+[(0,0)]
        if fields['orbitals'][:,2:].tolist() != [list(v) for v in expected_lm]*2:errors.append('Angular coverage differs: '+label)
        for key, value in dict(queries=len(qs), species=2, orbitals=2*count, samples=len(fields['values'])).items():
            if row[key] != value:errors.append('Oracle dimensions differ: '+label+' '+key)
    for label, row in geo_fixtures.items():
        g = {field:geometry[label+'_'+field] for field in (*basis_fields, *geometry_fields)}
        io = g['entry_orbital']-1;ia = g['orbital_atom'][io]-1;nsp = len(g['subpoint'])
        xyz = (g['offset'][g['entry_offset']-1, None, :]+g['subpoint'][None, :, :]-g['atom_offset'][ia, None, :]).reshape(-1, 3)
        qs = np.repeat(g['atom_species'][ia]-1, nsp);qo = np.repeat(g['orbital_local'][io]-1, nsp)
        raw = oracle_input(g, qs, qo, xyz)
        if hashlib.sha256(raw).hexdigest() != row['input_sha256']:errors.append('Geometry oracle input hash differs: '+label)
        for field in (*basis_fields, 'radius2'):
            if not np.array_equal(data['shells_l5_'+field], geometry[label+'_'+field]):errors.append('Geometry basis differs: '+label+'_'+field)
        for field, array in dict(query_species=qs, query_orbital=qo, xyz=xyz).items():
            if not np.array_equal(array, geometry[label+'_'+field]):errors.append('Geometry query differs: '+label+'_'+field)
        if not np.array_equal(geometry[label+'_expected'][4*len(qs):], xyz.ravel()):errors.append('Geometry oracle displacement differs: '+label)
        for key, value in dict(entries=len(io), nsp=nsp, orbitals=len(g['orbital_atom']), atoms=len(g['atom_species']), offsets=len(g['offset']), queries=len(qs)).items():
            if row[key] != value:errors.append('Geometry dimensions differ: '+label+' '+key)
    seen = set()
    for row in report['checks']:
        label, captured, cycles, fault = (row[k] for k in ('fixture', 'captured', 'cycles', 'fault'))
        key = (label, captured, cycles, fault);faults = []
        if key in seen:faults.append('Duplicate scenario')
        seen.add(key)
        kind = 'geometry' if label in geo_fixtures else 'query'
        if row['kind'] != kind:faults.append('Wrong scenario kind')
        arrays = geometry if kind == 'geometry' else data
        fields = (*basis_fields, *geometry_fields) if kind == 'geometry' else query_fields
        staged = sum(arrays[label+'_'+field].nbytes for field in fields)+4
        wanted = arrays[label+'_expected'];out = outputs[row['scenario']]
        allocated = staged+wanted.nbytes+sum(arrays[label+'_'+field].dtype.itemsize for field in fields if not arrays[label+'_'+field].size)
        exported = wanted.nbytes+4
        status = (2 if fault == 'offset' else 1) if kind == 'geometry' and fault else 0
        if kind == 'query':status = 1 if fault in index_faults else 2 if fault in (*value_faults, 'sticky') else -1 if fault == 'pi' else 0
        if fault == 'pi':
            if row['native_return'] != -1 or row['stage'] != 3 or row['exported'] != 0:faults.append('Invalid pi accepted')
        elif (row['native_return'] != 0 or row['status'] != status or row['stage'] != 4 or row['graph_nodes'] != int(captured) or
              row['forbidden_nodes'] != 0 or row['staged'] != staged or row['allocated'] != allocated or row['exported'] != exported):
            faults.append('Native status, graph or shape-derived traffic failed')
        if row['expected_status'] != status or row['expected_staged'] != staged or row['expected_allocated'] != allocated or row['expected_exported'] != exported:
            faults.append('Reported expectation differs')
        name = f'{label}_captured{int(captured)}_cycles{cycles}'+('_'+fault if fault else '')
        if row['scenario'] != name:faults.append('Scenario name differs')
        error = None;bit_differences = None;zero_sign_differences = None
        if fault is None or fault == 'sticky':
            if out.shape != wanted.shape or not np.isfinite(out).all():faults.append('Invalid output shape or values')
            else:
                delta = np.abs(out-wanted);error = float(delta.max());scaled = float((delta/(1e-10+1e-10*np.abs(wanted))).max())
                bits = out.view(np.uint64) != wanted.view(np.uint64)
                bit_differences = int(np.count_nonzero(bits))
                zero_sign_differences = int(np.count_nonzero(bits & (out == 0.) & (wanted == 0.)))
                if scaled > 1 or error != row['max_absolute_error'] or scaled != row['max_scaled_error']:faults.append('Numerical comparison does not reproduce')
                if kind == 'geometry' and not np.array_equal(out[4*(len(out)//7):], wanted[4*(len(out)//7):]):faults.append('Displacements do not agree exactly')
        if row['passed'] is not True:faults.append('Recorded scenario failed')
        checks.append(dict(scenario=name, passed=not faults, errors=faults, max_absolute_error=error,
            bit_pattern_differences=bit_differences, zero_sign_differences=zero_sign_differences))
    if seen != expected or len(seen) != 81 or len(outputs.files) != 81:errors.append('Required scenario coverage differs')
    artifacts = (path, oracle, folder/reference['arrays_file'], folder/reference['wrapper_file'], folder/report['geometry_file'], folder/report['output_file'])
    for artifact, digest in ((oracle, report['oracle_sha256']), (folder/reference['arrays_file'], report['oracle_arrays_sha256']),
            (folder/reference['arrays_file'], reference['arrays_sha256']), (folder/reference['wrapper_file'], reference['wrapper_sha256']),
            (folder/report['geometry_file'], report['geometry_sha256']), (folder/report['output_file'], report['output_sha256'])):
        if sha(artifact) != digest:errors.append('Artifact hash mismatch: '+artifact.name)
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors, checks=checks,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in artifacts}, verifier_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('run', type=Path);args = parser.parse_args()
    result = verify(args.run)
    (args.run/'orbital-basis-component-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'orbital-basis-component-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Orbital basis component audit:', len(result['checks']), 'scenarios, passed=', result['passed'])
    if not result['passed']:raise SystemExit(1)
