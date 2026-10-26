"""Focused energy-only sweep checks; reuse ordinary science/export comparators."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from siesta_remag.corpus import sha256
from siesta_remag.compare import compare, difference
from siesta_remag.density_history import policy, record_errors as history_errors
from siesta_remag.energy_only import publication_records, sweep_record_errors
from siesta_remag.orbital_basis import record_errors as orbital_errors
from siesta_remag.orientation import parse_sweep, write_direction_views
from siesta_remag.output import read_eigenvalues, read_geometry
from siesta_remag.frozen_reference import direction_checks
from comparison_inputs import verify_convergence
from verify_exports import compare_exports

def read(path):
    return json.loads(path.read_text())

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def axes(spec, plan):
    mat = plan['materials'][spec['material']]
    return [mat['axes'][i] for i in spec['axis_indices']], [mat['directions'][i] for i in spec['axis_indices']]

def rotations(directions):
    result = []
    for x, y, z in directions:
        q = [1+z, -y, x, 0]
        norm = math.sqrt(sum(v*v for v in q))
        if norm == 0:
            raise ValueError('This diagnostic uses the positive z reference')
        result.append([v/norm for v in q])
    return result

def verify_inputs(inputs, plan):
    """Prove effective single-axis input differs only by spin orientation/name."""
    receipt = {}
    for material, info in plan['materials'].items():
        normalized = []
        for name, direction in zip(info['axes'], info['directions']):
            folder = inputs/name
            text = (folder/'input.fdf').read_text()
            manifest = read(folder/'manifest.json')
            assert sha256(folder/'input.fdf') == manifest['input_sha256']
            assert all(sha256(folder/p) == digest for p, digest in manifest['pseudopotentials'].items())
            assert len(re.findall(r'^%block DM.InitSpin\s*$', text, re.M)) == 1
            block = re.search(r'%block DM.InitSpin\n(.*?)%endblock DM.InitSpin', text, re.S)
            magnitudes = []
            for row in block.group(1).strip().splitlines():
                fields = row.split(); assert len(fields) == 4
                theta, phi = map(math.radians, map(float, fields[2:]))
                actual = [math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)]
                assert max(abs(a-b) for a, b in zip(actual, direction)) < 1e-8
                magnitudes.append(' '.join(fields[:2]))
            assert not re.search(r'(?im)^\s*(DM.AllowReuse|REMAG.ReuseOrientationGeometry|%block REMAG.SpinRotations)\b', text)
            family = re.sub(r'^SystemName.*\n', '', text, flags=re.M)
            family = re.sub(r'%block DM.InitSpin.*?%endblock DM.InitSpin', '\n'.join(magnitudes), family, flags=re.S)
            normalized.append((family, manifest['pseudopotentials']))
        assert all(value == normalized[0] for value in normalized)
        receipt[material] = dict(inputs=[read(inputs/name/'manifest.json')['input_sha256'] for name in info['axes']],
            family_sha256=hashlib.sha256(normalized[0][0].encode()).hexdigest(),
            method='Only SystemName and DM.InitSpin angles differ; magnitudes and all other input/pseudopotentials match')
    return receipt

def regression(reference, candidate, refdir, dest):
    # These are scientific controls. Whole-process paired timing is evaluated
    # separately; per-direction times within a sweep remain unknown.
    checked = copy.deepcopy(candidate)
    checked['hardware'] = dict(checked['hardware'], timing_scope='scientific control only')
    science = compare(reference, checked, reference_dir=refdir, candidate_dir=dest, electronic_only=True)
    exports = compare_exports(refdir, dest)
    return dict(passed=science['passed'] and exports['passed'], science=science, exports=exports)

def directional(reference, candidate):
    rows = []
    for index in range(1, len(reference)):
        for quantity in ('total_energy_ev', 'free_energy_ev'):
            a = reference[index]['observables'].get(quantity); a0 = reference[0]['observables'].get(quantity)
            b = candidate[index]['observables'].get(quantity); b0 = candidate[0]['observables'].get(quantity)
            ref = a-a0 if a is not None and a0 is not None else None
            value = b-b0 if b is not None and b0 is not None else None
            rows.append(dict(index=index+1, quantity=quantity, reference_ev=ref, candidate_ev=value,
                             **difference(ref, value, 5e-6)))
    return rows

def cold_checks(folder, result, direction, nk):
    check = direction_checks(dict(spin_expectations=True, points=[dict(index=0, quaternion=[1,0,0,0],
        spin_muB=result['observables'].get('total_moment_muB'))]), [direction])
    convergence = verify_convergence(folder, expected={'DM.Tolerance':1e-9, 'SCF.DM.Tolerance':1e-9, 'SCF.H.Tolerance':1e-8})
    actual = re.findall(r'k-grid: Number of k-points\s*=\s*(\d+)', (folder/'stdout.log').read_text())
    cycles = result['telemetry'].get('gpu_device_cycle', {}).get('cycles', [])
    mesh = bool(actual and all(int(v) == nk for v in actual) and len(cycles) == 1 and
                cycles[0].get('eigensystems', {}).get('jobs') == nk and result['observables'].get('eigenvalues', {}).get('nk') == nk)
    return dict(passed=result['valid'] and check['passed'] and convergence['passed'] and mesh,
                direction=check, convergence=convergence, mesh_passed=mesh)

def sweep(folder, spec, plan, inputs, *, create_views=False):
    execution = read(folder/'execution.json')
    names, directions = axes(spec, plan)
    report = parse_sweep((folder/'stdout.log').read_text(errors='replace'), directions, rotations(directions),
                         returncode=execution['returncode'], timed_out=execution['timed_out'])
    if create_views:
        views = write_direction_views(folder, report)
    else:
        views = [folder/'direction-outputs'/f'{i:03d}' for i in range(1, len(names)+1)]
        for row, view in zip(report['directions'], views):
            try:
                row['observables']['eigenvalues'] = read_eigenvalues(view/'remag.EIG')
                row['observables']['geometry'] = read_geometry(view/'remag.XV')
            except (OSError, ValueError) as exc:
                row['valid'] = False; row['errors'].append(str(exc))
    def records(name):
        path = folder/name
        return read(path).get('cycles', []) if path.is_file() else []
    cycles = records('remag-device-scf.json')
    publication = read(folder/'remag-energy-only.json') if (folder/'remag-energy-only.json').exists() else {}
    published = publication_records(publication)
    rows = report['directions']
    errors = list(report['errors'])
    errors += sweep_record_errors(publication, cycles, [r['observables'] for r in rows],
                                  [r['stdout'] for r in rows], expected_directions=len(names))
    convergence = verify_convergence(folder, expected={'DM.Tolerance':1e-9, 'SCF.DM.Tolerance':1e-9, 'SCF.H.Tolerance':1e-8})
    errors += convergence['errors']
    initial = records('remag-initial-state.json')
    errors += history_errors(records('remag-density-history.json'), cycles, initial, published,
                             policy((folder/'input.fdf').read_text()), energy_only=True)
    nk = plan['materials'][spec['material']]['kpoints']
    actual = re.findall(r'k-grid: Number of k-points\s*=\s*(\d+)', (folder/'stdout.log').read_text())
    if not actual or any(int(v) != nk for v in actual):
        errors.append('Actual stdout k count differs from requested mesh')
    for i, (cycle, row) in enumerate(zip(cycles, rows)):
        if (cycle.get('producer_status') != 0 or cycle.get('forbidden_nodes') != 0 or
            cycle.get('eigensystems', {}).get('jobs') != nk or
            cycle.get('eigensystems', {}).get('solver_lanes') != 8 or
            cycle.get('iterations') != row['observables'].get('scf_iterations') or
            row['observables'].get('eigenvalues', {}).get('nk') != nk):
            errors.append(f'Direction {i+1}: cycle, iteration or eigensystem evidence differs')
    owner_evidence = {}
    for name in ('remag-pair-geometry.json', 'remag-nonlocal-geometry.json', 'remag-nonlocal-h0.json',
                 'remag-initial-mesh-production.json', 'remag-orbital-basis.json'):
        owners = records(name); owner_evidence[name] = owners
        if len(owners) != len(names):
            errors.append('Incomplete retained owner: '+name)
        for i, (owner, cycle) in enumerate(zip(owners, cycles)):
            if owner.get('epoch') != cycle.get('epoch') or owner.get('energy_only_complete') is not True:
                errors.append('Unaccepted electronic owner: '+name)
            if owner.get('complete', False) or owner.get('forces_complete', False) or owner.get('force_borrowed', False):
                errors.append('Energy-only owner claimed force consumption: '+name)
            if i and (owner.get('setup_bytes', owner.get('compact_setup_bytes')) != 0 or owner.get('device_allocation_bytes') != 0):
                errors.append('Immutable owner restaged or reallocated: '+name)
            if name == 'remag-orbital-basis.json':
                errors += orbital_errors(owner, cycle, shadow=False, energy_only=True, reused_geometry=i>0)
    densities = records('remag-final-density.json')
    expected = [(c['epoch'], kind) for c in cycles for kind in (1, 2)]
    if ([(r.get('epoch'), r.get('kind')) for r in densities] != expected or
        not densities or any(not all(r.get(k) is True for k in ('requested','exported','passed')) for r in densities)):
        errors.append('Requested densities were not exported for every direction')
    if any(folder.glob('remag-integral-force-*.json')) or any(folder.glob('remag-integral-grid-*.json')):
        errors.append('Energy-only sweep executed force consumers')
    results = []
    for name, row, view, cycle in zip(names, rows, views, cycles):
        result = dict(schema_version=1, valid=row['valid'] and not errors, benchmark=read(inputs/name/'manifest.json'),
            calculation_scope='energy_only', observables=row['observables'], hardware=execution['hardware'],
            runtime_settings=plan['settings'], wall_seconds=None, returncode=execution['returncode'],
            timed_out=execution['timed_out'], parse_errors=errors+row['errors'], executable=execution['executable'],
            input_derivation=dict(executed_input_sha256=execution['executed_input_sha256'],
                effective_input_sha256=sha256(inputs/name/'input.fdf'), initial_guess='independent rotated atomic',
                method='Verified fixed geometry/input family; absolute quaternion; DM.AllowReuse=false'),
            telemetry=dict(gpu_device_cycle=dict(cycles=[cycle]), final_density=[r for r in densities if r['epoch']==cycle['epoch']]))
        if create_views:
            save(view/'result.json', result)
        results.append(result)
    passed = not errors and len(results) == len(names) and all(r['valid'] for r in results)
    receipt = dict(passed=passed, errors=errors, convergence=convergence, owners=owner_evidence,
                   directions=[{k:v for k,v in row.items() if k not in ('stdout','observables')} for row in rows])
    return receipt, results, views
