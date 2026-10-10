#!/usr/bin/env python3
"""Independently audit retained-D nonlocal lifecycle probes and full outputs.

Reconstructs the advertised fixtures, sparse scalar/SOC selection, energy,
force/stress contractions and shape-derived owner copy boundaries. Probe D
uploads/readbacks are outside owner accounting: this is not a whole-process
CUDA residency audit. Angular SOC arithmetic is qualified separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENARIOS = {'normal', 'captured', 'recovery', 'empty', 'failed_reference',
             'failed_status', 'stale_setup', 'stale_staged'}
SPINS = {(1, 0), (2, 0), (4, 0), (8, 0), (8, 1)}
# The lifecycle contract, independent of the checker implementation.
RECOVERY = '''missing_end missing_bind bad_mode bad_volume
REMAG_NONLOCAL_FORCE_SHADOW REMAG_PAIR_FORCE_SHADOW REMAG_TWO_CENTER_SHADOW
bad_span overlapping_rows bad_sparse_column prepare_in_scf duplicate_prepare
early_bind early_center table registry_a registry_b null_xyz nonfinite_xyz
observe_in_scf initial_row initial_column initial_atom initial_offset
initial_angular_low initial_angular_high initial_epsilon initial_coverage
unfinished_center bind_without_epoch_advance duplicate_initial_end
center_after_staging duplicate_bind missing_retained missing_kinetic
changed_volume changed_consumer changed_split changed_shape changed_num
changed_ptr changed_list grid_fingerprint nonfinite_initial final_begin_in_scf
duplicate_final_begin early_final_end early_pending changed_registry changed_xyz
extra_query final_row final_column final_atom final_offset final_angular_low
final_angular_high final_epsilon final_coverage changed_local_row changed_center
extra_center nonfinite_reference stage_without_reference duplicate_reference
center_after_reference wrong_pending stage_pairs stage_xyz stage_count
stage_consumer apply_before_stage finish_before_stage duplicate_stage
finish_before_apply wrong_stream missing_status apply_in_scf duplicate_apply
duplicate_finish missing_final_begin'''.split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fixture(spins, geometry):
    return dict(
        num=[3, 2], ptr=[1, 5], list=[0, 1, 1, 2, 0, 1, 2, 0],
        pairs=[1, 1, 1, 2, 2, 1, 2, 2], epsilon=[.7, .7, -.4, -.4],
        xyz=(np.array([.4, -.3, .2, .4, -.3, .2, -.1, .6, -.4, -.1, .6, -.4])+.03*geometry).tolist(),
        integrals=(np.array([.3, .1, -.2, .05, .1, .05, .07, -.03,
                            .2, -.15, .1, .2, -.15, -.08, .12, -.04])+.02*geometry).tolist(),
        initial=[0.]*9+[.01*i for i in range(1, 16)],
        density=(.025*np.arange(1, spins+1)[:, None]+.01*np.arange(8)[None, :]+.125*geometry).ravel().tolist())


def contract(f, spins, split, empty):
    """Independent dense contraction from CSR; no owner/checker imports."""
    out = np.array(f['initial'])
    if empty:
        return out
    sparse = np.array(f['density']).reshape(spins, 8)
    dense = np.zeros((2, 2, spins))
    for row in range(2):
        entries = np.arange(f['ptr'][row], f['ptr'][row]+f['num'][row])
        for column in range(2):
            selected = entries[np.array(f['list'])[entries] == column+1]
            if len(selected):
                dense[row, column] = sparse[:, selected[-1]] if spins == 8 else sparse[:, selected].sum(axis=1)
    diagonal = dense[:, :, :min(2, spins)].sum(axis=2)
    integral = np.array(f['integrals']).reshape(2, 2, 4)
    value, gradient = integral[:, :, 0], integral[:, :, 1:]
    epsilon = np.array(f['epsilon'])[::2]
    operator = (value*epsilon) @ value.T
    if spins == 8 and not split:
        complex_density = dense[:, :, :2]+1j*dense[:, :, 4:6]
        energy = np.einsum('ij,ijs->s', operator, complex_density)
        out[1:3] += energy.real
        out[5:7] += energy.imag
    else:
        out[0] += np.sum(diagonal*operator)
    force = 2*np.einsum('ij,b,jb,iba->ia', diagonal, epsilon, value, gradient)
    out[9:15] -= force.ravel()
    out[12:15] += force.sum(axis=0)
    displacement = np.array(f['xyz']).reshape(2, 2, 3)[:, 0, :]
    out[15:] += (force.T @ displacement/2.3).ravel()
    return out


def compare(actual, expected, issues, label):
    a, b = np.array(actual, dtype=float), np.array(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)):
        issues.append(label+': incomplete/nonfinite output')
        return None
    delta = np.abs(a-b)
    tolerance = np.r_[1e-10+1e-10*np.abs(b[:9]), np.full(6, 1e-10), np.full(9, 1e-12)]
    if len(b) and np.any(delta > tolerance):
        issues.append(label+': energy/force/stress comparison failed')
    return [float(np.max(delta[s], initial=0)) for s in (slice(0, 9), slice(9, 15), slice(15, None))]


def verify(root):
    folder = root/'nonlocal-owner-check'
    path = folder/'check.json'
    data, metadata = read(path), read(root/'metadata.json')
    checks, errors, seen = [], [], set()
    if data.get('passed') is not True or data.get('checker_sha256') != metadata.get('nonlocal_owner_check_sha256'):
        errors.append('Check or executed checker binding differs')
    if len(RECOVERY) != 81 or len(set(RECOVERY)) != 81:
        errors.append('Verifier rejection contract is incomplete')
    for row in data['checks']:
        name, spins, split = row['scenario'], row['spins'], row['split']
        key = (name, spins, split)
        issues, cycles, maxima = [], [], np.zeros(3)
        if key in seen:
            issues.append('Duplicate scenario')
        seen.add(key)
        if name not in SCENARIOS or (spins, split) not in SPINS:
            issues.append('Unknown fixture')
        files = {k: folder/row[k+'_file'] for k in ('report', 'owner', 'residency', 'log')}
        if any(sha(p) != row[k+'_sha256'] for k, p in files.items()):
            issues.append('Raw evidence hash differs')
        probe, owner, raw = (read(files[k]) for k in ('report', 'owner', 'residency'))
        failed = name.startswith('failed_')
        stale = name.startswith('stale_')
        held, empty = failed or stale, name == 'empty'
        geometries = 2 if name == 'recovery' else 1
        rejections = RECOVERY*2 if name == 'recovery' else []
        if held:
            rejections = (['failed_completion', 'retry_completion'] if failed else
                          ['stale_begin'] if name == 'stale_setup' else ['stale_apply', 'stale_finish'])
            rejections += ['held_grid', 'held_overlap', 'held_reset', 'held_prepare']
        if (probe['scenario'], probe['spins'], probe['split'], probe['held'], probe['rejections']) != (name, spins, split, held, rejections):
            issues.append('Lifecycle rejection/recovery coverage differs')
        if owner['complete'] != (not held) or len(owner['cycles']) != geometries or len(probe['cycles']) != geometries:
            issues.append('Geometry coverage or owner completion differs')
        for g, (p, r) in enumerate(zip(probe['cycles'], owner['cycles'])):
            f = fixture(spins, g)
            if any(p[k] != v for k, v in f.items()) or p['retained'] != f['density']:
                issues.append('Executed fixture or retained-D snapshot differs')
            expected = contract(f, spins, split, empty)
            compare(p['expected'], expected, issues, 'Independent probe reference')
            status = -1 if stale else 7 if name == 'failed_status' else 0
            want_probe = dict(geometry=g, volume=2.3, queries=0 if empty else 4,
                              executed=not stale, completed=not held, status=status,
                              graph_nodes=2 if name == 'captured' else 0)
            if any(p.get(k) != v for k, v in want_probe.items()):
                issues.append('Execution/status/capture record differs')
            geometry_bytes = 8*2+4*8+(0 if empty else 28+16*2+24*2+32*4)
            allocated = geometry_bytes+(28+16+24+4+8 if empty else 0)
            staged = 0 if name == 'stale_setup' else 8*24
            allocated += staged+(8*8 if staged else 0)
            exported = 0 if stale else 8*24
            wanted = dict(epoch=g+1, consumer=3 if spins == 8 else 2, atoms=2, rows=2, columns=2,
                          capacity=8, components=0 if name == 'stale_setup' else spins,
                          queries=0 if empty else 4, centers=0 if empty else 1,
                          neighbors=0 if empty else 2, blocks=0 if empty else 2, volume=2.3,
                          geometry_bytes=geometry_bytes, setup_bytes=staged, device_allocation_bytes=allocated,
                          export_bytes=exported, electronic_upload_bytes=0, status=status,
                          complete=not stale, passed=not held)
            if any(r.get(k) != v for k, v in wanted.items()):
                issues.append('Owner shape/lifecycle/allocation record differs')
            initial = [] if name == 'stale_setup' else f['initial']
            reference = expected.copy()
            if name == 'failed_reference':
                reference[0] += 1.
            if r['initial'] != initial:
                issues.append('Owner initial state differs')
            compare(r['reference'], [] if name == 'stale_setup' else reference, issues, 'Owner reference')
            delta = compare(r['actual'], [] if stale else expected, issues, 'Owner actual')
            if delta is not None:
                maxima = np.maximum(maxima, delta)
            cycles.append(dict(geometry=g, geometry_bytes=geometry_bytes, setup_bytes=staged,
                               export_bytes=exported, device_allocation_bytes=allocated))
        if raw['geometry_epochs'] != (2 if stale else geometries) or raw['phase_markers_seen'] is not True or raw['schema_version'] != 2:
            issues.append('Geometry/phase/accounting evidence differs')
        if raw['strict_residency_verified'] is not False or any(raw[k] is not None for k in ('library_transfers', 'unified_memory_migrations', 'file_io')):
            issues.append('Explicit-only probe misrepresents independent residency coverage')
        boundaries = ('nonlocal_geometry', 'nonlocal_force_shadow')
        totals = {b: dict(host_to_device_bytes=0, device_to_host_bytes=0, copy_calls=0) for b in boundaries}
        for interval in raw['intervals']:
            for boundary, traffic in interval['boundaries'].items():
                if boundary not in totals:
                    continue
                if interval['phase'] != ('initialization' if boundary == 'nonlocal_geometry' else 'post_scf'):
                    issues.append('Nonlocal copy outside permitted phase')
                for k, v in traffic.items():
                    totals[boundary][k] += v
        wanted_totals = dict(
            nonlocal_geometry=dict(host_to_device_bytes=sum(c['geometry_bytes'] for c in cycles),
                                   device_to_host_bytes=0, copy_calls=(3 if empty else 8)*geometries),
            nonlocal_force_shadow=dict(host_to_device_bytes=sum(c['setup_bytes'] for c in cycles),
                                       device_to_host_bytes=sum(c['export_bytes'] for c in cycles),
                                       copy_calls=sum(bool(c['setup_bytes'])+bool(c['export_bytes']) for c in cycles)))
        if totals != wanted_totals:
            issues.append('Independent shape-derived owner transfer accounting failed')
        if row['native_return'] != 0 or row['passed'] is not True or row['errors']:
            issues.append('Native checker failed')
        checks.append(dict(scenario=name, spins=spins, split=split, passed=not issues, errors=issues,
                           rejections=len(rejections), geometries=geometries, cycles=cycles,
                           max_absolute_error=maxima.tolist(), transfers=totals,
                           evidence_sha256={str(p.relative_to(root)): sha(p) for p in files.values()}))
    if seen != {(n, s, split) for n in SCENARIOS for s, split in SPINS} or len(checks) != 40:
        errors.append('Missing scenario/spin/split coverage')
    return dict(scope=__doc__, passed=not errors and all(c['passed'] for c in checks), errors=errors,
                checks=checks, report_sha256=sha(path), metadata_sha256=sha(root/'metadata.json'),
                verifier_sha256=sha(Path(__file__)), independent_cuda_trace=False,
                native_graph_nodes_note='Owner graph_nodes is a structural constant; execution is checked using probe executed/status and captured-node inspection.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    (args.run/'nonlocal-owner-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'nonlocal-owner-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Nonlocal owner audit:', len(result['checks']), 'passed=', result['passed'])
    if not result['passed']:
        print(result['errors'], [c for c in result['checks'] if not c['passed']])
        raise SystemExit(1)
