"""Collect independent full-SCF directions from the native orientation loop.

Completion and direction attribution are checked here. Scientific equivalence
still requires the ordinary observable/export and directional E/F comparisons.
"""
import math
from pathlib import Path
import re
import shutil

import numpy as np

from .frozen_reference import direction_checks
from .output import parse_output, read_eigenvalues, read_geometry


def _quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError('A finite nonzero quaternion is required')
    q = q / max(abs(q))
    return q / np.linalg.norm(q)


def parse_sweep(text, directions, rotations, *, returncode, timed_out, label='remag'):
    """Use ordered axes/quaternions from the requested inputs, never stdout labels."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Unsupported orientation SystemLabel')
    if not directions or len(directions) != len(rotations) or len(directions) > 256:
        raise ValueError('Provide 1–256 requested directions and their absolute rotations')
    requested = [_quaternion(q) for q in rotations]
    binding = direction_checks(dict(spin_expectations=False, points=[
        dict(index=i, quaternion=q.tolist()) for i, q in enumerate(requested)]), directions)
    if not binding['passed']:
        raise ValueError('Requested absolute rotations do not match the target axes')

    events = list(re.finditer(r'^REMAG ORIENTATION ([^\n]*)$', text, re.M))
    expected = [(kind, i) for i in range(1, len(directions)+1) for kind in ('BEGIN', 'END')]
    parsed = []
    for event in events:
        fields = event.group(1).split()
        try:
            kind, index = fields[0], int(fields[1])
            if kind not in ('BEGIN', 'END') or len(fields) != (7 if kind == 'BEGIN' else 2):
                raise ValueError('Invalid orientation marker')
            if kind == 'BEGIN':
                q = [float(v.replace('D', 'E').replace('d', 'e')) for v in fields[3:]]
                _quaternion(q)
            else:
                q = None
            parsed.append(dict(kind=kind, index=index, label=fields[2] if q is not None else None,
                               quaternion=q, start=event.start(), end=event.end()))
        except (IndexError, ValueError, TypeError):
            parsed.append(dict(kind='INVALID', index=None, start=event.start(), end=event.end()))
    errors = []
    if [(e['kind'], e['index']) for e in parsed] != expected:
        errors.append('Orientation markers are missing, duplicated, malformed or out of order')
    completed = type(returncode) is int and returncode == 0 and timed_out is False and 'Job completed' in text
    if not completed:
        errors.append('Orientation process did not complete successfully')
    header = text[:events[0].start()] if events else ''
    metadata = parse_output(header)
    rows = []
    for index, axis in enumerate(directions, 1):
        starts = [j for j, e in enumerate(parsed) if e['kind'] == 'BEGIN' and e['index'] == index]
        row_errors = []
        body = ''; closed = False; quaternion = None
        prefix = f'{label}.o{index:03d}'
        if len(starts) != 1:
            row_errors.append('Expected exactly one start marker for this direction')
        else:
            j = starts[0];start = parsed[j]
            following = parsed[j+1] if j+1 < len(parsed) else None
            body = text[start['end']:following['start'] if following else len(text)]
            closed = bool(following and following['kind'] == 'END' and following['index'] == index)
            quaternion = start['quaternion']
            q = _quaternion(quaternion)
            if start['label'] != prefix or min(np.linalg.norm(q-requested[index-1]), np.linalg.norm(q+requested[index-1])) > 1e-12:
                row_errors.append('Direction label or quaternion differs from the requested input')
        if not closed:
            row_errors.append('Direction has no matching end marker')
        # Never inherit an earlier energy, moment or convergence marker from
        # another direction. Only static spin metadata comes from the preamble.
        obs = parse_output(body)
        obs.update(completed=completed and closed,
                   spin_components=metadata['spin_components'], spin_degenerate=metadata['spin_degenerate'])
        if not obs['converged']:
            row_errors.append('Direction lacks a converged SCF cycle')
        for field in ('total_energy_ev', 'free_energy_ev', 'fermi_energy_ev'):
            if type(obs[field]) not in (int, float) or not math.isfinite(obs[field]):
                row_errors.append('Direction lacks finite '+field)
        rows.append(dict(index=index, label=prefix, quaternion=quaternion, requested_direction=list(axis),
                         stdout=header+body, observables=obs, errors=row_errors))
    magnetic = direction_checks(dict(spin_expectations=True, points=[
        dict(index=i, quaternion=r['quaternion'], spin_muB=r['observables']['total_moment_muB'])
        for i, r in enumerate(rows)]), directions)
    for row, check in zip(rows, magnetic['points'], strict=True):
        row['direction_check'] = check
        if not check['passed']:
            row['errors'].append('Final moment or rotation does not match the requested direction')
        row['valid'] = not row['errors'] and row['observables']['completed']
    return dict(passed=not errors and all(r['valid'] for r in rows), errors=errors, directions=rows,
                physical_accuracy_qualified=False)


def write_direction_views(folder, report):
    """Keep source files intact; create ordinary per-direction comparison views.

    Refuse existing views to preserve previous/partial attempts. Whole-process
    timing remains with the caller; a direction has no inferred wall time.
    """
    folder = Path(folder)
    target = folder/'direction-outputs'
    target.mkdir(exist_ok=False)
    paths = []
    for row in report['directions']:
        view = target/f"{row['index']:03d}"
        view.mkdir()
        (view/'stdout.log').write_text(row['stdout'])
        prefix = row['label']+'.'
        for source in folder.iterdir():
            if source.is_file() and source.name.startswith(prefix):
                shutil.copyfile(source, view/('remag.'+source.name[len(prefix):]))
        for name, reader, key in [('remag.EIG', read_eigenvalues, 'eigenvalues'), ('remag.XV', read_geometry, 'geometry')]:
            try:
                row['observables'][key] = reader(view/name)
            except (OSError, ValueError, IndexError) as error:
                row['errors'].append(f'{name}: {error}')
                row['valid'] = False
        paths.append(view)
    report['passed'] = report['passed'] and all(r['valid'] for r in report['directions'])
    return paths
