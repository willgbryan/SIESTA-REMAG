"""Bounded physical questions, independent of numerical accuracy settings.

A query selects work; it never relaxes SCF acceptance or chooses an accuracy
default. Directions are Cartesian in the exported geometry's solver frame.
"""
import re

import numpy as np


def _vector(value, name):
    if not isinstance(value, list) or len(value) != 3 or any(type(x) not in (int, float) for x in value):
        raise ValueError(f'{name} must contain three finite numbers')
    v = np.array(value, float)
    norm = np.linalg.norm(v)
    if not np.isfinite(v).all() or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f'{name} must be finite and nonzero')
    return v/norm


def validate_query(query):
    if not isinstance(query, dict) or query.get('kind') not in ('scalar', 'anisotropy', 'directions', 'angular'):
        raise ValueError('Query kind must be scalar, anisotropy, directions or angular')
    kind = query['kind']
    fields = {'scalar': {'kind'}, 'anisotropy': {'kind'},
              'directions': {'kind', 'directions'}, 'angular': {'kind', 'axis', 'tangent', 'angles_deg'}}[kind]
    if set(query) != fields:
        raise ValueError('Missing or unknown query fields')
    if kind in ('scalar', 'anisotropy'):
        return {'kind': kind}
    if kind == 'directions':
        directions = query['directions']
        if not isinstance(directions, list) or not 1 <= len(directions) <= 37:
            raise ValueError('Select 1 to 37 independent SOC directions')
        rows = []
        for row in directions:
            if not isinstance(row, dict) or set(row) != {'label', 'vector'}:
                raise ValueError('Direction requires label and vector')
            label = row['label']
            if not isinstance(label, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', label):
                raise ValueError('Direction label must be a short safe identifier')
            vector = _vector(row['vector'], 'Direction')
            if any(label == r['label'] or np.allclose(vector, r['vector'], atol=1e-12, rtol=0) for r in rows):
                raise ValueError('Duplicate direction label or orientation')
            rows.append(dict(label=label, vector=vector.tolist()))
        return dict(kind=kind, directions=rows)
    axis, tangent = _vector(query['axis'], 'Axis'), _vector(query['tangent'], 'Tangent')
    if abs(np.dot(axis, tangent)) > 1e-10:
        raise ValueError('Angular axis and tangent must be perpendicular')
    angles = query['angles_deg']
    if (not isinstance(angles, list) or not 3 <= len(angles) <= 37
            or any(type(x) not in (int, float) or not np.isfinite(x) or not 0 <= x <= 180 for x in angles)
            or len(set(angles)) != len(angles) or 0 not in angles):
        raise ValueError('Angular sweep needs 3 to 37 distinct angles in [0,180], including 0')
    x = np.sin(np.deg2rad(angles))**2
    if np.linalg.matrix_rank(np.array([np.ones(len(x)), x, x*x]).T) < 3:
        raise ValueError('Angles cannot identify K1 and K2')
    return dict(kind=kind, axis=axis.tolist(), tangent=tangent.tolist(), angles_deg=sorted(angles))


def directions(query, default_axes):
    q = validate_query(query)
    if q['kind'] == 'scalar':
        return {}
    if q['kind'] == 'anisotropy':
        return default_axes
    if q['kind'] == 'directions':
        return {row['label']: row['vector'] for row in q['directions']}
    a, t = np.array(q['axis']), np.array(q['tangent'])
    return {f'angle{i:03d}': (np.cos(np.deg2rad(theta))*a + np.sin(np.deg2rad(theta))*t).tolist()
            for i, theta in enumerate(q['angles_deg'])}


def diagnostic_labels(query, requested=None):
    """Keep legacy z/x/y diagnostics; new query types require explicit selection."""
    q = validate_query(query)
    selected = (['z', 'x', 'y'] if q['kind'] == 'anisotropy' else []) if requested is None else requested
    available = directions(q, dict(z=[0,0,1], x=[1,0,0], y=[0,1,0]))
    if (not isinstance(selected, list) or any(not isinstance(x, str) for x in selected)
            or len(set(selected)) != len(selected) or any(x not in available for x in selected)):
        raise ValueError('Diagnostic stages must be distinct requested SOC directions')
    return selected


def angular_fit(query, observations):
    """Fit E and F separately along one declared plane, with a free offset.

Residuals are model diagnostics, not SCF error bars. A one-plane fit cannot
identify azimuthal terms or demonstrate a globally uniaxial energy surface.
"""
    q = validate_query(query)
    if q['kind'] != 'angular':
        raise ValueError('Expected an angular query')
    x = np.sin(np.deg2rad(q['angles_deg']))**2
    design = np.array([np.ones(len(x)), x, x*x]).T
    fits = {}
    for quantity in ('total_energy_ev', 'free_energy_ev'):
        energies = np.array([observations[f'angle{i:03d}'][quantity] for i in range(len(x))], float)
        if not np.isfinite(energies).all():
            raise ValueError('Angular fit requires finite E/F at every angle')
        delta = energies-energies[0]
        coeff, _, _, _ = np.linalg.lstsq(design, delta, rcond=None)
        residual = delta-design@coeff
        fits[quantity] = dict(offset_ev_cell=float(coeff[0]), K1_ev_cell=float(coeff[1]),
            K2_ev_cell=float(coeff[2]), delta_ev_cell=delta.tolist(), residual_ev_cell=residual.tolist(),
            max_abs_residual_ev_cell=float(np.max(abs(residual))), numerical_uncertainty_ev_cell=None)
    return dict(format='REMAG-ANGULAR-FIT/1', query=q, fits=fits,
        model='offset + K1 sin(theta)^2 + K2 sin(theta)^4',
        scope='One Cartesian plane; azimuthal terms and higher orders untested',
        physical_accuracy_qualified=False)
