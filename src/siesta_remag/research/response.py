"""Identifiable volume/shape response models with exact structural provenance."""
from pathlib import Path

import numpy as np

from ..corpus import sha256
from ..discovery.common import digest, load
from .convergence import numerical_configuration


def coordinates(reference, structure):
    """Return log volume and c/a changes for matched orthogonal tetragonal cells.

    Exact sites, species, Hall setting and units must match before numerical
    descriptors are formed. These coordinates describe structure, not orbitals.
    """
    from gpac import ir
    from gpac.physical import verify_export
    ref, child = [verify_export(s) for s in (reference, structure)]
    a, b = [ir.execute(s['source']['program']) for s in (ref, child)]
    if a.program.hall != b.program.hall or list(a.sites) != list(b.sites):
        raise ValueError('Response controls must retain Hall setting, species and exact fractional sites')
    for e in (a, b):
        if any(e.gram[i][j] for i in range(3) for j in range(3) if i != j) or e.gram[0][0] != e.gram[1][1]:
            raise ValueError('Orthogonal tetragonal metrics required')
    ratios = [b.gram[i][i]/a.gram[i][i] for i in range(3)]
    qv = .5 * np.log(float(ratios[0]*ratios[1]*ratios[2]))
    qs = .5 * np.log(float(ratios[2]/ratios[0]))
    return dict(log_volume_ratio=float(qv), log_c_over_a_ratio=float(qs))


def _design(points):
    v = np.array([p['log_volume_ratio'] for p in points])
    s = np.array([p['log_c_over_a_ratio'] for p in points])
    return np.column_stack([np.ones(len(points)), v, s, v*v, s*s])


def fit(reference, reports):
    """Fit an additive quadratic with no inferred mixed interaction.

    A pure c-strain series cannot identify separate volume and shape responses;
    rank deficiency is rejected rather than resolved by a minimum-norm guess.
    At least five measured structures with common numerical controls are needed.
    """
    if len(reports) < 5:
        raise ValueError('At least five independent measured controls required')
    points, values, configs, sources = [], [], [], []
    for item in reports:
        path = Path(item) if isinstance(item, (str, Path)) else None
        r = load(path) if path else item
        if (r.get('format') != 'REMAG-EXPERIMENT-RESULT/1' or r.get('status') != 'measured'
                or r['request'].get('fixed_geometry') is not True
                or r['request']['source_sha256'] != r['source_sha256']
                or r['structure']['source']['sha256'] != r['source_sha256']
                or [s['stage'] for s in r['stages']] != ['scalar','soc-z','soc-x','soc-y']
                or not all(s['accepted'] for s in r['stages'])):
            raise ValueError('Source-bound measured fixed-geometry reports required')
        points.append(coordinates(reference, r['structure']))
        z, x = [r['observations']['soc-'+axis] for axis in ('z','x')]
        values.append([x[k]-z[k] for k in ('total_energy_ev', 'free_energy_ev')])
        config = numerical_configuration(r); del config['source_sha256']; configs.append(config)
        sources.append(dict(source_sha256=r['source_sha256'], report_sha256=sha256(path) if path else None))
    if len({s['source_sha256'] for s in sources}) != len(sources):
        raise ValueError('Repeated structures are not independent controls')
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError('Numerical settings or executable differ')
    design = _design(points); y = np.array(values)
    if not np.isfinite(y).all(): raise ValueError('Finite directional energies required')
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0): raise ValueError('Volume/shape response is not identifiable')
    normalized = design/scales
    coefficients, _, rank, singular = np.linalg.lstsq(normalized, y, rcond=1e-10)
    if rank != 5: raise ValueError('Volume/shape response is not identifiable')
    coefficients /= scales[:, None]
    result = dict(format='REMAG-RESPONSE/1', model='additive quadratic in log volume and log c/a',
        terms=['1','log(V/V0)','log((c/a)/(c/a)0)','log(V/V0)^2','log((c/a)/(c/a)0)^2'],
        observables=['energy_anisotropy_ev','free_energy_anisotropy_ev'],
        coefficients=coefficients.tolist(), reference=reference, training=sources, coordinates=points,
        training_observations_ev=y.tolist(),
        training_residual_ev=(y-design@coefficients).tolist(), normalized_condition_number=float(singular[0]/singular[-1]),
        numerical_configuration=configs[0], physical_accuracy_qualified=False, mechanism_verified=False,
        scope='Geometric response hypothesis; mixed volume-shape term is unmeasured; no electronic mechanism or uncertainty inferred')
    result['sha256'] = digest(result)
    return result


def predict(model, structure):
    """Bind a prediction to exact structure and immutable fitted model bytes."""
    body = {k:v for k,v in model.items() if k != 'sha256'}
    if model.get('format') != 'REMAG-RESPONSE/1' or model.get('sha256') != digest(body):
        raise ValueError('Response model hash mismatch')
    point = coordinates(model['reference'], structure)
    values = (_design([point]) @ np.asarray(model['coefficients']))[0]
    return dict(format='REMAG-RESPONSE-PREDICTION/1', model_sha256=model['sha256'],
        source_sha256=structure['source']['sha256'], coordinates=point,
        **dict(zip(model['observables'], map(float, values))), uncertainty_ev=None,
        numerical_configuration=model['numerical_configuration'],
        mechanism_verified=False, physical_accuracy_qualified=False)


def baseline_predictions(model, structure):
    """Constant, shape-only and volume-only fits to exactly the same outcomes."""
    bound = predict(model, structure)  # verifies model and structural provenance
    y = np.array(model['training_observations_ev'])
    values = {'constant': y.mean(axis=0)}
    for name, key in [('shape_only', 'log_c_over_a_ratio'), ('volume_only', 'log_volume_ratio')]:
        x = np.array([p[key] for p in model['coordinates']])
        scale = max(abs(x))
        matrix = np.column_stack([np.ones(len(x)), x/scale, (x/scale)**2])
        coef, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=1e-10)
        if rank != 3: raise ValueError('Baseline quadratic is not identifiable')
        q = bound['coordinates'][key]/scale
        values[name] = np.array([1,q,q*q])@coef
    return {name: dict(model_sha256=model['sha256'], source_sha256=bound['source_sha256'],
                      **dict(zip(model['observables'], map(float, row)))) for name,row in values.items()}
