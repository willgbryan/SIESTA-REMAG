from copy import deepcopy

import numpy as np
import pytest

from siesta_remag.research.response import coordinates, fit, predict, baseline_predictions
from test_research_convergence import report


@pytest.fixture(scope='module')
def structures():
    from gpac.physical import fept_reference, export_geometry, isotropic_scale, volume_preserving_tetragonal, strain
    p = fept_reference()
    ref = export_geometry(p, length_unit='angstrom')
    controls = [ref]
    for operation in (isotropic_scale, volume_preserving_tetragonal):
        for scale in ('399/400', '401/400'):
            child = operation(p, scale, expected_sha256=p.sha256)['child']['program']
            controls.append(export_geometry(child, length_unit='angstrom'))
    axial = [export_geometry(strain(p, 2, q, expected_sha256=p.sha256)['child']['program'], length_unit='angstrom')
             for q in ('-1/100', '-1/200', '0', '1/200', '1/100')]
    return ref, controls, axial


def synthetic(reference, structures):
    rows = []
    for structure in structures:
        p = coordinates(reference, structure); v,s = p.values()
        energy = .003 + .1*v + .2*s + 3*v*v - 4*s*s
        row = report(24, energy, energy*.9)
        row['structure'] = structure
        row['source_sha256'] = row['request']['source_sha256'] = structure['source']['sha256']
        rows.append(row)
    return rows


def test_controls_identify_response_and_bind_prediction(structures):
    ref, controls, axial = structures
    for i, child in enumerate(controls):
        p = coordinates(ref, child)
        if i in (1,2): assert abs(p['log_c_over_a_ratio']) < 1e-14
        if i in (3,4): assert abs(p['log_volume_ratio']) < 1e-14
    model = fit(ref, synthetic(ref, controls))
    assert np.allclose(np.array(model['coefficients'])[:,0], [.003,.1,.2,3,-4])
    result = predict(model, axial[-1])
    expected = synthetic(ref, [axial[-1]])[0]['anisotropy']['mae_ev_cell']
    assert result['energy_anisotropy_ev'] == pytest.approx(expected, abs=1e-14)
    assert result['uncertainty_ev'] is None and not result['mechanism_verified']
    baseline = baseline_predictions(model, axial[-1])
    assert set(baseline) == {'constant','shape_only','volume_only'}
    for name in baseline:
        assert baseline[name]['model_sha256'] == model['sha256']
        assert abs(baseline[name]['energy_anisotropy_ev']-expected) > 1e-5
    model['coefficients'][0][0] += .001
    with pytest.raises(ValueError, match='hash'): predict(model, axial[-1])


def test_c_axis_only_data_cannot_separate_volume_and_shape(structures):
    ref, _, axial = structures
    with pytest.raises(ValueError, match='not identifiable'): fit(ref, synthetic(ref, axial))


def test_mixed_settings_and_repeated_controls_rejected(structures):
    ref, controls, _ = structures
    rows = synthetic(ref, controls)
    rows[-1]['request']['settings']['mesh_ry'] = 500
    with pytest.raises(ValueError, match='settings'): fit(ref, rows)
    rows = synthetic(ref, controls); rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match='Repeated'): fit(ref, rows)
