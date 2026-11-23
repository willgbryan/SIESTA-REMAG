"""Query routing uses the existing native acceptance harness, not fake DFT evidence."""
import numpy as np
import pytest

from test_research import experiment
from test_discovery import backend
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.queries import validate_query, directions, angular_fit, diagnostic_labels


def execute_query(experiment, tmp_path, query, fail=None):
    request = load(experiment/'experiment.json')
    request['query'] = validate_query(query)
    save(experiment/'experiment.json', request)
    executable = tmp_path/'binary'; executable.write_text('test backend')
    calls = []
    result = run_experiment(experiment, tmp_path/'output', executable,
                           executable_sha256=sha256(executable), runner=backend(calls, fail=fail))
    return result, calls


def test_scalar_query_never_runs_soc_or_claims_anisotropy(experiment, tmp_path):
    result, calls = execute_query(experiment, tmp_path, {'kind': 'scalar'})
    assert result['status'] == 'measured'
    assert [c[0] for c in calls] == ['scalar']
    assert result['anisotropy'] is None
    assert set(result['observations']) == {'scalar'}


def test_selected_soc_query_and_failed_branch(experiment, tmp_path):
    result, calls = execute_query(experiment, tmp_path,
        dict(kind='directions', directions=[dict(label='diagonal', vector=[1, 0, 1])]))
    assert result['status'] == 'measured', result.get('reason')
    assert [c[0] for c in calls] == ['scalar', 'soc-diagonal']
    assert result['stages'][-1]['branch']['passed']
    assert result['anisotropy'] is None
    assert result['axes_cartesian']['diagonal'] == pytest.approx([2**-.5, 0, 2**-.5])


def test_requested_failure_stops_without_fallback(experiment, tmp_path):
    result, calls = execute_query(experiment, tmp_path,
        dict(kind='directions', directions=[dict(label='z', vector=[0, 0, 1]), dict(label='x', vector=[1, 0, 0])]),
        fail='soc-z')
    assert result['status'] == 'held_soc-z'
    assert len(calls) == 2
    assert result['anisotropy'] is None


def test_angular_sweep_runs_requested_plane(experiment, tmp_path):
    result, calls = execute_query(experiment, tmp_path,
        dict(kind='angular', axis=[0,0,1], tangent=[1,0,0], angles_deg=[90,0,45]))
    assert result['status'] == 'measured', result.get('reason')
    assert [c[0] for c in calls] == ['scalar', 'soc-angle000', 'soc-angle001', 'soc-angle002']
    assert result['anisotropy'] is None
    assert result['angular_fit']['fits']['total_energy_ev']['K1_ev_cell'] == pytest.approx(0)


def test_angular_coefficients_from_independent_energy_model():
    q = dict(kind='angular', axis=[0,0,1], tangent=[1,0,0], angles_deg=[0,30,45,60,90])
    rows = {}
    for i, theta in enumerate(q['angles_deg']):
        x = np.sin(np.deg2rad(theta))**2
        rows[f'angle{i:03d}'] = dict(total_energy_ev=-23+.002*x-.0003*x*x,
                                    free_energy_ev=-23.1+.0018*x+.0001*x*x)
    r = angular_fit(q, rows)
    assert r['fits']['total_energy_ev']['K1_ev_cell'] == pytest.approx(.002)
    assert r['fits']['total_energy_ev']['K2_ev_cell'] == pytest.approx(-.0003)
    assert r['fits']['free_energy_ev']['K2_ev_cell'] == pytest.approx(.0001)
    assert r['fits']['total_energy_ev']['max_abs_residual_ev_cell'] < 1e-14
    assert r['fits']['total_energy_ev']['numerical_uncertainty_ev_cell'] is None


@pytest.mark.parametrize('query', [
    {'kind':'scalar', 'directions':[]},
    {'kind':'directions', 'directions':[{'label':'../x','vector':[0,0,1]}]},
    {'kind':'directions', 'directions':[{'label':'x','vector':[0,0,0]}]},
    {'kind':'directions', 'directions':[{'label':'x','vector':[1,0,0]}, {'label':'y','vector':[2,0,0]}]},
    {'kind':'angular', 'axis':[0,0,1], 'tangent':[0,0,1], 'angles_deg':[0,45,90]},
    {'kind':'angular', 'axis':[0,0,1], 'tangent':[1,0,0], 'angles_deg':[0,90,180]},
    {'kind':'angular', 'axis':[0,0,1], 'tangent':[1,0,0], 'angles_deg':[0,45,float('nan')]},
])
def test_invalid_or_unidentifiable_query_rejected(query):
    with pytest.raises(ValueError): validate_query(query)


def test_diagnostics_cannot_request_uncomputed_stages():
    assert diagnostic_labels({'kind':'scalar'}) == []
    assert diagnostic_labels({'kind':'anisotropy'}) == ['z','x','y']
    with pytest.raises(ValueError): diagnostic_labels({'kind':'scalar'}, ['z'])
    query = dict(kind='directions', directions=[dict(label='diagonal', vector=[1,0,1])])
    assert diagnostic_labels(query, ['diagonal']) == ['diagonal']
