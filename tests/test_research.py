import copy
from pathlib import Path

import numpy as np
import pytest

from siesta_remag.research.electronic import mulliken_weights, finite_difference, bloch_matrices
from siesta_remag.research.memory import Notebook, evaluate_holdout
from siesta_remag.research.experiments import prepare_experiment, run_experiment, verify_experiment
from siesta_remag.discovery.common import load
from siesta_remag.corpus import sha256


def test_nonorthogonal_population_and_spinor_assembly():
    from scipy.linalg import eigh
    s = np.array([[1, .3], [.3, 1]])
    h = np.diag([1., 2.])
    _, c = eigh(h, s)
    weights = mulliken_weights(c, s)
    assert np.allclose(weights.sum(axis=0), 1)
    assert not np.allclose(abs(c)**2, weights)
    with pytest.raises(ValueError): mulliken_weights(np.eye(2), s)
    raw = dict(header=[1, 1, 8], counts=np.array([1]), columns=np.array([1]), offsets=np.array([0, 0, 0]),
               overlap=np.array([1.]), h=np.array([[2], [3], [.2], [.4], [0], [0], [.2], [.4]]), precision=1)
    hk, sk = bloch_matrices(raw, [0, 0, 0])
    assert hk[0, 1].imag < 0
    assert np.allclose(hk, hk.conj().T) and np.allclose(sk, np.eye(2))


def test_unequal_finite_difference_and_unknown_error():
    points = [dict(q=q, value=7+2*q+3*q*q) for q in [-.02, 0, .01]]
    r = finite_difference(points, parameter_unit='strain', property_unit='eV/cell')
    assert r['first'] == pytest.approx(2)
    assert r['second'] == pytest.approx(6)
    assert r['propagated_error_bound'] is None and r['truncation_error'] is None
    for p in points: p['error_bound'] = 1e-6
    assert finite_difference(points, parameter_unit='strain', property_unit='eV')['propagated_error_bound']['first'] > 0


def test_complex_offsite_spinor_uses_native_spatial_index_order():
    # Independent explicit matrix from SIESTA diag3kp.F90 setup_k's (spin,j,spin,i).
    # A one-orbital or real collinear control cannot detect this transpose error.
    from siesta_remag.research.electronic import RY_EV
    raw = dict(header=[2, 2, 8], counts=np.array([2, 2]), columns=np.array([1, 2, 1, 2]),
        offsets=np.array([0, 0, 0]), overlap=np.array([1., 0, 0, 1]), precision=1,
        h=np.array([[1,0,0,2], [3,0,0,4], [.2,.8,.6,1], [.1,.9,.7,.2],
                    [0,-.3,.3,0], [0,.4,-.4,0], [.2,.6,.8,1], [.1,.7,.9,.2]]))
    expected = np.array([[1,.2-.1j,.3j,.6-.7j], [.2+.1j,3,.8+.9j,-.4j],
                         [-.3j,.8-.9j,2,1-.2j], [.6+.7j,.4j,1+.2j,4]])
    h, s = bloch_matrices(raw, [0, 0, 0])
    assert np.allclose(h/RY_EV, expected)
    assert np.allclose(s, np.eye(4))


def test_viewer_withholds_uncompared_or_failed_spectra(tmp_path):
    from siesta_remag.research.workspace import electronic_preview
    from siesta_remag.discovery.common import save
    p = tmp_path/'electronic-z'; p.mkdir()
    for residual, passed in [(None, False), (1.2, False), (1.2, True)]:
        save(p/'electronic.json', dict(native_max_abs_error_ev=residual, native_comparison_passed=passed))
        assert electronic_preview(tmp_path) is None


def test_predictions_cannot_follow_outcomes(tmp_path):
    n = Notebook(tmp_path/'research.sqlite')
    h = n.record_hypothesis('strain changes near-Fermi states', predictions=['positive slope'], alternatives=['branch change'])
    e = n.propose_experiment(h, structure_sha256='a'*64, action='strain', rationale='distinguish response')
    n.record_prediction(e, observable='MAE', value=1., unit='meV/cell', uncertainty=.2)
    with pytest.raises(ValueError): n.record_prediction(e, observable='MAE', value=2., unit='meV/cell')
    n.record_outcome(e, observable='MAE', value=1.1, unit='meV/cell', report_sha256='b'*64, status='measured')
    assert n.evaluate(e)['within_predicted_uncertainty']
    with pytest.raises(ValueError): n.record_prediction(e, observable='MAE', value=1.1, unit='meV/cell')
    assert len(n.retrieve('strain')) == 2
    n.close()
    r = evaluate_holdout([[-1, 0], [0, 1], [1, 2]], [[.5, 1.5]], agent_predictions=[1.6])
    assert r['agent']['mae'] == pytest.approx(.1)
    assert r['interpolation']['mae'] == 0


@pytest.fixture
def experiment(tmp_path):
    ir = pytest.importorskip('gpac.ir')
    from gpac.physical import export_geometry
    p = ir.author_program(dict(hall=1, gram=[[9, 0, 0], [0, 9, 0], [0, 0, 9]], orbits=[
        dict(z=26, wyckoff_index=0, parameters=[0, 0, 0]),
        dict(z=26, wyckoff_index=0, parameters=['1/2', '1/2', '1/2'])]))
    pseudo = tmp_path/'pseudos'; pseudo.mkdir()
    (pseudo/'Fe.psf').write_text(' Fe pb rel pcec\nfixture only\n')
    b = tmp_path/'bundle'
    prepare_experiment(export_geometry(p, length_unit='angstrom'), pseudo, b, mesh=[3, 3, 3])
    return b


def test_fixed_experiment_and_provenance(experiment, tmp_path):
    from test_discovery import backend
    exe = tmp_path/'binary'; exe.write_text('fixture')
    calls = []
    report = run_experiment(experiment, tmp_path/'result', exe, executable_sha256=sha256(exe), runner=backend(calls))
    assert report['status'] == 'measured', report.get('reason', report['stages'])
    assert [c[0] for c in calls] == ['scalar', 'soc-z', 'soc-x', 'soc-y']
    assert report['anisotropy']['mae_ev_cell'] == pytest.approx(.0001)
    assert report['anisotropy']['axes_cartesian'] == dict(z=[0.,0.,1.], x=[1.,0.,0.], y=[0.,1.,0.])
    assert not report['physical_accuracy_qualified']
    assert 'MD.NumCGsteps 0' in (experiment/'scalar/input.fdf').read_text()
    assert 'MD.VariableCell' not in (experiment/'scalar/input.fdf').read_text()
    assert '0.0000000000000000 0.0000000000000000 0.0000000000000000 1' in (experiment/'scalar/input.fdf').read_text()
    assert run_experiment(experiment, tmp_path/'result', exe, executable_sha256=sha256(exe), runner=backend(calls))['status'] == 'measured'
    assert len(calls) == 4
    (experiment/'scalar/input.fdf').write_text('tampered')
    with pytest.raises(ValueError): verify_experiment(experiment)


def test_fixed_failure_retains_no_anisotropy(experiment, tmp_path):
    from test_discovery import backend
    exe = tmp_path/'binary'; exe.write_text('fixture'); calls = []
    r = run_experiment(experiment, tmp_path/'result', exe, executable_sha256=sha256(exe), runner=backend(calls, fail='soc-x'))
    assert r['status'] == 'held_soc-x' and r['anisotropy'] is None
    assert len(calls) == 3
