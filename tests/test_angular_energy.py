"""Angular identifiability, omitted terms and model torque, independent of GPUs."""
import itertools

import numpy as np
import pytest

from siesta_remag.angular_energy import (angular_energy, angular_features, angular_torque,
                                        fit_angular_energy, unit_directions)

CUBIC_AXES = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]


def fit(directions, energies, model='cubic6'):
    return fit_angular_energy(directions, energies, model, energy_kind='free', energy_unit='eV/f.u.')


def test_three_cubic_axes_identify_coefficients_without_testing_model():
    k1, k2 = 35.e-6, 290.e-6
    values = [0., k1/4, k1/3+k2/27]
    r = fit(CUBIC_AXES, values)
    assert r['coefficients'] == pytest.approx(dict(K1=k1, K2=k2))
    assert r['coefficient_error_amplification'] == pytest.approx(dict(K1=8., K2=72.))
    assert r['residual_degrees_of_freedom'] == 0
    assert not r['overdetermined_angular_fit']
    assert r['coefficient_uncertainties'] is None and not r['physical_accuracy_qualified']


def test_equivalent_axes_are_controls_not_new_angular_information():
    directions = CUBIC_AXES+[[1, 0, 0], [0, -1, 0]]
    values = [0., 1.e-5, 3.e-5, 0., 1.e-6]
    r = fit(directions, values)
    assert r['symmetry_equivalent_samples'] == 2
    assert r['residual_degrees_of_freedom'] == 2
    assert r['max_absolute_residual'] > 5.e-7
    assert not r['overdetermined_angular_fit']


def test_unseen_angle_exposes_higher_order_term_hidden_by_three_point_fit():
    directions = CUBIC_AXES+[[2, 1, 0]]
    features = angular_features(directions, 'cubic6')
    values = features@np.array([30.e-6, 290.e-6])+1.e-3*features[:, 0]**2
    three = fit(directions[:3], values[:3])
    four = fit(directions, values)
    assert three['max_absolute_residual'] < 1.e-18
    prediction = angular_energy(directions[-1:], 'cubic6', list(three['coefficients'].values()))[0]
    prediction += three['offset_relative_to_origin']+three['energy_origin']
    assert abs(prediction-values[-1]) > 1.e-5
    assert four['overdetermined_angular_fit'] and four['max_absolute_residual'] > 1.e-6


def test_cubic_energy_obeys_axis_permutations_and_time_reversal():
    directions = [np.array(p)*s for p in itertools.permutations([1, 2, 3])
                  for s in itertools.product([-1, 1], repeat=3)]
    e = angular_energy(directions, 'cubic6', [1.3, -2.7])
    assert np.max(abs(e-e[0])) < 2.e-16


@pytest.mark.parametrize('model', ['cubic6', 'axial4'])
def test_torque_matches_independent_rotation_finite_difference(model):
    m = unit_directions([[2, -3, 5], [1, 0, 0], [0, 0, 1]])
    axis = unit_directions([[1, 2, -1]])[0]
    k = [1.3, -.7]
    def rotated(theta):
        return m*np.cos(theta)+np.cross(axis, m)*np.sin(theta)+np.outer(m@axis, axis)*(1-np.cos(theta))
    step = 1.e-4
    # Five-point derivative avoids allowing a large truncation error.
    derivative = (8*(angular_energy(rotated(step), model, k)-angular_energy(rotated(-step), model, k))
                  -(angular_energy(rotated(2*step), model, k)-angular_energy(rotated(-2*step), model, k)))/(12*step)
    torque = angular_torque(m, model, k)
    np.testing.assert_allclose(-torque@axis, derivative, atol=2.e-11, rtol=0)
    np.testing.assert_allclose(np.sum(torque*m, axis=1), 0, atol=2.e-16)


def test_45_degree_shortcut_belongs_to_axial_quartic_model():
    k = [1.3, -.7]
    m = [[1, 0, 1]]
    delta = angular_energy([[1, 0, 0]], 'axial4', k)[0]-angular_energy([[0, 0, 1]], 'axial4', k)[0]
    assert -angular_torque(m, 'axial4', k)[0, 1] == pytest.approx(delta)
    cubic_delta = angular_energy([[1, 0, 0]], 'cubic6', k)[0]-angular_energy([[0, 0, 1]], 'cubic6', k)[0]
    assert cubic_delta == 0  # Symmetry-equivalent cubic directions.
    assert delta != 0


def test_axial_fit_requires_intermediate_angle():
    with pytest.raises(ValueError, match='independently determine'):
        fit([[0, 0, 1], [1, 0, 0]], [0, 1.e-3], 'axial4')
    r = fit([[0, 0, 1], [1, 0, 1], [1, 0, 0]], [0, .5+.25*2, 3], 'axial4')
    assert r['coefficients'] == pytest.approx(dict(A2=1, A4=2))


def test_large_energy_origin_does_not_pollute_fit_residuals():
    values = np.array([-7573., -7573.+8.e-6, -7573.+22.e-6])
    r = fit(CUBIC_AXES, values)
    assert r['coefficients']['K1'] == pytest.approx(4*(values[1]-values[0]), abs=1.e-18)
    assert r['coefficients']['K2'] == pytest.approx(27*(values[2]-values[0])-36*(values[1]-values[0]), abs=1.e-18)
    assert r['max_absolute_residual'] < 1.e-18


@pytest.mark.parametrize('directions,energies', [
    ([[0, 0, 0], [1, 1, 0], [1, 1, 1]], [0, 1, 2]),
    ([[0, 0, np.nan], [1, 1, 0], [1, 1, 1]], [0, 1, 2]),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]),
    (CUBIC_AXES, [0, 1]), (CUBIC_AXES, [0, np.inf, 2]),
    (CUBIC_AXES, [1.e308, -1.e308, 0]),
])
def test_invalid_or_unidentifiable_inputs_fail(directions, energies):
    with pytest.raises(ValueError):
        fit(directions, energies)


def test_complex_data_are_not_silently_discarded():
    with pytest.raises(ValueError, match='real'):
        fit(CUBIC_AXES, np.array([0, 1, 2], complex)+1j)
    with pytest.raises(ValueError, match='real'):
        unit_directions(np.array(CUBIC_AXES, complex))
    with pytest.raises(ValueError, match='real'):
        angular_energy(CUBIC_AXES, 'cubic6', np.array([1, 2], complex))


def test_direction_scaling_does_not_change_energy_or_torque():
    directions = np.array([[1, -2, 3], [2, 3, -1]], float)
    extreme = directions*np.array([1.e300, 1.e-300])[:, None]
    np.testing.assert_allclose(angular_energy(extreme, 'cubic6', [1, 2]),
                               angular_energy(directions, 'cubic6', [1, 2]), atol=1.e-15)
    np.testing.assert_allclose(angular_torque(extreme, 'cubic6', [1, 2]),
                               angular_torque(directions, 'cubic6', [1, 2]), atol=1.e-15)
