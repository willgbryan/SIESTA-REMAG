"""Internal angular-model analysis; not a Kohn–Sham torque implementation.

Directions are Cartesian vectors in an orthonormal CRYSTAL frame. The caller
must choose a symmetry model justified by the structure and use actual final
moment directions. No symmetry, constraint, physical convergence or uncertainty
is inferred from a successful fit. Fit E, F and frozen band estimators separately.
"""
import numpy as np

MODELS = {
    'cubic6': ('K1', 'K2'),  # K1*(x²y²+y²z²+z²x²) + K2*x²y²z²
    'axial4': ('A2', 'A4'),  # A2*sin²(theta) + A4*sin⁴(theta); ignores azimuth
}


def unit_directions(directions):
    if np.iscomplexobj(directions):
        raise ValueError('Magnetization directions must be real')
    a = np.asarray(directions, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Directions must be a nonempty finite (N,3) array')
    scale = np.max(abs(a), axis=1)
    if np.any(scale == 0):
        raise ValueError('Zero magnetization has no direction')
    a = a/scale[:, None]
    return a/np.linalg.norm(a, axis=1)[:, None]


def angular_features(directions, model):
    """Two anisotropy invariants, without the orientation-independent constant."""
    x, y, z = (unit_directions(directions)**2).T
    if model == 'cubic6':
        return np.column_stack((x*y+y*z+z*x, x*y*z))
    if model == 'axial4':
        transverse = x+y
        return np.column_stack((transverse, transverse**2))
    raise ValueError('Select cubic6 or axial4 explicitly')


def coefficients_array(coefficients):
    if np.iscomplexobj(coefficients):
        raise ValueError('Anisotropy coefficients must be real')
    a = np.asarray(coefficients, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError('Two finite anisotropy coefficients are required')
    return a


def angular_energy(directions, model, coefficients):
    """Model energy, in the supplied coefficient units, excluding a constant."""
    return angular_features(directions, model) @ coefficients_array(coefficients)


def angular_torque(directions, model, coefficients):
    """Model torque -m cross grad(E), per radian, in coefficient energy units.

    For rotation about unit axis a, dE/dtheta = -a dot torque. This is the
    derivative of the chosen fitted polynomial, not an electronic torque or
    a constraint correction. It cannot validate the polynomial itself.
    """
    m = unit_directions(directions)
    k1, k2 = coefficients_array(coefficients)
    if model == 'cubic6':
        squared = m*m
        others = squared[:, [1, 2, 0]]+squared[:, [2, 0, 1]]
        product = squared[:, [1, 2, 0]]*squared[:, [2, 0, 1]]
        gradient = 2*m*(k1*others+k2*product)
    elif model == 'axial4':
        transverse = m[:, 0]**2+m[:, 1]**2
        gradient = np.zeros_like(m)
        gradient[:, 2] = -2*m[:, 2]*(k1+2*k2*transverse)
    else:
        raise ValueError('Select cubic6 or axial4 explicitly')
    return -np.cross(m, gradient)


def fit_angular_energy(directions, energies, model, *, energy_kind, energy_unit):
    """Fit constant + two invariants without subtracting large fitted totals.

    Coefficients retain the input energy normalization (cell, formula unit,
    volume, etc.). No conversion is performed. The uncertainty amplification
    is an L1 bound: independent per-point energy errors bounded by epsilon can
    shift a coefficient by at most amplification*epsilon, for fixed directions.
    This does not assign an error bar when the input error is unknown.
    """
    features = angular_features(directions, model)
    if np.iscomplexobj(energies):
        raise ValueError('Angular energies must be real')
    y = np.asarray(energies, dtype=float)
    if y.shape != (len(features),) or not np.isfinite(y).all():
        raise ValueError('One finite energy is required per direction')
    if energy_kind not in ('total', 'free', 'frozen_band', 'frozen_band_free'):
        raise ValueError('Declare one energy estimator; never mix E/F or SCF/frozen data')
    if not isinstance(energy_unit, str) or not energy_unit.strip():
        raise ValueError('Declare the energy unit and normalization')
    design = np.column_stack((np.ones(len(y)), features))
    with np.errstate(over='ignore', invalid='ignore'):
        centered = y-y[0]
    if not np.isfinite(centered).all():
        raise ValueError('Relative energies overflow')
    fit, _, rank, singular = np.linalg.lstsq(design, centered, rcond=1.e-12)
    if rank != 3:
        raise ValueError('Angular samples do not independently determine both coefficients and a constant')
    residual = centered-design@fit
    if not np.isfinite(fit).all() or not np.isfinite(residual).all():
        raise ValueError('Nonfinite angular fit')
    inverse = np.linalg.pinv(design, rcond=1.e-12)
    names = MODELS[model]
    unique = len(np.unique(np.round(features, decimals=12), axis=0))
    return dict(model=model, energy_kind=energy_kind, energy_unit=energy_unit,
                energy_origin=float(y[0]), offset_relative_to_origin=float(fit[0]),
                coefficients={name: float(value) for name, value in zip(names, fit[1:])},
                coefficient_error_amplification={name: float(value) for name, value in
                    zip(names, np.sum(abs(inverse[1:]), axis=1))},
                observations=len(y), distinct_invariant_points=unique,
                overdetermined_angular_fit=unique > 3,
                symmetry_equivalent_samples=len(y)-unique,
                residual_degrees_of_freedom=int(len(y)-rank),
                condition_number=float(singular[0]/singular[-1]),
                relative_energies=centered.tolist(), fitted_relative_energies=(design@fit).tolist(),
                residuals=residual.tolist(), max_absolute_residual=float(np.max(abs(residual))),
                coefficient_uncertainties=None, physical_accuracy_qualified=False)
