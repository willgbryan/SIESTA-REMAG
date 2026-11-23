"""Operator conventions shared by screening methods; no host eigensolver."""
import numpy as np
from ..angular_energy import unit_directions

RY_EV = 13.605693122994
BOHR_ANGSTROM = 0.529177210903


def native_lower_hermitian(a):
    """Exact LOWER triangle consumed by pinned SIESTA/cuSOLVER captures.

    Independently accumulated upper entries are diagnostics, not a second
    definition of the operator. Do not average them into the native spectrum.
    """
    a = np.asarray(a, dtype=np.complex128)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        raise ValueError('Finite native square matrices required')
    lower = np.tril(a,-1)
    out=lower+lower.swapaxes(-1,-2).conj()
    ids=np.arange(a.shape[-1]);out[...,ids,ids]=a[...,ids,ids].real
    return out


def hermitian(a, name, tolerance=1.e-10):
    a = np.asarray(a, dtype=np.complex128)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite square matrices required")
    error = float(np.max(abs(a-a.swapaxes(-1, -2).conj())))
    if error > tolerance:
        raise ValueError(f"{name}: Hermiticity error {error:g} exceeds {tolerance:g}")
    return (a+a.swapaxes(-1, -2).conj())*.5


def spin_trace(a):
    """Spatial matrix: half the spin trace, interleaved orbital/up/down order."""
    return .5*(a[..., ::2, ::2]+a[..., 1::2, 1::2])


def spin_scalar(a):
    result = np.zeros(a.shape[:-2]+(2*a.shape[-2], 2*a.shape[-1]), dtype=np.complex128)
    result[..., ::2, ::2] = result[..., 1::2, 1::2] = a
    return result


def decompose(reference, offsite, *, reference_has_soc=True):
    """Native offsite export only; never difference two independently SCF H's.

    A full offsite lj operator can contain a scalar term. That trace stays in
    base; only its spin-traceless part is identified as SOC. The residual spin
    vector is the exchange field for supported XC-only collinear references.
    """
    h = hermitian(reference, "reference H")
    offsite = hermitian(offsite, "native offsite operator")
    if h.shape != offsite.shape or h.shape[-1] % 2:
        raise ValueError("Matching even-dimensional spinor matrices required")
    soc = offsite-spin_scalar(spin_trace(offsite))
    non_soc = h-(soc if reference_has_soc else 0)
    base = spin_scalar(spin_trace(non_soc))
    magnetic = non_soc-base
    return base, magnetic, soc


def rotation_between(reference, target):
    a, b = unit_directions([reference, target])
    dot = float(np.clip(a@b, -1., 1.))
    if dot < -1+1.e-14:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        return np.r_[0., axis/np.linalg.norm(axis)]
    q = np.r_[1+dot, np.cross(a, b)]
    return q/np.linalg.norm(q)


def rotated_direction(reference, quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError("Finite nonzero quaternion required")
    q = q/np.max(abs(q)); q = q/np.linalg.norm(q)
    a = unit_directions([reference])[0]
    return a+2*np.cross(q[1:], np.cross(q[1:], a)+q[0]*a)
