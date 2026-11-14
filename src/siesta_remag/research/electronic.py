"""Nonorthogonal-basis diagnostics from the existing pinned HSX v2 export.

These are post-SCF diagnostics of the exported Hamiltonian, not a new SCF solver
or a unique partition of observables. Native EIG comparison qualifies each use.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import struct

import numpy as np
from scipy.linalg import eigh

from ..corpus import sha256
from ..hsx import read_hsx
from ..output import read_eigenvalues
from ..discovery.common import save

RY_EV = 13.605693122994


def orbital_metadata(hsx):
    n_atoms, n_orbitals, _, n_species = map(int, hsx['header'][:4])
    endian = hsx['endian']
    species = []
    for i in range(n_species):
        header = hsx['basis'][0][32*i:32*(i+1)]
        label = header[:20].decode('ascii').strip()
        shells = np.frombuffer(hsx['basis'][i+1], dtype=endian+'i4').reshape(-1, 3)
        species.append((label, shells))
    isa = hsx['labels'][:n_atoms]
    lasto = hsx['labels'][n_atoms:]
    rows = []
    for atom, (kind, end) in enumerate(zip(isa, lasto)):
        label, shells = species[kind-1]
        if len(shells) != end-len(rows):
            raise ValueError('HSX atom orbital ranges differ from basis metadata')
        for local, (n, l, zeta) in enumerate(shells):
            rows.append(dict(index=len(rows), atom_index=atom, species=label,
                             atom_orbital_index=local, n=int(n), l=int(l), zeta=int(zeta)))
    if len(rows) != n_orbitals:
        raise ValueError('HSX orbital count differs')
    return rows


def bloch_matrices(hsx, k_fractional, *, collinear_spin=0):
    """H(k), S(k) in the lattice gauge, interleaved up/down for spinors.

    HSX sparse row i / column j entries populate matrix (j,i), following
    pinned upstream diag3kp.F90 setup_k. Their phase is exp(-2π i k·R).
    Transposing spatial indices while preserving spin indices is essential:
    a scalar-only test cannot detect the incorrect spin-orbit convention.
    Atomic-position phases are removed by a lattice-gauge transform.
    """
    _, n, spins = map(int, hsx['header'][:3])
    if n > 2048:
        raise ValueError('Dense diagnostic is bounded to 2048 spatial orbitals')
    k = np.asarray(k_fractional, dtype=float)
    if k.shape != (3,) or not np.isfinite(k).all():
        raise ValueError('One finite reciprocal fractional k point required')
    row = np.repeat(np.arange(n), hsx['counts'])
    col = (hsx['columns']-1) % n
    image = (hsx['columns']-1) // n
    translations = hsx['offsets'].reshape(-1, 3)
    phase = np.exp(-2j*np.pi*(translations[image] @ k))
    def dense(values):
        result = np.zeros((n, n), complex)
        np.add.at(result, (col, row), values*phase)
        return result
    overlap = dense(hsx['overlap'])
    h = hsx['h']
    if spins <= 2:
        if type(collinear_spin) is not int or not 0 <= collinear_spin < spins:
            raise ValueError('Invalid collinear spin channel')
        matrix = dense(h[collinear_spin])
    else:
        matrix = np.zeros((2*n, 2*n), complex)
        matrix[0::2, 0::2] = dense(h[0] + (1j*h[4] if spins == 8 else 0))
        matrix[1::2, 1::2] = dense(h[1] + (1j*h[5] if spins == 8 else 0))
        matrix[0::2, 1::2] = dense(h[2]-1j*h[3])
        matrix[1::2, 0::2] = dense(h[6]+1j*h[7] if spins == 8 else h[2]+1j*h[3])
        overlap = np.kron(overlap, np.eye(2))
    for name, a in [('H', matrix), ('S', overlap)]:
        error = np.max(abs(a-a.conj().T))
        # Native sparse integration leaves small row-pair asymmetry even in
        # double exports. Bound it before explicitly taking the Hermitian part.
        tolerance = (2e-6 if not hsx['precision'] else 1e-7)*max(1., np.max(abs(a)))
        if error > tolerance:
            raise ValueError(f'{name}(k) is not Hermitian: {error}')
    return (matrix+matrix.conj().T)*(.5*RY_EV), (overlap+overlap.conj().T)*.5


def mulliken_weights(coefficients, overlap):
    """Re[c*_mu (S c)_mu], normalized in the generalized eigenproblem.

    Negative components are allowed and never clipped. This basis-dependent
    population is not a probability density or a uniquely measurable partition.
    """
    c, s = np.asarray(coefficients), np.asarray(overlap)
    if c.ndim != 2 or s.shape != (c.shape[0],)*2 or not np.isfinite(c).all() or not np.isfinite(s).all():
        raise ValueError('Finite compatible coefficient and overlap matrices required')
    gram = c.conj().T @ s @ c
    if not np.allclose(gram, np.eye(c.shape[1]), atol=2e-7, rtol=0):
        raise ValueError('States are not S-orthonormal')
    return np.real(c.conj() * (s @ c))


def spectrum(hsx_path, kpoints, weights, output, *, collinear_spin=0, native_eigenvalues_ev=None):
    """Store numerical arrays as NPZ plus small JSON metadata; never inline vectors."""
    hsx = read_hsx(hsx_path)
    ks, ws = np.asarray(kpoints, float), np.asarray(weights, float)
    if ks.ndim != 2 or ks.shape[1:] != (3,) or not 0 < len(ks) <= 4096 or not np.isfinite(ks).all():
        raise ValueError('Finite (N,3) reciprocal fractional k points required')
    if ws.shape != (len(ks),) or not np.isfinite(ws).all() or np.any(ws < 0) or not np.isclose(ws.sum(), 1, atol=1e-10):
        raise ValueError('Nonnegative normalized k weights required')
    energies, projections, spin_expectations = [], [], []
    spins = int(hsx['header'][2])
    for k in ks:
        h, s = bloch_matrices(hsx, k, collinear_spin=collinear_spin)
        values, vectors = eigh(h, s)
        pop = mulliken_weights(vectors, s)
        energies.append(values)
        if spins > 2:
            projections.append(pop[0::2]+pop[1::2])
            up, down = vectors[0::2], vectors[1::2]
            spatial_s = s[0::2, 0::2]
            cross = np.sum(up.conj()*(spatial_s @ down), axis=0)
            spin_expectations.append(np.array([2*cross.real, 2*cross.imag, np.sum(pop[0::2]-pop[1::2], axis=0)]).T)
        else:
            projections.append(pop)
    energies = np.array(energies)
    residual = None
    if native_eigenvalues_ev is not None:
        native = np.asarray(native_eigenvalues_ev, float)
        if native.shape != energies.shape or not np.isfinite(native).all():
            raise ValueError('Native EIG comparison dimensions differ')
        residual = float(np.max(abs(native-energies)))
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    temperature = hsx['temperature_ry']*RY_EV
    fermi = hsx['fermi_ry']*RY_EV
    occupations = (1/(1+np.exp(np.clip((energies-fermi)/temperature, -700, 700))) if temperature > 0
                   else (energies < fermi).astype(float)) * (2 if spins == 1 else 1)
    np.savez_compressed(root/'spectrum.npz', eigenvalues_ev=energies, k_fractional=ks, k_weights=ws,
        occupations=occupations, orbital_mulliken=np.array(projections), spin_pauli=np.array(spin_expectations))
    metadata = dict(format='REMAG-ELECTRONIC/1', hsx_sha256=sha256(hsx_path),
        artifact='spectrum.npz', artifact_sha256=sha256(root/'spectrum.npz'),
        fermi_ev=fermi, temperature_ev=temperature, energy_units='eV', spin_components=spins,
        collinear_spin=collinear_spin if spins <= 2 else None, orbitals=orbital_metadata(hsx),
        projection='Mulliken Re(c* S c); basis-dependent; negative contributions retained',
        occupations_model='Fermi-Dirac at HSX chemical potential and temperature; not native occupation export',
        spin_frame='Cartesian solver frame; interleaved up/down', lattice_gauge='H[j,i] += H_sparse[i,j,R] exp(-2pi i k.R)',
        native_max_abs_error_ev=residual, native_spectrum_compared=residual is not None,
        native_comparison_tolerance_ev=1e-3,
        native_comparison_passed=residual is not None and residual <= 1e-3,
        hermitian_policy='reject relative skew >1e-7 (double) or 2e-6 (single); then (A+A†)/2',
        physical_accuracy_qualified=False)
    save(root/'electronic.json', metadata)
    return metadata


def finite_difference(points, *, parameter_unit, property_unit):
    """Unequal-step three-point local derivative with conservative supplied error bounds."""
    if len(points) != 3:
        raise ValueError('Three distinct ordered samples required')
    q = np.array([p['q'] for p in points], float)
    y = np.array([p['value'] for p in points], float)
    if not np.isfinite(q).all() or not np.isfinite(y).all() or np.any(np.diff(q) <= 0):
        raise ValueError('Finite, distinct, increasing parameter samples required')
    x = q-q[1]
    inverse = np.linalg.inv(np.array([np.ones(3), x, x*x]).T)
    fit = inverse @ (y-y[1])
    errors = [p.get('error_bound') for p in points]
    bound = None
    if all(e is not None for e in errors):
        e = np.array(errors, float)
        if not np.isfinite(e).all() or np.any(e < 0): raise ValueError('Invalid error bounds')
        bound = dict(first=float(abs(inverse[1])@e), second=float(2*abs(inverse[2])@e))
    return dict(at=float(q[1]), first=float(fit[1]), second=float(2*fit[2]),
        first_unit=f'{property_unit}/{parameter_unit}', second_unit=f'{property_unit}/{parameter_unit}^2',
        propagated_error_bound=bound, truncation_error=None, approximation='local quadratic three-point finite difference')


def spectrum_from_run(directory, output, *, collinear_spin=0):
    """Use native .KP (Cartesian bohr^-1) sampling and compare native .EIG.

    Both files are text and limit the comparison precision. Weight normalization
    corrects only bounded decimal-output rounding, never a missing k point.
    """
    root = Path(directory)
    hsx = read_hsx(root/'remag.HSX')
    lines = (root/'remag.KP').read_text().splitlines()
    count = int(lines[0])
    kp = np.array([[float(x) for x in line.split()] for line in lines[1:]])
    if kp.shape != (count, 5) or not np.isfinite(kp).all() or not np.array_equal(kp[:, 0], np.arange(1, count+1)):
        raise ValueError('Invalid native KP sampling')
    eigen = read_eigenvalues(root/'remag.EIG')
    native = np.array(eigen['values_ev'])
    if eigen['nk'] != count or eigen['nspin'] != int(hsx['header'][2]):
        raise ValueError('Native KP/EIG/HSX dimensions differ')
    if eigen['nspin'] <= 2:
        native = native[:, collinear_spin*eigen['nbands']:(collinear_spin+1)*eigen['nbands']]
    cell_bohr = hsx['cell'].reshape(3, 3)
    fractional = kp[:, 1:4] @ cell_bohr.T / (2*np.pi)
    weights, total, rounding_bound = _native_weights([line.split()[4] for line in lines[1:]])
    result = spectrum(root/'remag.HSX', fractional, weights, output,
                      collinear_spin=collinear_spin, native_eigenvalues_ev=native)
    result.update(kp_sha256=sha256(root/'remag.KP'), eig_sha256=sha256(root/'remag.EIG'),
                  native_kp_weight_sum=total, native_kp_weight_rounding_bound=rounding_bound,
                  sampling='native SCF k mesh; KP decimal rounding retained',
                  native_comparison_limit='EIG/KP printed precision; not micro-eV qualification')
    save(Path(output)/'electronic.json', result)
    return result


def _native_weights(tokens):
    """Normalize only a discrepancy covered by the printed decimal precision.

    SIESTA's 512 equal weights print as 0.195312E-02; their 2.56e-6
    sum discrepancy is exactly half a last-place unit per printed weight.
    Counting missing samples or accepting coarsely printed weights is excluded.
    """
    try:
        values = [Decimal(token) for token in tokens]
    except (InvalidOperation, ValueError):
        raise ValueError('Invalid native k-point weight') from None
    if not values or any(not v.is_finite() or v < 0 or abs(v.as_tuple().exponent) > 100 for v in values):
        raise ValueError('Invalid native k-point weight')
    total = sum(values)
    bound = sum(Decimal(10)**v.as_tuple().exponent/2 for v in values)
    if total <= 0 or bound > Decimal('0.0001') or abs(total-1) > bound:
        raise ValueError('Native k-point weights do not sum to one within text rounding')
    return np.array([float(v/total) for v in values]), float(total), float(bound)
