#!/usr/bin/env python3
"""Independent selected-k oracle for experimental frozen-generator captures.

Uses NumPy SU(2) matrix multiplication and SciPy/LAPACK, never the native
rotation helper. This is an implementation diagnostic, not a full-SCF oracle
or a physical-accuracy qualification. All energies in captures/CSVs are Ry.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

RY_EV = 13.605693122994
MATRIX_ATOL_RY = 1.e-10
SPECTRUM_ATOL_EV = 1.e-7


def lower_hermitian(a):
    """The matrix actually consumed by Zhegvd(LOWER), including real diagonal."""
    lower = np.tril(a, -1)
    return lower + lower.conj().T + np.diag(a.diagonal().real)


def spin_unitary(quaternion, orbitals):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
        raise ValueError('Invalid quaternion')
    q = q / np.max(abs(q))
    w, x, y, z = q / np.linalg.norm(q)
    u = np.array([[w-1j*z, -y-1j*x], [y-1j*x, w+1j*z]])
    return np.kron(np.eye(orbitals), u)


def local_file(parent, name):
    if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Capture references must be local file names')
    return parent / name


def check_capture(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    if (meta.get('schema') != 1 or meta.get('dtype') != '<c16' or meta.get('order') != 'F'
            or meta.get('energy_unit') != 'Ry'
            or meta.get('reference') != 'Final SCF generator Hamiltonian'
            or meta.get('matrices') != ['generator', 'overlap', 'soc', 'rotated']):
        raise ValueError('Unsupported matrix capture format or reference')
    n, k, nk = (meta.get(key) for key in ('bands', 'k', 'kpoints'))
    if (any(type(v) is not int for v in (n, k, nk)) or n < 2 or n % 2
            or not 0 <= k < nk or 4*n*n*16 > 64*1024*1024):
        raise ValueError('Invalid matrix capture dimensions')
    binary = local_file(path.parent, meta['matrix_file'])
    if binary.stat().st_size != 4*n*n*16:
        raise ValueError('Truncated or oversized matrix capture')
    matrices = np.fromfile(binary, dtype='<c16').reshape(4, n*n)
    if not np.isfinite(matrices).all():
        raise ValueError('Nonfinite matrix capture')
    h, s, soc, actual = [a.reshape(n, n, order='F') for a in matrices]
    energies, generator_energies = {}, {}
    with local_file(path.parent, meta['raw_spectrum']).open() as stream:
        for row in csv.DictReader(stream):
            if int(row['k']) == k:
                band = int(row['band'])
                if band in energies:
                    raise ValueError('Duplicate sampled band')
                energies[band] = float(row['energy_ry'])
                generator_energies[band] = float(row['generator_scf_energy_ry'])
    if set(energies) != set(range(n)):
        raise ValueError('Incomplete sampled spectrum')
    observed = np.array([energies[i] for i in range(n)])
    generator_observed = np.array([generator_energies[i] for i in range(n)])
    if not np.isfinite([observed, generator_observed]).all():
        raise ValueError('Nonfinite sampled spectrum')
    u = spin_unitary(meta['quaternion'], n//2)
    # Rotate the full complex matrices first, independently of sparse Pauli
    # packing. Also quantify the noncommuting lower-triangle interpretation.
    independent = u @ (h-soc) @ u.conj().T + soc
    sh = lower_hermitian(s)
    canonical = u @ lower_hermitian(h-soc) @ u.conj().T + lower_hermitian(soc)
    raw_e = eigvalsh(lower_hermitian(independent), sh, driver='gvd')
    actual_e = eigvalsh(lower_hermitian(actual), sh, driver='gvd')
    canonical_e = eigvalsh(canonical, sh, driver='gvd')
    generator_e = eigvalsh(lower_hermitian(h), sh, driver='gvd')
    # Removing SOC must leave a globally spin-rotation-invariant generalized
    # spectrum. This control alone cannot qualify the force theorem.
    exchange = lower_hermitian(h-soc)
    without_soc = eigvalsh(exchange, sh, driver='gvd')
    rotated_without_soc = eigvalsh(u @ exchange @ u.conj().T, sh, driver='gvd')
    maximum = lambda a: float(np.max(abs(a)))
    checks = dict(
        matrix_rotation=maximum(actual-independent) <= MATRIX_ATOL_RY,
        generator_spectrum=maximum(generator_e-generator_observed)*RY_EV <= SPECTRUM_ATOL_EV,
        rotated_spectrum=maximum(raw_e-observed)*RY_EV <= SPECTRUM_ATOL_EV,
        exported_matrix_spectrum=maximum(actual_e-observed)*RY_EV <= SPECTRUM_ATOL_EV,
        canonical_rotation_spectrum=maximum(canonical_e-observed)*RY_EV <= SPECTRUM_ATOL_EV,
        spin_independent_overlap=maximum(sh-u@sh@u.conj().T) <= MATRIX_ATOL_RY,
        no_soc_rotation_invariance=maximum(without_soc-rotated_without_soc)*RY_EV <= SPECTRUM_ATOL_EV,
    )
    return dict(capture=str(path), k=k, bands=n, direction=meta['direction'], checks=checks,
                passed=all(checks.values()), physical_accuracy_qualified=False,
                matrix_rotation_max_abs_error_ry=maximum(actual-independent),
                generator_spectrum_max_abs_error_ev=maximum(generator_e-generator_observed)*RY_EV,
                rotated_spectrum_max_abs_error_ev=maximum(raw_e-observed)*RY_EV,
                exported_matrix_spectrum_max_abs_error_ev=maximum(actual_e-observed)*RY_EV,
                canonical_rotation_spectrum_max_abs_error_ev=maximum(canonical_e-observed)*RY_EV,
                no_soc_rotation_max_abs_error_ev=maximum(without_soc-rotated_without_soc)*RY_EV,
                hermiticity_max_abs_error_ry={key: maximum(a-a.conj().T)
                                            for key, a in [('generator', h), ('soc', soc), ('rotated', actual)]},
                overlap_hermiticity_max_abs_error=maximum(s-s.conj().T),
                overlap_condition_number=float(np.linalg.cond(sh)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Calculation or run containing completed captures')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.rglob('remag-frozen-matrices-*.json'))
    if not paths:
        parser.error('No frozen matrix captures found')
    reports, coverage = [], []
    for parent in sorted({p.parent for p in paths}):
        try:
            probe = json.loads((parent/'remag-frozen-potential.json').read_text())
            nk = probe['kpoints']
            points = probe['points']
            expected = {(p['index'], k) for p in points for k in {0, nk//2, nk-1}}
            found = []
            for path in [p for p in paths if p.parent == parent]:
                meta = json.loads(path.read_text())
                if meta['quaternion'] != points[meta['direction']]['quaternion']:
                    raise ValueError('Capture rotation differs from the probe')
                found.append((meta['direction'], meta['k']))
            complete = (probe.get('status') == 'complete' and probe.get('matrix_samples') is True
                        and bool(points) and all(p['completed'] and p['status'] == 0 for p in points)
                        and len(found) == len(expected) and set(found) == expected)
            coverage.append(dict(calculation=str(parent), passed=complete))
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            coverage.append(dict(calculation=str(parent), passed=False, error=str(error)))
    for path in paths:
        try:
            reports.append(check_capture(path))
        except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as error:
            reports.append(dict(capture=str(path), passed=False, error=str(error)))
    report = dict(scope='Selected-k frozen-generator implementation check, not full-SCF or physical qualification',
                  matrix_atol_ry=MATRIX_ATOL_RY, spectrum_atol_ev=SPECTRUM_ATOL_EV,
                  passed=all(r['passed'] for r in reports+coverage), coverage=coverage, captures=reports)
    content = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
