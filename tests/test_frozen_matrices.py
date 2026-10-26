"""Exercise the independent matrix oracle, including realistic failure modes."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigvalsh

spec = importlib.util.spec_from_file_location('check_frozen_matrices',
    Path(__file__).resolve().parents[1]/'scripts/check_frozen_matrices.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


@pytest.fixture
def capture(tmp_path):
    rng = np.random.default_rng(704)
    orbitals = 4
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    def hermitian():
        a = rng.normal(size=(orbitals, orbitals))+1j*rng.normal(size=(orbitals, orbitals))
        return a+a.conj().T
    scalar = np.kron(hermitian(), np.eye(2))
    fields = [hermitian() for _ in range(3)]
    exchange = scalar+sum(np.kron(b, sigma) for b, sigma in zip(fields, pauli))
    soc = .2*sum(np.kron(hermitian(), sigma) for sigma in pauli)
    # Active +90 degrees around y: x->-z, z->x. Deliberately construct
    # independently from the oracle's quaternion/Kronecker multiplication.
    rotated = scalar+np.kron(fields[2], pauli[0])+np.kron(fields[1], pauli[1])-np.kron(fields[0], pauli[2])+soc
    a = hermitian()
    overlap = np.kron(a@a+np.eye(orbitals), np.eye(2))
    h = exchange+soc
    meta = dict(schema=1, dtype='<c16', order='F', reference='Final SCF generator Hamiltonian',
                energy_unit='Ry', matrices=['generator', 'overlap', 'soc', 'rotated'],
                bands=8, k=0, kpoints=1, direction=1, quaternion=[1., 0., 1., 0.],
                matrix_file='remag-frozen-matrices-001-k000000.bin', raw_spectrum='remag-frozen-001.csv')
    path = tmp_path/'remag-frozen-matrices-001-k000000.json'
    path.write_text(json.dumps(meta))
    def write(h=h, overlap=overlap, soc=soc, rotated=rotated, energy_shift=0.):
        np.concatenate([a.flatten(order='F') for a in (h, overlap, soc, rotated)]).astype('<c16').tofile(tmp_path/meta['matrix_file'])
        energies = eigvalsh(rotated, overlap, driver='gvd')+energy_shift
        generator = eigvalsh(h, overlap, driver='gvd')
        with (tmp_path/meta['raw_spectrum']).open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['k', 'band', 'energy_ry', 'generator_scf_energy_ry'])
            writer.writerows((0, i, e, g) for i, (e, g) in enumerate(zip(energies, generator)))
    write()
    return path, meta, (h, overlap, soc, rotated), write


def test_exact_rotation_and_generalized_spectrum(capture):
    path, _, _, _ = capture
    report = oracle.check_capture(path)
    assert report['passed'] and all(report['checks'].values())
    assert report['physical_accuracy_qualified'] is False


def test_detects_accidental_rotation_of_lattice_soc(capture):
    path, meta, (h, s, soc, _), write = capture
    u = oracle.spin_unitary(meta['quaternion'], 4)
    write(rotated=u@h@u.conj().T)
    report = oracle.check_capture(path)
    assert not report['passed']
    assert not report['checks']['matrix_rotation']
    assert not report['checks']['rotated_spectrum']
    assert report['checks']['exported_matrix_spectrum']


def test_detects_spectrum_from_different_reference(capture):
    path, _, _, write = capture
    write(energy_shift=1.e-5)
    report = oracle.check_capture(path)
    assert not report['passed']
    assert report['checks']['matrix_rotation']
    assert not report['checks']['exported_matrix_spectrum']


def test_measures_rotation_before_lower_triangle_difference(capture):
    path, meta, (h, s, soc, _), write = capture
    # Independently assembled upper/lower spin entries need not be identical.
    # Algebraic agreement alone is insufficient if canonical spectra differ.
    h[0, 1] += 1.e-4j
    u = oracle.spin_unitary(meta['quaternion'], 4)
    write(rotated=u@(h-soc)@u.conj().T+soc)
    report = oracle.check_capture(path)
    assert report['checks']['matrix_rotation']
    assert report['checks']['rotated_spectrum']
    assert not report['checks']['canonical_rotation_spectrum']
    assert not report['passed']


def test_uses_solver_overlap_triangle(capture):
    path, _, (h, s, soc, rotated), write = capture
    s[0, 2] += 1.e-5j  # Upper triangle is ignored by the generalized solver.
    write()
    report = oracle.check_capture(path)
    assert report['passed']
    assert report['overlap_hermiticity_max_abs_error'] > 1.e-6


def test_cli_keeps_partial_capture_failure(capture):
    path, meta, _, _ = capture
    report_path = path.parent/'check.json'
    command = [sys.executable, str(Path(oracle.__file__)), str(path.parent), '--output', str(report_path)]
    missing = subprocess.run(command, capture_output=True, text=True)
    report = json.loads(report_path.read_text())
    assert missing.returncode == 1 and not report['passed']
    assert report['captures'][0]['passed']  # A good sample does not erase missing coverage.
    probe = dict(status='complete', matrix_samples=True, kpoints=1,
                 points=[dict(index=0, completed=True, status=0, quaternion=[1, 0, 0, 0]),
                         dict(index=1, completed=True, status=0, quaternion=meta['quaternion'])])
    (path.parent/'remag-frozen-potential.json').write_text(json.dumps(probe))
    missing_identity = subprocess.run(command, capture_output=True, text=True)
    assert missing_identity.returncode == 1
    assert not json.loads(report_path.read_text())['coverage'][0]['passed']


@pytest.mark.parametrize('invalid', ['size', 'nan', 'reference', 'path', 'quaternion', 'bands', 'missing_band'])
def test_rejects_bad_capture(capture, invalid):
    path, meta, _, _ = capture
    binary = path.parent/meta['matrix_file']
    if invalid == 'size':
        binary.write_bytes(binary.read_bytes()[:-16])
    elif invalid == 'nan':
        data = np.fromfile(binary, dtype='<c16'); data[0] = np.nan; data.tofile(binary)
    elif invalid == 'reference':
        meta['reference'] = 'terminal HSX'
    elif invalid == 'path':
        meta['matrix_file'] = '../outside.bin'
    elif invalid == 'quaternion':
        meta['quaternion'] = [0., 0., 0., 0.]
    elif invalid == 'bands':
        meta['bands'] = 2**40
    else:
        raw = path.parent/meta['raw_spectrum']
        raw.write_text('\n'.join(raw.read_text().splitlines()[:-1])+'\n')
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        oracle.check_capture(path)


@pytest.mark.parametrize('reference_scale', [0, 1])
def test_soc_free_generator_capture_and_mislabelling(capture, reference_scale):
    path, meta, (h, s, soc, rotated), write = capture
    h = h-(1-reference_scale)*soc
    meta.update(reference_soc_scale=reference_scale, target_soc_scale=1)
    path.write_text(json.dumps(meta))
    write(h=h)
    assert oracle.check_capture(path)['passed']
    meta['reference_soc_scale'] = 1-reference_scale
    path.write_text(json.dumps(meta))
    assert not oracle.check_capture(path)['passed']


@pytest.mark.parametrize('scale', [-1, .5, True, '0', None])
def test_rejects_invalid_reference_soc_scale(capture, scale):
    path, meta, _, _ = capture
    meta['reference_soc_scale'] = scale
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='SOC scales'):
        oracle.check_capture(path)
