"""Direction attribution must survive partial runs without borrowing later data."""
import copy

import pytest

from siesta_remag.orientation import parse_sweep, write_direction_views


AXES = [[0, 0, 1], [1, 0, 0], [0, 0, 1]]
ROTATIONS = [[1, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]]


def log():
    text = 'redata: Number of spin components = 8\nredata: Spin configuration = spin-orbit\n'
    for n, (axis, q) in enumerate(zip(AXES, ROTATIONS), 1):
        text += f'REMAG ORIENTATION BEGIN {n} remag.o{n:03d} '+ ' '.join(map(str, q))+'\n'
        text += f'scf: 1\nSCF cycle converged\nsiesta: Total = {-12+n*.001}\nsiesta: FreeEng = {-12.1+n*.001}\nsiesta: Fermi = -2\n'
        text += 'spin moment: {S} , |S| = {'+' '.join(str(3*v) for v in axis)+'}\n'
        text += 'REMAG: energy-only finalization; forces and stress not calculated\n'
        text += f'REMAG ORIENTATION END {n}\n'
    return text+'Job completed\n'


def parse(text, **kwargs):
    return parse_sweep(text, AXES, ROTATIONS, returncode=kwargs.get('returncode', 0), timed_out=kwargs.get('timed_out', False))


def test_independent_direction_observables_and_request_binding():
    report = parse(log())
    assert report['passed']
    assert [r['observables']['total_moment_muB'] for r in report['directions']] == [[0, 0, 3], [3, 0, 0], [0, 0, 3]]
    assert [r['observables']['scf_iterations'] for r in report['directions']] == [1, 1, 1]
    assert [r['label'] for r in report['directions']] == ['remag.o001', 'remag.o002', 'remag.o003']
    assert not report['physical_accuracy_qualified']
    assert parse(log().replace('remag.o002 1 0 1 0', 'remag.o002 -2 0 -2 0'))['passed']
    with pytest.raises(ValueError, match='target axes'):
        parse_sweep(log(), [[0, 0, 1], [1, 1, 0], [0, 0, 1]], ROTATIONS, returncode=0, timed_out=False)


@pytest.mark.parametrize('defect', ['wrong_label', 'wrong_rotation', 'zero_rotation', 'bad_marker',
                                  'duplicate_start', 'missing_end', 'reordered', 'wrong_end',
                                  'missing_convergence', 'missing_energy', 'zero_moment', 'wrong_moment'])
def test_later_pass_does_not_hide_earlier_failure(defect):
    text = log()
    if defect == 'wrong_label':text = text.replace('BEGIN 2 remag.o002', 'BEGIN 2 remag.o003')
    elif defect == 'wrong_rotation':text = text.replace('remag.o002 1 0 1 0', 'remag.o002 1 0 0 0')
    elif defect == 'zero_rotation':text = text.replace('remag.o002 1 0 1 0', 'remag.o002 0 0 0 0')
    elif defect == 'bad_marker':text = text.replace('END 2', 'END 2 junk')
    elif defect == 'duplicate_start':text = text.replace('REMAG ORIENTATION END 1', 'REMAG ORIENTATION BEGIN 1 remag.o001 1 0 0 0\nREMAG ORIENTATION END 1')
    elif defect == 'missing_end':text = text.replace('REMAG ORIENTATION END 1\n', '')
    elif defect == 'reordered':text = text.replace('BEGIN 2', 'BEGIN 7').replace('END 2', 'END 7')
    elif defect == 'wrong_end':text = text.replace('END 1', 'END 9')
    elif defect == 'missing_convergence':text = text.replace('SCF cycle converged\n', '', 1)
    elif defect == 'missing_energy':text = text.replace('siesta: Total = -11.999\n', '', 1)
    elif defect == 'zero_moment':text = text.replace('{0 0 3}', '{0 0 0}', 1)
    elif defect == 'wrong_moment':text = text.replace('{0 0 3}', '{0 0 -3}', 1)
    report = parse(text)
    assert not report['passed']
    assert report['directions'][-1]['valid']  # Retain the real last-point success separately.


@pytest.mark.parametrize('returncode,timed_out', [(1, False), (0, True), (None, False)])
def test_success_markers_cannot_override_process_failure(returncode, timed_out):
    report = parse(log(), returncode=returncode, timed_out=timed_out)
    assert not report['passed'] and all(not r['valid'] for r in report['directions'])


def test_partial_last_direction_is_retained_without_inventing_missing_outputs():
    text = log().split('REMAG ORIENTATION BEGIN 3')[0]+'REMAG ORIENTATION BEGIN 3 remag.o003 1 0 0 0\nscf: 1\n'
    report = parse(text, returncode=-1, timed_out=True)
    assert not report['passed']
    last = report['directions'][-1]
    assert last['observables']['total_energy_ev'] is None and last['observables']['total_moment_muB'] is None
    assert not last['observables']['converged'] and 'scf: 1' in last['stdout']
    assert report['directions'][0]['observables']['total_energy_ev'] == -11.999


def test_views_keep_original_files_separate_and_reject_missing_exports(tmp_path):
    report = parse(log())
    # Small native-format EIG/XV records with distinguishable per-axis values.
    for i in (1, 2, 3):
        (tmp_path/f'remag.o{i:03d}.EIG').write_text(f'-2\n1 8 1\n1 {i}\n')
        (tmp_path/f'remag.o{i:03d}.XV').write_text('1 0 0\n0 1 0\n0 0 1\n1\n1 26 0 0 0 0 0 0\n')
        (tmp_path/f'remag.o{i:03d}.DM').write_bytes(bytes([i]))
    (tmp_path/'remag.o002.XV').unlink()
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    paths = write_direction_views(tmp_path, report)
    assert not report['passed'] and not report['directions'][1]['valid']
    assert [r['observables']['eigenvalues']['values_ev'] for r in report['directions']] == [[[1]], [[2]], [[3]]]
    assert [p.joinpath('remag.DM').read_bytes() for p in paths] == [b'\x01', b'\x02', b'\x03']
    assert all((tmp_path/name).read_bytes() == contents for name, contents in original.items())
    with pytest.raises(FileExistsError):write_direction_views(tmp_path, copy.deepcopy(report))
