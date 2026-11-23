import numpy as np
import pytest

from siesta_remag.research import electronic
from siesta_remag.research.__main__ import main
from siesta_remag.discovery.common import load


def chain(tmp_path, monkeypatch, hopping, *, spins=2):
    # Analytic S(k) = diag(1 + 2t cos(2πk), 2). Its least stable point
    # is k=1/2, not Gamma. The second orbital fixes the maximum eigenvalue.
    raw = dict(header=[1, 2, spins], counts=np.array([3, 1]),
        columns=np.array([1, 3, 5, 2]), offsets=np.array([0, 0, 0, 1, 0, 0, -1, 0, 0]),
        overlap=np.array([1., hopping, hopping, 2.]), h=np.zeros((spins, 4)),
        cell=2*np.pi*np.eye(3), precision=1)
    monkeypatch.setattr(electronic, 'read_hsx', lambda path: raw)
    (tmp_path/'remag.HSX').write_bytes(b'analytic test fixture')
    (tmp_path/'remag.KP').write_text('3\n1 0 0 0 0.333333333333\n'
        '2 .25 0 0 0.333333333333\n3 .5 0 0 0.333333333333\n')
    return raw


@pytest.mark.parametrize('spins', [2, 8])
def test_all_native_points_and_spatial_spinor_conditioning(tmp_path, monkeypatch, spins):
    chain(tmp_path, monkeypatch, .499999, spins=spins)
    r = electronic.overlap_from_run(tmp_path)
    assert r['spatial_orbitals'] == 2 and r['native_kpoints'] == 3
    assert r['positive_definite'] and r['nonpositive_kpoints'] == 0
    assert r['min_eigenvalue'] == pytest.approx(2e-6)
    assert r['max_condition_number'] == pytest.approx(1e6)
    assert r['min_relative_eigenvalue'] == pytest.approx(1e-6)
    assert r['min_eigenvalue_kpoint_index'] == r['max_condition_kpoint_index'] == 3
    assert not r['physical_accuracy_qualified']


@pytest.mark.parametrize('hopping', [.5, .6])
def test_nonpositive_overlap_is_retained_without_clipping(tmp_path, monkeypatch, hopping):
    chain(tmp_path, monkeypatch, hopping)
    path = tmp_path/'overlap.json'
    assert main(['overlap-run', str(tmp_path), '--output', str(path)]) == 2
    r = load(path)
    assert not r['positive_definite'] and r['nonpositive_kpoints'] == 1
    assert r['min_eigenvalue'] == pytest.approx(1-2*hopping)
    assert r['max_condition_number'] is None
    with pytest.raises(FileExistsError): main(['overlap-run', str(tmp_path), '--output', str(path)])


def test_overlap_sampling_does_not_hide_missing_points(tmp_path, monkeypatch):
    chain(tmp_path, monkeypatch, .2)
    (tmp_path/'remag.KP').write_text('2\n1 0 0 0 0.333333333333\n2 .25 0 0 0.333333333333\n')
    with pytest.raises(ValueError, match='weights'): electronic.overlap_from_run(tmp_path)
    (tmp_path/'remag.KP').write_text('2\n1 0 0 0 0.500000\n3 .5 0 0 0.500000\n')
    with pytest.raises(ValueError, match='sampling'): electronic.overlap_from_run(tmp_path)
