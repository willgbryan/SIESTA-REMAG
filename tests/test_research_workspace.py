from copy import deepcopy

import numpy as np
import pytest

from siesta_remag.research.workspace import authored_family, comparison_key, dataset, electronic_preview
from siesta_remag.discovery.common import save
from test_research_convergence import report


def test_spectrum_must_belong_to_the_displayed_hamiltonian(tmp_path):
    save(tmp_path/'electronic-z/electronic.json', dict(native_comparison_passed=True,
        native_max_abs_error_ev=.0001, hsx_sha256='a'*64))
    with pytest.raises(ValueError, match='Hamiltonian'):
        electronic_preview(tmp_path, expected_hsx_sha256='b'*64)


def test_comparison_requires_physical_runtime_but_not_deadlines():
    base = report(16, .003, .0028)
    other = deepcopy(base)
    other['request']['settings']['total_dft_seconds'] *= 2
    other['request']['settings']['max_kpoints'] *= 2
    other['source_sha256'] = 'b' * 64
    assert comparison_key(base) == comparison_key(other)
    for key, value in [('mesh_ry', 500), ('smearing_k', 100), ('mixing_weight', .1)]:
        other = deepcopy(base)
        other['request']['settings'][key] = value
        assert comparison_key(base) != comparison_key(other)
    other = deepcopy(base)
    other['environment'] = {'REMAG_DEVICE_SCF': '1'}
    assert comparison_key(base) != comparison_key(other)


def test_independent_families_keep_exact_sources_and_geometry(tmp_path):
    pytest.importorskip('gpac.physical')
    families = {name: authored_family(name) for name in ['axial', 'volume', 'shape']}
    ref = next(r for r in families['axial'] if r['strain'] == 0)
    cell = np.array(ref['structure']['geometry']['cell'])
    for name in ['volume', 'shape']:
        assert families[name][1]['structure']['source']['sha256'] == ref['structure']['source']['sha256']
        for row in families[name]:
            changed = np.array(row['structure']['geometry']['cell'])
            if name == 'shape':
                assert np.linalg.det(changed) == pytest.approx(np.linalg.det(cell), abs=1e-10)
            else:
                assert changed[2, 2]/changed[0, 0] == pytest.approx(cell[2, 2]/cell[0, 0])
    assert {r['exact_strain'] for r in families['axial']} >= {'-1/400', '1/400'}
    chosen = families['shape'][0]
    value = report(16, .003, .0028)
    value['source_sha256'] = value['request']['source_sha256'] = chosen['structure']['source']['sha256']
    save(tmp_path/'shape'/'report.json', value)
    state = dataset(tmp_path)
    measured = [(f['id'], r['exact_strain']) for f in state['families'] for r in f['rows'] if r['result']]
    assert measured == [('shape', '-1/400')]
    assert state['rows'] == state['families'][0]['rows']
