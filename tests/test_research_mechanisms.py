"""Analytic metric/gauge/subspace controls and a small native-format handoff."""
import struct

import numpy as np
import pytest
from scipy.linalg import eigh

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.electronic import RY_EV
from siesta_remag.research.mechanisms import decompose, coupling_summary, state_correspondence, export_matrices, compare_matrices
from siesta_remag.research.tools import ResearchTools


def test_nonorthogonal_density_trace_and_lowdin_energy_gauge():
    s = np.array([[1., .3], [.3, 1.]])
    w, v = eigh(s); root = (v*np.sqrt(w)) @ v.T
    h_orth = np.array([[-1., .2j], [-.2j, 1.]])
    h = root @ h_orth @ root
    d = decompose(h, s, 0., .025, degeneracy=2)
    assert np.allclose(d['lowdin_hamiltonian_ev'], h_orth)
    assert np.allclose(d['eigenvalues_ev'], [-np.sqrt(1.04), np.sqrt(1.04)])
    assert np.trace(d['density'] @ s).real == pytest.approx(2)
    assert np.trace(d['density'] @ h).real == pytest.approx(d['occupations'] @ d['eigenvalues_ev'])
    assert not np.isclose(np.trace(d['density']), 2)
    assert d['generalized_residual_ev'] < 1e-14
    shifted = decompose(h+3*s, s, 3., .025, degeneracy=2)
    assert np.allclose(d['density'], shifted['density'])
    groups = [dict(indices=[0]), dict(indices=[1])]
    for result in (d, shifted):
        coupling = coupling_summary(result['lowdin_hamiltonian_ev'], groups)['strongest_couplings'][0]
        assert (coupling['left'], coupling['right']) == (0, 1)
        assert coupling['frobenius_ev'] == pytest.approx(.2, rel=0, abs=1e-12)
        assert coupling['largest_singular_value_ev'] == pytest.approx(.2, rel=0, abs=1e-12)


def test_degenerate_projectors_survive_arbitrary_complex_rotation():
    c = np.eye(3, dtype=complex)
    rotation = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
    d = c.copy(); d[:2, :2] = rotation
    states, clusters, weights = state_correspondence(c, d, np.eye(3), np.array([0, 0, 1.]), np.array([0, 0, 1.]))
    assert np.allclose(weights[:2, :2], .5)
    assert [x['assignment_ambiguous'] for x in states] == [True, True, False]
    assert clusters[0]['matches'][0]['projector_trace'] == pytest.approx(2)
    assert clusters[0]['matches'][0]['principal_overlap_singular_values'] == pytest.approx([1, 1])
    d *= np.exp(1j*np.array([.8, .1, .3]))
    _, phase_clusters, _ = state_correspondence(c, d, np.eye(3), np.array([0, 0, 1.]), np.array([0, 0, 1.]))
    assert phase_clusters[0]['matches'][0]['projector_trace'] == pytest.approx(2)


def test_assignment_follows_states_through_energy_order_swap():
    states, _, _ = state_correspondence(np.eye(2), np.eye(2)[:, ::-1], np.eye(2), np.array([-1., 1]), np.array([-.5, .5]))
    assert [(x['before_band'], x['after_band']) for x in states] == [(0, 1), (1, 0)]
    assert [x['energy_shift_ev'] for x in states] == [1.5, -1.5]


def test_nondegenerate_overlap_ties_are_not_unique_state_matches():
    rotated = np.array([[1, 1], [-1, 1]])/np.sqrt(2)
    states, _, _ = state_correspondence(np.eye(2), rotated, np.eye(2), np.array([-1., 1]), np.array([-1., 1]))
    assert all(x['assignment_ambiguous'] and not x['energy_degeneracy_present'] for x in states)


@pytest.mark.parametrize('s', [np.diag([1., 0]), np.diag([1., -1])])
def test_no_overlap_regularization(s):
    with pytest.raises(ValueError, match='positive definite'):
        decompose(np.eye(2), s, 0, .025)


def native_fixture(root, shift=0):
    root.mkdir()
    stage = root/'soc-z'; stage.mkdir()
    s = np.array([[1., .2], [.2, 1.]])
    overlap = np.kron(s, np.eye(2))
    h = np.array([[-1., .1j, .2, .03j], [-.1j, -.8, .02j, .1],
                  [.2, -.02j, 1., .05j], [-.03j, .1, -.05j, 1.2]])+shift*overlap
    # HSX stores the spatial transpose separately from the spin indices.
    blocks = [h[0::2, 0::2].real, h[1::2, 1::2].real, h[0::2, 1::2].real, -h[0::2, 1::2].imag,
              h[0::2, 0::2].imag, h[1::2, 1::2].imag, h[1::2, 0::2].real, h[1::2, 0::2].imag]
    with (stage/'remag.HSX').open('wb') as f:
        def record(x):
            f.write(struct.pack('<i', len(x))); f.write(x); f.write(struct.pack('<i', len(x)))
        def array(values, kind): record(np.array(values, dtype='<'+kind).tobytes())
        array([2], 'i4'); array([1], 'i4'); array([2, 2, 8, 2, 1, 1, 1], 'i4')
        array([*np.eye(3).ravel(), 0, 2, .025/RY_EV], 'f8')
        record(np.zeros(3, dtype='<i4').tobytes()+np.array([0, 0, 0, .5, .5, .5], dtype='<f8').tobytes()
               +np.array([1, 2, 1, 2], dtype='<i4').tobytes())
        record(b'Fe'.ljust(20)+struct.pack('<di', 8., 1)+b'Pt'.ljust(20)+struct.pack('<di', 10., 1))
        array([1, 0, 1], 'i4'); array([1, 0, 1], 'i4'); record(bytes(60))
        array([2, 2], 'i4'); array([1, 2], 'i4'); array([1, 2], 'i4')
        for block in blocks:
            for row in block.T/RY_EV: array(row, 'f8')
        for row in s.T: array(row, 'f8')
    for symbol in ('Fe', 'Pt'):
        (stage/(symbol+'.ion.xml')).write_text(f'<ion><symbol>{symbol}</symbol><paos><orbital n="1" l="0" z="1">'
            '<radfunc><npts>2</npts><delta>0.5</delta><cutoff>0.5</cutoff><data>0 1 0.5 0</data></radfunc></orbital></paos></ion>')
    (stage/'remag.KP').write_text('2\n1 0 0 0 0.500000000000\n2 .5 0 0 0.500000000000\n')
    values = eigh(h, overlap, eigvals_only=True)
    (stage/'remag.EIG').write_text('0\n4 8 2\n'+'\n'.join(str(i)+' '+' '.join(map(str, values)) for i in (1, 2))+'\n')
    report = dict(format='REMAG-EXPERIMENT-RESULT/1', source_sha256='a'*64, request=dict(source_sha256='a'*64),
        executable_sha256='b'*64, stages=[dict(stage='soc-z', accepted=True, branch=dict(passed=True))],
        artifacts={'soc-z/'+p.name: sha256(p) for p in stage.iterdir()})
    save(root/'report.json', report)
    return root/'report.json', stage


def test_native_export_and_agent_comparison_preserve_partial_sampling(tmp_path):
    a, native_a = native_fixture(tmp_path/'a'); b, native_b = native_fixture(tmp_path/'b', .01)
    tool = ResearchTools(tmp_path)
    left = tool.call('export_electronic_matrices', dict(report='a/report.json', stage='soc-z', native_directory='a/soc-z', output='left', k_indices=[1]))
    right = export_matrices(b, 'soc-z', native_b, tmp_path/'right', k_indices=[1])
    assert left['native_max_abs_error_ev'] < 1e-14
    assert left['selected_k_weight'] == .5 and left['native_k_count'] == 2
    result = tool.call('compare_electronic_matrices', dict(before='left/matrices.json', after='right/matrices.json', output='comparison'))
    assert all(x['overlap_squared'] == pytest.approx(1) for x in result['kpoints'][0]['states'])
    assert all(x['delta_h_expectation_ev'] == pytest.approx(.01) for x in result['kpoints'][0]['states'])
    assert result['weighted_selected_band_sum_change_ev'] == pytest.approx(.5*result['kpoints'][0]['band_sum_change_ev'])
    assert result['common_basis_max_abs_differences']['radial_functions'] == 0
    assert not result['mechanism_verified']
    with pytest.raises(FileExistsError):
        export_matrices(a, 'soc-z', native_a, tmp_path/'left')


def test_changed_radial_basis_refuses_state_overlap_even_if_labels_match(tmp_path):
    a, native_a = native_fixture(tmp_path/'a'); b, native_b = native_fixture(tmp_path/'b')
    path = native_b/'Pt.ion.xml'; path.write_text(path.read_text().replace('0.5 0</data>', '0.5 0.1</data>'))
    report = load(b); report['artifacts']['soc-z/Pt.ion.xml'] = sha256(path); save(b, report)
    export_matrices(a, 'soc-z', native_a, tmp_path/'left')
    export_matrices(b, 'soc-z', native_b, tmp_path/'right')
    with pytest.raises(ValueError, match='radial_functions'):
        compare_matrices(tmp_path/'left/matrices.json', tmp_path/'right/matrices.json', tmp_path/'compare')
    assert not (tmp_path/'compare').exists()


def test_artifact_tamper_and_failed_branch_rejected_before_output(tmp_path):
    p, native = native_fixture(tmp_path/'a')
    (native/'Fe.ion.xml').write_text('changed')
    with pytest.raises(ValueError, match='binding'):
        export_matrices(p, 'soc-z', native, tmp_path/'out')
    assert not (tmp_path/'out').exists()
    p, native = native_fixture(tmp_path/'b')
    report = load(p); report['stages'][0]['branch']['passed'] = False; save(p, report)
    with pytest.raises(ValueError, match='branch'):
        export_matrices(p, 'soc-z', native, tmp_path/'out')


def test_selected_native_eigenvalues_must_agree(tmp_path):
    p, native = native_fixture(tmp_path/'a')
    eig = native/'remag.EIG'; lines = eig.read_text().splitlines(); tokens = lines[2].split()
    tokens[1] = str(float(tokens[1])+.02); lines[2] = ' '.join(tokens); eig.write_text('\n'.join(lines)+'\n')
    report = load(p); report['artifacts']['soc-z/remag.EIG'] = sha256(eig); save(p, report)
    with pytest.raises(ValueError, match='eigenvalue comparison'):
        export_matrices(p, 'soc-z', native, tmp_path/'out')


def test_workspace_matrix_preview_binds_report_stage_and_atoms(tmp_path):
    from siesta_remag.research.catalog import matrix_view
    p, native = native_fixture(tmp_path/'a')
    r = export_matrices(p, 'soc-z', native, tmp_path/'out', k_indices=[0, 1])
    path = tmp_path/'out/matrices.json'
    view = matrix_view(path, sha256(p), 'soc-z', r['evidence']['hsx'], 2)
    assert view['k_indices'] == [0, 1] and len(view['coupling']) == 2
    assert view['coupling'][0]['strongest_couplings'][0]['frobenius_ev'] > 0
    for report, stage, hsx in [('c'*64, 'soc-z', r['evidence']['hsx']), (sha256(p), 'soc-x', r['evidence']['hsx']), (sha256(p), 'soc-z', 'c'*64)]:
        with pytest.raises(ValueError, match='another displayed'):
            matrix_view(path, report, stage, hsx, 2)
    with pytest.raises(ValueError, match='does not belong'):
        matrix_view(path, sha256(p), 'soc-z', r['evidence']['hsx'], 1)
