"""Synthetic native Ni/Co observations through the real local tool/notebook loop."""
from pathlib import Path

import pytest

from test_research_relaxation_execution import discovery, DiscoveryTransport, run, refresh_bundle
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.discovery_observation import import_relaxation_result
from siesta_remag.research.tools import ResearchTools, _verified_report, calculation_observation
from siesta_remag.research.toolhost import ToolHost


def test_prospective_prediction_native_import_score_and_revision(discovery, tmp_path, monkeypatch):
    tools = ResearchTools(tmp_path)
    source = load(tmp_path/'structure.json')['source']['sha256']
    hypothesis = tools.call('record_hypothesis', dict(statement='Synthetic relaxed magnetization clears the screen',
        predictions=['Js_T > 0.4 T'], alternatives=['Low moment']))
    experiment = tools.call('propose_experiment', dict(hypothesis_id=hypothesis, structure_sha256=source,
        action='relax and screen', rationale='Measure the relaxed geometry and magnetization'))
    tools.call('record_prediction', dict(experiment_id=experiment, observable='relaxation.Js_T',
        value=1., unit='T'))
    transport = DiscoveryTransport(monkeypatch, relaxed_cell_scale=.98)
    before = (tmp_path/'structure.json').read_bytes()
    result = run(tmp_path, transport)
    assert result['status'] == 'measured'
    raw_hash = sha256(tmp_path/'attempt/collected/archive-manifest.json')
    imported = tools.call('import_relaxation_result', dict(attempt='attempt'))
    report = Path(imported['report'])
    observed = tools.call('inspect_calculation', dict(report=str(report)))
    assert observed['status'] == 'measured' and observed['native_status'] == 'screened'
    assert observed['request']['fixed_geometry'] is False
    assert observed['evaluated_geometry']['atomic_numbers'] == load(tmp_path/'structure.json')['geometry']['atomic_numbers']
    assert observed['evaluated_geometry']['cell'][0][0] == pytest.approx(2.94)
    assert load(report)['structure']['geometry']['cell'][0][0] == 3
    assert observed['stages'][0]['stage'] == 'scalar' and observed['stages'][0]['native_stage'] == 'dft-relax'
    assert observed['relaxation_metrics']['Js_T'] > .4
    assert observed['physical_stability_qualified'] is False
    outcome = tools.call('record_calculation_outcome', dict(experiment_id=experiment, report=str(report),
        stage='relaxation', observable='Js_T'))
    assert isinstance(outcome, str)
    score = tools.call('evaluate_hypothesis', dict(experiment_id=experiment))
    assert score['status'] == 'evaluated', score
    revision = tools.call('record_analysis', dict(parent_id=experiment, evidence=str(report),
        conclusion='The measured relaxed moment passed; assess stability before candidate promotion.', status='evaluated'))
    history = tools.call('retrieve_research_history', {})
    assert any(row['id'] == revision and row['payload']['report_sha256'] == sha256(report) for row in history)
    energy = calculation_observation(report, source, 'anisotropy', 'total_energy_ev')
    assert energy['status'] == 'measured' and energy['value'] == pytest.approx(.0001)
    assert tools.call('import_relaxation_result', dict(attempt='attempt')) == imported
    assert sha256(tmp_path/'attempt/collected/archive-manifest.json') == raw_hash
    assert (tmp_path/'structure.json').read_bytes() == before
    assert 'WriteKpoints true' in (report.parent/'scalar/input.fdf').read_text()
    ion = next((report.parent/'scalar').glob('*.ion.xml'))
    assert sha256(ion) == load(report)['artifacts']['scalar/'+ion.name]
    assert len(transport.calls) == 1 and len(transport.native) == 4
    assert 'import_relaxation_result' in ToolHost(tmp_path).catalog()['tools']
    from siesta_remag.research.workspace import dataset, comparison_key
    save(tmp_path/'workspace.json', dict(format='REMAG-RESEARCH-WORKSPACE/1', title='Synthetic discovery', structures=[
        dict(id='candidate', label='Non-FePt fixture', structure='structure.json',
             calculations=[dict(report=str(report.relative_to(tmp_path)))])]))
    row = dataset(tmp_path)['families'][0]['rows'][0]
    assert row['structure']['geometry']['cell'][0][0] == 3
    assert row['result']['evaluated_geometry']['cell'][0][0] == pytest.approx(2.94)
    saved = load(report)
    assert comparison_key(saved) != comparison_key(saved | dict(format='REMAG-EXPERIMENT-RESULT/1'))


def test_low_moment_is_measured_but_missing_soc_is_inconclusive(discovery, tmp_path, monkeypatch):
    request = load(discovery/'request.json'); request['settings']['minimum_js_t'] = 100
    save(discovery/'request.json', request); refresh_bundle(discovery)
    transport = DiscoveryTransport(monkeypatch); assert run(tmp_path, transport)['status'] == 'measured'
    report = Path(import_relaxation_result(tmp_path/'attempt')['report'])
    source = load(report)['source_sha256']
    assert calculation_observation(report, source, 'scalar', 'free_energy_ev')['status'] == 'measured'
    assert calculation_observation(report, source, 'relaxation', 'Js_T')['status'] == 'measured'
    outcome = calculation_observation(report, source, 'anisotropy', 'total_energy_ev')
    assert outcome['status'] == 'inconclusive' and outcome['value'] is None
    assert len(transport.native) == 1 and load(report)['anisotropy'] is None


@pytest.mark.parametrize('failure', ['native', 'optimizer', 'branch'])
def test_native_failures_remain_failed_observations(discovery, tmp_path, monkeypatch, failure):
    transport = DiscoveryTransport(monkeypatch, failure=failure)
    assert run(tmp_path, transport)['status'] == 'held_native_review'
    report = Path(import_relaxation_result(tmp_path/'attempt')['report'])
    outcome = calculation_observation(report, load(report)['source_sha256'], 'anisotropy', 'total_energy_ev')
    assert outcome['status'] == 'failed' and outcome['value'] is None
    assert len(transport.calls) == 1


@pytest.mark.parametrize('target', ['source', 'geometry', 'copy', 'raw', 'runtime', 'shutdown', 'summary'])
def test_changed_evidence_is_rejected(discovery, tmp_path, monkeypatch, target):
    run(tmp_path, DiscoveryTransport(monkeypatch))
    report = Path(import_relaxation_result(tmp_path/'attempt')['report'])
    if target in ('source', 'geometry'):
        content = load(report)
        if target == 'source': content['source_sha256'] = 'f'*64
        else: content['evaluated_geometry']['cell'][0][0] += .1
        save(report, content)
    elif target in ('copy', 'raw'):
        path = report.parent/'scalar/stdout.log' if target == 'copy' else tmp_path/'attempt/collected/query/dft-relax/stdout.log'
        path.write_text(path.read_text()+'edited')
    elif target == 'runtime':
        path = tmp_path/'attempt/runtime.json'; content = load(path); content['threads'] = 8; save(path, content)
    else:
        path = tmp_path/'attempt/execution.json'; content = load(path)
        if target == 'shutdown': content['allocation_stopped'] = False
        else: content['review']['observations']['dft-relax']['total_energy_ev'] += 1
        save(path, content)
    with pytest.raises(ValueError): _verified_report(report)


def test_replay_uses_captured_producer_identity_not_unrelated_current_code(discovery, tmp_path, monkeypatch):
    run(tmp_path, DiscoveryTransport(monkeypatch))
    from siesta_remag.research import relaxation_execution, experiments
    monkeypatch.setattr(relaxation_execution, 'source_identity', lambda: {'new.py': 'f'*64})
    monkeypatch.setattr(experiments, 'experiment_code', lambda: {'new.py': 'f'*64})
    report = Path(import_relaxation_result(tmp_path/'attempt')['report'])
    assert _verified_report(report)['status'] == 'measured'
    with pytest.raises(ValueError, match='producer code'):
        relaxation_execution.review_relaxation(tmp_path/'attempt', load(discovery/'request.json'), load(tmp_path/'runtime.json'))


def test_withheld_sources_rejected_before_import_and_profiles_preserved(discovery, tmp_path, monkeypatch):
    run(tmp_path, DiscoveryTransport(monkeypatch))
    source = load(tmp_path/'structure.json')['source']['sha256']
    blocked = ResearchTools(tmp_path, withheld_sources=[source])
    with pytest.raises(ValueError, match='held-out'): blocked.call('import_relaxation_result', dict(attempt='attempt'))
    assert not (tmp_path/'attempt/research-observation').exists()
    scalar = ResearchTools(tmp_path, information_profile='scalar')
    imported = scalar.call('import_relaxation_result', dict(attempt='attempt'))
    observed = scalar.call('inspect_calculation', dict(report=imported['report']))
    assert 'eigenvalues' not in observed['observations']['scalar']
    with pytest.raises(ValueError, match='source differs'):
        calculation_observation(imported['report'], 'f'*64, 'scalar', 'total_energy_ev')
    with pytest.raises(ValueError, match='Electronic diagnostics'):
        scalar.call('export_electronic_matrices', dict(report=imported['report'], stage='soc-z',
            native_directory='attempt/research-observation/soc-z', output='blocked'))
    assert not (tmp_path/'blocked').exists()


def test_native_spectrum_tool_exports_inspectable_saved_evidence(tmp_path):
    from test_research_mechanisms import native_fixture
    report, native = native_fixture(tmp_path/'native')
    tools = ResearchTools(tmp_path)
    arguments = dict(report=str(report), stage='soc-z', native_directory=str(native), output='spectrum')
    result = tools.call('export_electronic_spectrum', arguments)
    assert result['native_max_abs_error_ev'] < 1e-14
    assert result['report_sha256'] == sha256(report)
    assert result['source_sha256'] == 'a'*64 and result['native_stage'] == 'soc-z'
    inspected = tools.call('inspect_electronic_structure', dict(spectrum='spectrum/electronic.json'))
    assert inspected
    with pytest.raises(FileExistsError): tools.call('export_electronic_spectrum', arguments)
    with pytest.raises(ValueError, match='accepted native stage'):
        tools.call('export_electronic_spectrum', arguments | dict(stage='soc-x', output='bad-stage'))
    assert not (tmp_path/'bad-stage').exists()
    (native/'remag.KP').write_text('changed sampling')
    with pytest.raises(ValueError, match='artifact binding'):
        tools.call('export_electronic_spectrum', arguments | dict(output='bad-files'))
    assert not (tmp_path/'bad-files').exists()
    scalar = ToolHost(tmp_path, information_profile='scalar')
    assert 'export_electronic_spectrum' not in scalar.catalog()['tools']
    assert 'export_electronic_spectrum' in ToolHost(tmp_path).catalog()['tools']
