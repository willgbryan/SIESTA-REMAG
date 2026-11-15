import json
import subprocess
import sys

import pytest

from test_research import experiment
from test_discovery import backend
from test_research_inspection import spectrum_fixture
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.tools import ResearchTools


def test_saved_result_hypothesis_loop_persists_and_rejects_late_predictions(experiment,tmp_path):
    executable=tmp_path/'binary'; executable.write_text('fixture')
    report=run_experiment(experiment,tmp_path/'result',executable,executable_sha256=sha256(executable),runner=backend([]))
    t=ResearchTools(tmp_path)
    h=t.call('record_hypothesis',dict(statement='Positive anisotropy', predictions=['E[x]>E[z]'],alternatives=['Negative anisotropy']))
    e=t.call('propose_experiment',dict(hypothesis_id=h,structure_sha256=report['source_sha256'],action='anisotropy',rationale='test direction'))
    t.call('record_prediction',dict(experiment_id=e,observable='anisotropy.total_energy_ev',value=.0001,unit='eV/cell'))
    t.call('record_calculation_outcome',dict(experiment_id=e,report='result/report.json',stage='anisotropy',observable='total_energy_ev'))
    # A fresh process/tool object sees the same immutable history.
    t=ResearchTools(tmp_path)
    r=t.call('evaluate_hypothesis',dict(experiment_id=e))
    assert r['absolute_error']<1e-12 and not r['mechanism_verified']
    with pytest.raises(ValueError):t.call('record_prediction',dict(experiment_id=e,observable='anisotropy.total_energy_ev',value=0,unit='eV/cell'))
    other=t.call('propose_experiment',dict(hypothesis_id=h,structure_sha256='b'*64,action='other',rationale='different source'))
    with pytest.raises(ValueError,match='source differs'):
        t.call('record_calculation_outcome',dict(experiment_id=other,report='result/report.json',stage='scalar',observable='total_energy_ev'))
    inspected=t.call('inspect_calculation',dict(report='result/report.json'))
    assert inspected['status']=='measured'
    (tmp_path/'result/scalar/stdout.log').write_text('tampered')
    with pytest.raises(ValueError,match='binding differs'):t.call('inspect_calculation',dict(report='result/report.json'))


def test_structural_inspection_and_preparation_do_not_run_solver(experiment,tmp_path):
    t=ResearchTools(tmp_path)
    q=t.call('prepare_query',dict(structure='bundle/structure.json',pseudos='bundle/pseudos',
        output='new-query',query=dict(kind='scalar'),mesh=[3,3,3]))
    assert q['query']['kind']=='scalar'
    assert not (tmp_path/'new-query/report.json').exists()
    p=load(experiment/'structure.json')['source']['program']
    assert t.call('structural_operation',dict(request=dict(op='physical-dofs',program=p)))['ok']


def test_worker_protocol_and_bounded_electronic_selection(tmp_path):
    spectrum_fixture(tmp_path/'a')
    requests=[dict(op='catalog'),dict(op='inspect_electronic_structure',args=dict(spectrum='a/electronic.json',selection=dict(limit=1))),
              dict(op='inspect_calculation',args=dict(report='../outside.json'))]
    result=subprocess.run([sys.executable,'-m','siesta_remag.research','tools','--workspace',str(tmp_path)],
        input='\n'.join(json.dumps(r) for r in requests)+'\n',text=True,capture_output=True,check=True)
    replies=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(replies)==3 and replies[0]['ok'] and replies[1]['ok']
    assert len(replies[1]['result']['states'])==1
    assert not replies[2]['ok'] and 'escapes' in replies[2]['error']['message']
    assert not (tmp_path/'research-memory.sqlite').exists()


def test_compact_saved_collection_keeps_omissions_explicit(experiment,tmp_path):
    from siesta_remag.discovery.collection import pack,unpack
    binary=tmp_path/'binary';binary.write_text('fixture')
    run_experiment(experiment,tmp_path/'result',binary,executable_sha256=sha256(binary),runner=backend([]))
    unpack(pack(tmp_path/'result',include_arrays=True),tmp_path/'collected')
    t=ResearchTools(tmp_path)
    with pytest.raises(FileNotFoundError):t.call('inspect_calculation',dict(report='collected/report.json'))
    value=t.call('inspect_calculation',dict(report='collected/report.json',collection='collected/archive-manifest.json'))
    assert value['status']=='measured'
    assert value['verification']['mode']=='compact collection'
    assert any(name.endswith('.DM') for name in value['verification']['omitted_artifacts'])
    (tmp_path/'collected/scalar/stdout.log').write_text('changed')
    with pytest.raises(ValueError,match='Collected artifact binding differs'):
        t.call('inspect_calculation',dict(report='collected/report.json',collection='collected/archive-manifest.json'))
