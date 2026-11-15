import json
import sys

import pytest

from test_research import experiment
from test_discovery import backend
from siesta_remag.corpus import sha256
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.agent import run_session
from siesta_remag.research.memory import Notebook


def test_policy_drives_prediction_outcome_revision_loop(experiment,tmp_path):
    exe=tmp_path/'binary';exe.write_text('fixture')
    report=run_experiment(experiment,tmp_path/'result',exe,executable_sha256=sha256(exe),runner=backend([]))
    policy=tmp_path/'policy.py'
    policy.write_text('''import json,sys
c=json.load(sys.stdin);h=c['history'];n=len(h)
if n==0: a={'tool':'record_hypothesis','args':{'statement':'Positive anisotropy','predictions':['E[x]>E[z]'],'alternatives':['Negative anisotropy']}}
elif n==1: a={'tool':'propose_experiment','args':{'hypothesis_id':h[0]['response']['result'],'structure_sha256':SOURCE,'action':'inspect directional response','rationale':'compare alternatives'}}
elif n==2: a={'tool':'record_prediction','args':{'experiment_id':h[1]['response']['result'],'observable':'anisotropy.total_energy_ev','value':.0001,'unit':'eV/cell'}}
elif n==3: a={'tool':'record_calculation_outcome','args':{'experiment_id':h[1]['response']['result'],'report':'result/report.json','stage':'anisotropy','observable':'total_energy_ev'}}
elif n==4: a={'tool':'evaluate_hypothesis','args':{'experiment_id':h[1]['response']['result']}}
elif n==5:
 assert h[4]['response']['result']['absolute_error']<1e-10
 a={'tool':'record_analysis','args':{'parent_id':h[1]['response']['result'],'conclusion':'Synthetic fixture agrees; no mechanism verified','evidence':'result/report.json'}}
else:a={'finish':'Engineering loop completed; scientific conclusion remains unverified.'}
if 'tool' in a:a['rationale']='Next hypothesis-driven step'
print(json.dumps(a))
'''.replace('SOURCE',repr(report['source_sha256'])))
    result=run_session(tmp_path,'session',objective='Test the research loop on a synthetic fixture',
        policy_command=[sys.executable,str(policy)],max_steps=8,max_seconds=60)
    assert result['status']=='completed',result
    assert len(result['steps'])==7 and result['native_submissions']==0
    assert not result['scientific_success_assessed']
    notebook=Notebook(tmp_path/'research-memory.sqlite')
    events=notebook.retrieve();notebook.close()
    assert {e['kind'] for e in events}=={'hypothesis','experiment','prediction','outcome','analysis'}
    assert (tmp_path/'session/003/action.json').exists()
    with pytest.raises(FileExistsError):run_session(tmp_path,'session',objective='No retry',policy_command=[sys.executable,str(policy)])


@pytest.mark.parametrize('decision,status',[
    ({'tool':'inspect_calculation','args':{'report':'missing.json'},'rationale':'test failure'},'held_tool_error'),
    ({'tool':'run_native','args':{},'rationale':'unavailable'},'held_error'),
    ({'tool':'prepare_relaxation','args':{},'rationale':'CPU not enabled'},'held_error'),
    ({'tool':'retrieve_research_history','args':{},'rationale':'bounded lookup'},'held_step_limit')])
def test_failure_or_limit_does_not_queue_another_action(tmp_path,decision,status):
    cmd=[sys.executable,'-c','import json;print(json.dumps('+repr(decision)+'))']
    result=run_session(tmp_path,'session',objective='Bounded check',policy_command=cmd,max_steps=1,max_seconds=30)
    assert result['status']==status and len(result['steps'])==1
    assert not (tmp_path/'session/001').exists()


def test_policy_timeout_retains_evidence(tmp_path):
    result=run_session(tmp_path,'session',objective='Check timeout',
        policy_command=[sys.executable,'-c','import time;time.sleep(30)'],max_seconds=1)
    assert result['status']=='held_error'
    assert result['error']['type']=='TimeoutExpired'
    assert (tmp_path/'session/000/policy.stdout').exists()
    assert result['wall_seconds']<5


def test_new_tools_are_not_silently_added_to_session_permissions(tmp_path,monkeypatch):
    from siesta_remag.research import agent
    monkeypatch.setitem(agent.TOOLS,'future_native_tool',{'required':[],'optional':[]})
    decision={'tool':'future_native_tool','args':{},'rationale':'must remain unavailable'}
    result=run_session(tmp_path,'session',objective='Permission boundary',
        policy_command=[sys.executable,'-c','import json;print(json.dumps('+repr(decision)+'))'])
    assert result['status']=='held_error'
    assert not (tmp_path/'session/000/action.json').exists()
