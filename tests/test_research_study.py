import json
from pathlib import Path
import sys
import subprocess

import pytest

from test_research_execution import prepared
from test_research_evaluation import endpoints
from siesta_remag.discovery.common import load,save
from siesta_remag.research.tools import ResearchTools,ELECTRONIC_TOOLS
from siesta_remag.research.memory import Notebook
from siesta_remag.research.study import register_study,run_study_arm,grade_study,evaluate_study,_released


def study_fixture(root,*,response=False):
    for arm in ('scalar','physics'):
        p=root/('arm-'+arm);p.mkdir();save(p/'common.json',dict(source_sha256='c'*64,structural_dofs='same fixture for both arms'))
    policy=root/'policy.py'
    policy.write_text('''import sys,json
c=json.load(sys.stdin);h=c['history'];n=len(h)
assert 'structural_operation' in c['tools'] and c['withheld_sources']==['c'*64]
assert ('inspect_electronic_structure' in c['tools'])==(c['information_profile']=='physics')
if n==0:a={'tool':'record_hypothesis','args':{'statement':'Synthetic heldout sign','predictions':['positive anisotropy'],'alternatives':['negative anisotropy']}}
elif n==1:a={'tool':'propose_experiment','args':{'hypothesis_id':h[0]['response']['result'],'structure_sha256':'c'*64,'action':'heldout prediction','rationale':'compare before exposing result'}}
elif n==2:
 a={'tool':'record_prediction','args':{'experiment_id':h[1]['response']['result'],'observable':'anisotropy.total_energy_ev','unit':'eV/cell','value':.0001 if c['information_profile']=='physics' else 0}}
else:a={'finish':'Synthetic protocol demonstration; no scientific advantage inferred.'}
if 'tool' in a:a['rationale']='Next fixture step'
print(json.dumps(a))
''')
    definition=dict(format='REMAG-PAIRED-STUDY/1',objective='Synthetic protocol comparison',policy_command=[sys.executable,str(policy)],
        policy_artifacts=['policy.py'],common_inputs=['common.json'],limits=dict(max_steps=8,max_seconds=30,allow_cpu_relaxation=False),
        arms={arm:dict(workspace='arm-'+arm,execution_config=None) for arm in ('scalar','physics')},
        evaluation=dict(heldout_sources=['c'*64],observables={'anisotropy.total_energy_ev':dict(unit='eV/cell',resolution=1e-5)}))
    if response:
        target=dict(kind='anisotropy_response',baseline_source_sha256='a'*64,metric='total_energy_ev')
        old="a={'tool':'record_prediction','args':{'experiment_id':h[1]['response']['result'],'observable':'anisotropy.total_energy_ev','unit':'eV/cell','value':.0001 if c['information_profile']=='physics' else 0}}"
        new="a={'tool':'record_response_prediction','args':{'experiment_id':h[1]['response']['result'],'observable':'delta-E','target':"+repr(target)+",'value':0}}"
        policy.write_text(policy.read_text().replace(old,new))
        definition['evaluation']['observables']={'delta-E':dict(unit='eV/cell',resolution=1e-5,target=target)}
    return definition


def test_scalar_profile_exposes_same_authoring_but_blocks_electronic_dispatch(tmp_path):
    scalar=ResearchTools(tmp_path,information_profile='scalar');physics=ResearchTools(tmp_path)
    for key in ('structural_operation','prepare_query','record_hypothesis','propose_experiment'):
        assert scalar.catalog()[key]==physics.catalog()[key]
    assert not ELECTRONIC_TOOLS & set(scalar.catalog())
    for tool in ELECTRONIC_TOOLS:
        with pytest.raises(ValueError,match='scalar profile'):scalar.call(tool,{})
    with pytest.raises(ValueError,match='cannot request spectra'):
        scalar.call('run_prepared_query',dict(bundle='none',output='none',spectra=[]))
    with pytest.raises(ValueError,match='separate evaluator'):
        scalar.call('record_response_outcome',dict(before=dict(spectrum='secret')))
    observation=scalar._observations({'scalar':dict(total_energy_ev=-2,total_moment_muB=[0,0,1],eigenvalues=[1],orbital_density=[99])})
    assert observation=={'scalar':dict(total_energy_ev=-2,total_moment_muB=[0,0,1])}


def test_withheld_source_blocks_known_reports_and_launch_before_execution_config(endpoints,tmp_path):
    t=ResearchTools(tmp_path,withheld_sources=['c'*64])
    with pytest.raises(ValueError,match='held-out evaluator'):t.call('inspect_calculation',dict(report='after/report.json'))
    with pytest.raises(ValueError,match='held-out evaluator'):t.call('run_prepared_query',dict(bundle='other',output='attempt'))
    assert not (tmp_path/'attempt').exists()


def test_paired_sessions_separate_history_grade_shared_evidence_and_compare(endpoints,tmp_path):
    definition=study_fixture(tmp_path);registration=register_study(tmp_path,definition,'study')
    assert registration['native_budget_per_arm'] is None
    assert not (tmp_path/'arm-scalar/session').exists()
    scalar=run_study_arm(tmp_path,'study','scalar');assert scalar['status']=='completed',scalar
    assert not (tmp_path/'study/physics.json').exists() and not (tmp_path/'arm-physics/session').exists()
    with pytest.raises(ValueError,match='already attempted'):run_study_arm(tmp_path,'study','scalar')
    outcomes=dict(format='REMAG-STUDY-OUTCOMES/1',cases={'c'*64:{'anisotropy.total_energy_ev':dict(report='after/report.json',stage='anisotropy',observable='total_energy_ev')}})
    with pytest.raises(ValueError,match='Both arms'):grade_study(tmp_path,'study',outcomes)
    physics=run_study_arm(tmp_path,'study','physics');assert physics['status']=='completed',physics
    pending=evaluate_study(tmp_path,'study','pending.json')
    assert pending['heldout']['scalar']['scores']['metrics']['anisotropy.total_energy_ev']['pending_count']==1
    assert not pending['empirical_equal_compute_verified']
    grade=grade_study(tmp_path,'study',outcomes);assert grade['status']=='completed' and len(grade['recorded_outcome_ids'])==2
    scores=evaluate_study(tmp_path,'study','scores.json')
    pair=scores['paired_prediction_errors']['anisotropy.total_energy_ev']
    assert pair['paired_count']==1 and pair['mae']['physics']<1e-10
    assert pair['physics_minus_scalar_mae']==pytest.approx(-.0001)
    assert all(a['native_execution_attempts']==0 and a['reserved_cents']==0 and a['actual_gpu_hours'] is None for a in scores['arms'].values())
    assert not scores['blinding_verified'] and not scores['scientific_advantage_demonstrated']
    assert scores['independent_candidate_validation'] is None
    with pytest.raises(ValueError,match='already attempted'):grade_study(tmp_path,'study',outcomes)
    with pytest.raises(ValueError,match='must be new'):evaluate_study(tmp_path,'study','scores.json')
    assert load(tmp_path/'arm-scalar/session/request.json')['tools']['structural_operation']==load(tmp_path/'arm-physics/session/request.json')['tools']['structural_operation']
    proc=subprocess.run([sys.executable,'-m','siesta_remag.research','study-evaluate','--workspace',str(tmp_path),
        '--study','study','--output','cli-scores.json'],capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)['paired_prediction_errors']==scores['paired_prediction_errors']


@pytest.mark.parametrize('mutation,match',[('common','Common initial inputs differ'),('overlap','disjoint'),('session','previous sessions'),
    ('policy','policy artifact'),('initial','initial knowledge'),('heldout','heldout calculation')])
def test_registration_and_execution_refuse_changed_or_contaminated_inputs(tmp_path,mutation,match):
    d=study_fixture(tmp_path)
    if mutation=='common':save(tmp_path/'arm-scalar/common.json',{'different':True})
    elif mutation=='overlap':d['arms']['physics']['workspace']='arm-scalar'
    elif mutation=='session':(tmp_path/'arm-scalar/session').mkdir()
    elif mutation=='heldout':save(tmp_path/'arm-scalar/leak.json',dict(format='REMAG-EXPERIMENT-RESULT/1',source_sha256='c'*64))
    if mutation in ('policy','initial'):
        register_study(tmp_path,d,'study')
        path=tmp_path/('policy.py' if mutation=='policy' else 'arm-scalar/common.json');path.write_text('changed')
        with pytest.raises(ValueError,match=match):run_study_arm(tmp_path,'study','scalar')
    else:
        with pytest.raises(ValueError,match=match):register_study(tmp_path,d,'study')


def test_late_prediction_and_unregistered_outcome_cannot_enter_comparison(tmp_path):
    d=study_fixture(tmp_path);register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):assert run_study_arm(tmp_path,'study',arm)['status']=='completed'
    n=Notebook(tmp_path/'arm-scalar/research-memory.sqlite');e=next(x for x in n.retrieve() if x['kind']=='experiment')
    n.record_outcome(e['id'],observable='anisotropy.total_energy_ev',value=0,unit='eV/cell',report_sha256='b'*64,status='measured');n.close()
    with pytest.raises(ValueError,match='outside the registered grading'):evaluate_study(tmp_path,'study','invalid.json')


def test_new_predictions_and_tool_changes_after_session_are_rejected(tmp_path,monkeypatch):
    import siesta_remag.research.study as study
    d=study_fixture(tmp_path);register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):run_study_arm(tmp_path,'study',arm)
    n=Notebook(tmp_path/'arm-scalar/research-memory.sqlite');e=next(x for x in n.retrieve() if x['kind']=='experiment')
    n.record_prediction(e['id'],observable='late',value=0,unit='eV/cell');n.close()
    with pytest.raises(ValueError,match='added after'):evaluate_study(tmp_path,'study','invalid.json')
    monkeypatch.setattr(study,'_implementation',lambda:{'different':True})
    with pytest.raises(ValueError,match='tool implementation changed'):evaluate_study(tmp_path,'study','invalid.json')


def test_evaluation_cannot_occupy_future_grading_receipt(tmp_path):
    d=study_fixture(tmp_path);register_study(tmp_path,d,'study')
    with pytest.raises(ValueError,match='outside study ownership'):evaluate_study(tmp_path,'study','study/grading.json')
    assert not (tmp_path/'study/grading.json').exists()


def test_native_budgets_must_match_and_registration_does_not_reserve(tmp_path):
    d=study_fixture(tmp_path);d['limits']['max_seconds']=900
    common=tmp_path/'synthetic-binary';common.write_text('not executed')
    from siesta_remag.corpus import sha256
    for arm in ('scalar','physics'):
        where=tmp_path/('arm-'+arm);prepared.__wrapped__(where)
        runtime=load(where/'runtime.json');runtime['executable']=str(common);runtime['executable_sha256']=sha256(common);save(where/'runtime.json',runtime)
        c=load(where/'execution-config.json');c['runtime_sha256']=sha256(where/'runtime.json');save(where/'execution-config.json',c)
        d['arms'][arm]['execution_config']='execution-config.json'
    r=register_study(tmp_path,d,'study');assert r['native_budget_per_arm']['max_queries']==2
    assert all(load(tmp_path/('arm-'+a)/'budget.json')['reservations']==[] for a in ('scalar','physics'))
    c=load(tmp_path/'arm-physics/execution-config.json');c['max_queries']=1;save(tmp_path/'arm-physics/execution-config.json',c)
    with pytest.raises(ValueError,match='must match'):register_study(tmp_path,d,'different')


def test_missing_or_unstopped_native_owner_never_releases_following_arm(tmp_path,monkeypatch):
    d=study_fixture(tmp_path);d['arms']['scalar']['execution_config']='config.json'
    where=tmp_path/'arm-scalar';save(where/'session/report.json',dict(native_execution_attempts=1))
    save(where/'.research-execution/owner.json',dict(status='measured',allocation_stopped=True,app_id='ap-fixture'))
    calls=[]
    def info(self,app):calls.append(app);return dict(app_id=app,lifecycle=dict(state='running'))
    monkeypatch.setattr('siesta_remag.research.study.ModalExecution.info',info)
    with pytest.raises(ValueError,match='authoritatively STOPPED'):_released(tmp_path,d,'scalar')
    assert calls==['ap-fixture']
    (where/'session/report.json').unlink()
    with pytest.raises(ValueError,match='terminal session evidence'):_released(tmp_path,d,'scalar')


def test_session_artifact_mutation_is_not_accepted_as_a_fair_comparison(tmp_path):
    d=study_fixture(tmp_path);register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):run_study_arm(tmp_path,'study',arm)
    p=tmp_path/'arm-scalar/session/000/response.json';p.write_text('{}')
    with pytest.raises(ValueError,match='response changed'):evaluate_study(tmp_path,'study','invalid.json')


def test_shared_frozen_response_definition_is_graded_from_saved_endpoints(endpoints,tmp_path):
    definition=study_fixture(tmp_path,response=True);register_study(tmp_path,definition,'study')
    for arm in ('scalar','physics'):assert run_study_arm(tmp_path,'study',arm)['status']=='completed'
    outcomes=dict(format='REMAG-STUDY-OUTCOMES/1',cases={'c'*64:{'delta-E':dict(before=dict(report='before/report.json'),after=dict(report='after/report.json'))}})
    result=grade_study(tmp_path,'study',outcomes);assert result['status']=='completed'
    scores=evaluate_study(tmp_path,'study','scores.json');assert scores['paired_prediction_errors']['delta-E']['mae']=={'scalar':0,'physics':0}


def test_native_enabled_arms_reuse_guarded_controller_and_account_for_attempts(tmp_path,monkeypatch):
    from siesta_remag.research import agent
    from siesta_remag.research.execution import execute_query
    from test_research_execution import SyntheticTransport
    from siesta_remag.corpus import sha256
    d=study_fixture(tmp_path);d['limits']['max_seconds']=900
    common=tmp_path/'synthetic-binary';common.write_text('not executed');transports={}
    for arm in ('scalar','physics'):
        where=tmp_path/('arm-'+arm);prepared.__wrapped__(where)
        runtime=load(where/'runtime.json');runtime.update(executable=str(common),executable_sha256=sha256(common));save(where/'runtime.json',runtime)
        c=load(where/'execution-config.json');c['runtime_sha256']=sha256(where/'runtime.json');save(where/'execution-config.json',c)
        d['arms'][arm]['execution_config']='execution-config.json';transports[where]=SyntheticTransport()
    policy=tmp_path/'policy.py';s=policy.read_text();s=s.replace("n=len(h)","n=len(h)-1")
    s=s.replace("if n==0:a=", "if n==-1:a={'tool':'run_prepared_query','args':{'bundle':'bundle','output':c['information_profile']+'attempt'}}\nelif n==0:a=")
    s=s.replace("h[0]['response']['result']","h[1]['response']['result']").replace("h[1]['response']['result'],'observable'","h[2]['response']['result'],'observable'")
    policy.write_text(s)
    original=agent._process
    def process(command,payload,deadline,folder,label,**kw):
        if label=='tool' and payload['op']=='run_prepared_query':
            where=Path(command[command.index('--workspace')+1])
            return dict(ok=True,result=execute_query(where,'execution-config.json',**payload['args'],transport=transports[where]))
        return original(command,payload,deadline,folder,label,**kw)
    monkeypatch.setattr(agent,'_process',process)
    observations=[]
    def info(self,app):observations.append(app);return dict(app_id=app,lifecycle=dict(state='stopped'))
    monkeypatch.setattr('siesta_remag.research.study.ModalExecution.info',info)
    register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):
        result=run_study_arm(tmp_path,'study',arm);assert result['status']=='completed',result
    assert observations==['ap-scalarattempt']
    result=evaluate_study(tmp_path,'study','scores.json')
    for arm in ('scalar','physics'):
        a=result['arms'][arm];assert a['native_execution_attempts']==1 and a['completed_experiments']==1 and a['reserved_cents']==10
        assert a['actual_gpu_hours'] is None and a['actual_billed_usd'] is None
        assert len(transports[tmp_path/('arm-'+arm)].calls)==1


def test_failed_heldout_without_directional_energies_is_retained(endpoints,tmp_path):
    d=study_fixture(tmp_path);register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):run_study_arm(tmp_path,'study',arm)
    p=endpoints[1]['report'];r=load(p);r['status']='held_experiment';r['anisotropy']=None
    r['observations']={'scalar':r['observations']['scalar']};r['stages']=r['stages'][:1];save(p,r)
    outcomes=dict(format='REMAG-STUDY-OUTCOMES/1',cases={'c'*64:{'anisotropy.total_energy_ev':dict(report='after/report.json',stage='anisotropy',observable='total_energy_ev')}})
    grade_study(tmp_path,'study',outcomes);scores=evaluate_study(tmp_path,'study','scores.json')
    for arm in ('scalar','physics'):
        m=scores['heldout'][arm]['scores']['metrics']['anisotropy.total_energy_ev']
        assert m['failed_count']==1 and m['measured_count']==0 and m['mae'] is None
    assert scores['paired_prediction_errors']['anisotropy.total_energy_ev']['paired_count']==0
