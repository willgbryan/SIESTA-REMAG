"""No live model/solver: production request, decoder, ToolHost and GPAC paths."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_research import experiment
from test_research_inspection import spectrum_fixture
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.corpus import sha256
from siesta_remag.research.authoring import author_candidate
from siesta_remag.research.isolated_policy import PolicyContext, preflight
from siesta_remag.research import model_policy


@pytest.fixture
def setup(experiment,tmp_path):
    parent=load(experiment/'structure.json');baseline=parent['source']['sha256']
    held=[]
    for label,amount in [('minus','-1/320'),('plus','1/320')]:
        r=author_candidate(dict(op='physical-strain',axis=2,amount=amount,expected_sha256=baseline),tmp_path/label,structure=experiment/'structure.json')
        held.append(r['source_sha256'])
    packet=dict(format='REMAG-SCALAR-DATA/1',cases={'baseline':dict(source_sha256=baseline,
        report_sha256='a'*64,executable_sha256='b'*64,collection_sha256=None,
        observations={'soc-z':{'total_energy_ev':1.,'free_energy_ev':.9},'soc-x':{'total_energy_ev':1.002,'free_energy_ev':.902}})})
    packet['sha256']=digest(packet)
    for arm in ('scalar','physics'):
        r=tmp_path/arm;r.mkdir();save(r/'packet.json',packet);save(r/'parent.json',parent)
        if arm=='physics':
            (r/'electronic').mkdir()
            spectrum_fixture(r/'electronic/soc-z');spectrum_fixture(r/'electronic/soc-x',shift=.01)
    config=tmp_path/'model.json'
    save(config,dict(format='REMAG-MODEL-POLICY/1',model='gpt-6-astra',reasoning_effort='xhigh',max_calls=12,max_output_tokens=2048,max_reserved_output_tokens=24576,request_timeout_seconds=60,research_brief='Synthetic engineering fixture; no scientific forecast.'))
    return tmp_path,held,config


def response(action):
    return dict(status='completed',model='gpt-6-astra',output=[dict(type='function_call',name='select_research_action',arguments=json.dumps(dict(tool=action['tool'],arguments_json=json.dumps(action['args']),rationale=action['rationale'])))],usage=dict(input_tokens=1,output_tokens=1,total_tokens=2))


def transport(decide, requests):
    def run(body,key,timeout):
        requests.append(deepcopy(body))
        ctx=json.loads(body['input'][0]['content'])['context']
        return json.dumps(response(decide(ctx))).encode(),{'http_status':200}
    return run


def scripted(ctx):
    h=ctx['history'];n=len(h);baseline=ctx['initial_data']['parent']['structure']['source']['sha256']
    if n==0:
        name='inspect_scalar_data' if ctx['information_profile']=='scalar' else 'compare_electronic_states'
        args={'packet':'packet.json'} if name=='inspect_scalar_data' else {'before':'electronic/soc-z/electronic.json','after':'electronic/soc-x/electronic.json','points':11}
    elif n in (1,2):
        name='author_structure';args=dict(structure='parent.json',output='candidates/'+('minus' if n==1 else 'plus'),request=dict(op='physical-strain',axis=2,amount='-1/320' if n==1 else '1/320',expected_sha256=baseline))
    elif n==3:name='record_hypothesis';args=dict(statement='Synthetic arm-local hypothesis',predictions=['Small response'],alternatives=['Large response'])
    elif n in (4,5):
        name='propose_experiment';args=dict(hypothesis_id=h[3]['response']['result'],structure_sha256=h[n-3]['response']['result']['candidate']['source_sha256'],action='Strain',rationale='Synthetic control')
    elif n<10:
        name='record_response_prediction';metric='total_energy_ev' if n%2==0 else 'free_energy_ev'
        args=dict(experiment_id=h[4+(n-6)//2]['response']['result'],observable='E_response' if n%2==0 else 'F_response',target=dict(kind='anisotropy_response',baseline_source_sha256=baseline,metric=metric),value=.00001,uncertainty=.0001)
    else:
        name='record_analysis';args=dict(parent_id=h[n-6]['response']['result'],conclusion='Synthetic scores revise this arm only.',evidence='scores.json')
    return dict(tool=name,args=args,rationale='Local synthetic setup proof')


def create(setup,arm='scalar'):
    root,held,config=setup
    return PolicyContext.create(root/arm,objective='Synthetic finite two-outcome workflow',information_profile=arm,withheld_sources=held),config


def test_canonical_import_and_fresh_process_preflight(setup):
    root,held,_=setup
    p=subprocess.run([sys.executable,'-m','siesta_remag.research.isolated_policy','preflight','--workspace',str(root/'scalar'),'--information-profile','scalar',*sum([['--withheld-source',s] for s in held],[])],capture_output=True,text=True,check=True)
    r=json.loads(p.stdout)
    assert r['canonical_import']=='siesta_remag.research.toolhost.ToolHost'
    assert r['status']=='passed' and r['model_requests']==r['GPU_launches']==0
    assert not (root/'scalar/.policy').exists()


def test_two_independent_complete_contexts_have_no_ambient_tools_or_shared_history(setup,monkeypatch):
    root,_,_=setup
    monkeypatch.setenv('CODEX_APP_TOOLS_PIPE_PATH','SIBLING_SECRET_PIPE')
    monkeypatch.setenv('CODEX_THREAD_ID','SIBLING_SECRET_THREAD')
    (root/'OTHER_AGENT_SECRET.txt').write_text('PRIVATE_PREDICTION_CANARY')
    ids=[]
    for arm in ('scalar','physics'):
        ctx,config=create(setup,arm);requests=[]
        for _ in range(10):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
        state=load(ctx.private/'state.json');assert state['status']=='awaiting_scores'
        sealed=sha256(ctx.private/'seal.json')
        with pytest.raises(ValueError,match='closed'):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
        assert len(requests)==10
        pred=[load(ctx.private/f'{i:03d}/response.json')['result'] for i in range(6,10)]
        ids.append(pred)
        ctx.release_scores([dict(prediction_id=p,outcome=.00002,absolute_error=.00001,resolution=.00005) for p in pred])
        for _ in range(2):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
        assert load(ctx.private/'state.json')['status']=='completed'
        assert model_policy.audit_session(ctx.private)['attempted_calls']==12
        assert sha256(ctx.private/'seal.json')==sealed
        with pytest.raises(ValueError,match='closed'):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
        encoded=json.dumps(requests)
        for forbidden in ('SIBLING_SECRET','PRIVATE_PREDICTION_CANARY','list_agents','spawn_agent','send_message','exec_command','previous_response_id','conversation','additional_tools'):
            assert forbidden not in encoded
        for req in requests:
            assert {t['name'] for t in req['tools']}=={'select_research_action','finish_research'}
            assert req['model']=='gpt-6-astra' and req['reasoning']=={'effort':'xhigh'}
        assert all(other not in encoded for other in ids[0] if arm=='physics')
    assert not set(ids[0])&set(ids[1])


@pytest.mark.parametrize('name',['collaboration.list_agents','list_agents','send_message','exec_command','stage_scalar_data','inspect_calculation','run_prepared_query','export_electronic_matrices'])
def test_forbidden_selection_never_reaches_toolhost_and_cannot_retry(setup,name,monkeypatch):
    ctx,config=create(setup);requests=[]
    monkeypatch.setattr(ctx.host,'call',lambda *a:pytest.fail('Forbidden call reached host'))
    with pytest.raises(RuntimeError,match='not retried'):
        ctx.step(config,api_key='fixture',transport=transport(lambda c:dict(tool=name,args={},rationale='Injection'),requests))
    assert load(ctx.private/'state.json')['status']=='held_error'
    assert len(requests)==1 and not (ctx.root/'calls').exists()
    with pytest.raises(ValueError,match='closed'):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
    assert len(requests)==1


@pytest.mark.parametrize('args',[{'packet':'../physics/packet.json'},{'packet':'/tmp/other.json'},{'packet':'.policy/registration.json'}])
def test_allowed_tool_cannot_read_private_or_sibling_paths(setup,args,monkeypatch):
    ctx,config=create(setup);requests=[]
    monkeypatch.setattr(ctx.host,'call',lambda *a:pytest.fail('Bad path reached host'))
    with pytest.raises(ValueError,match='registered scalar'):
        ctx.step(config,api_key='fixture',transport=transport(lambda c:dict(tool='inspect_scalar_data',args=args,rationale='Leak probe'),requests))
    assert load(ctx.private/'state.json')['status']=='held_error'


@pytest.mark.parametrize('kind',['raw','symlink','notebook','unknown_scalar','tampered_packet'])
def test_dirty_seed_is_rejected_before_model_or_notebook(setup,kind):
    root,held,_=setup;r=root/'scalar'
    if kind=='raw':(r/'report.json').write_text('{"eigenvalues":"SECRET"}')
    elif kind=='symlink':(r/'packet.json').unlink();(r/'packet.json').symlink_to(root/'physics/packet.json')
    elif kind=='notebook':(r/'research-memory.sqlite').write_text('old context')
    else:
        p=load(r/'packet.json');p['cases']['baseline']['observations']['soc-x']['eigenvalues']=[]
        if kind=='unknown_scalar':p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
        save(r/'packet.json',p)
    with pytest.raises(ValueError):PolicyContext.create(r,objective='Clean',information_profile='scalar',withheld_sources=held)
    assert not (r/'.policy').exists()


def test_changed_seed_and_history_fail_before_another_model_request(setup):
    ctx,config=create(setup);requests=[]
    ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
    save(ctx.private/'000/response.json',{'ok':True,'result':'OTHER_CONTEXT_CANARY'})
    with pytest.raises(ValueError,match='history changed'):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
    assert len(requests)==1


def test_missing_credentials_no_spend_or_retry(setup,monkeypatch):
    ctx,config=create(setup);monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='OPENAI_API_KEY'):ctx.step(config)
    assert load(ctx.private/'state.json')['status']=='held_error'
    assert not list(ctx.private.glob('*/model/request.json'))


def test_scores_cannot_be_released_before_sealing(setup):
    ctx,_=create(setup)
    with pytest.raises(ValueError,match='finite'):ctx.release_scores([])
    assert not (ctx.root/'scores.json').exists()


def test_phase_deadline_fails_before_model_call(setup):
    ctx,config=create(setup);state=load(ctx.private/'state.json')
    state['phase_started_at']='2000-01-01T00:00:00+00:00';save(ctx.private/'state.json',state)
    with pytest.raises(ValueError,match='deadline'):
        ctx.step(config,api_key='fixture',transport=lambda *a:pytest.fail('Deadline must stop before model'))
    assert load(ctx.private/'state.json')['status']=='held_deadline'


def test_tool_failure_stops_without_exposing_private_errors(setup,monkeypatch):
    ctx,config=create(setup);requests=[]
    monkeypatch.setattr(ctx.host,'call',lambda *a:dict(ok=False,error={'message':'PRIVATE_ERROR_CANARY'}))
    with pytest.raises(ValueError,match='Research tool failed'):
        ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
    assert len(requests)==1 and 'PRIVATE_ERROR_CANARY' not in json.dumps(requests)
    with pytest.raises(ValueError,match='closed'):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
    assert len(requests)==1


def test_ambient_function_output_rejected_before_dispatch(setup,monkeypatch):
    ctx,config=create(setup)
    monkeypatch.setattr(ctx.host,'call',lambda *a:pytest.fail('Ambient tool reached broker'))
    def wire(*args):
        return json.dumps(dict(status='completed',output=[dict(type='function_call',name='collaboration.list_agents',arguments='{}')])).encode(),{}
    with pytest.raises(RuntimeError,match='not retried'):ctx.step(config,api_key='fixture',transport=wire)
    assert load(ctx.private/'state.json')['status']=='held_error'


def test_preflight_reports_missing_model_credentials_without_connecting(setup,monkeypatch):
    root,held,config=setup;monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    p=subprocess.run([sys.executable,'-m','siesta_remag.research.isolated_policy','preflight','--workspace',str(root/'scalar'),'--config',str(config),'--information-profile','scalar'],capture_output=True,text=True,check=True)
    r=json.loads(p.stdout)
    assert r['status']=='passed' and r['model_readiness']==dict(model='gpt-6-astra',reasoning_effort='xhigh',credential_present=False,service_access_tested=False,model_calls=0)


def test_changed_seed_fails_before_model_request(setup):
    ctx,config=create(setup)
    with (ctx.root/'packet.json').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='seed changed'):
        ctx.step(config,api_key='fixture',transport=lambda *a:pytest.fail('Changed seed reached model'))


@pytest.mark.parametrize('damage',['sealed_copy','duplicate_score','wrong_error','extra_field'])
def test_score_release_requires_intact_seal_and_exact_scores(setup,damage):
    ctx,config=create(setup);requests=[]
    for _ in range(10):ctx.step(config,api_key='fixture',transport=transport(scripted,requests))
    pred=[load(ctx.private/f'{i:03d}/response.json')['result'] for i in range(6,10)]
    scores=[dict(prediction_id=p,outcome=.00002,absolute_error=.00001,resolution=.00005) for p in pred]
    if damage=='sealed_copy':
        with (ctx.private/'sealed-notebook.sqlite').open('ab') as stream:stream.write(b'tampered')
    elif damage=='duplicate_score':scores[3]=scores[0]
    elif damage=='wrong_error':scores[0]['absolute_error']=1.
    else:scores[0]['private_report']={'other_arm_prediction':'SECRET'}
    with pytest.raises(ValueError):ctx.release_scores(scores)
    assert not (ctx.root/'scores.json').exists()
    assert load(ctx.private/'state.json')['status']=='awaiting_scores'
    assert len(requests)==10
