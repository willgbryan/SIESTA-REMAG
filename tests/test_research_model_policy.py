"""Model protocol fixtures: no service request, model inference or solver run."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib import error

import pytest

from siesta_remag.discovery.common import load, save
from siesta_remag.research import agent, model_policy as policy
from siesta_remag.research.memory import Notebook
from siesta_remag.research.tools import TOOLS


@pytest.fixture
def config(tmp_path):
    path=tmp_path/'policy.json'
    save(path,dict(format='REMAG-MODEL-POLICY/1',model='explicit-fixture-model',max_calls=8,
        max_output_tokens=1024,max_reserved_output_tokens=8192,request_timeout_seconds=20,
        research_brief='Synthetic protocol check; no empirical research claim.'))
    return path


def context(history=()):
    return dict(objective='Synthetic policy integration',tools={k:TOOLS[k] for k in
        ('record_hypothesis','propose_experiment','record_prediction','retrieve_research_history')},
        history=list(history),information_profile='physics',withheld_sources=[],seconds_remaining=60,decisions_remaining=8-len(history))


def response(decision, **updates):
    if 'finish' in decision:name='finish_research';args=dict(conclusion=decision['finish'])
    else:name='select_research_action';args=dict(tool=decision['tool'],arguments_json=json.dumps(decision['args']),rationale=decision['rationale'])
    value=dict(id='resp_fixture',model='explicit-fixture-model',status='completed',error=None,incomplete_details=None,
        output=[dict(type='function_call',name=name,arguments=json.dumps(args),status='completed',call_id='call_fixture')],
        usage=dict(input_tokens=100,output_tokens=20,total_tokens=120))
    return value | updates


def wire(value):
    def transport(payload,key,timeout):return json.dumps(value).encode(),dict(http_status=200,request_id='req_fixture')
    return transport


def step(tmp_path,index=0):
    path=tmp_path/'session'/f'{index:03d}';path.mkdir(parents=True);return path


def test_request_and_receipt_pin_model_permissions_and_report_usage(config,tmp_path):
    c=context();d=dict(tool='retrieve_research_history',args={'limit':4},rationale='Find earlier hypotheses')
    path=step(tmp_path)
    calls=[]
    def transport(payload,key,timeout):
        calls.append(payload)
        assert key=='not-a-real-key' and timeout==20
        assert load(path/'model/receipt.json')['status']=='intent'
        return wire(response(d))(payload,key,timeout)
    assert policy.decide(c,config,path,api_key='not-a-real-key',transport=transport)==d
    payload=calls[0]
    assert payload['store'] is False and payload['parallel_tool_calls'] is False and payload['stream'] is False
    assert payload['tool_choice']=='required' and payload['model']=='explicit-fixture-model'
    assert all(x['type']=='function' and x['strict'] and not x['parameters']['additionalProperties'] for x in payload['tools'])
    assert 'run_prepared_query' not in payload['tools'][0]['parameters']['properties']['tool']['enum']
    audit=policy.audit_session(path.parent)
    assert audit['attempted_calls']==1 and audit['reserved_output_tokens']==1024
    assert audit['reported_usage']==dict(input_tokens=100,output_tokens=20,total_tokens=120)
    assert audit['attempts'][0]['transport']=='injected-test' and audit['actual_billed_usd'] is None
    assert all('not-a-real-key' not in p.read_text() for p in path.parent.rglob('*.json'))


@pytest.mark.parametrize('change',[
    {'status':'incomplete'}, {'error':{'message':'failed'}}, {'incomplete_details':{'reason':'max_output_tokens'}},
    {'output':[]}, {'output':[{'type':'message','content':[{'type':'refusal','refusal':'refused'}]}]},
    {'output':[{'type':'web_search_call'}]}, {'usage':{'input_tokens':100,'output_tokens':True,'total_tokens':101}},
    {'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':121}},
])
def test_failed_model_reply_keeps_reservation_and_never_retries(config,tmp_path,change):
    c=context();path=step(tmp_path);calls=[]
    def transport(*a):calls.append(1);return wire(response({'finish':'Done'},**change))(*a)
    with pytest.raises(RuntimeError,match='not retried'):policy.decide(c,config,path,api_key='fixture',transport=transport)
    assert calls==[1]
    assert load(path/'model/receipt.json')['status']=='held_error'
    audit=policy.audit_session(path.parent);assert audit['reserved_output_tokens']==1024 and audit['unresolved_calls']==1
    with pytest.raises(ValueError,match='unresolved'):policy.decide(c,config,path,api_key='fixture',transport=transport)
    assert calls==[1]


def test_ambiguous_transport_failure_is_retained_without_secrets(config,tmp_path):
    path=step(tmp_path);calls=[]
    def broken(*a):calls.append(1);raise TimeoutError('secret-api-key-must-not-be-logged')
    with pytest.raises(RuntimeError):policy.decide(context(),config,path,api_key='fixture',transport=broken)
    receipt=load(path/'model/receipt.json')
    assert receipt['error_type']=='TimeoutError' and not receipt['retry_authorized'] and receipt['usage'] is None
    assert 'secret-api-key' not in (path/'model/receipt.json').read_text()
    assert policy.audit_session(path.parent)['reported_usage'] is None
    assert calls==[1]


@pytest.mark.parametrize('decision',[
    dict(tool='run_prepared_query',args={},rationale='unauthorized'),
    dict(tool='record_hypothesis',args={'statement':'missing fields'},rationale='invalid'),
    dict(tool='retrieve_research_history',args={'shell':'bad'},rationale='invalid'),
    dict(tool='retrieve_research_history',args={},rationale=''),
    dict(finish=''),
])
def test_bad_decisions_are_not_actions(decision):
    with pytest.raises(ValueError):policy.decode_decision(response(decision),context()['tools'])


def test_multiple_calls_extra_fields_and_nonfinite_arguments_are_rejected():
    value=response(dict(tool='retrieve_research_history',args={},rationale='test'))
    value['output']*=2
    with pytest.raises(ValueError,match='exactly one'):policy.decode_decision(value,context()['tools'])
    for raw in ('{"limit":1,"limit":2}','{"limit":NaN}','{"limit":1e999}'):
        value=response(dict(tool='retrieve_research_history',args={},rationale='test'))
        args=json.loads(value['output'][0]['arguments']);args['arguments_json']=raw
        value['output'][0]['arguments']=json.dumps(args)
        with pytest.raises(ValueError):policy.decode_decision(value,context()['tools'])


def test_config_change_budget_and_missing_key_fail_before_transport(config,tmp_path,monkeypatch):
    path=step(tmp_path);c=context();calls=[]
    action=dict(tool='retrieve_research_history',args={},rationale='Read notebook')
    def transport(*a):calls.append(1);return wire(response(action))(*a)
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='OPENAI_API_KEY'):policy.decide(c,config,path,transport=transport)
    assert not (path/'model').exists() and not calls
    v=load(config);v['max_calls']=1;save(config,v)
    d=policy.decide(c,config,path,api_key='fixture',transport=transport)
    save(path/'decision.json',d);save(path/'response.json',{'ok':True})
    c=context([dict(decision=d,response={'ok':True})]);second=step(tmp_path,1)
    with pytest.raises(ValueError,match='ceiling'):policy.decide(c,config,second,api_key='fixture',transport=transport)
    v['max_calls']=2;save(config,v)
    with pytest.raises(ValueError,match='policy changed'):policy.decide(c,config,second,api_key='fixture',transport=transport)
    assert calls==[1] and not (second/'model').exists()


def test_output_reservation_is_not_refunded_after_small_reply(config,tmp_path):
    v=load(config);v['max_reserved_output_tokens']=1024;save(config,v)
    action=dict(tool='retrieve_research_history',args={},rationale='Read notebook');path=step(tmp_path)
    d=policy.decide(context(),config,path,api_key='fixture',transport=wire(response(action)))
    save(path/'decision.json',d);save(path/'response.json',{})
    with pytest.raises(ValueError,match='ceiling'):
        policy.decide(context([dict(decision=d,response={})]),config,step(tmp_path,1),api_key='fixture',transport=wire(response(d)))


def test_changed_context_and_observation_are_rejected_before_call(config,tmp_path):
    path=step(tmp_path);action=dict(tool='retrieve_research_history',args={},rationale='Read notebook')
    d=policy.decide(context(),config,path,api_key='fixture',transport=wire(response(action)))
    save(path/'decision.json',d);save(path/'response.json',{'ok':False})
    c=context([dict(decision=d,response={'ok':True})]);second=step(tmp_path,1)
    with pytest.raises(ValueError,match='observation history changed'):
        policy.decide(c,config,second,api_key='fixture',transport=wire(response({'finish':'Done'})))
    c['objective']='A different objective'
    with pytest.raises(ValueError,match='policy changed'):
        policy.decide(c,config,second,api_key='fixture',transport=wire(response({'finish':'Done'})))
    assert not (second/'model').exists()


def test_finished_session_cannot_be_resumed(config,tmp_path):
    d=policy.decide(context(),config,step(tmp_path),api_key='fixture',transport=wire(response({'finish':'Done'})))
    with pytest.raises(ValueError,match='cannot be resumed'):
        policy.decide(context([dict(decision=d,response={})]),config,step(tmp_path,1),api_key='fixture',transport=wire(response(d)))


def test_response_tampering_and_unfinished_intent_are_audited(config,tmp_path):
    path=step(tmp_path)
    policy.decide(context(),config,path,api_key='fixture',transport=wire(response({'finish':'Done'})))
    p=path/'model/response.json';original=p.read_bytes();p.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):policy.audit_session(path.parent)
    p.write_bytes(original)
    receipt=load(path/'model/receipt.json');receipt['status']='intent';receipt['usage']=None;save(path/'model/receipt.json',receipt)
    assert policy.audit_session(path.parent)['unresolved_calls']==1


def test_http_transport_uses_fixed_endpoint_and_rejects_redirects_without_retry(monkeypatch):
    seen=[]
    class FakeOpener:
        def open(self,req,timeout):
            seen.append(req)
            assert req.full_url==policy.ENDPOINT and req.method=='POST'
            assert req.get_header('Authorization')=='Bearer fixture'
            raise error.HTTPError(req.full_url,429,'secret server detail',{},io.BytesIO(b'secret'))
    def opener(handler):
        assert handler.redirect_request(None,None,302,'',{},'https://elsewhere.invalid') is None
        return FakeOpener()
    monkeypatch.setattr(policy.request,'build_opener',opener)
    with pytest.raises(policy.TransportError) as caught:policy.post_responses({},'fixture',1)
    assert caught.value.status==429 and 'secret' not in str(caught.value) and len(seen)==1


def install_fixture_policy(monkeypatch,config):
    original=agent._process
    requests=[]
    def process(command,payload,deadline,folder,label,**kwargs):
        if label!='policy':return original(command,payload,deadline,folder,label,**kwargs)
        def transport(body,key,timeout):
            requests.append(body)
            c=json.loads(body['input'][0]['content'])['context'];h=c['history'];n=len(h)
            if n==0:d=dict(tool='record_hypothesis',args=dict(statement='Fixture positive sign',predictions=['E[x]>E[z]'],alternatives=['Negative sign']),rationale='Set competing hypotheses')
            elif n==1:d=dict(tool='propose_experiment',args=dict(hypothesis_id=h[0]['response']['result'],structure_sha256='c'*64,action='heldout prediction',rationale='Compare alternatives'),rationale='Record a discriminating test')
            elif n==2:d=dict(tool='record_prediction',args=dict(experiment_id=h[1]['response']['result'],observable='anisotropy.total_energy_ev',value=.0001,unit='eV/cell'),rationale='Predict before observing')
            else:d={'finish':'Synthetic model-protocol fixture complete; no inference service was called.'}
            return wire(response(d))(body,key,timeout)
        return policy.decide(payload,config,folder,api_key='fixture',transport=transport)
    monkeypatch.setattr(agent,'_process',process)
    return requests


def test_model_decisions_use_real_tool_processes_and_notebook(config,tmp_path,monkeypatch):
    requests=install_fixture_policy(monkeypatch,config)
    result=agent.run_session(tmp_path,'session',objective='Protocol fixture',policy_command=['fixture'],max_steps=6,max_seconds=30)
    assert result['status']=='completed',result
    assert result['native_submissions']==0 and result['model_policy']['attempted_calls']==4
    assert result['model_policy']['reported_usage']['total_tokens']==480
    assert len(requests)==4 and len(json.loads(requests[-1]['input'][0]['content'])['context']['history'])==3
    n=Notebook(tmp_path/'research-memory.sqlite');events=n.retrieve();n.close()
    assert {e['kind'] for e in events}=={'hypothesis','experiment','prediction'}
    assert not result['scientific_success_assessed']


def test_paired_study_carries_model_usage_and_rejects_changed_receipts(config,tmp_path,monkeypatch):
    from test_research_study import study_fixture
    from siesta_remag.research.study import register_study,run_study_arm,evaluate_study
    d=study_fixture(tmp_path);d['policy_artifacts'].append('policy.json')
    install_fixture_policy(monkeypatch,config)
    register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):assert run_study_arm(tmp_path,'study',arm)['status']=='completed'
    scores=evaluate_study(tmp_path,'study','scores.json')
    for arm in scores['arms'].values():
        assert arm['model_policy']['reported_usage']['total_tokens']==480
        assert arm['model_policy']['attempted_calls']==4 and arm['model_policy']['actual_billed_usd'] is None
    assert not scores['scientific_advantage_demonstrated']
    path=tmp_path/'arm-physics/session/000/model/receipt.json';v=load(path);v['returned_model']='changed';save(path,v)
    with pytest.raises(ValueError,match='model policy audit differs'):evaluate_study(tmp_path,'study','invalid.json')


def test_cli_requires_runner_and_key_without_exposing_values(config,tmp_path):
    env=dict(os.environ);env.pop('REMAG_RESEARCH_POLICY_STEP',None);env['OPENAI_API_KEY']='not-a-real-secret'
    p=subprocess.run([sys.executable,'-m','siesta_remag.research.model_policy','--config',str(config)],
        input=json.dumps(context()),capture_output=True,text=True,env=env)
    assert p.returncode==1 and not p.stdout and 'not-a-real-secret' not in p.stderr
    assert not (tmp_path/'session').exists()


def test_policy_process_receives_step_directory(tmp_path):
    script='import os,json;print(json.dumps({"finish":os.environ["REMAG_RESEARCH_POLICY_STEP"]}))'
    result=agent.run_session(tmp_path,'session',objective='Environment handoff',policy_command=[sys.executable,'-c',script],max_seconds=20)
    assert result['status']=='completed' and result['policy_conclusion']==str(tmp_path/'session/000')
