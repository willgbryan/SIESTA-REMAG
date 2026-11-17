"""No network, credentials, model calls or compute: real SDK over mock HTTP."""
import json
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip('ouro')
pytest.importorskip('jsonschema')
import httpx
from ouro import Ouro
from ouro.models.route import Route

from siesta_remag.discovery.common import digest, load, save
from siesta_remag.research.ouro import (
    collect_action, contract, observe_action, prepare_request, price_ceiling_cents,
    submit_request, verify_request,
)
from siesta_remag.research.tools import ResearchTools
from siesta_remag.research.toolhost import ToolHost


def uid(n):
    return str(UUID(int=n))


@pytest.fixture
def route():
    return Route.model_validate(dict(id=uid(1), parent_id=uid(2), user_id=uid(3),
        org_id=uid(4), team_id=uid(5), visibility='monetized', asset_type='route',
        created_at='2026-11-17T00:00:00Z', last_updated='2026-11-17T00:00:00Z',
        monetization='pay-per-use', cost_accounting='fixed', unit_cost_usd=.12,
        route=dict(path='/relax', method='POST', execution_mode='async',
            input_assets={'structure': {'asset_type': 'file', 'file_extensions': ['cif']}},
            output_assets={'relaxed': {'asset_type': 'file'},
                'report': {'asset_type': 'post', 'when': {'include_report': True}}},
            request_body={'content': {'application/json': {'schema': {
                'type': 'object', 'required': ['structure', 'fmax'], 'additionalProperties': False,
                'properties': {'structure': {'type': 'object', 'required': ['url','filename']},
                    'fmax': {'type':'number', 'exclusiveMinimum':0},
                    'include_report': {'type':'boolean', 'default':False}}}}}})))


@pytest.fixture
def prepared(tmp_path, route):
    f = tmp_path / 'Ni.cif'
    f.write_text('data_fixture_Ni\n')
    summary = prepare_request(route, {'structure': f}, tmp_path/'prepared', body={'fmax':.05})
    return tmp_path/'prepared', summary


class Service:
    def __init__(self, route):
        self.route = route.model_dump(mode='json')
        self.requests = []
        self.mode = 'ok'
        self.action = dict(id=uid(10), route_id=uid(1), user_id=uid(3), status='queued',
            response={'energy': -2.0}, output_assets=[])

    def __call__(self, request):
        self.requests.append((request.method, request.url.path, request.content, dict(request.headers)))
        path = request.url.path
        if path == '/user':
            return httpx.Response(200, json={'data': {'id':uid(3), 'email':'fixture@example.invalid'}})
        if path == '/routes/'+uid(1):
            return httpx.Response(200, json={'data':self.route})
        if path == '/files/upload-url':
            return httpx.Response(200, json={'data': {'upload_id':uid(11),
                'upload_url':'https://storage.example.invalid/scratch', 'method':'PUT'}})
        if path == '/scratch':
            return httpx.Response(200)
        if path.endswith('/use'):
            if self.mode == 'timeout':
                raise httpx.ReadTimeout('uncertain secret URL must not enter receipt', request=request)
            if self.mode == 'unauthorized':
                return httpx.Response(401, json={'error':'expired'})
            return httpx.Response(202, json={'action':self.action, 'metadata':{'requiresPolling':True}})
        if path == '/actions/'+uid(10):
            return httpx.Response(200, json={'data':self.action})
        if path == '/assets/'+uid(12)+'/download':
            return httpx.Response(200, content=b'data_relaxed_fixture\n',
                headers={'content-disposition':'attachment; filename="../../escaped.cif"'})
        raise AssertionError('Unexpected request: '+str(request.url))

    def count(self, suffix):
        return sum(p.endswith(suffix) for _,p,_,_ in self.requests)


@pytest.fixture
def sdk(route, monkeypatch):
    service = Service(route)
    original = httpx.Client
    transport = httpx.MockTransport(service)
    monkeypatch.setattr(httpx, 'Client', lambda **kw: original(transport=transport, **kw))
    def storage(method, url, **kw):
        with original(transport=transport) as client:
            return client.request(method,url,**kw)
    monkeypatch.setattr(httpx, 'request', storage)
    client = Ouro(access_token='local-fixture-token', organization='',
                  base_url='https://api.example.invalid')
    yield client, service
    client._raw_client.close()


def submit(prepared, tmp_path, sdk, **kwargs):
    directory, summary = prepared
    client, _ = sdk
    return submit_request(directory, tmp_path/'attempt', client,
        expected_sha256=kwargs.pop('expected_sha256',summary['request_sha256']),
        maximum_cents=kwargs.pop('maximum_cents',12),
        admit=kwargs.pop('admit', lambda h,c: {'fixture_lease':'held', 'request_sha256':h, 'cents':c}), **kwargs)


def test_preparation_keeps_bytes_independent_of_original(prepared,tmp_path):
    directory, summary = prepared
    (tmp_path/'Ni.cif').unlink()
    req = verify_request(directory)
    assert req['price_ceiling_cents']==12
    assert not summary['uploaded'] and not summary['executed']
    assert (directory/req['inputs']['structure']['path']).read_bytes()==b'data_fixture_Ni\n'
    assert 'id' not in req['inputs']['structure']
    assert req['scientific_acceptance']=='not_evaluated'


@pytest.mark.parametrize('change,match',[
    ('variable','Variable-cost'), ('string','object field'), ('asset_id','metadata unavailable'),
    ('extension','extension'), ('required','required'), ('external_ref','local JSON'),
    ('post','post/comment'), ('unknown_price','pricing'), ('missing_usd','USD'),
    ('unbounded_runtime','billing ceiling'), ('negative_price','Invalid USD'),
])
def test_preflight_rejects_before_writing(tmp_path,route,change,match):
    raw=route.model_dump(mode='json'); schema=raw['route']['request_body']['content']['application/json']['schema']
    if change=='variable':raw['cost_accounting']='variable'
    if change=='string':schema['properties']['structure']={'type':'string'}
    if change=='asset_id':schema['properties']['structure']['required'].append('id')
    if change=='extension':raw['route']['input_assets']['structure']['file_extensions']=['xyz']
    if change=='required':schema['required'].append('missing')
    if change=='external_ref':schema['$defs']={'unused':{'$ref':'https://example.invalid/schema'}}
    if change=='post':schema['properties']['include_report']['default']=True
    if change=='unknown_price':raw['monetization']='subscription'
    if change=='missing_usd':raw['unit_cost_usd']=None;raw['price_currency']='btc'
    if change=='unbounded_runtime':raw['cost_accounting']='runtime'
    if change=='negative_price':raw['unit_cost_usd']=-1
    f=tmp_path/'input.cif';f.write_text('fixture')
    with pytest.raises(Exception,match=match):
        prepare_request(raw,{'structure':f},tmp_path/'out',body={'fmax':.05})
    assert not (tmp_path/'out').exists()


def test_runtime_and_free_pricing_are_explicit(route):
    raw=contract(route);raw.update(cost_accounting='runtime',unit_cost_usd=.0011,max_billable_seconds=101)
    assert price_ceiling_cents(raw)==12  # round up, no floating-cent underreservation
    raw.update(visibility='public',monetization=None,unit_cost_usd=None)
    assert price_ceiling_cents(raw)==0
    raw['visibility']='inherit'
    with pytest.raises(ValueError,match='pricing'):price_ceiling_cents(raw)


@pytest.mark.parametrize('fault',['body','bytes','digest','cap','admission','contract','refresh'])
def test_submission_preconditions_precede_upload(prepared,tmp_path,sdk,fault):
    directory,_=prepared;client,service=sdk;kwargs={}
    if fault=='body':
        req=load(directory/'request.json');req['body']['fmax']=.1;save(directory/'request.json',req)
    if fault=='bytes':(directory/'inputs/00.cif').write_text('changed')
    if fault=='digest':kwargs['expected_sha256']='0'*64
    if fault=='cap':kwargs['maximum_cents']=11
    if fault=='admission':kwargs['admit']=None
    if fault=='contract':service.route['unit_cost_usd']=.13
    if fault=='refresh':client.api_key='fake-auto-refresh-key'
    with pytest.raises(ValueError):submit(prepared,tmp_path,sdk,**kwargs)
    assert service.count('/files/upload-url')==service.count('/use')==0


def test_real_sdk_upload_and_submit_exactly_once(prepared,tmp_path,sdk):
    state=submit(prepared,tmp_path,sdk);client,service=sdk
    assert state['action_id']==uid(10) and state['state']=='queued'
    assert not state['allocation_release_verified']
    assert verify_request(tmp_path/'attempt')['contract_sha256']==verify_request(prepared[0])['contract_sha256']
    uploads=[json.loads(b) for _,p,b,_ in service.requests if p=='/files/upload-url']
    assert uploads==[{'file_name':'Ni.cif','visibility':'private','purpose':'input'}]
    assert next(b for _,p,b,_ in service.requests if p=='/scratch')==b'data_fixture_Ni\n'
    payload,headers=next((json.loads(b),h) for _,p,b,h in service.requests if p.endswith('/use'))
    assert payload['config']['input_assets']['structure']=={'upload_id':uid(11),'file_name':'Ni.cif'}
    assert payload['config']['output']=={'visibility':'private'} and payload['currency']=='usd'
    assert headers['prefer']=='respond-async'
    assert service.count('/routes/'+uid(1))==1
    with pytest.raises(FileExistsError):submit(prepared,tmp_path,sdk)
    assert service.count('/use')==1
    assert all(p not in ('/files','/posts') for _,p,_,_ in service.requests)


@pytest.mark.parametrize('mode',['timeout','unauthorized'])
def test_uncertain_submission_is_retained_and_never_retried(prepared,tmp_path,sdk,mode):
    client,service=sdk;service.mode=mode
    with pytest.raises(Exception):submit(prepared,tmp_path,sdk)
    state=load(tmp_path/'attempt/state.json')
    assert state['state']=='submission_uncertain' and state['action_id'] is None
    assert state['upload_refs']['structure']['upload_id']==uid(11)
    assert 'secret' not in (tmp_path/'attempt/state.json').read_text()
    with pytest.raises(FileExistsError):submit(prepared,tmp_path,sdk)
    with pytest.raises(ValueError,match='unknown action'):observe_action(tmp_path/'attempt',client)
    assert service.count('/use')==1 and service.count('/users/get-token')==0


@pytest.mark.parametrize('status',['success','error','timed-out','cancelled'])
def test_collection_retains_raw_response_and_native_file_without_promoting(prepared,tmp_path,sdk,status):
    submit(prepared,tmp_path,sdk);client,service=sdk
    service.action.update(status=status,output_assets=[{'name':'relaxed','asset':{'id':uid(12),'asset_type':'file'}}])
    observe_action(tmp_path/'attempt',client)
    result=collect_action(tmp_path/'attempt',client,tmp_path/'collection')
    assert result['complete'] and result['service_status']==status
    assert result['scientific_acceptance']=='not_evaluated' and not result['allocation_release_verified']
    assert load(tmp_path/'collection/action.json')['response']=={'energy':-2.0}  # no SDK final_data mutation
    assert (tmp_path/'collection'/result['files'][0]['path']).read_bytes()==b'data_relaxed_fixture\n'
    assert not (tmp_path/'escaped.cif').exists()
    assert service.count('/use')==1


def test_poll_is_bounded_and_rejects_identity_change(prepared,tmp_path,sdk):
    submit(prepared,tmp_path,sdk);client,service=sdk
    assert observe_action(tmp_path/'attempt',client)['state']=='queued'
    assert service.count('/actions/'+uid(10))==1
    with pytest.raises(ValueError,match='still running'):collect_action(tmp_path/'attempt',client,tmp_path/'collection')
    assert not (tmp_path/'collection').exists()
    service.action['id']=uid(99)
    with pytest.raises(ValueError,match='identity'):observe_action(tmp_path/'attempt',client)
    assert load(tmp_path/'attempt/state.json')['action_id']==uid(10)


def test_local_tool_paths_and_policy_isolation(tmp_path,route):
    save(tmp_path/'route.json',route.model_dump(mode='json'))
    (tmp_path/'input.cif').write_text('fixture')
    args={'route':'route.json','inputs':{'structure':'input.cif'},'output':'prepared','body':{'fmax':.05}}
    assert ResearchTools(tmp_path).call('prepare_ouro_request',args)['executed'] is False
    for profile in [dict(information_profile='scalar'),dict(withheld_sources=['a'*64])]:
        with pytest.raises(ValueError):ResearchTools(tmp_path,**profile).call('prepare_ouro_request',args)
    args['inputs']['structure']='../outside.cif'
    with pytest.raises(ValueError,match='escapes'):ResearchTools(tmp_path).call('prepare_ouro_request',args)
    assert not any('submit_ouro' in x for x in ResearchTools(tmp_path).catalog())
    assert ToolHost(tmp_path).catalog()['tools']['prepare_ouro_request']['effect']=='prepare'
    assert 'prepare_ouro_request' not in ToolHost(tmp_path,withheld_sources=['a'*64]).catalog()['tools']


@pytest.mark.parametrize('case',['missing','undeclared','duplicate','disabled'])
def test_incomplete_or_unexpected_outputs_are_preserved_not_accepted(prepared,tmp_path,sdk,case):
    submit(prepared,tmp_path,sdk);client,service=sdk
    service.action.update(status='success',output_assets=[])
    asset={'name':'relaxed','asset':{'id':uid(12),'asset_type':'file'}}
    if case=='undeclared':service.action['output_assets']=[dict(asset,name='surprise')]
    if case=='duplicate':service.action['output_assets']=[asset,asset]
    if case=='disabled':service.action['output_assets']=[{'name':'report','asset':{'id':uid(12),'asset_type':'post'}}]
    with pytest.raises(ValueError):collect_action(tmp_path/'attempt',client,tmp_path/'collection')
    assert load(tmp_path/'collection/action.json')['status']=='success'
    assert not load(tmp_path/'collection/manifest.json')['complete']
    assert service.count('/use')==1


def test_local_schema_ref_and_explicit_parameter_preflight(tmp_path,route):
    raw=route.model_dump(mode='json')
    schema=raw['route']['request_body']['content']['application/json']['schema']
    schema['$defs']={'file':schema['properties']['structure']}
    schema['properties']['structure']={'$ref':'#/$defs/file'}
    raw['route']['parameters']=[{'name':'mode','in':'query','required':True,'schema':{'enum':['relax']}}]
    (tmp_path/'input.cif').write_text('fixture')
    with pytest.raises(ValueError,match='Missing required'):
        prepare_request(raw,{'structure':tmp_path/'input.cif'},tmp_path/'failed',body={'fmax':.05})
    prepare_request(raw,{'structure':tmp_path/'input.cif'},tmp_path/'ok',body={'fmax':.05},query={'mode':'relax'})
    assert verify_request(tmp_path/'ok')['query']=={'mode':'relax'}


def test_denied_admission_and_failed_upload_do_not_submit(prepared,tmp_path,sdk):
    client,service=sdk
    def denied(*_):raise ValueError('Another allocation is owned')
    with pytest.raises(ValueError,match='owned'):submit(prepared,tmp_path,sdk,admit=denied)
    assert service.count('/files/upload-url')==service.count('/use')==0
    assert load(tmp_path/'attempt/state.json')['state']=='admission_pending'


def test_sdk_upload_failure_preserves_partial_attempt(prepared,tmp_path,sdk,monkeypatch):
    client,service=sdk
    def fail(*_,**kw):raise httpx.ConnectError('No storage')
    monkeypatch.setattr(httpx,'request',fail)
    with pytest.raises(httpx.ConnectError):submit(prepared,tmp_path,sdk)
    state=load(tmp_path/'attempt/state.json')
    assert state['state']=='uploading' and state['action_id'] is None
    assert service.count('/files/upload-url')==1 and service.count('/use')==0
    assert (tmp_path/'attempt/inputs/00.cif').read_bytes()==b'data_fixture_Ni\n'
