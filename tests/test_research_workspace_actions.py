"""Workspace handoff tests use prepared files and injected synthetic observations."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from test_research import experiment
from test_research_authoring import http_request
from test_research_series import measured, observables
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.research.authoring import register_structure, save_workspace_candidate
from siesta_remag.research.experiments import verify_experiment
from siesta_remag.research.series import analyze_series, verify_series
from siesta_remag.research.workspace import dataset, make_server, snapshot
from siesta_remag.research.workspace_actions import register_research, prepare_workspace_research


@pytest.fixture
def configured(experiment, tmp_path):
    register_structure(tmp_path, 'bundle/structure.json', 'base', 'Baseline')
    profile = register_research(tmp_path, 'profile', 'bundle', 'saved', 'Saved numerical settings')
    return dict(kind='query', source_id='base', expected_source_sha256=load(experiment/'structure.json')['source']['sha256'],
        profile_id='saved', expected_profile_sha256=profile['sha256'], query={'kind':'scalar'}, label='Ground state')


def test_query_preserves_scientific_settings_and_candidate_lineage(experiment, tmp_path, configured):
    frozen = sha256(experiment/'experiment.json')
    child = save_workspace_candidate(tmp_path, dict(source_id='base', expected_source_sha256=configured['expected_source_sha256'],
        request={'op':'physical-tetragonal', 'scale':'101/100'}, label='Shape variant'))
    payload = configured | dict(source_id=child['entry']['id'], expected_source_sha256=child['candidate']['source_sha256'])
    result = prepare_workspace_research(tmp_path, payload)
    prepared = verify_experiment(tmp_path/result['entry']['path'])
    original = verify_experiment(experiment)
    for key in ('settings', 'kmesh', 'basis_size', 'basis_energy_shift_ry', 'pseudopotentials'):
        assert prepared[key] == original[key]
    assert prepared['query'] == {'kind':'scalar'}
    assert prepared['authorship']['source_sha256'] == child['candidate']['source_sha256']
    assert all(result[k] is False for k in ('calculation_performed','queued','reservation_created'))
    assert sha256(experiment/'experiment.json') == frozen
    assert not list((tmp_path/'prepared').rglob('report.json'))
    assert len(dataset(tmp_path)['research']) == 2


def test_workspace_response_preparation_is_exact_and_unqueued(experiment, tmp_path, configured):
    request = configured | dict(kind='series', design=dict(format='REMAG-RESPONSE-DESIGN/1', parameter='q',
        values=['-1/200','0','1/100'], control={'kind':'axial_strain','axis':2}))
    response = prepare_workspace_research(tmp_path, request)
    series = verify_series(tmp_path/response['entry']['path'])
    assert series['parent_source_sha256'] == configured['expected_source_sha256']
    assert series['design']['values'] == ['-1/200','0','1/100']
    for point in series['points']:
        q = verify_experiment(tmp_path/response['entry']['path']/point['bundle'])
        assert q['query'] == configured['query'] and q['settings'] == verify_experiment(experiment)['settings']
    assert dataset(tmp_path)['research'][-1]['point_count'] == 3


@pytest.mark.parametrize('change',[
    {'source_id':'missing'}, {'profile_id':'missing'}, {'expected_source_sha256':'f'*64},
    {'expected_profile_sha256':'f'*64}, {'kind':'execute'}, {'output':'../escape'},
    {'query':{'kind':'retry'}}, {'label':''},
    {'kind':'series','design':{'format':'REMAG-RESPONSE-DESIGN/1','parameter':'q','values':[-.1,0,.1],'control':{'kind':'axial_strain','axis':2}}},
])
def test_invalid_requests_do_not_prepare_or_modify_registry(experiment,tmp_path,configured,change):
    before=sha256(tmp_path/'workspace.json')
    with pytest.raises(Exception):prepare_workspace_research(tmp_path, configured | change)
    assert sha256(tmp_path/'workspace.json')==before
    assert not (tmp_path/'prepared').exists()


def test_altered_profile_and_escaping_preparation_are_rejected(experiment,tmp_path,configured):
    pseudo=next((experiment/'pseudos').iterdir());original=pseudo.read_bytes();pseudo.write_text('altered')
    with pytest.raises(ValueError,match='artifact changed'):prepare_workspace_research(tmp_path,configured)
    assert not (tmp_path/'prepared').exists()
    pseudo.write_bytes(original)
    outside=tmp_path.parent/(tmp_path.name+'-outside');outside.mkdir()
    (tmp_path/'prepared').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError,match='escapes'):prepare_workspace_research(tmp_path,configured)
    assert list(outside.iterdir())==[]


def test_registration_is_explicit_immutable_bounded_and_confined(experiment,tmp_path,configured):
    before=sha256(tmp_path/'workspace.json')
    with pytest.raises(ValueError,match='duplicate'):register_research(tmp_path,'profile','bundle','saved','Other')
    with pytest.raises(ValueError,match='escapes'):register_research(tmp_path,'profile','../outside','bad','Bad')
    assert sha256(tmp_path/'workspace.json')==before
    request=load(experiment/'experiment.json');request['kmesh']=[9,9,9];save(experiment/'experiment.json',request)
    with pytest.raises(ValueError,match='Registered research artifact changed'):dataset(tmp_path)


@pytest.mark.parametrize('mode',['missing','failed','branch'])
def test_saved_response_keeps_exact_series_gaps_failures_and_branch_holds(experiment,tmp_path,configured,mode):
    series,evidence=measured(experiment,tmp_path,fail='point-002' if mode=='failed' else None,branch='point-002' if mode=='branch' else None)
    if mode=='missing':del evidence['point-002']
    response=analyze_series(tmp_path/'series',evidence,observables(),tmp_path/'analysis')
    before=sha256(tmp_path/'workspace.json')
    with pytest.raises(ValueError,match='Register the exact response series'):register_research(tmp_path,'response','analysis/response.json','response','Response')
    assert sha256(tmp_path/'workspace.json')==before
    register_research(tmp_path,'series','series','strain','c strain')
    register_research(tmp_path,'response','analysis/response.json','response','Response')
    view=dataset(tmp_path)['research'][-1]
    assert view['parent_source_sha256']==configured['expected_source_sha256']
    assert view['result']==response and 'not recollected' in view['verification']
    p=view['result']['observables']['K']['points'][2]
    assert p['status']=={'missing':'pending','failed':'failed','branch':'measured'}[mode]
    assert not p['derivative_eligible']
    assert view['result']['observables']['K']['derivatives'][0]['status']=='inconclusive'
    saved=snapshot(tmp_path,tmp_path/'view.html')
    assert saved['sha256']==sha256(tmp_path/'view.html')
    assert 'saved-state' in (tmp_path/'view.html').read_text()
    response['observables']['K']['points'][2]['source_sha256']='b'*64
    response['sha256']=digest({k:v for k,v in response.items() if k!='sha256'})
    save(tmp_path/'changed-response.json',response)
    with pytest.raises(ValueError,match='another exact source'):register_research(tmp_path,'response','changed-response.json','bad','Bad')


def test_registration_cli_and_client_protocol(experiment,tmp_path,configured):
    result=subprocess.run([sys.executable,'-m','siesta_remag.research','workspace-register','--workspace',str(tmp_path),
        '--kind','query','--path','bundle','--identifier','existing','--label','Existing query'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['id']=='existing'
    if shutil.which('node') is None:pytest.skip('Node needed for client DOM protocol check')
    series,evidence=measured(experiment,tmp_path,branch='point-002')
    analyze_series(tmp_path/'series',evidence,observables(),tmp_path/'analysis')
    register_research(tmp_path,'series','series','strain','Strain')
    register_research(tmp_path,'response','analysis/response.json','response','Saved response')
    save(tmp_path/'state.json',dataset(tmp_path)|{'authoring_enabled':True})
    base=Path(__file__).resolve().parents[1]
    run=subprocess.run(['node',str(base/'tests/fixtures/workspace_research_dom.cjs'),str(base/'src/siesta_remag/research/workspace.html'),str(tmp_path/'state.json')],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stdout+run.stderr


@pytest.mark.parametrize('editable',[False,True])
def test_local_workspace_preparation_requires_same_origin_and_editable_mode(experiment,tmp_path,configured,editable):
    server=make_server(tmp_path,0,editable=editable);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        body=json.dumps(configured);origin='http://127.0.0.1:'+str(server.server_port)
        for headers in ({'Content-Type':'application/json'},{'Content-Type':'application/json','Origin':'https://elsewhere.invalid'}):
            assert http_request(server,'POST','/api/prepare',body,**headers)[0]==403
        assert not (tmp_path/'prepared').exists()
        headers={'Content-Type':'application/json','Origin':origin}
        status,raw=http_request(server,'POST','/api/prepare',body,**headers)
        assert status==(201 if editable else 403),raw
        if editable:
            assert json.loads(raw)['result']['calculation_performed'] is False
            assert http_request(server,'POST','/api/prepare','{"kind":"query","kind":"series"}',**headers)[0]==400
            assert http_request(server,'POST','/api/prepare',body,**(headers|{'Host':'elsewhere.invalid'}))[0]==403
            assert len(list((tmp_path/'prepared').iterdir()))==1
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


def test_profile_manifest_symlink_cannot_escape_workspace(experiment,tmp_path,configured):
    outside=tmp_path.parent/(tmp_path.name+'-experiment.json')
    shutil.copyfile(experiment/'experiment.json',outside)
    (experiment/'experiment.json').unlink();(experiment/'experiment.json').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):dataset(tmp_path)
    with pytest.raises(ValueError,match='escapes'):register_research(tmp_path,'profile','bundle','escape','Escape')
    with pytest.raises(ValueError,match='escapes'):prepare_workspace_research(tmp_path,configured)
    assert not (tmp_path/'prepared').exists()
