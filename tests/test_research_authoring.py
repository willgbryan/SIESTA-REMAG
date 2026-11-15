import copy
import http.client
import json
from pathlib import Path
import sys
import threading
import shutil
import subprocess

import numpy as np
import pytest

from test_research import experiment
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load,save
from siesta_remag.research.authoring import author_candidate,verify_candidate,register_structure
from siesta_remag.research.experiments import verify_experiment
from siesta_remag.research.tools import ResearchTools
from siesta_remag.research.agent import run_session
from siesta_remag.research.workspace import dataset,make_server


def edit(experiment,op='physical-tetragonal'):
    return dict(op=op,expected_sha256=load(experiment/'structure.json')['source']['sha256'],scale='101/100')


def test_exact_candidate_handoff_keeps_parent_and_prepared_lineage(experiment,tmp_path):
    tools=ResearchTools(tmp_path);original=sha256(experiment/'structure.json')
    inspected=tools.call('inspect_structure',dict(structure='bundle/structure.json'))
    result=tools.call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment),output='child'))
    child=load(tmp_path/result['structure_path'])
    assert child['source']['sha256']!=inspected['structure']['source']['sha256']
    assert np.linalg.det(child['geometry']['cell'])==pytest.approx(np.linalg.det(inspected['structure']['geometry']['cell']))
    record=verify_candidate(tmp_path/result['candidate_path'])
    assert record['parent_sha256']==inspected['structure']['source']['sha256']
    assert not record['calculation_performed'] and not record['physical_accuracy_qualified']
    assert sha256(experiment/'structure.json')==original
    q=tools.call('prepare_query',dict(structure=result['structure_path'],candidate=result['candidate_path'],
        pseudos='pseudos',output='query',query={'kind':'scalar'},mesh=[3,3,3]))
    assert q['authorship']['parent_sha256']==record['parent_sha256']
    assert q['authorship']['source_sha256']==record['source_sha256']
    assert verify_experiment(tmp_path/'query')==q
    assert not (tmp_path/'query/report.json').exists()
    # The prepared copy is independent of later mutation at the authoring path.
    (tmp_path/'child/program.xcl').write_text('changed')
    assert verify_experiment(tmp_path/'query')==q
    with pytest.raises(ValueError,match='artifact changed'):verify_candidate(tmp_path/result['candidate_path'])
    (tmp_path/'query/authorship/program.xcl').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):verify_experiment(tmp_path/'query')


@pytest.mark.parametrize('change',[
    {'expected_sha256':'a'*64}, {'scale':1.01}, {'scale':True},
    {'scale':'0'}, {'program':'conflicting'}, {'op':'compile-program'},
])
def test_invalid_or_stale_edits_leave_no_candidate_files(experiment,tmp_path,change):
    tools=ResearchTools(tmp_path)
    with pytest.raises(Exception):tools.call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment)|change,output='bad'))
    assert not (tmp_path/'bad').exists()


def test_candidate_and_workspace_never_replace_existing_evidence(experiment,tmp_path):
    tools=ResearchTools(tmp_path)
    result=tools.call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment),output='child'))
    before={p.name:sha256(p) for p in (tmp_path/'child').iterdir()}
    with pytest.raises(FileExistsError):tools.call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment),output='child'))
    entry=tools.call('register_structure',dict(structure=result['structure_path'],candidate=result['candidate_path'],identifier='child',label='Shape candidate'))
    assert entry['id']=='child'
    catalog=sha256(tmp_path/'workspace.json')
    with pytest.raises(ValueError,match='already exists'):tools.call('register_structure',dict(structure='bundle/structure.json',identifier='child',label='Changed'))
    assert sha256(tmp_path/'workspace.json')==catalog
    assert {p.name:sha256(p) for p in (tmp_path/'child').iterdir()}==before
    state=dataset(tmp_path);r=state['families'][0]['rows'][0]
    assert r['candidate']['parent_sha256']==load(experiment/'structure.json')['source']['sha256']
    assert r['result'] is None and r['calculations']==[]
    with pytest.raises(ValueError,match='escapes'):
        tools.call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment),output='../outside'))


def test_root_import_and_parameter_edit_delegate_to_exact_gpac(experiment,tmp_path):
    from gpac.physical import describe_dofs
    program=load(experiment/'structure.json')['source']['program']
    root=author_candidate(dict(op='inspect',program=program),tmp_path/'root')
    assert root['parent_sha256'] is None
    parameters=describe_dofs(program)['exact_parameters']
    request=dict(op='apply',changes={'metric.g33':'16'},expected_sha256=root['source_sha256'],expected_space_sha256=parameters['space_sha256'])
    child=author_candidate(request,tmp_path/'edited',structure=tmp_path/'root/structure.json')
    assert load(tmp_path/'edited/structure.json')['source']['gram'][2][2]=='16'
    assert child['parent_sha256']==root['source_sha256']
    correspondence=load(tmp_path/'edited/response.json')['result']['correspondence']
    assert len(correspondence)==2
    request['expected_space_sha256']='b'*64
    with pytest.raises(Exception):author_candidate(request,tmp_path/'stale-space',structure=tmp_path/'root/structure.json')
    assert not (tmp_path/'stale-space').exists()


def test_candidate_cannot_be_attached_to_another_query(experiment,tmp_path):
    tools=ResearchTools(tmp_path)
    result=tools.call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment),output='child'))
    with pytest.raises(ValueError,match='another prepared structure'):
        tools.call('prepare_query',dict(structure='bundle/structure.json',candidate=result['candidate_path'],pseudos='pseudos',output='wrong',query={'kind':'scalar'}))
    assert not (tmp_path/'wrong').exists()


def test_prepare_cli_carries_candidate_provenance(experiment,tmp_path):
    result=ResearchTools(tmp_path).call('author_structure',dict(structure='bundle/structure.json',request=edit(experiment),output='child'))
    save(tmp_path/'query.json',{'kind':'scalar'})
    subprocess.run([sys.executable,'-m','siesta_remag.research','prepare',str(tmp_path/result['structure_path']),
        '--candidate',str(tmp_path/result['candidate_path']),'--pseudos',str(tmp_path/'pseudos'),
        '--output',str(tmp_path/'cli-query'),'--query',str(tmp_path/'query.json'),'--mesh','3','3','3'],check=True,capture_output=True,text=True)
    assert verify_experiment(tmp_path/'cli-query')['authorship']['source_sha256']==result['candidate']['source_sha256']


def test_agent_reads_edits_registers_and_prepares_without_external_file_writes(experiment,tmp_path):
    policy=tmp_path/'policy.py'
    policy.write_text('''import json,sys
c=json.load(sys.stdin);h=c['history'];n=len(h)
if n==0:d={'tool':'inspect_structure','args':{'structure':'bundle/structure.json'}}
elif n==1:d={'tool':'author_structure','args':{'structure':'bundle/structure.json','output':'candidate','request':{'op':'physical-tetragonal','scale':'101/100','expected_sha256':h[0]['response']['result']['structure']['source']['sha256']}}}
elif n==2:
 r=h[1]['response']['result'];d={'tool':'register_structure','args':{'structure':r['structure_path'],'candidate':r['candidate_path'],'identifier':'new','label':'Authored candidate'}}
elif n==3:
 r=h[1]['response']['result'];d={'tool':'prepare_query','args':{'structure':r['structure_path'],'candidate':r['candidate_path'],'pseudos':'pseudos','output':'new-query','query':{'kind':'scalar'},'mesh':[3,3,3]}}
else:d={'finish':'Candidate and query prepared; no calculation or scientific claim.'}
if 'tool' in d:d['rationale']='Exercise the exact authoring-to-query handoff'
print(json.dumps(d))
''')
    report=run_session(tmp_path,'session',objective='Engineering handoff',policy_command=[sys.executable,str(policy)],max_steps=6,max_seconds=30)
    assert report['status']=='completed',report
    assert report['native_submissions']==0 and len(report['steps'])==5
    prepared=verify_experiment(tmp_path/'new-query')
    assert prepared['authorship']['source_sha256']==load(tmp_path/'candidate/candidate.json')['source_sha256']
    assert dataset(tmp_path)['families'][0]['id']=='new'


def http_request(server,method,path,body=None,**headers):
    conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
    try:
        conn.request(method,path,body=body,headers=headers)
        r=conn.getresponse();return r.status,r.read()
    finally:conn.close()


@pytest.mark.parametrize('editable',[False,True])
def test_local_workspace_edits_are_explicit_and_same_origin(experiment,tmp_path,editable):
    register_structure(tmp_path,'bundle/structure.json','base','Baseline')
    server=make_server(tmp_path,0,editable=editable);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        status,raw=http_request(server,'GET','/api/state');assert status==200
        state=json.loads(raw);assert state['authoring_enabled'] is editable
        payload=dict(source_id='base',expected_source_sha256=load(experiment/'structure.json')['source']['sha256'],label='New shape',request={'op':'physical-tetragonal','scale':'101/100'})
        body=json.dumps(payload)
        origin='http://127.0.0.1:'+str(server.server_port)
        for headers in ({'Content-Type':'application/json'},{'Content-Type':'application/json','Origin':'https://elsewhere.invalid'}):
            assert http_request(server,'POST','/api/author',body,**headers)[0]==403
        assert http_request(server,'GET','/api/state',Host='elsewhere.invalid')[0]==403
        assert not (tmp_path/'candidates').exists()
        status,raw=http_request(server,'POST','/api/author',body,**{'Content-Type':'application/json','Origin':origin})
        assert status==(201 if editable else 403),raw
        if editable:
            result=json.loads(raw)['result'];assert result['candidate']['parent_sha256']==payload['expected_source_sha256']
            assert len(dataset(tmp_path)['families'])==2
            assert load(tmp_path/'workspace.json')['structures'][0]['id']=='base'
            payload['expected_source_sha256']='a'*64
            assert http_request(server,'POST','/api/author',json.dumps(payload),**{'Content-Type':'application/json','Origin':origin})[0]==400
            assert len(list((tmp_path/'candidates').iterdir()))==1
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


def test_editable_mode_requires_registered_sources(tmp_path):
    with pytest.raises(ValueError,match='registered workspace'):make_server(tmp_path,0,editable=True)


def test_shipped_workspace_client_preserves_exact_inputs_and_parent_selection(experiment,tmp_path):
    if shutil.which('node') is None:pytest.skip('Node is needed for the client DOM protocol check')
    register_structure(tmp_path,'bundle/structure.json','base','Baseline')
    save(tmp_path/'state.json',dataset(tmp_path)|{'authoring_enabled':True})
    base=Path(__file__).resolve().parents[1]
    result=subprocess.run(['node',str(base/'tests/fixtures/workspace_authoring_dom.cjs'),
        str(base/'src/siesta_remag/research/workspace.html'),str(tmp_path/'state.json')],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


def test_existing_motif_construct_result_can_become_a_query_source(tmp_path):
    gpac=pytest.importorskip('gpac')
    presets=load(Path(gpac.__file__).parent/'viewer/v2/presets.json')
    record=author_candidate(dict(op='construct',recipe=presets[1]['recipe']),tmp_path/'motif')
    assert verify_candidate(tmp_path/'motif/candidate.json')==record
    assert record['parent_sha256'] is None
    assert load(tmp_path/'motif/response.json')['result']['performed'][0]=='exact motif construction'
