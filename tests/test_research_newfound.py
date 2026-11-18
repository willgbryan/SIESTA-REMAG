import json
import threading

import pytest

from test_research import experiment
from test_research_authoring import http_request
from siesta_remag.discovery.common import load, save
from siesta_remag.research import newfound
from siesta_remag.research.workspace import make_server


def catalog(root):
    export = load(root/'bundle/structure.json')
    save(root/'workspace.json', dict(format='REMAG-RESEARCH-WORKSPACE/1', title='lab',
         structures=[dict(id=name, label=name, structure='bundle/structure.json') for name in ('a','b')]))
    return export['source']['sha256']


class Service:
    def __init__(self):
        self.calls = []
        self.pending = False

    def call(self, tool, args):
        self.calls.append(tool)
        if tool == 'job_status':
            return {'job_id':args['job_id'], 'status':'completed'}
        if tool == 'synthetic_accessibility' and self.pending:
            self.pending = False
            return {'formula':'Fe','by_temperature':[{'job_id':'old', 'status':'running'}]}
        return {'status':'completed', 'formula':'Fe'}


def test_reuse_scopes_and_verify_received_inputs(experiment, tmp_path, monkeypatch):
    source = catalog(tmp_path)
    from gpac import ir, physical
    other = ir.author_program(dict(hall=1, gram=[[10,0,0],[0,9,0],[0,0,9]],
        orbits=[dict(z=26,wyckoff_index=0,parameters=[0,0,0]),
                dict(z=26,wyckoff_index=0,parameters=['1/2','1/2','1/2'])]))
    exported = physical.export_geometry(other,length_unit='angstrom')
    save(tmp_path/'b.json',exported)
    manifest=load(tmp_path/'workspace.json')
    manifest['structures'][1]['structure']='b.json'
    save(tmp_path/'workspace.json',manifest)
    service = Service()
    monkeypatch.setattr(newfound, 'Newfound', lambda key: service)
    assert newfound.view(tmp_path,'a',source)['status'] == 'empty'
    for name in ('a','b','a'):
        assert not newfound.screen(tmp_path,name,source if name=='a' else exported['source']['sha256'])['missing_steps']
    assert service.calls == ['relax_structure','synthetic_accessibility','synthesis_recipes','relax_structure']
    a_calls = load(newfound.selection_path(tmp_path,'a',source))['calls']
    b_calls = load(newfound.selection_path(tmp_path,'b',exported['source']['sha256']))['calls']
    assert a_calls['structure'] != b_calls['structure'] and a_calls['recipes'] == b_calls['recipes']
    with pytest.raises(ValueError,match='source has changed'):
        newfound.screen(tmp_path,'a','0'*64)
    selection = newfound.selection_path(tmp_path,'a',source)
    rec = load(selection)
    rec['calls']['structure'] = rec['calls']['recipes']
    save(selection,rec)
    with pytest.raises(ValueError,match='another candidate'):
        newfound.view(tmp_path,'a',source)


def test_poll_existing_jobs_and_keep_failed_attempt(experiment,tmp_path,monkeypatch):
    source = catalog(tmp_path)
    service=Service()
    service.pending=True
    monkeypatch.setattr(newfound,'Newfound',lambda key:service)
    result=newfound.screen(tmp_path,'b',source)
    assert not result['jobs_pending']
    assert service.calls==['relax_structure','synthetic_accessibility','job_status','synthetic_accessibility','synthesis_recipes']
    class Offline:
        def call(self,*args):
            raise newfound.NewfoundError('Unavailable.')
    args={'formula':'Co'}
    assert newfound.cached_call(tmp_path,'synthesis_recipes',args,Offline())['status']=='error'
    assert newfound.cached_call(tmp_path,'synthesis_recipes',args,service,refresh=True)['status']=='error'
    assert service.calls.count('synthesis_recipes')==1


def test_workspace_read_never_submits_and_post_requires_same_origin(experiment,tmp_path,monkeypatch):
    source=catalog(tmp_path)
    service=Service()
    monkeypatch.setattr(newfound,'Newfound',lambda key:service)
    for editable in (False,True):
        server=make_server(tmp_path,0,editable=editable)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url='/api/synthesis'
            assert http_request(server,'GET',f'{url}?candidate=a&source={source}')[0]==200
            assert not service.calls
            body=json.dumps({'candidate':'a','source':source})
            headers={'Content-Type':'application/json','Origin':'https://elsewhere.invalid'}
            assert http_request(server,'POST',url,body,**headers)[0]==403
            headers['Origin']='http://127.0.0.1:'+str(server.server_port)
            status,_=http_request(server,'POST',url,body,**headers)
            assert status==(201 if editable else 403)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5)
