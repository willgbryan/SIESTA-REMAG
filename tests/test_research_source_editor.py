import json
import threading
import pytest
from siesta_remag.research.authoring import register_structure, workspace_source, verify_candidate
from siesta_remag.research.workspace import make_server, dataset
from siesta_remag.discovery.common import load
from siesta_remag.corpus import sha256
from test_research import experiment
from test_research_authoring import http_request


@pytest.fixture
def editor(experiment,tmp_path):
    register_structure(tmp_path,'bundle/structure.json','parent','Parent')
    source=load(experiment/'structure.json')['source']
    # Generate a valid exact source using the same GPAC operation Astra uses.
    from gpac.physical import isotropic_scale
    child=isotropic_scale(source['program'],'101/100',expected_sha256=source['sha256'])['child']['program']
    return dict(source_id='parent',expected_source_sha256=source['sha256'],program=child)


def test_preview_is_read_only_and_save_is_an_independent_exact_root(experiment,tmp_path,editor):
    before={str(p):sha256(p) for p in tmp_path.rglob('*') if p.is_file() and p.suffix!='.lock'}
    preview=workspace_source(tmp_path,editor)
    assert preview['saved'] is False and preview['calculation_performed'] is False
    assert {str(p):sha256(p) for p in tmp_path.rglob('*') if p.is_file() and p.suffix!='.lock'}==before
    result=workspace_source(tmp_path,editor|dict(label='Design'),save_source=True)
    record=verify_candidate(tmp_path/result['entry']['candidate'])
    assert record['source_sha256']==preview['structure']['source']['sha256']
    assert record['parent_sha256'] is None  # No invented edit mapping from free source text.
    state=dataset(tmp_path);row=state['families'][-1]['rows'][0]
    assert row['structure']['source']['program']==editor['program']
    assert row['calculations']==[] and row['result'] is None
    for path,h in before.items():
        if not path.endswith('/workspace.json'):assert sha256(path)==h


@pytest.mark.parametrize('change',[{'program':'not a program'},{'expected_source_sha256':'f'*64},{'source_id':'absent'},{'program':'a'*32769},{'output':'../elsewhere'},{'program':1.5}])
def test_invalid_source_does_not_create_a_candidate(experiment,tmp_path,editor,change):
    catalog=sha256(tmp_path/'workspace.json')
    with pytest.raises(Exception):workspace_source(tmp_path,editor|change|dict(label='No'),save_source=True)
    assert sha256(tmp_path/'workspace.json')==catalog and not (tmp_path/'candidates').exists()


@pytest.mark.parametrize('editable',[False,True])
def test_source_http_guards_and_exact_preview(experiment,tmp_path,editor,editable):
    server=make_server(tmp_path,0,editable=editable);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    origin=f'http://127.0.0.1:{server.server_port}'
    headers={'Origin':origin,'Content-Type':'application/json'}
    try:
        assert http_request(server,'GET','/ide')[0]==200
        assert http_request(server,'POST','/api/source/preview',json.dumps(editor),**{'Content-Type':'application/json'})[0]==403
        code,body=http_request(server,'POST','/api/source/preview',json.dumps(editor),**headers)
        if editable:assert code==201 and json.loads(body)['result']['saved'] is False
        else:assert code==403
        assert not (tmp_path/'candidates').exists()
    finally:server.shutdown();server.server_close();thread.join()
