"""Read-only activity projection: receipt binding, confinement and real refresh."""
import hashlib
import json
import threading

import pytest

from siesta_remag.research.activity import activity_state
from siesta_remag.research.workspace import make_server
from test_research_authoring import http_request


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recorded(tmp_path):
    write(tmp_path/'research-activity.json', dict(format='REMAG-RESEARCH-ACTIVITY/1',
        status='Stopped', policies=[dict(label='Physics', calls='policy/calls', status='Sealed')]))
    request=tmp_path/'policy/calls/000/request.json'
    checksum=write(request, dict(format='REMAG-DIRECT-TOOL-CALL/1',
        request=dict(tool='record_response_prediction', rationale='<script>public note</script>', args=dict(value=.1)),
        model_context='MUST NOT DISCLOSE', implementation_sha256={'secret-path':'a'}))
    return tmp_path, request, checksum


def test_read_only_projection_retains_pending_then_bound_completion(recorded):
    root, request, checksum=recorded
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    state=activity_state(root);action=state['policies'][0]['actions'][0]
    assert action['status']=='pending' and action['record']==dict(value=.1)
    assert 'MUST NOT DISCLOSE' not in json.dumps(state) and 'secret-path' not in json.dumps(state)
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    write(request.with_name('response.json'), dict(format='REMAG-DIRECT-TOOL-RESULT/1',
        request_sha256=checksum, ok=True, result={'private':'not shown'}))
    state=activity_state(root)
    assert state['policies'][0]['actions'][0]['status']=='passed'
    assert 'private' not in json.dumps(state)
    request.write_text(request.read_text()+' ')
    with pytest.raises(ValueError,match='another request'):activity_state(root)


def test_non_notebook_arguments_not_exposed(recorded):
    root,request,_=recorded
    write(request,dict(format='REMAG-DIRECT-TOOL-CALL/1',request=dict(tool='extension_tool',args={'token':'secret'})))
    assert 'secret' not in json.dumps(activity_state(root))


def test_missing_and_escaping_or_oversized_activity(tmp_path):
    assert activity_state(tmp_path)==dict(available=False)
    manifest=tmp_path/'research-activity.json'
    write(manifest,dict(format='REMAG-RESEARCH-ACTIVITY/1',policies=[dict(label='Bad',calls='../outside')]))
    with pytest.raises(ValueError,match='inside'):activity_state(tmp_path)
    manifest.write_bytes(b' '*262145)
    with pytest.raises(ValueError,match='256 KiB'):activity_state(tmp_path)


def test_symlink_cannot_disclose_external_receipt(recorded,tmp_path_factory):
    root,request,_=recorded
    outside=tmp_path_factory.mktemp('outside')/'request.json';outside.write_text(request.read_text())
    request.unlink();request.symlink_to(outside)
    with pytest.raises(ValueError,match='inside'):activity_state(root)


def test_live_endpoint_refresh_and_no_host_or_write_bypass(recorded):
    root,request,checksum=recorded
    # Avoid unrelated default-family generation; this endpoint only needs its explicit receipts.
    write(root/'workspace.json',{})
    server=make_server(root,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        status,raw=http_request(server,'GET','/api/activity')
        assert status==200 and json.loads(raw)['status']=='Stopped'
        write(request.with_name('response.json'),dict(format='REMAG-DIRECT-TOOL-RESULT/1',request_sha256=checksum,ok=False))
        status,raw=http_request(server,'GET','/api/activity')
        assert status==200 and json.loads(raw)['policies'][0]['actions'][0]['status']=='failed'
        assert http_request(server,'GET','/api/activity',Host='evil.invalid')[0]==403
        assert http_request(server,'POST','/api/activity')[0]==404
        request.write_text('{')
        assert http_request(server,'GET','/api/activity')[0]==503
    finally:server.shutdown();server.server_close();thread.join()
