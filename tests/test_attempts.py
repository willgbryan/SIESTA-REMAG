"""Cloud rescheduling must preserve evidence and isolate hardware attempts."""
import io
import json
import tarfile
import pytest
from siesta_remag.attempts import prepare_attempt,complete_attempt,package_evidence,invocation_digest,MARKER

def test_interrupted_attempts_are_preserved_and_completed_response_repackages(tmp_path):
    root=tmp_path/'run';identity=invocation_digest({'cases':['a']},{'source':'hash'},{'binary':'hash'})
    first,done=prepare_attempt(root,identity);assert not done
    (root/'ledger.json').write_text('[{"passed":true}]')
    second,done=prepare_attempt(root,identity);assert not done and first['attempt_id']!=second['attempt_id']
    assert not (root/'ledger.json').exists()
    old=tmp_path/second['interrupted_attempts'][0]
    assert json.loads((old/'ledger.json').read_text())==[{'passed':True}]
    # A worker can disappear after the rename but before preparing its successor.
    preserved=root.with_name(root.name+'-interrupted-'+second['attempt_id']);root.rename(preserved)
    third,done=prepare_attempt(root,identity);assert len(third['interrupted_attempts'])==2
    (root/'ledger.json').write_text('[{"passed":false}]')
    complete_attempt(root);completed,done=prepare_attempt(root,identity)
    assert done and completed['attempt_id']==third['attempt_id']
    with tarfile.open(fileobj=io.BytesIO(package_evidence(root))) as archive:
        assert json.load(archive.extractfile('ledger.json'))==[{'passed':False}]
        assert json.load(archive.extractfile(MARKER))['state']=='complete'
        assert not any('interrupted' in name for name in archive.getnames())

def test_run_identity_collision_preserves_existing_outputs(tmp_path):
    root=tmp_path/'run';prepare_attempt(root,'first');(root/'evidence.json').write_text('original')
    with pytest.raises(ValueError,match='different invocation'):prepare_attempt(root,'other')
    assert (root/'evidence.json').read_text()=='original'
    (root/MARKER).unlink()
    with pytest.raises(ValueError,match='lacks an attempt identity'):prepare_attempt(root,'first')
    assert (root/'evidence.json').read_text()=='original'

def test_identity_includes_source_binary_and_parameters():
    baseline=invocation_digest({'x':1},{'a':'b'},{'c':'d'})
    assert all(value!=baseline for value in (
        invocation_digest({'x':2},{'a':'b'},{'c':'d'}),
        invocation_digest({'x':1},{'a':'other'},{'c':'d'}),
        invocation_digest({'x':1},{'a':'b'},{'c':'other'})))
