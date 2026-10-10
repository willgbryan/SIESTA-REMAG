"""A Modal retry must not silently combine scientific artifacts across attempts."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_live_fetch_rejects_attempt_rollover_and_tainted_reuse(tmp_path):
    stub = tmp_path/'stub'
    stub.mkdir()
    # Mirror the asynchronous SDK namespace used by the strict block reader.
    # No file downloads or real Modal credentials are needed for attempt checks.
    package = stub/'modal'
    (package/'_utils').mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'_utils'/'__init__.py').write_text('')
    (package/'_utils'/'http_utils.py').write_text('class ClientSessionRegistry: pass\n')
    (package/'exception.py').write_text('class ExecutionError(Exception): pass\n')
    (stub/'modal_proto.py').write_text('class api_pb2: pass\n')
    (package/'volume.py').write_text('''
import json,os
class _Volume:
    count=0
    async def hydrate(self): pass
    async def read_file(self,path):
        self.count+=1
        changed=os.environ.get('FAKE_ROLLOVER')=='1' and self.count>1
        yield json.dumps(dict(attempt_id=('b' if changed else 'a')*32,
            invocation_sha256='same-inputs',state='running')).encode()
    async def listdir(self,*args,**kwargs):return []
    @classmethod
    def from_name(cls,name):return cls()
def retry(**kwargs): return lambda fn: fn
def unexpected_download(*args,**kwargs):
    raise AssertionError('The attempt-only fixture must not download blocks')
_expected_block_lengths = _read_block_body = unexpected_download
_raise_on_block_response_error = _block_download_timeout = unexpected_download
''')
    script = Path(__file__).resolve().parents[1]/'scripts/fetch_evidence.py'
    target = tmp_path/'checkpoint'
    command = [sys.executable,str(script),'test-run','--sources-only',
               '--attempt-id','a'*32,'--output',str(target)]
    env = dict(os.environ,PYTHONPATH=str(stub),FAKE_ROLLOVER='1')
    first = subprocess.run(command,env=env,capture_output=True,text=True)
    assert first.returncode != 0 and 'Remote attempt changed' in first.stderr
    evidence = target/'fetch-attempt.json'
    assert json.loads(evidence.read_text())['consistent'] is False
    before = evidence.read_bytes()
    env['FAKE_ROLLOVER'] = '0'
    retry = subprocess.run(command,env=env,capture_output=True,text=True)
    assert retry.returncode != 0 and 'Local checkpoint crossed attempts' in retry.stderr
    assert evidence.read_bytes() == before
    clean = tmp_path/'clean'
    command[-1] = str(clean)
    result = subprocess.run(command,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((clean/'fetch-attempt.json').read_text())['consistent'] is True
