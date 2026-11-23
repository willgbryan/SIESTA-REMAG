"""One bounded authored magnetic screen; invoked only by scripts/run_modal.py."""
from pathlib import Path
import json
import os
import modal

REPO = Path(__file__).resolve().parents[1]
RUNTIME_FILE = Path(os.environ.get('REMAG_MAGNETIC_RUNTIME','/opt/magnetic-job/runtime.json'))
REQUEST_FILE = Path(os.environ.get('REMAG_MAGNETIC_REQUEST','/opt/magnetic-job/request.json'))
runtime = json.loads(RUNTIME_FILE.read_text())
app = modal.App('siesta-remag-single-gpu')
volume = modal.Volume.from_name(runtime['volume'])
templates = modal.Volume.from_name(runtime['template_volume'])
image = (modal.Image.from_id(runtime['image'])
    .add_local_dir(REPO/'src/siesta_remag','/opt/remag/siesta_remag',copy=True,ignore=['__pycache__'])
    .add_local_file(RUNTIME_FILE,'/opt/magnetic-job/runtime.json',copy=True)
    .add_local_file(REQUEST_FILE,'/opt/magnetic-job/request.json',copy=True)
    .env({'PYTHONPATH':'/opt/remag','REMAG_MAGNETIC_RUNTIME':'/opt/magnetic-job/runtime.json',
          'REMAG_MAGNETIC_REQUEST':'/opt/magnetic-job/request.json'}))


@app.function(image=image,gpu='A100:1',cpu=8,memory=32768,
    timeout=runtime['timeout_seconds'],max_containers=1,min_containers=0,
    buffer_containers=0,scaledown_window=2,retries=0,
    volumes={'/results':volume,'/templates':templates})
def run(run_id):
    import hashlib
    import io
    import subprocess
    import tarfile
    import time
    import traceback
    from siesta_remag.attempts import prepare_attempt, complete_attempt, invocation_digest
    from siesta_remag.magnetic.workflow import run_screen
    root=Path('/results')/run_id
    request=json.loads(Path('/opt/magnetic-job/request.json').read_text())
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in Path('/opt/remag/siesta_remag/magnetic').glob('*.py')}
    prepare_attempt(root,invocation_digest(request,sources,{}))
    with tarfile.open(root/'implementation.tar.gz','w:gz') as archive:
        archive.add('/opt/remag/siesta_remag',arcname='siesta_remag')
    started=time.perf_counter()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).splitlines()
    if len(gpu)!=1 or 'A100' not in gpu[0]:raise RuntimeError('Expected exactly one A100')
    try:
        configured=dict(runtime,template=str(Path('/templates')/runtime['template_relative_path']))
        result=run_screen(request,configured,root,checkpoint=volume.commit)
    except Exception as error:
        ledger=json.loads((root/'ledger.json').read_text()) if (root/'ledger.json').is_file() else []
        converged=ledger[0]['valid'] if ledger else None
        result=dict(format='REMAG-MAGNETIC-RUN/1',status='failed',error=str(error),
                    traceback=traceback.format_exc(),reference_converged=converged,screens=[])
        print(result['traceback'],flush=True)
    result.update(run_id=run_id,volume=runtime['volume'],image=runtime['image'],
                  gpu=gpu,sources=sources,seconds=time.perf_counter()-started)
    (root/'magnetic-run.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    complete_attempt(root);volume.commit()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.log','.fdf'):
                archive.add(path,arcname=str(path.relative_to(root)))
    return buffer.getvalue()


@app.local_entrypoint()
def main():
    import datetime
    import io
    import tarfile
    import uuid
    output=Path(os.environ['REMAG_MAGNETIC_OUTPUT'])
    run_id='magnetic-screen-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    (output/'owned-app.json').write_text(json.dumps(dict(app_id=app.app_id,run_id=run_id,volume=runtime['volume']),indent=2)+'\n')
    data=run.remote(run_id)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        archive.extractall(output,filter='data')
    print('Collected magnetic screen:',output,flush=True)
