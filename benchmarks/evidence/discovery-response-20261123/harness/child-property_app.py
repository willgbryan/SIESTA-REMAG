from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if modal.is_local() else Path('/opt/remag')
app=modal.App('siesta-remag-single-gpu')
volume=modal.Volume.from_name('siesta-remag-magnetic-results')
history=modal.Volume.from_name('siesta-remag-results')
image=(modal.Image.from_id('im-v2SJzLdYwhBNWghlBeQgMq').apt_install('gdb')
    .add_local_dir(HERE,'/opt/discovery-stage',copy=True,ignore=['__pycache__','*.log','run-status.json'])
    
    .add_local_dir(HERE/'python/siesta_remag','/opt/remag/siesta_remag',copy=True))
@app.function(image=image,gpu='A100:1',cpu=8,memory=32768,timeout=900,max_containers=1,min_containers=0,
    buffer_containers=0,scaledown_window=2,retries=0,volumes={'/results':volume,'/history':history})
def run(run_id):
    import sys,json,time,traceback,subprocess,hashlib,tarfile
    sys.path.insert(0,'/opt/discovery-stage')
    import material_response
    from siesta_remag.attempts import prepare_attempt,complete_attempt,package_evidence,invocation_digest
    root=Path('/results')/run_id
    if root.exists():raise RuntimeError('Existing attempt cannot be overwritten')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/opt/remag/siesta_remag/magnetic').glob('*.py')}
    prepare_attempt(root,invocation_digest({'purpose':'GPAC FePt dense MFT and real-space exchange'},sources,{}))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).splitlines()
    assert len(gpu)==1 and 'A100' in gpu[0]
    start=time.perf_counter()
    with tarfile.open(root/'implementation.tar.gz','w:gz') as tar:tar.add('/opt/discovery-stage',arcname='source')
    try:
        report=material_response.run(root,volume);summary=dict(status='complete',**report)
    except Exception as exc:
        summary=dict(status='failed',error=str(exc),traceback=traceback.format_exc());print(summary['traceback'],flush=True)
    summary.update(seconds=time.perf_counter()-start,gpu=gpu,sources=sources)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    complete_attempt(root);volume.commit();
    import io
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.log','.fdf'):
                archive.add(path,arcname=str(path.relative_to(root)))
    return buffer.getvalue()
@app.local_entrypoint()
def main():
    import json,datetime,io,tarfile
    run_id='discovery-child-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (HERE/'run-status.json').write_text(json.dumps(dict(run_id=run_id,app_id=app.app_id),indent=2)+'\n')
    data=run.remote(run_id);target=ROOT/'results'/run_id;target.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:archive.extractall(target,filter='data')
    print('Results:',target,flush=True)
