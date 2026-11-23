from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if modal.is_local() else Path('/opt/remag')
app=modal.App('siesta-remag-single-gpu')
volume=modal.Volume.from_name('siesta-remag-magnetic-results')
image=(modal.Image.from_id('im-ToLkc13u6TqJPGEHDbU4yd')
    .add_local_dir(HERE,'/opt/exchange-comparison',copy=True,ignore=['__pycache__','*.log','run-status.json']))

@app.function(image=image,cpu=8,memory=32768,timeout=900,max_containers=1,
    min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,volumes={'/results':volume})
def run(run_id):
    import os,json,time,subprocess,shutil,tarfile,io,hashlib
    root=Path('/results')/run_id;root.mkdir(exist_ok=False)
    gpu=[];diagnostic=None;started=time.perf_counter()
    with (root/'benchmark.log').open('w') as out:
        p=subprocess.run(['python','/opt/exchange-comparison/compare_exchange.py',str(root)],stdout=out,stderr=subprocess.STDOUT,timeout=720)
    summary=dict(status='complete' if p.returncode==0 else 'failed',benchmark_returncode=p.returncode,
        gpu=gpu,seconds=time.perf_counter()-started,startup_diagnostic_returncode=diagnostic,
        source_sha256={str(f.relative_to('/opt/exchange-comparison')):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path('/opt/exchange-comparison').glob('*.py')})
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with tarfile.open(root/'implementation.tar.gz','w:gz') as a:a.add('/opt/exchange-comparison',arcname='source')
    volume.commit()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as a:
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.log','.fdf'):a.add(f,arcname=str(f.relative_to(root)))
    return buffer.getvalue()

@app.local_entrypoint()
def main():
    import json,datetime,tarfile,io
    run_id='exchange-comparison-parallel-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (HERE/'run-status.json').write_text(json.dumps(dict(run_id=run_id,app_id=app.app_id),indent=2)+'\n')
    data=run.remote(run_id);out=ROOT/'results'/run_id;out.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as a:a.extractall(out,filter='data')
    print(out)
