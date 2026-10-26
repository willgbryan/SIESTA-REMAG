from pathlib import Path
import json
import time
import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/opt/remag')
PLAN = json.loads((HERE/'experiment.json' if modal.is_local() else Path('/opt/energy-sweep-stage/experiment.json')).read_text())
if modal.is_local():
    status = json.loads((HERE/'run-status.json').read_text())
    if (PLAN['authorized_gpu_seconds'] != 1200 or status['status'] != 'approved' or
        status['authorized_gpu_seconds'] != 1200 or status.get('run_id') or
        not status.get('deadline_epoch') or status['deadline_epoch']-time.time() < 360):
        raise RuntimeError('Fresh 20-minute H100 authorization and fixed deadline required before image build')
app = modal.App('siesta-remag-single-gpu')
volume = modal.Volume.from_name('siesta-remag-results')
image = (modal.Image.from_id(PLAN['base_image'])
    .add_local_dir(HERE, '/opt/energy-sweep-stage', copy=True, ignore=['__pycache__','run-status.json'])
    .run_commands('python /opt/energy-sweep-stage/rebuild.py')
    .add_local_dir(ROOT/'src/siesta_remag', '/opt/remag/siesta_remag', copy=True))

@app.function(image=image, gpu='H100!:1', cpu=8, memory=65536, timeout=1200,
    max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
    retries=0, volumes={'/results':volume})
def benchmark(run_id, deadline_epoch):
    import copy, os, platform, shutil, signal, subprocess, sys, tarfile, threading, traceback
    import psutil
    sys.path[:0] = ['/opt/remag', '/opt/energy-sweep-stage']
    from siesta_remag.attempts import prepare_attempt, complete_attempt, invocation_digest, package_evidence
    from siesta_remag.corpus import sha256
    from siesta_remag.runner import run_case
    import analyze
    stage = Path('/opt/energy-sweep-stage')
    assert all(sha256(stage/name) == digest for name,digest in PLAN['diagnostic_sources'].items())
    start = time.time(); deadline = min(deadline_epoch, start+1200)-PLAN['reserve_seconds']
    assert PLAN['authorized_gpu_seconds'] == 1200 and deadline-start >= 90
    source = Path('/results')/PLAN['source_run']
    marker = analyze.read(source/'run-attempt.json')
    assert marker['state'] == 'complete' and marker['attempt_id'] == PLAN['source_attempt']
    assert analyze.read(source/'summary.json')['status'] == 'validation_complete'
    root = Path('/results')/run_id
    if root.exists():
        raise RuntimeError('Preserve the prior attempt; no automatic rerun')
    exe = Path('/opt/remag-candidate/Src/siesta'); lib = Path('/opt/hopper-build/libremag_gpu.so')
    build = analyze.read(stage/'build.json')
    assert sha256(exe) == build['candidate_executable_sha256'] and sha256(lib) == build['candidate_library_sha256']
    assert build['native_sources'] == PLAN['candidate_native_sources']
    attempt, _ = prepare_attempt(root, invocation_digest(PLAN, PLAN['diagnostic_sources'], {str(exe):sha256(exe),str(lib):sha256(lib)}))
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,uuid,driver_version','--format=csv,noheader'],text=True).splitlines()
    assert len(gpu) == 1 and 'H100' in gpu[0] and '9.0' in gpu[0]
    hardware = dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
                    blas_threads=1, hostname=platform.node(), runtime_instance_id=attempt['attempt_id'])
    analyze.save(root/'metadata.json', dict(plan=PLAN, attempt=attempt, gpu=gpu, build=build,
        allocation_start_epoch=start, deadline_epoch=deadline_epoch, hardware=hardware))
    with tarfile.open(root/'implementation.tar.gz','w:gz') as tar:
        for folder in (stage, Path('/opt/hopper-native'), Path('/opt/remag/siesta_remag')):
            tar.add(folder, arcname=str(folder).removeprefix('/opt/'))
        for name in build['candidate_fortran']:
            tar.add('/opt/remag-candidate-siesta-src/Src/'+name, arcname='candidate-fortran/'+name)
    shutil.copytree(source/'inputs', root/'inputs')
    analyze.save(root/'input-family-verification.json', analyze.verify_inputs(root/'inputs', PLAN))
    for key in list(os.environ):
        if key.startswith('REMAG_'):
            os.environ.pop(key)
    env = {**os.environ, **PLAN['settings'], 'REMAG_HOST_ID':attempt['attempt_id']}
    assert not any(key.startswith('REMAG_FROZEN_') for key in env)
    linked = subprocess.check_output(['ldd',str(exe)],env=env,text=True)
    assert str(lib) in linked and 'not found' not in linked
    (root/'linked-libraries.log').write_text(linked)
    ledger = []; checks = []; paired = []; results = {}
    summary = dict(status='running', physical_accuracy_qualified=False, speedup_over_upstream_gpu=None)
    def checkpoint():
        for name,value in [('ledger.json',ledger),('current-comparisons.json',checks),('paired-timings.json',paired),('summary.json',summary)]:
            analyze.save(root/name,value)
        volume.commit()

    def execute_sweep(spec, dest, seconds):
        names, directions = analyze.axes(spec, PLAN)
        shutil.copytree(root/'inputs'/names[0], dest)
        text = (dest/'input.fdf').read_text()
        text += '\nDM.AllowReuse false\nREMAG.ReuseOrientationGeometry true\n%block REMAG.SpinRotations\n'
        text += '\n'.join(' '.join(format(v,'.17g') for v in q) for q in analyze.rotations(directions))
        text += '\n%endblock REMAG.SpinRotations\n'
        (dest/'input.fdf').write_text(text)
        manifest = analyze.read(dest/'manifest.json')
        manifest.update(input_sha256=sha256(dest/'input.fdf'), spin_rotation_directions=directions)
        analyze.save(dest/'manifest.json', manifest)
        # Match run_case's 100-ms CPU and 200-ms GPU observers and timeout policy.
        samples=[]; gpu_lines=[]; timed_out=False
        with (dest/'input.fdf').open('rb') as inp, (dest/'stdout.log').open('wb') as out, (dest/'stderr.log').open('wb') as err:
            t0=time.perf_counter()
            proc=subprocess.Popen([str(exe)],stdin=inp,stdout=out,stderr=err,cwd=dest,env=env,start_new_session=True)
            monitor=psutil.Process(proc.pid); monitor.cpu_percent()
            observer=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used',
                '--format=csv,nounits','--loop-ms=200'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            def drain():
                with observer.stdout:
                    gpu_lines.extend(observer.stdout)
            reader=threading.Thread(target=drain,daemon=True); reader.start()
            try:
                while proc.poll() is None:
                    elapsed=time.perf_counter()-t0
                    try:
                        cpu=monitor.cpu_times()
                        samples.append(dict(seconds=elapsed,cpu_percent=monitor.cpu_percent(),rss_bytes=monitor.memory_info().rss,cpu_seconds=cpu.user+cpu.system))
                    except psutil.NoSuchProcess:
                        break
                    if elapsed>seconds:
                        timed_out=True; os.killpg(proc.pid,signal.SIGTERM)
                        try: proc.wait(5)
                        except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
                        break
                    time.sleep(.1)
                code=proc.wait(); wall=time.perf_counter()-t0
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                observer.terminate(); observer.wait(); reader.join()
                (dest/'gpu.csv').write_bytes(b''.join(gpu_lines))
        execution=dict(returncode=code,timed_out=timed_out,wall_seconds=wall,hardware=hardware,
            executable=dict(path=str(exe),sha256=sha256(exe)),library_sha256=sha256(lib),
            executed_input_sha256=manifest['input_sha256'],cpu_samples=samples,
            timing_scope='Whole SIESTA process including requested exports; same observers as cold run_case')
        analyze.save(dest/'execution.json',execution)
        # Publish raw completion independently of parsing, including partials.
        analyze.save(dest/'result.json',dict(valid=False,execution=execution,status='analysis_pending'))
        volume.commit()
        receipt,outputs,paths=analyze.sweep(dest,spec,PLAN,root/'inputs',create_views=True)
        analyze.save(dest/'sweep-analysis.json',receipt)
        analyze.save(dest/'result.json',dict(valid=receipt['passed'],execution=execution,analysis=receipt,status='collected'))
        return outputs,paths,receipt['passed'],wall

    checkpoint()
    try:
        for spec in PLAN['cases']:
            if deadline-time.time()<90:
                raise TimeoutError('Calculation budget reached; preserve evidence reserve')
            label=spec['label']; dest=root/label
            names,directions=analyze.axes(spec,PLAN)
            row=dict(**spec,status='started',passed=False,wall_seconds=None)
            ledger.append(row); checkpoint()
            print('START',label,'remaining',round(deadline-time.time()),flush=True)
            if spec['role']=='cold':
                result=run_case(root/'inputs'/names[0],exe,dest,threads=1,timeout=min(420,deadline-time.time()),extra_env=env)
                outputs,paths,passed,wall=[result],[dest],result['valid'],result['wall_seconds']
                control=analyze.cold_checks(dest,result,directions[0],PLAN['materials'][spec['material']]['kpoints'])
                row.update(control=control)
                passed=passed and control['passed']
            else:
                outputs,paths,passed,wall=execute_sweep(spec,dest,min(600,deadline-time.time()))
            row.update(status='executed',passed=passed,wall_seconds=wall)
            checkpoint()
            references=[]
            for name,result,path in zip(names,outputs,paths):
                refdir=source/name/'energy_only'; reference=analyze.read(refdir/'result.json'); references.append(reference)
                check=analyze.regression(reference,result,refdir,path)
                checks.append(dict(label=label,case=name,reference_role='historical_energy_only',**check))
                if spec['material']=='fept':
                    cpu=source/name/'cpu'
                    checks.append(dict(label=label,case=name,reference_role='historical_pristine_cpu',
                        **analyze.regression(analyze.read(cpu/'result.json'),result,cpu,path)))
            diffs=analyze.directional(references,outputs)
            row.update(directional=diffs,passed=passed and all(c['passed'] for c in checks) and all(d['passed'] for d in diffs))
            results[label]=(outputs,paths)
            checkpoint()
            print('DONE',label,'passed',row['passed'],'seconds',round(wall,3),flush=True)
            if not row['passed']:
                raise RuntimeError('Scientific, export, direction or owner validation failed: '+label)
        # Compare matched counts from the same allocation and executable only.
        for material in PLAN['materials']:
            for rep in sorted({s['repetition'] for s in PLAN['cases'] if s['material']==material and s['repetition'] is not None}):
                cold=[r for r in ledger if r['material']==material and r['repetition']==rep and r['role']=='cold']
                hot=next(r for r in ledger if r['material']==material and r['repetition']==rep and r['role']=='sweep')
                cold.sort(key=lambda r:r['axis_indices'])
                refs=[results[r['label']][0][0] for r in cold]
                output,paths=results[hot['label']]
                direct=[analyze.regression(ref,result,results[row['label']][1][0],path)
                        for ref,result,row,path in zip(refs,output,cold,paths)]
                diffs=analyze.directional(refs,output)
                valid=all(c['passed'] for c in direct+diffs) and all(r['passed'] for r in cold+[hot])
                cold_seconds=sum(r['wall_seconds'] for r in cold)
                paired.append(dict(material=material,repetition=rep,passed=valid,science=direct,directional=diffs,
                    cold_seconds=cold_seconds,sweep_seconds=hot['wall_seconds'],
                    incremental_ratio=cold_seconds/hot['wall_seconds'] if valid else None,
                    upstream_gpu_speedup=None,scope='Matched requested electronic outputs; independent atomic guesses; same-process geometry reuse'))
        checkpoint()
        if not all(r['passed'] for r in paired):
            raise RuntimeError('Fresh cold-versus-sweep comparison failed')
        summary.update(status='validation_complete',stopping_reason='Prepared correctness and matched timing batch completed')
    except Exception as exc:
        summary.update(status='budget_limited' if isinstance(exc,TimeoutError) else 'validation_failed',
            stopping_reason=str(exc),exception=traceback.format_exc())
        print(summary['exception'],flush=True)
    summary['elapsed_seconds']=time.time()-start
    checkpoint(); complete_attempt(root); volume.commit()
    return package_evidence(root)

@app.local_entrypoint()
def main():
    import datetime,io,tarfile
    status_path=HERE/'run-status.json'; status=json.loads(status_path.read_text())
    if status['status']!='approved' or status.get('run_id') or status['deadline_epoch']-time.time()<360:
        raise RuntimeError('Fresh authorization and sufficient remaining shared budget required')
    run='mae-energy-sweep-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    status.update(status='launching',run_id=run,app_id=app.app_id,allocation_released=False)
    status_path.write_text(json.dumps(status,indent=2)+'\n'); print('Run ID:',run,flush=True)
    call=benchmark.spawn(run,status['deadline_epoch'])
    status.update(status='running',function_call_id=call.object_id)
    status_path.write_text(json.dumps(status,indent=2)+'\n'); print('Function:',call.object_id,flush=True)
    payload=call.get(); target=ROOT/'results'/run; target.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        archive.extractall(target,filter='data')
    status=json.loads(status_path.read_text()); status.update(status='analysis_pending',results=str(target))
    status_path.write_text(json.dumps(status,indent=2)+'\n'); print('Results:',target,flush=True)
