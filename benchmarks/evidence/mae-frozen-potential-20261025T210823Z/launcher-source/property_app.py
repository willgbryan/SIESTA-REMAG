from pathlib import Path
import json
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1] if modal.is_local() else Path('/opt/remag')
PLAN=json.loads((HERE/'experiment.json' if modal.is_local() else Path('/opt/frozen-stage/experiment.json')).read_text())
if modal.is_local() and PLAN['authorized_gpu_seconds']!=1200:
    raise RuntimeError('Frozen-potential stage awaits explicit new 20-minute H100 authorization')
app=modal.App('siesta-remag-single-gpu')
volume=modal.Volume.from_name('siesta-remag-results')
image=(modal.Image.from_id(PLAN['runtime_image'])
    .add_local_dir(HERE,'/opt/frozen-stage',copy=True,ignore=['__pycache__','run-status.json','failed-build-1','*-allocation-release.json'])
    .add_local_dir(ROOT/'src/siesta_remag','/opt/remag/siesta_remag',copy=True))

@app.function(image=image,gpu='H100!:1',cpu=8,memory=65536,timeout=1200,
    max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,volumes={'/results':volume})
def benchmark(run,deadline_epoch):
    import math,os,re,shutil,subprocess,sys,tarfile,time,traceback
    sys.path[:0]=['/opt/remag','/opt/frozen-stage']
    from siesta_remag.attempts import prepare_attempt,complete_attempt,invocation_digest,package_evidence
    from siesta_remag.runner import run_case
    from siesta_remag.compare import compare,difference
    from siesta_remag.corpus import sha256
    from comparison_inputs import verify_convergence
    from verify_exports import compare_exports
    import analyze
    start=time.time();deadline=min(deadline_epoch,start+1200)-PLAN['reserve_seconds']
    assert PLAN['authorized_gpu_seconds']==1200 and deadline>start
    source=Path('/results')/PLAN['source_run']
    marker=json.loads((source/'run-attempt.json').read_text())
    assert marker['state']=='complete' and marker['attempt_id']==PLAN['source_attempt']
    assert json.loads((source/'summary.json').read_text())['status']=='validation_complete'
    root=Path('/results')/run
    if root.exists():raise RuntimeError('Preserve existing evidence; no automatic restart')
    exe=Path('/opt/remag-candidate/Src/siesta');lib=Path('/opt/hopper-build/libremag_gpu.so')
    build=json.loads(Path('/opt/frozen-stage/build.json').read_text())
    assert build==PLAN['accepted_runtime_build']
    assert sha256(exe)==build['candidate_executable_sha256'] and sha256(lib)==build['candidate_library_sha256']
    attempt,_=prepare_attempt(root,invocation_digest(PLAN,{'app':sha256(Path('/opt/frozen-stage/property_app.py'))},
        {str(exe):sha256(exe),str(lib):sha256(lib)}))
    def save(name,value):(root/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,uuid,driver_version','--format=csv,noheader'],text=True).splitlines()
    assert len(gpu)==1 and 'H100' in gpu[0] and '9.0' in gpu[0]
    save('metadata.json',dict(plan=PLAN,attempt=attempt,gpu=gpu,build=build,allocation_start_epoch=start,deadline_epoch=deadline_epoch))
    with tarfile.open(root/'implementation.tar.gz','w:gz') as tar:
        for folder in ('/opt/frozen-stage','/opt/hopper-native','/opt/remag/siesta_remag'):
            tar.add(folder,arcname=folder.removeprefix('/opt/'))
    shutil.copytree(source/'inputs',root/'inputs')
    for key in list(os.environ):
        if key.startswith('REMAG_'):os.environ.pop(key)
    ledger=[];checks=[];biases={};results={};probes={};modes={}
    summary=dict(status='running',physical_accuracy_qualified=False,speedup_over_upstream_gpu=None)
    def checkpoint():
        save('ledger.json',ledger);save('current-comparisons.json',checks);save('method-bias.json',biases);save('summary.json',summary);volume.commit()
    def run(name,role,material):
        if deadline-time.time()<90:raise TimeoutError('Shared stage budget/evidence reserve reached')
        inp=root/'inputs'/name;dest=root/name/role
        env={**PLAN['settings'],'REMAG_HOST_ID':attempt['attempt_id']}
        if role.startswith('frozen_'):
            env.update(REMAG_FROZEN_ROTATIONS='/opt/frozen-stage/'+material+'-rotations.txt',
                       REMAG_FROZEN_SPIN_EXPECTATIONS='1' if role=='frozen_spin' else '0')
        print('START',material,name,role,'remaining',round(deadline-time.time()),flush=True)
        result=run_case(inp,exe,dest,threads=1,timeout=min(500,deadline-time.time()),extra_env=env)
        conv=verify_convergence(dest,expected={'DM.Tolerance':1e-9,'SCF.DM.Tolerance':1e-9,'SCF.H.Tolerance':1e-8})
        cycles=result['telemetry'].get('gpu_device_cycle',{}).get('cycles',[])
        actual=re.findall(r'k-grid: Number of k-points\s*=\s*(\d+)',(dest/'stdout.log').read_text())
        nk=PLAN['materials'][material]['kpoints']
        mesh=bool(actual and int(actual[-1])==nk and len(cycles)==1 and cycles[0]['eigensystems']['jobs']==nk)
        moment=result['observables'].get('total_moment_muB')
        manifest=result['benchmark'];axis=manifest.get('direction_vector') or ([1,0,0] if name.endswith('_x') else [0,0,1])
        angle=None
        if moment is not None:
            norm=math.sqrt(sum(x*x for x in moment))*math.sqrt(sum(x*x for x in axis))
            if norm:angle=math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(moment,axis))/norm))))
        valid=bool(result['valid'] and conv['passed'] and mesh and angle is not None and angle<=1)
        ledger.append(dict(case=name,role=role,valid=valid,wall_seconds=result['wall_seconds'],
            convergence=conv,mesh_passed=mesh,direction_error_degrees=angle,parse_errors=result['parse_errors']))
        checkpoint()
        if not valid:raise RuntimeError('Fresh full-SCF reference failed')
        refdir=source/name/'energy_only';ref=json.loads((refdir/'result.json').read_text())
        reference=compare(ref,result,reference_dir=refdir,candidate_dir=dest,electronic_only=True)
        # These are historical outputs, and probes perform additional work.
        reference.update(end_to_end_speedup=None,performance_comparable=False,
                         performance_reasons=['Historical SCF regression; frozen probe adds work and exports'])
        exports=compare_exports(refdir,dest)
        checks.append(dict(case=name,role=role,reference_role='previous_energy_only',science=reference,exports=exports,
                           passed=reference['passed'] and exports['passed']))
        if material=='fept':
            cpu_dir=source/name/'cpu';cpu=json.loads((cpu_dir/'result.json').read_text())
            cpu_check=compare(cpu,result,reference_dir=cpu_dir,candidate_dir=dest,electronic_only=True)
            cpu_check.update(end_to_end_speedup=None,performance_comparable=False)
            checks.append(dict(case=name,role=role,reference_role='previous_pristine_cpu',science=cpu_check,passed=cpu_check['passed']))
        checkpoint()
        if not all(row['passed'] for row in checks):raise RuntimeError('Full-SCF science/export regression failed')
        if role.startswith('frozen_'):
            check=analyze.probe(dest,result,nk,role=='frozen_spin')
            save(material+'-'+role+'-analysis.json',check)
            checks.append(dict(case=name,role=role,reference_role='native_frozen_probe',passed=check['passed'],errors=check['errors']))
            checkpoint()
            if not check['passed']:raise RuntimeError('Frozen potential implementation or final-direction check failed')
            probes[material,role]=check['report']
        print('DONE',name,role,flush=True)
        return result
    checkpoint()
    try:
        for material,p in PLAN['materials'].items():
            anchors=[]
            anchors.append(run(p['reference'],'frozen_spin',material))
            for name in p['axes'][1:]:anchors.append(run(name,'full_scf',material))
            # Direct directional regression against the preserved full-SCF points.
            refs=[json.loads((source/name/'energy_only/result.json').read_text()) for name in p['axes']]
            for i in range(1,len(anchors)):
                for key in ('total_energy_ev','free_energy_ev'):
                    check=difference(refs[i]['observables'][key]-refs[0]['observables'][key],
                                     anchors[i]['observables'][key]-anchors[0]['observables'][key],5e-6)
                    checks.append(dict(material=material,axis=i,quantity=key,**check))
            checkpoint()
            if not all(x['passed'] for x in checks):raise RuntimeError('Full-SCF directional regression failed')
            biases[material]=analyze.bias(probes[material,'frozen_spin'],anchors);checkpoint()
            run(p['reference'],'frozen_values',material)
            modes[material]=analyze.compare_modes(root/p['reference']/'frozen_spin',root/p['reference']/'frozen_values')
            save('eigenvalues-only-comparisons.json',modes)
            if not modes[material]['passed']:raise RuntimeError('Eigenvalues-only directional comparison failed')
        summary.update(status='validation_complete',stopping_reason='Implementation checks and selected full-SCF method-bias measurements completed; physical convergence remains open')
    except Exception as exc:
        summary.update(status='budget_limited' if isinstance(exc,TimeoutError) else 'validation_failed',stopping_reason=str(exc),exception=traceback.format_exc())
        print(summary['exception'],flush=True)
    summary['elapsed_seconds']=time.time()-start
    checkpoint();complete_attempt(root);volume.commit()
    return package_evidence(root)

@app.local_entrypoint()
def main():
    import datetime,io,tarfile,time
    status_path=HERE/'run-status.json';status=json.loads(status_path.read_text())
    if status['status']!='approved' or status.get('run_id'):raise RuntimeError('Explicit authorization and fresh attempt required')
    if status['deadline_epoch'] is None:
        status['deadline_epoch']=time.time()+PLAN['authorized_gpu_seconds']
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    if status['deadline_epoch']-time.time()<360:raise RuntimeError('Insufficient remaining approved budget')
    run='mae-frozen-potential-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    status.update(status='launching',run_id=run,app_id=app.app_id,allocation_released=False)
    status_path.write_text(json.dumps(status,indent=2)+'\n');print('Run ID:',run,flush=True)
    call=benchmark.spawn(run,status['deadline_epoch']);status.update(status='running',function_call_id=call.object_id)
    status_path.write_text(json.dumps(status,indent=2)+'\n');print('Function:',call.object_id,flush=True)
    payload=call.get();target=ROOT/'results'/run;target.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(target,filter='data')
    status.update(status='analysis_pending',results=str(target));status_path.write_text(json.dumps(status,indent=2)+'\n')
    print('Results:',target,flush=True)
