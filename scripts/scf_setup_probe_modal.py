"""Operator-only, finite setup diagnostic against a retained native library."""
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import modal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src') if modal.is_local() else '/opt/remag-probe/python')
from siesta_remag.discovery.common import load,save,digest
from siesta_remag.research.budget import BudgetLedger
from siesta_remag.research.scf_setup_probe import validate_policy,summarize_case

if modal.is_local():
    folder=Path(os.environ['REMAG_SETUP_PROBE']).resolve()
    policy=validate_policy(load(folder/'input/policy.json'))
    for name,sha in policy['files'].items():
        if hashlib.sha256((folder/'input'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Changed frozen diagnostic input: '+name)
    if hashlib.sha256((ROOT/'src/siesta_remag/research/scf_setup_probe.py').read_bytes()).hexdigest()!=policy['helper_sha256']:
        raise ValueError('Changed diagnostic helper')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=policy['module_sha256']:
        raise ValueError('Changed diagnostic module')
    budget=BudgetLedger(folder/'budget.json')
    if budget.remaining_cents()<policy['reservation_cents']:raise ValueError('Insufficient diagnostic reservation')
    image=(modal.Image.from_id(policy['image'])
        .add_local_dir(folder/'input','/opt/remag-probe/input',copy=True)
        .add_local_dir(ROOT/'src','/opt/remag-probe/python',copy=True,ignore=['__pycache__']))
else:
    policy=validate_policy(load('/opt/remag-probe/input/policy.json'));image=None
app=modal.App('siesta-remag-single-gpu')
volume=modal.Volume.from_name(policy['volume'],create_if_missing=False)

@app.function(image=image,gpu='A100-80GB:1',cpu=4,memory=16384,
    timeout=420,max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,volumes={'/results':volume})
def probe(run_id):
    import subprocess,time,resource
    from siesta_remag.discovery.collection import pack
    root=Path('/results')/run_id;root.mkdir(exist_ok=False)
    inp=Path('/opt/remag-probe/input');runtime=load(inp/'runtime.json')
    report={'status':'running','scope':'Synthetic SCF graph setup only; no cycle launch or material/reference calculation',
        'invocation_sha256':digest(policy),'cases':[],'native_runtime_repaired':False,'scientific_acceptance':False}
    def flush():save(root/'report.json',report);volume.commit()
    flush()
    try:
        import siesta_remag.research.scf_setup_probe as helper
        if hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()!=policy['helper_sha256']:
            raise ValueError('Uploaded diagnostic helper changed')
        for name,sha in policy['files'].items():
            if hashlib.sha256((inp/name).read_bytes()).hexdigest()!=sha:raise ValueError('Uploaded input changed')
        library=Path(runtime['capabilities']['library'])
        if hashlib.sha256(library.read_bytes()).hexdigest()!=runtime['capabilities']['library_sha256']:
            raise ValueError('Retained library hash differs')
        report['library_sha256']=runtime['capabilities']['library_sha256']
        # Compile only the diagnostic translation unit. No REMAG or SIESTA rebuild.
        binary=root/'probe-scf-setup'
        env={k:v for k,v in os.environ.items() if not k.startswith('REMAG_')}
        env.update(runtime['environment'])
        resource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        compile_cmd=['g++','-std=c++17','-O2','-I'+str(inp),'-I/usr/local/cuda/include',str(inp/'probe_scf_setup.cpp'),
            '-L'+str(library.parent),'-L/usr/local/cuda/lib64','-Wl,-rpath,'+str(library.parent),
            '-lremag_gpu','-lcudart','-o',str(binary)]
        report['compile_command']=compile_cmd
        with (root/'compile-stdout.log').open('w') as out,(root/'compile-stderr.log').open('w') as err:
            result=subprocess.run(compile_cmd,env=env,stdout=out,stderr=err,timeout=60)
        report['compile_returncode']=result.returncode;flush()
        if result.returncode:raise RuntimeError('Diagnostic compilation failed; no probe execution')
        report['probe_executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        with (root/'nvidia-smi.log').open('w') as out:subprocess.run(['nvidia-smi'],stdout=out,stderr=subprocess.STDOUT,timeout=10,check=True)
        for case in policy['cases']:
            case_dir=root/case['name'];case_dir.mkdir()
            command=[str(binary),str(case['dimension']),str(case['kpoints']),str(case['mesh'])]
            before=time.monotonic();row=dict(case,command=command,timed_out=False)
            with (case_dir/'stdout.log').open('w') as out,(case_dir/'stderr.log').open('w') as err:
                try:
                    proc=subprocess.run(command,cwd=case_dir,env=env,stdout=out,stderr=err,timeout=60)
                    row['returncode']=proc.returncode
                except subprocess.TimeoutExpired:
                    row.update(returncode=None,timed_out=True)
            row['wall_seconds']=time.monotonic()-before
            row.update(summarize_case(case,(case_dir/'stdout.log').read_text(),row['returncode'],row['timed_out']))
            if (case_dir/'probe-maps.txt').exists():(case_dir/'probe-maps.txt').rename(case_dir/'probe-maps.log')
            save(case_dir/'execution-status.json',row);report['cases'].append(row);flush()
            # Distinct preregistered shapes are independent diagnostic controls,
            # not retries. A timeout stops the finite diagnostic batch.
            if row['timed_out']:break
        report['status']='diagnostic_complete' if len(report['cases'])==len(policy['cases']) else 'diagnostic_incomplete'
    except Exception as error:report.update(status='diagnostic_error',error={'type':type(error).__name__,'message':str(error)})
    flush()
    package=pack(root);volume.commit()
    return package

@app.local_entrypoint()
def main():
    from siesta_remag.discovery.collection import collect
    receipt=folder/'receipt.json';output=folder/'collected'
    if receipt.exists() or output.exists():raise ValueError('Existing diagnostic attempt; never resubmit')
    run_id='setup-probe-'+uuid.uuid4().hex
    budget.reserve(run_id,policy['reservation_cents'],metadata={'app_id':app.app_id,'receipt':str(receipt),'policy_sha256':digest(policy)})
    row={'status':'submitting','run_id':run_id,'app_id':app.app_id,'volume':policy['volume'],'reserved_cents':policy['reservation_cents']}
    save(receipt,row)
    call=probe.spawn(run_id);row.update(status='running',function_call_id=call.object_id);save(receipt,row);budget.record(run_id,row)
    collect(receipt,output,timeout=480);budget.record(run_id,load(receipt))
