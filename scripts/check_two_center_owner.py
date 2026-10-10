#!/usr/bin/env python3
"""Exercise staged two-center snapshots, force ordering, geometry and failure ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENARIOS=('normal','soc','empty','reuse','snapshot','missing','table_order','duplicate','bad_map','bad_dimensions','nonfinite_table','mode_stage','scf_stage','missing_t','wrong_begin','duplicate_begin','wrong_table','index','null_xyz','wrong_finish','mode_consume','scf_consume','duplicate_finish','stale_active','mismatch','nonfinite')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def errors_for(probe,owner,trace,scenario):
    errors=[];failure=scenario in ('stale_active','mismatch','nonfinite');epochs=2 if scenario in ('reuse','stale_active') else 1
    count=3 if scenario=='stale_active' else 2 if scenario in ('mismatch','nonfinite') else 0 if scenario in SCENARIOS[:5] else 1
    if probe!=dict(scenario=scenario,passed=True,epoch=epochs,rejections=[-250]*count):errors.append('Native rejection/recovery differs')
    rows=owner.get('cycles',[]);tables=owner.get('tables',[]);consumers=[1] if failure else [1,3 if scenario=='soc' else 2,4]*(2 if scenario=='reuse' else 1)
    if owner.get('complete')!=(not failure) or [r['consumer'] for r in rows]!=consumers:errors.append('Consumer coverage differs')
    if len(tables)!=2 or [r['setup_bytes'] for r in tables]!=[16440]*2:errors.append('Table staging differs')
    n=0 if scenario=='empty' else 1
    for i,r in enumerate(rows):
        status=-1 if scenario=='stale_active' else 3 if scenario=='mismatch' else 2 if scenario=='nonfinite' else 0
        complete=scenario!='stale_active';executed=complete and n>0
        wanted=dict(epoch=1+i//3,table=2 if i%3==0 else 1,queries=n,status=status,complete=complete,passed=not failure,
                    graph_nodes=2 if executed else 0,setup_bytes=124 if executed else 0,device_allocation_bytes=156 if executed else 0,
                    export_bytes=60 if executed else 0,host_reference_bytes=32*n,compared=4*n if complete else 0,
                    mismatches=int(scenario=='mismatch'),nonfinite=int(scenario=='nonfinite'))
        if any(r.get(k)!=v for k,v in wanted.items()):errors.append('Owner execution/traffic differs')
    traffic=[b for interval in trace.get('intervals',[]) for b in interval['boundaries'].values()]
    if sum(b['host_to_device_bytes'] for b in traffic)!=32880+sum(r['setup_bytes'] for r in rows) or sum(b['device_to_host_bytes'] for b in traffic)!=sum(r['export_bytes'] for r in rows):errors.append('Explicit counters differ')
    if any(i['boundaries'] for i in trace['intervals'] if i['phase']=='scf'):errors.append('Rejected SCF call performed transfers')
    return errors

def run(executable,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for scenario in SCENARIOS:
        folder=output.parent/scenario;folder.mkdir();files={k:folder/v for k,v in dict(report='owner.json',owner='remag-two-center.json',residency='remag-residency.json',log='worker.log').items()}
        with files['log'].open('w') as stream:
            p=subprocess.run([str(executable.resolve()),scenario,str(files['report'])],cwd=folder,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
        errors=['Native check failed'] if p.returncode else errors_for(*(json.loads(files[k].read_text()) for k in ('report','owner','residency')),scenario)
        checks.append(dict(scenario=scenario,passed=not errors,errors=errors,native_return=p.returncode,
                           **{k+'_file':str(v.relative_to(output.parent)) for k,v in files.items()},**{k+'_sha256':sha(v) if v.is_file() else None for k,v in files.items()}))
    result=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable))
    output.write_text(json.dumps(result,indent=2)+'\n');print('Two-center owner:',len(checks),'passed=',result['passed'])
    if not result['passed']:print([c for c in checks if not c['passed']]);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('executable','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.executable,a.output)
