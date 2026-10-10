#!/usr/bin/env python3
"""Exercise real grid ownership and force preparation without topology readbacks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENARIOS=('normal','missing','count','shape','stale','restage','ends','orbs','phi','nsc','lease','topology','topology-lease')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def errors_for(probe,residency,scenario):
    stages=2 if scenario in ('stale','restage') else 1
    topology=scenario in ('topology','topology-lease')
    wanted=dict(scenario=scenario,passed=True,stages=stages,epoch=stages,entries=2 if scenario=='restage' else 1,
                rho=0. if topology else .5625 if scenario=='restage' else .25,rejections=[] if scenario in ('normal','topology') else [-230])
    errors=[]
    if probe!=wanted:errors.append('Owner validation/recovery differs')
    intervals=residency.get('intervals',[])
    if not intervals or residency.get('geometry_epochs')!=stages:errors.append('Missing execution evidence')
    for interval in intervals:
        if interval['phase']=='post_scf' and interval['boundaries']:errors.append('Force preparation performed explicit transfers')
    exports=sum(b.get('device_to_host_bytes',0) for i in intervals for b in i['boundaries'].values())
    if exports!=(0 if topology else 8*stages):errors.append('Unexpected exports beyond initialization density fixture')
    if topology and any(b.get('host_to_device_bytes',0) for i in intervals for k,b in i['boundaries'].items() if k!='grid_topology_basis'):
        errors.append('Topology-only preparation uploaded electronic state')
    return errors

def run(executable,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for scenario in SCENARIOS:
        folder=output.parent/scenario;folder.mkdir();report=folder/'owner.json';log=folder/'worker.log';residency=folder/'remag-residency.json'
        with log.open('w') as stream:
            p=subprocess.run([str(executable.resolve()),scenario,str(report)],cwd=folder,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
        errors=['Native check failed'] if p.returncode else errors_for(json.loads(report.read_text()),json.loads(residency.read_text()),scenario)
        files=dict(report=report,owner=residency,log=log)
        checks.append(dict(scenario=scenario,passed=not errors,errors=errors,native_return=p.returncode,
            **{k+'_file':str(v.relative_to(output.parent)) for k,v in files.items()},
            **{k+'_sha256':sha(v) if v.is_file() else None for k,v in files.items()}))
    result=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable))
    output.write_text(json.dumps(result,indent=2)+'\n');print('Grid owner:',len(checks),'passed=',result['passed'])
    if not result['passed']:print([c for c in checks if not c['passed']]);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('executable','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.executable,a.output)
