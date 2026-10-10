#!/usr/bin/env python3
"""Qualify removal of force topology readbacks with raw phase and owner evidence.

Only successful explicit bridge copies are measured. This does not establish
complete CUDA visibility or remove the remaining electronic matrix exports.
"""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def traffic(folder):
    raw=folder/'remag-residency.json';r=json.loads(raw.read_text());v=dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
    for i in r['intervals']:
        if i['phase']=='post_scf':
            for k,x in i['boundaries'].get('grid_output',{}).items():v[k]+=x
    return v

def verify(root,baseline):
    checks=[];errors=[];proof={}
    for name in ('source-verification','component-evidence-verification','grid-owner-verification','device-cycle-verification','grid-force-verification','final-grid-verification','final-density-verification','scientific-recheck'):
        p=root/(name+'.json');r=json.loads(p.read_text());proof[p.name]=sha(p)
        if r.get('passed') is not True:errors.append(name+' failed')
    meta=json.loads((root/'metadata.json').read_text())
    for case in meta['cases']:
        folder=root/case/'candidate';prior=baseline/case/'candidate';r=json.loads((folder/'result.json').read_text());b=json.loads((prior/'result.json').read_text());faults=[]
        records=[json.loads(p.read_text()) for p in sorted(folder.glob('remag-grid-force-*.json'))];old=[json.loads(p.read_text()) for p in sorted(prior.glob('remag-grid-force-*.json'))]
        if not r['valid'] or not b['valid'] or len(records)!=len(old) or not records:faults.append('Invalid candidate/baseline geometry coverage')
        if r['benchmark']['input_sha256']!=b['benchmark']['input_sha256']:faults.append('Input provenance differs')
        for row in records:
            if row.get('topology_validation')!='retained_geometry_owner' or row.get('retained_entries')!=row['entries'] or row.get('topology_export_bytes')!=0:faults.append('Topology did not use retained geometry metadata')
            if row['export_bytes']!=8*(3*row['atoms']+9)+4:faults.append('Unexpected force export payload')
        current=traffic(folder);previous=traffic(prior);n=len(records)
        wanted=dict(host_to_device_bytes=0,device_to_host_bytes=sum(row['export_bytes'] for row in records),copy_calls=3*n)
        before=dict(host_to_device_bytes=0,device_to_host_bytes=sum(8*(3*row['atoms']+9)+8 for row in old),copy_calls=4*len(old))
        if current!=wanted or previous!=before:faults.append('Raw post-SCF grid-output traffic differs')
        if previous['device_to_host_bytes']-current['device_to_host_bytes']!=4*n:faults.append('Readback reduction differs')
        checks.append(dict(case=case,passed=not faults,errors=faults,geometries=n,traffic=current,baseline_traffic=previous,
            removed_bytes=4*n,removed_copy_calls=n,evidence_sha256={str(p):sha(p) for p in (folder/'result.json',folder/'remag-residency.json',prior/'result.json',prior/'remag-residency.json')}))
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,qualification_sha256=proof,baseline=baseline.name,verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();r=verify(a.run,a.baseline)
    (a.run/'force-topology-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'force-topology-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Force topology audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r);raise SystemExit(1)
