#!/usr/bin/env python3
"""Independently audit two-center lifetime probes, full comparisons and copy boundaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    folder=root/'two-center-owner-check';path=folder/'check.json';data=json.loads(path.read_text());checks=[];seen=set()
    expected={'normal','soc','empty','reuse','snapshot','missing','table_order','duplicate','bad_map','bad_dimensions','nonfinite_table','mode_stage','scf_stage','missing_t','wrong_begin','duplicate_begin','wrong_table','index','null_xyz','wrong_finish','mode_consume','scf_consume','duplicate_finish','stale_active','mismatch','nonfinite'}
    for row in data['checks']:
        name=row['scenario'];errors=[];files={k:folder/row[k+'_file'] for k in ('report','owner','residency','log')}
        if name in seen:errors.append('Duplicate scenario')
        seen.add(name)
        if any(sha(v)!=row[k+'_sha256'] for k,v in files.items()):errors.append('Evidence hash mismatch')
        probe,owner,raw=(json.loads(files[k].read_text()) for k in ('report','owner','residency'))
        failure=name in ('stale_active','mismatch','nonfinite');epoch=2 if name in ('reuse','stale_active') else 1
        rejections=3 if name=='stale_active' else 2 if name in ('mismatch','nonfinite') else 0 if name in ('normal','soc','empty','reuse','snapshot') else 1
        if probe!=dict(scenario=name,passed=True,epoch=epoch,rejections=[-250]*rejections):errors.append('Native ownership rejection/recovery differs')
        rows=owner['cycles'];stale=name=='stale_active';n=0 if name=='empty' else 1;ng=2 if name=='reuse' else 1
        if owner['complete']!=(not failure) or owner['atol']!=1e-10 or owner['rtol']!=1e-10:errors.append('Completion or tolerances differ')
        wanted=[(1,1)] if failure else [(e,c) for e in range(1,ng+1) for c in (1,3 if name=='soc' else 2,4)]
        if [(r['epoch'],r['consumer']) for r in rows]!=wanted:errors.append('Consumer/geometry coverage differs')
        if len(owner['tables'])!=2:errors.append('Missing S/T snapshot')
        for i,t in enumerate(owner['tables']):
            if t['kind']!=i+1 or t['shape']!=[1,1,1,1,1,0,0,1,1,1,1] or t['delta']!=.1 or t['pi']!=math.pi or t['setup_bytes']!=16440 or t['device_allocation_bytes']!=16440:
                errors.append('Staged fixture shape/constants differ')
            if t['uses']!=sum(r['table']==i+1 and r['complete'] for r in rows):errors.append('Table use lifetime differs')
        for r in rows:
            status=-1 if stale else 3 if name=='mismatch' else 2 if name=='nonfinite' else 0;executed=n and not stale
            want=dict(table=2 if r['consumer']==1 else 1,queries=n,status=status,complete=not stale,passed=not failure,
                      graph_nodes=2 if executed else 0,setup_bytes=64*n+60 if executed else 0,device_allocation_bytes=96*n+60 if executed else 0,
                      export_bytes=60 if executed else 0,host_reference_bytes=32*n,compared=4*n if not stale else 0,
                      mismatches=int(name=='mismatch'),nonfinite=int(name=='nonfinite'))
            if any(r.get(k)!=v for k,v in want.items()):errors.append('Full comparison or execution record differs')
            for key in ('max_absolute_error','max_scaled_error'):
                v=r[key]
                if len(v)!=2 or any(not math.isfinite(x) or x<0 for x in v):errors.append('Invalid error maxima')
                if not failure and (max(v)>1 if key=='max_scaled_error' else max(v)>1e-10):errors.append('Analytic fixture comparison failed')
            if name=='mismatch' and (r['max_absolute_error'][0]<.99 or r['max_scaled_error'][0]<=1):errors.append('Mismatch was not measured')
        if raw['geometry_epochs']!=epoch or raw['phase_markers_seen'] is not True:errors.append('Missing geometry/phase coverage')
        totals={k:dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0) for k in ('two_center_setup','two_center_shadow')}
        for interval in raw['intervals']:
            for boundary,traffic in interval['boundaries'].items():
                if boundary not in totals:errors.append('Unexpected transfer boundary');continue
                if interval['phase']!=('initialization' if boundary=='two_center_setup' else 'post_scf'):errors.append('Copy in wrong phase')
                for k,v in traffic.items():totals[boundary][k]+=v
        count=sum(r['complete'] and r['queries']>0 for r in rows)
        if totals!={'two_center_setup':dict(host_to_device_bytes=32880,device_to_host_bytes=0,copy_calls=14),
                    'two_center_shadow':dict(host_to_device_bytes=124*count,device_to_host_bytes=60*count,copy_calls=7*count)}:
            errors.append('Independent explicit-copy reconciliation failed')
        if row['native_return']!=0 or row['passed'] is not True or row['errors']:errors.append('Recorded check failed')
        checks.append(dict(scenario=name,passed=not errors,errors=errors))
    errors=[] if seen==expected and len(checks)==26 else ['Scenario coverage differs']
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'two-center-owner-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'two-center-owner-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Two-center owner audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
