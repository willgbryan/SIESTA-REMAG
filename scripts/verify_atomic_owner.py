#!/usr/bin/env python3
"""Audit isolated atomic owner lifecycle records and retained failure recovery."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root):
    folder=root/'atomic-owner-check';path=folder/'check.json';report=json.loads(path.read_text());checks=[];errors=[];seen=set()
    scenarios=('normal','reuse','snapshot','begin_without_stage','stage_in_scf','wrong_mode','bad_shape','duplicate_pair','duplicate_stage',
      'unconsumed_epoch','begin_in_scf','stale_epoch','wrong_shape','wrong_kind','out_of_order','mode_change','finish_before_begin',
      'prepare_before_begin','duplicate_begin','invalid_point','unstaged_pair','duplicate_point','finish_before_generation',
      'generate_before_prepare','selected_count','duplicate_prepare','point_after_prepare','wrong_stream','duplicate_generation',
      'missing_term','finish_in_scf','duplicate_finish','sticky','device_error','missing_point','unlaunched','both_modes','cpu_callback')
    expected={(s,c,h) for s in scenarios for c in (0,1) for h in (False,True) if (c or s!='missing_term') and (not h or s!='cpu_callback') and (h or s not in ('invalid_point','unstaged_pair','duplicate_point','point_after_prepare','missing_point','unlaunched'))}
    for item in report['checks']:
        s,c,h=item['scenario'],item['core'],item['shadow'];key=(s,c,h);issues=[]
        if key in seen:issues.append('Duplicate scenario')
        seen.add(key);files={k:folder/item[k+'_file'] for k in ('report','owner','log')}
        if any(sha(p)!=item[k+'_sha256'] for k,p in files.items()):issues.append('Retained hash differs')
        probe=json.loads(files['report'].read_text());raw=json.loads(files['owner'].read_text());records=raw['cycles']
        terminal=s=='unconsumed_epoch';failed=s in ('sticky','device_error','missing_point','unlaunched');count=2 if s=='reuse' else 1;status=-1 if terminal else 7 if s=='sticky' else (3 if h else 2) if s=='device_error' else 3 if s=='missing_point' else 0
        rejects=[] if s in ('normal','reuse','snapshot') else [s,'stale_after_epoch_change'] if terminal else [s]
        summary={k:v for k,v in probe.items() if k!='probes'}
        if summary!=dict(scenario=s,core=c,shadow=h,passed=True,graphs=0 if terminal else (c+1)*count,status=status,rejections=rejects):issues.append('Execution/rejections differ')
        probes=probe.get('probes',[])
        if len(probes)!=(0 if h or terminal else (c+1)*count):issues.append('Generated array coverage differs')
        for i,p in enumerate(probes):
            x=.4 if s=='reuse' and i>=c+1 else .2
            wanted=[x,1,0,1,2,1,1]+([0.] if s=='device_error' else [.2])+[0.]*5+[x,0.,0.,2.,0.,0.]
            if len(p)!=len(wanted) or any(not isinstance(a,(int,float)) or not math.isfinite(a) or abs(a-b)>1e-14 for a,b in zip(p,wanted)):issues.append('Generated arrays differ from staged analytic fixture')
        if len(records)!=count or raw['atol']!=1e-10 or raw['rtol']!=1e-10 or raw['displacements_exact'] is not True:issues.append('Record count/tolerances differ')
        for i,r in enumerate(records):
            if (r['epoch']!=i+1 or r['shape']!=[1,4,2,1,1,2,1,2,1,2,1,1,c] or r['complete']!=(not terminal and not failed) or
                r['compact_setup_bytes']!=244 or r['device_allocation_bytes']!=244+(0 if terminal else (392 if h else 136)*(c+1))):issues.append('Epoch/shape/traffic differs')
            if [t['kind'] for t in r['terms']]!=([1,2] if c else [2]):issues.append('Required consumers differ')
            if r.get('shadow')!=h or r.get('host_derivatives_authoritative')!=h:issues.append('Owner mode differs')
            for t in r['terms']:
                if any(t[k]!=(not terminal) for k in ('begun','generated','consumed')) or t['passed']!=(not terminal and not failed) or t['status']!=status:issues.append('Consumer lifecycle differs')
                if (t['reference_selected']!=((0 if terminal or s=='missing_point' else 1) if h else None) or t['shadow_setup_bytes']!=(256 if h and not terminal else 0) or
                    t['device_allocation_bytes']!=(0 if terminal else 392 if h else 136) or t['shadow_export_bytes']!=(120 if h and not terminal else 0)):issues.append('Consumer accounting differs')
                v=t['comparison']
                if terminal or not h:
                    if v is not None:issues.append('Unperformed comparison recorded')
                    continue
                wanted=dict(points=2,selected=1,reference_selected=1,compared=11,index_compared=2,mismatches=0,nonfinite=0,mask_mismatches=0,index_mismatches=0)
                if s=='unlaunched':wanted=dict.fromkeys(wanted,0)
                if s=='missing_point':wanted.update(reference_selected=0,compared=8,index_compared=0,mismatches=2,mask_mismatches=1)
                if s=='device_error':wanted['mismatches']=2
                if any(v[k]!=x for k,x in wanted.items()):issues.append('Full comparison counters differ')
                if v['absolute'][2]!=0 or v['scaled'][2]!=0:issues.append('Exact displacement differs')
                if s not in ('device_error','missing_point') and max(v['scaled'])>1:issues.append('Numerical comparison failed')
        if item['native_return']!=0 or item['passed'] is not True or item['errors']:issues.append('Recorded scenario failed')
        checks.append(dict(scenario=s,core=c,shadow=h,passed=not issues,errors=issues,evidence_sha256={k:sha(p) for k,p in files.items()}))
    if seen!=expected or len(checks)!=136:errors.append('Scenario coverage differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'atomic-owner-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'atomic-owner-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Atomic owner audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
