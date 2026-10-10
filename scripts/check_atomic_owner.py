#!/usr/bin/env python3
"""Isolated process checks for atomic geometry ownership and full shadow consumption."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENARIOS=('normal','reuse','snapshot','begin_without_stage','stage_in_scf','wrong_mode','bad_shape','duplicate_pair','duplicate_stage',
 'unconsumed_epoch','begin_in_scf','stale_epoch','wrong_shape','wrong_kind','out_of_order','mode_change','finish_before_begin',
 'prepare_before_begin','duplicate_begin','invalid_point','unstaged_pair','duplicate_point','finish_before_generation',
 'generate_before_prepare','selected_count','duplicate_prepare','point_after_prepare','wrong_stream','duplicate_generation',
 'missing_term','finish_in_scf','duplicate_finish','sticky','device_error','missing_point','unlaunched','both_modes','cpu_callback')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def errors_for(report,records,scenario,core,shadow):
    errors=[];terminal=scenario=='unconsumed_epoch';failed=scenario in ('sticky','device_error','missing_point','unlaunched')
    count=2 if scenario=='reuse' else 1;terms=core+1;graphs=0 if terminal else count*terms
    rejects=0 if scenario in ('normal','reuse','snapshot') else 2 if terminal else 1
    status=-1 if terminal else 7 if scenario=='sticky' else (3 if shadow else 2) if scenario=='device_error' else 3 if scenario=='missing_point' else 0
    if (report.get('scenario')!=scenario or report.get('core')!=core or report.get('shadow')!=shadow or report.get('passed') is not True or report.get('graphs')!=graphs or
        report.get('status')!=status or len(report.get('rejections',[]))!=rejects):errors.append('Owner execution/rejection mismatch')
    if len(records)!=count:errors.append('Owner record count differs')
    for i,r in enumerate(records):
        if r.get('epoch')!=i+1 or r.get('shape')!=[1,4,2,1,1,2,1,2,1,2,1,1,core] or r.get('complete')!=(not terminal and not failed):errors.append('Owner geometry/completion mismatch')
        if r.get('compact_setup_bytes')!=244 or r.get('device_allocation_bytes')!=244+(0 if terminal else terms*(392 if shadow else 136)):errors.append('Owner staging/allocation mismatch')
        if [t.get('kind') for t in r.get('terms',[])]!=([1,2] if core else [2]):errors.append('Owner required terms differ')
        if r.get('shadow')!=shadow or r.get('host_derivatives_authoritative')!=shadow:errors.append('Owner mode differs')
        for t in r.get('terms',[]):
            if any(t.get(k)!=(not terminal) for k in ('begun','generated','consumed')) or t.get('passed')!=(not terminal and not failed) or t.get('status')!=status:errors.append('Owner consumer state differs')
            m=(0 if terminal or scenario=='missing_point' else 1) if shadow else None
            if t.get('reference_selected')!=m or t.get('shadow_setup_bytes')!=(256 if shadow and not terminal else 0) or t.get('shadow_export_bytes')!=(120 if shadow and not terminal else 0) or t.get('device_allocation_bytes')!=(0 if terminal else 392 if shadow else 136):errors.append('Owner consumer traffic differs')
            c=t.get('comparison')
            if terminal or not shadow:
                if c is not None:errors.append('Unperformed comparison is not null')
                continue
            n=0 if scenario=='unlaunched' else 2
            expected=dict(points=n,selected=0 if scenario=='unlaunched' else 1,reference_selected=0 if scenario in ('unlaunched','missing_point') else 1,
                          compared=0 if scenario=='unlaunched' else 8 if scenario=='missing_point' else 11,index_compared=0 if scenario in ('unlaunched','missing_point') else 2,
                          mask_mismatches=1 if scenario=='missing_point' else 0,index_mismatches=0)
            if not isinstance(c,dict) or any(c.get(k)!=v for k,v in expected.items()):errors.append('Owner comparison coverage differs');continue
            if scenario=='missing_point':
                if c.get('mismatches')!=2 or c.get('nonfinite')!=0:errors.append('Missing point not detected')
            elif scenario=='device_error':
                if c.get('mismatches')!=2 or c.get('nonfinite')!=0:errors.append('Invalid radial input not rejected before comparison')
            elif c.get('mismatches')!=0 or c.get('nonfinite')!=0 or c.get('absolute',[1,1,1])[2]!=0 or max(c.get('scaled',[2,2,2]))>1:
                errors.append('Valid owner comparison failed')
    probes=report.get('probes',[])
    if len(probes)!=(0 if shadow or terminal else graphs):errors.append('Owner generated-array probe coverage differs')
    for i,p in enumerate(probes):
        x=.4 if scenario=='reuse' and i>=terms else .2
        wanted=[x,1,0,1,2,1,1]+([0.] if scenario=='device_error' else [.2])+[0.]*5+[x,0.,0.,2.,0.,0.]
        if len(p)!=19 or any(abs(a-b)>1e-14 for a,b in zip(p,wanted)):errors.append('Generated probe differs from analytic fixture')
    return errors

def run(executable,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for shadow in (False,True):
        for core in (0,1):
            for scenario in SCENARIOS:
                if not core and scenario=='missing_term':continue
                if shadow and scenario=='cpu_callback':continue
                if not shadow and scenario in ('invalid_point','unstaged_pair','duplicate_point','point_after_prepare','missing_point','unlaunched'):continue
                folder=output.parent/f'shadow{int(shadow)}-core{core}-{scenario}';folder.mkdir();target=folder/'owner.json';log=folder/'worker.log'
                with log.open('w') as stream:
                    process=subprocess.run([str(executable.resolve()),scenario,str(core),str(int(shadow)),str(target)],cwd=folder,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
                owner=folder/'remag-atomic-basis.json';report=json.loads(target.read_text()) if target.is_file() else {};records=json.loads(owner.read_text()).get('cycles',[]) if owner.is_file() else []
                errors=(['Native owner check failed'] if process.returncode else [])+errors_for(report,records,scenario,core,shadow)
                checks.append(dict(scenario=scenario,core=core,shadow=shadow,passed=not errors,errors=errors,native_return=process.returncode,
                    **{k+'_file':str(p.relative_to(output.parent)) for k,p in (('report',target),('owner',owner),('log',log))},
                    **{k+'_sha256':sha(p) if p.is_file() else None for k,p in (('report',target),('owner',owner),('log',log))}))
    result=dict(scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable))
    output.write_text(json.dumps(result,indent=2)+'\n');print('Atomic owner:',len(checks),'passed=',result['passed'])
    if not result['passed']:print([r for r in checks if not r['passed']]);raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.executable,a.output)
