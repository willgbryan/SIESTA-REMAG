#!/usr/bin/env python3
"""Isolated process checks of orbital owner phases, epochs, modes and recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

SCENARIOS=('normal','reuse','borrow_without_stage','stage_in_scf','duplicate_stage','unconsumed_epoch','stale_borrow',
 'wrong_shape','borrow_in_scf','finish_before_borrow','finish_before_generation','generate_before_borrow','duplicate_borrow',
 'duplicate_generation','wrong_stream','finish_in_scf','duplicate_finish','mode_change','device_error','both_modes','missing_comparison')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def errors_for(report,records,scenario,shadow):
    errors=[];terminal=scenario=='unconsumed_epoch';failed=scenario in ('device_error','missing_comparison')
    count=2 if scenario=='reuse' else 1;graphs=0 if terminal else count
    rejects=0 if scenario in ('normal','reuse') else 2 if terminal else 4 if scenario=='wrong_shape' else 1
    if (report.get('scenario')!=scenario or report.get('shadow')!=shadow or report.get('passed') is not True or
        report.get('graphs')!=graphs or len(report.get('rejections',[]))!=rejects or
        report.get('status')!=(-1 if terminal else 2 if scenario=='device_error' else 0)):errors.append('Owner probe execution/rejection mismatch')
    if len(records)!=count:errors.append('Owner record count differs')
    for i,r in enumerate(records):
        if r.get('epoch')!=i+1 or r.get('shape')!=[1,1,2,1,1,1,1,1,1]:errors.append('Owner epoch/shape differs')
        if r.get('generated')!=(not terminal) or r.get('consumed')!=(not terminal) or r.get('passed')!=(not terminal and not failed):errors.append('Owner terminal state differs')
        if r.get('status')!=(-1 if terminal else 2 if scenario=='device_error' else 0):errors.append('Owner status differs')
        extra=72 if shadow and not terminal else 0
        if (r.get('compact_setup_bytes')!=164 or r.get('derivative_bytes')!=56 or r.get('shadow_setup_bytes')!=extra or
            r.get('shadow_export_bytes')!=extra or r.get('device_allocation_bytes')!=164+(0 if terminal else 56)+extra):errors.append('Owner allocation/traffic differs')
        compared=shadow and not terminal
        if r.get('host_derivatives_authoritative')!=shadow or r.get('comparison_performed')!=compared:errors.append('Owner mode differs')
        if compared:
            if r.get('compared')!=(0 if scenario=='missing_comparison' else 7) or r.get('mismatches')!=0 or r.get('nonfinite')!=0:errors.append('Shadow completion count differs')
            if (r.get('max_absolute_error')!=[0.,0.,0.] or r.get('max_scaled_error')!=[0.,0.,0.] or
                r.get('atol')!=1e-10 or r.get('rtol')!=1e-10 or r.get('displacements_exact') is not True):errors.append('Shadow errors/tolerances differ')
        elif any(r.get(k) is not None for k in ('compared','mismatches','nonfinite','max_absolute_error','max_scaled_error','atol','rtol','displacements_exact')):
            errors.append('Unperformed owner comparison is not null')
    if not terminal and scenario!='device_error':
        if abs(report['first_phi']-(.3+.2*.2)*np.sqrt(1/(4*np.pi)))>1e-14:errors.append('Initial geometry result differs')
        if scenario=='reuse' and abs(report['second_phi']-(.3+.2*.4)*np.sqrt(1/(4*np.pi)))>1e-14:errors.append('Second geometry result differs')
    return errors


def run(executable,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for shadow in (False,True):
        for scenario in SCENARIOS:
            if not shadow and scenario=='missing_comparison':continue
            name=f'shadow{int(shadow)}-{scenario}';folder=output.parent/name;folder.mkdir();target=folder/'owner.json';log=folder/'worker.log'
            with log.open('w') as stream:
                process=subprocess.run([str(executable.resolve()),scenario,str(int(shadow)),str(target)],cwd=folder,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
            owner=folder/'remag-orbital-basis.json';errors=[];report=json.loads(target.read_text()) if target.exists() else {}
            records=json.loads(owner.read_text()).get('cycles',[]) if owner.exists() else []
            if process.returncode:errors.append('Native owner check failed')
            errors.extend(errors_for(report,records,scenario,shadow))
            checks.append(dict(scenario=scenario,shadow=shadow,passed=not errors,errors=errors,native_return=process.returncode,
                report_file=str(target.relative_to(output.parent)),report_sha256=sha(target) if target.exists() else None,
                owner_file=str(owner.relative_to(output.parent)),owner_sha256=sha(owner) if owner.exists() else None,
                log_file=str(log.relative_to(output.parent)),log_sha256=sha(log)))
    result=dict(scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable))
    output.write_text(json.dumps(result,indent=2)+'\n');print('Orbital owner checks:',len(checks),'passed=',result['passed'])
    if not result['passed']:
        print([r for r in checks if not r['passed']]);raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.executable,a.output)
