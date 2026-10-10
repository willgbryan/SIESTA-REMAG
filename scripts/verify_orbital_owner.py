#!/usr/bin/env python3
"""Audit retained isolated owner lifecycle failures, recovery and geometry reuse.

This checks raw reports and shape-derived accounting independently of the probe
controller. Kernel numerical correctness has a separate upstream-oracle gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(root):
    folder=root/'orbital-owner-check';path=folder/'check.json';report=json.loads(path.read_text())
    scenarios=('normal','reuse','borrow_without_stage','stage_in_scf','duplicate_stage','unconsumed_epoch',
        'stale_borrow','wrong_shape','borrow_in_scf','finish_before_borrow','finish_before_generation',
        'generate_before_borrow','duplicate_borrow','duplicate_generation','wrong_stream','finish_in_scf',
        'duplicate_finish','mode_change','device_error','both_modes','missing_comparison')
    expected={(name,shadow) for name in scenarios for shadow in (False,True) if shadow or name!='missing_comparison'}
    seen=set();errors=[];checks=[]
    with tarfile.open(root/'implementation.tar.gz') as archive:
        source=archive.extractfile('scripts/check_orbital_owner.py').read()
        if hashlib.sha256(source).hexdigest()!=report['checker_sha256']:errors.append('Controller provenance differs')
    for row in report['checks']:
        scenario,shadow=row['scenario'],row['shadow'];key=(scenario,shadow);issues=[]
        if key in seen:issues.append('Duplicate owner scenario')
        seen.add(key);files={name:folder/row[name+'_file'] for name in ('report','owner','log')}
        for name,p in files.items():
            if not p.is_file() or sha(p)!=row[name+'_sha256']:issues.append('Raw evidence hash differs: '+name)
        native=json.loads(files['report'].read_text());records=json.loads(files['owner'].read_text())['cycles']
        terminal=scenario=='unconsumed_epoch';failed=scenario in ('device_error','missing_comparison');count=2 if scenario=='reuse' else 1
        status=-1 if terminal else 2 if scenario=='device_error' else 0
        rejection_count=0 if scenario in ('normal','reuse') else 2 if terminal else 4 if scenario=='wrong_shape' else 1
        if (row.get('passed') is not True or row.get('native_return')!=0 or native.get('passed') is not True or
            native.get('scenario')!=scenario or native.get('shadow')!=shadow or native.get('status')!=status or
            native.get('graphs')!=(0 if terminal else count) or len(native.get('rejections',[]))!=rejection_count):
            issues.append('Native execution/rejections differ')
        if len(records)!=count:issues.append('Owner epoch coverage differs')
        for i,r in enumerate(records):
            shadow_bytes=72 if shadow and not terminal else 0
            wanted=dict(epoch=i+1,shape=[1,1,2,1,1,1,1,1,1],status=status,generated=not terminal,consumed=not terminal,
                passed=not terminal and not failed,host_derivatives_authoritative=shadow,comparison_performed=shadow and not terminal,
                compact_setup_bytes=164,derivative_bytes=56,device_allocation_bytes=164+(0 if terminal else 56)+shadow_bytes,
                shadow_setup_bytes=shadow_bytes,shadow_export_bytes=shadow_bytes)
            if any(k not in r or r[k]!=v for k,v in wanted.items()):issues.append('Owner state or traffic differs')
            comparison=dict(compared=0 if scenario=='missing_comparison' else 7,mismatches=0,nonfinite=0,
                max_absolute_error=[0.,0.,0.],max_scaled_error=[0.,0.,0.],atol=1e-10,rtol=1e-10,displacements_exact=True)
            if any(k not in r or r[k]!=(v if shadow and not terminal else None) for k,v in comparison.items()):
                issues.append('Comparison state differs')
        if not terminal and scenario!='device_error':
            for key,x in [('first_phi',.2)]+([('second_phi',.4)] if scenario=='reuse' else []):
                wanted=(.3+.2*x)/math.sqrt(4*math.pi);actual=native.get(key)
                if not isinstance(actual,(int,float)) or not math.isfinite(actual) or abs(actual-wanted)>1e-14:
                    issues.append('Analytic s-orbital/geometry reuse differs')
        checks.append(dict(scenario=scenario,shadow=shadow,passed=not issues,errors=issues,
            evidence_sha256={str(p.relative_to(root)):sha(p) for p in files.values()}))
    if seen!=expected or len(checks)!=41 or report.get('passed') is not True:errors.append('Owner scenario coverage or recorded success differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
        report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();result=verify(a.run)
    (a.run/'orbital-owner-verifier.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'orbital-owner-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Orbital owner audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
