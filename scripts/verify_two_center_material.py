#!/usr/bin/env python3
"""Audit material two-center shadow coverage, lifetime ordering and explicit copies.

Material arrays are compared fully on device; this audits their counters, not an
independent replay of every material derivative. CPU forces stay authoritative.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.two_center import record_errors,traffic

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def verify(root):
    meta=read(root/'metadata.json');ledger=read(root/'ledger.json');errors=[];checks=[];hashes={}
    validator=Path(inspect.getsourcefile(record_errors))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        if hashlib.sha256(archive.extractfile('src/siesta_remag/two_center.py').read()).hexdigest()!=sha(validator):raise ValueError('Validator differs from executed archive')
    for name in ('source-verification.json','component-evidence-verification.json','two-center-component-verification.json',
                 'two-center-comparison-verification.json','two-center-owner-verification.json','scientific-recheck.json',
                 'grid-force-verification.json','device-cycle-verification.json'):
        p=root/name
        if not p.is_file() or read(p).get('passed') is not True:errors.append('Failed/missing prerequisite: '+name)
        if p.is_file():hashes[name]=sha(p)
    expected={'fe2_forces','fe2_relax','fe2_noncollinear_gamma','fept2_soc_gamma_z','fept2_soc_gamma_x','fept2_soc_z','fept2_soc_x','fe16_noncollinear','fe54_collinear'}
    if meta.get('two_center_shadow') is not True or set(meta['cases'])!=expected or len(ledger)!=9 or any(r.get('passed') is not True for r in ledger):errors.append('Full material/scientific coverage failed')
    for case in [row['benchmark'] for row in ledger]:
        folder=root/case/'candidate';issues=[];result=read(folder/'result.json');report=read(folder/'remag-two-center.json');raw=read(folder/'remag-residency.json');manifest=read(folder/'manifest.json');telemetry=result['telemetry']
        forces=[read(p) for p in sorted(folder.glob('remag-grid-force-*.json'))];epochs=[r['epoch'] for r in forces]
        args=[report,epochs,manifest['spin']=='spin-orbit',meta.get('pair_force_shadow',False),meta.get('nonlocal_force_shadow',False)]
        if meta.get('pair_geometry_prescf'):args.append(True)
        issues.extend(record_errors(*args))
        if result['runtime_settings'].get('REMAG_TWO_CENTER_SHADOW')!='1' or result.get('valid') is not True:issues.append('Shadow runtime failed')
        if report!=telemetry.get('two_center') or traffic(report)!=telemetry.get('gpu_two_center'):issues.append('Raw owner differs from telemetry')
        if len(forces)!=len(telemetry.get('gpu_device_cycle',{}).get('cycles',[])) or telemetry['residency']['accounting_errors']:issues.append('Cycle or transfer reconciliation failed')
        pairs=[r for r in ledger if r['benchmark']==case]
        if len(pairs)!=1 or pairs[0].get('end_to_end_speedup') is not None or pairs[0].get('performance_comparable') is not False:issues.append('Shadow incorrectly qualifies performance')
        totals={k:dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0) for k in ('two_center_setup','two_center_shadow')}
        for interval in raw['intervals']:
            for boundary,b in interval['boundaries'].items():
                if boundary not in totals:continue
                phase='initialization' if boundary=='two_center_setup' else 'post_scf'
                if interval['phase']!=phase:issues.append('Two-center copy in wrong phase')
                for k,v in b.items():totals[boundary][k]+=v
        table_bytes=sum(t['setup_bytes'] for t in report['tables']);table_calls=0
        for t in report['tables']:
            nf,nr,nt,na,nb,*_=t['shape'];table_calls+=3+3*bool(nt)+bool(nr)
        active=[r for r in report['cycles'] if r['queries']>0 or (meta.get('pair_force_shadow') and r['consumer'] in (1,4)) or (meta.get('nonlocal_force_shadow') and r['consumer'] in (2,3))];queries=sum(r['queries'] for r in active)
        wanted={'two_center_setup':dict(host_to_device_bytes=table_bytes,device_to_host_bytes=0,copy_calls=table_calls),
                'two_center_shadow':dict(host_to_device_bytes=64*queries+60*len(active),device_to_host_bytes=60*len(active),copy_calls=sum(4+3*bool(r['queries']) for r in active))}
        if meta.get('nonlocal_force_shadow'):
            nonlocal_rows=[r for r in active if r['consumer'] in (2,3)]
            wanted['two_center_shadow']['host_to_device_bytes']-=32*sum(r['queries'] for r in nonlocal_rows)
            wanted['two_center_shadow']['copy_calls']-=2*sum(bool(r['queries']) for r in nonlocal_rows)
        if meta.get('pair_geometry_prescf'):
            pair_rows=[r for r in active if r['consumer'] in (1,4)]
            wanted['two_center_shadow']['host_to_device_bytes']-=32*sum(r['queries'] for r in pair_rows)
            wanted['two_center_shadow']['copy_calls']-=2*sum(bool(r['queries']) for r in pair_rows)
        if totals!=wanted:issues.append('Independent shape/phase copy accounting differs')
        evidence=[folder/p for p in ('result.json','remag-two-center.json','remag-residency.json','manifest.json')]+sorted(folder.glob('remag-grid-force-*.json'))
        checks.append(dict(case=case,passed=not issues,errors=issues,geometries=len(epochs),consumers=len(report['cycles']),queries=queries,
                           compared=sum(r['compared'] for r in report['cycles']),max_absolute_error=[max(r['max_absolute_error'][i] for r in report['cycles']) for i in (0,1)],
                           table_setup_bytes=table_bytes,diagnostic_setup_bytes=wanted['two_center_shadow']['host_to_device_bytes'],diagnostic_export_bytes=60*len(active),
                           evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,prerequisite_sha256=hashes,
                validator_sha256=sha(validator),verifier_sha256=sha(Path(__file__)),independent_cuda_trace=False,host_derivatives_authoritative=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'two-center-material-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'two-center-material-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Two-center material audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r['errors'],[c for c in r['checks'] if not c['passed']]);raise SystemExit(1)
