#!/usr/bin/env python3
"""Audit all material kinetic/overlap contributions and retained-field boundaries.

Recomputes numerical differences from complete recorded CPU/GPU energy, force
and stress vectors. Does not replay every material integral or sparse array.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def verify(root):
    meta=read(root/'metadata.json');ledger=read(root/'ledger.json');checks=[];errors=[];hashes={}
    expected={'fe2_forces','fe2_relax','fe2_noncollinear_gamma','fept2_soc_gamma_z','fept2_soc_gamma_x','fept2_soc_z','fept2_soc_x','fe16_noncollinear','fe54_collinear'}
    if meta.get('pair_force_shadow') is not True or set(meta['cases'])!=expected or len(ledger)!=9 or any(r.get('passed') is not True for r in ledger):errors.append('Full material/scientific coverage failed')
    for name in ('source-verification.json','component-evidence-verification.json','scientific-recheck.json','pair-force-component-verification.json',
                 'pair-handoff-verification.json','two-center-component-verification.json','two-center-comparison-verification.json','two-center-owner-verification.json',
                 'grid-force-verification.json','device-cycle-verification.json'):
        path=root/name
        if not path.is_file() or read(path).get('passed') is not True:errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():hashes[name]=sha(path)
    for case in [row['benchmark'] for row in ledger]:
        folder=root/case/'candidate';result=read(folder/'result.json');pair=read(folder/'remag-pair-force.json');integrals=read(folder/'remag-two-center.json');raw=read(folder/'remag-residency.json')
        forces=[read(p) for p in sorted(folder.glob('remag-grid-force-*.json'))];epochs=[f['epoch'] for f in forces];rows=pair['cycles'];issues=[]
        prescf=meta.get('pair_geometry_prescf') is True
        geometry=read(folder/'remag-pair-geometry.json') if prescf else None
        if prescf and (geometry.get('complete') is not True or pair.get('geometry_staged_before_scf') is not True or [g.get('epoch') for g in geometry.get('cycles',[])]!=epochs):issues.append('Pre-SCF geometry coverage failed')
        if pair.get('complete') is not True or [(r['epoch'],r['consumer']) for r in rows]!=[(e,c) for e in epochs for c in (1,4)]:issues.append('Missing or unordered material consumers')
        if result.get('valid') is not True or result['runtime_settings'].get('REMAG_PAIR_FORCE_SHADOW')!='1':issues.append('Failed material runtime')
        if result['telemetry'].get('pair_force')!=pair or result['telemetry']['residency']['accounting_errors']:issues.append('Telemetry or transfer accounting differs')
        maxima=[0.,0.,0.];uploads=downloads=copies=queries=0
        for row in rows:
            e,c=row['epoch'],row['consumer'];f=next(f for f in forces if f['epoch']==e);t=next(t for t in integrals['cycles'] if (t['epoch'],t['consumer'])==(e,c))
            na,nr,nc,cap,ns,n=(row[k] for k in ('atoms','rows','columns','capacity','spinor','queries'));size=3*na+10
            if (nc,nr,cap)!=(f['grid_shape'][0],f['grid_shape'][3],f['grid_shape'][4]) or na!=f['atoms'] or ns!=min(2,f['spins']) or row['components']!=(f['spins'] if c==1 else min(4,f['spins'])):issues.append('Electronic shape differs')
            staged=16+8*size if prescf else 16*n+8*nr+4*cap+16+8*size;exported=8*size
            wanted=dict(status=0,complete=True,passed=True,electronic_inputs_resident=True,electronic_upload_bytes=0,contraction_graph_nodes=1,
                        setup_bytes=staged,device_allocation_bytes=staged+(12 if n==0 and not prescf else 0),export_bytes=exported)
            if any(row.get(k)!=v for k,v in wanted.items()) or n!=t['queries'] or t['graph_nodes']!=(3 if n else 1) or t['passed'] is not True:issues.append('Device contraction/integral evidence differs')
            if not math.isfinite(row['volume']) or row['volume']<=0:issues.append('Invalid volume')
            arrays=[row[k] for k in ('initial','reference','actual')]
            if any(len(a)!=size or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in a) for a in arrays):issues.append('Incomplete/nonfinite output vectors');continue
            initial,reference,actual=arrays
            if initial[0]!=0 or (c==4 and (actual[0]!=0 or reference[0]!=0)):issues.append('Energy convention changed')
            for i,(a,b) in enumerate(zip(actual,reference)):
                group=0 if i==0 else 1 if i<=3*na else 2;err=abs(a-b);maxima[group]=max(maxima[group],err)
                tolerance=(1e-10+1e-10*abs(b)) if group==0 else (1e-10 if group==1 else 1e-12)
                if err>tolerance:issues.append('Independent energy/force/stress comparison failed')
            uploads+=staged;downloads+=exported;copies+=3 if prescf else 6+3*bool(n);queries+=n
        geometry_bytes=geometry_copies=0
        if prescf:
            for g in geometry.get('cycles',[]):
                consumers=[r for r in rows if r['epoch']==g['epoch']]
                if [r['consumer'] for r in consumers]!=[1,4] or any(any(g[k]!=r[k] for k in ('atoms','rows','columns','capacity','queries','volume')) for r in consumers):issues.append('Pre-SCF topology differs between consumers')
                n=g['queries'];staged=8*g['rows']+4*g['capacity']+48*n
                wanted=dict(consumer_mask=5,complete=True,setup_bytes=staged,device_allocation_bytes=staged+(24 if n==0 else 0))
                if any(g.get(k)!=v for k,v in wanted.items()):issues.append('Pre-SCF geometry lifecycle/allocation differs')
                geometry_bytes+=staged;geometry_copies+=3+5*bool(n)
            if result['telemetry'].get('pair_geometry')!=geometry:issues.append('Geometry telemetry differs')
        total=dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
        preparation=dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
        for interval in raw['intervals']:
            b=interval['boundaries'].get('pair_force_shadow')
            if b:
                if interval['phase']!='post_scf':issues.append('Pair diagnostic copies outside post-SCF')
                for k,v in b.items():total[k]+=v
            b=interval['boundaries'].get('pair_geometry')
            if b:
                if interval['phase']!='initialization':issues.append('Pair geometry copied after initial staging')
                for k,v in b.items():preparation[k]+=v
        if total!=dict(host_to_device_bytes=uploads,device_to_host_bytes=downloads,copy_calls=copies):issues.append('Independent shape-derived transfer reconciliation failed')
        if preparation!=dict(host_to_device_bytes=geometry_bytes,device_to_host_bytes=0,copy_calls=geometry_copies):issues.append('Independent pre-SCF geometry transfer reconciliation failed')
        if result['telemetry'].get('gpu_pair_force')!=dict(host_to_device_bytes=uploads+geometry_bytes,device_to_host_bytes=downloads):issues.append('Pair telemetry traffic differs')
        l=next(r for r in ledger if r['benchmark']==case)
        if l.get('performance_comparable') is not False or l.get('end_to_end_speedup') is not None:issues.append('Shadow incorrectly qualifies speedup')
        evidence=[folder/p for p in ('result.json','remag-pair-force.json','remag-two-center.json','remag-residency.json')]+sorted(folder.glob('remag-grid-force-*.json'))
        if prescf:evidence.append(folder/'remag-pair-geometry.json')
        checks.append(dict(case=case,passed=not issues,errors=issues,geometries=len(epochs),consumers=len(rows),queries=queries,max_absolute_error=maxima,
                           geometry_setup_bytes=geometry_bytes,diagnostic_setup_bytes=uploads,diagnostic_export_bytes=downloads,evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,prerequisite_sha256=hashes,verifier_sha256=sha(Path(__file__)),independent_cuda_trace=False,cpu_pair_outputs_authoritative=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'pair-force-material-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'pair-force-material-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Pair material audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r['errors'],[c for c in r['checks'] if not c['passed']]);raise SystemExit(1)
