#!/usr/bin/env python3
"""Reproduce masked contraction inputs and outputs against retained pristine arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root):
    folder=root/'atomic-masked-force-check';path=folder/'check.json';report=json.loads(path.read_text());oracle=folder/report['oracle_file'];ref=json.loads(oracle.read_text())
    data=np.load(folder/ref['arrays_file']);arrays=np.load(folder/report['output_file']);fixtures={v['fixture']:v for v in ref['checks']};checks=[];errors=[]
    for label,row in fixtures.items():
        values=[data[label+'_'+key] for key in ('gradient','displacement','potential','auxiliary','deformation','initial_force','initial_stress')]
        raw=(np.array([row['kind'],row['entries'],row['points'],row['atoms'],row['nsd'],row['forces'],row['stress']],np.int32).tobytes()+
             np.array([row['dvol'],row['volume']]).tobytes()+data[label+'_points'].tobytes()+data[label+'_atoms'].tobytes()+b''.join(v.tobytes(order='F') for v in values))
        if hashlib.sha256(raw).hexdigest()!=row['input_sha256']:errors.append('Pristine oracle input hash differs: '+label)
    expected=set();names=set();input_map={}
    faults=('all_inactive','bad_mask','bad_point','bad_atom','gradient','displacement','sticky','inactive_fields')
    for kind,nsd in ((1,1),(1,2),(2,1)):
        for n in (1,127,128,129,257,258,131073):
            for f in (None,*(faults if n in (129,131073) else ())):names.add(f'kind{kind}_nsd{nsd}_n{n}_{f}')
    for item in report['inputs']:
        label=item['fixture'];name=item['name'];n=item['entries'];fault=item['fault'];source=fixtures[label];issues=[]
        if name in input_map:errors.append('Duplicate input');continue
        input_map[name]=item
        m=1 if n<257 else 257;slots=np.linspace(0,n-1,m,dtype=np.int32) if m>1 else np.array([n-1],np.int32)
        if source['entries']!=m or item['slots']!=slots.tolist() or name!=f"kind{source['kind']}_nsd{source['nsd']}_n{n}_{fault}":issues.append('Padded input shape/mapping differs')
        fields=dict(mask=np.zeros(n,np.int32),points=np.full(n,-1,np.int32),atoms=np.full(n,-1,np.int32),gradient=np.full((3,n),np.nan),displacement=np.full((3,n),np.inf))
        fields['mask'][slots]=1
        for k in ('points','atoms'):fields[k][slots]=data[label+'_'+k]
        for k in ('gradient','displacement'):fields[k][:,slots]=data[label+'_'+k]
        for k in ('potential','auxiliary','deformation','initial_force','initial_stress'):fields[k]=data[label+'_'+k].copy()
        if fault in ('all_inactive','inactive_fields'):fields['mask'][:]=0
        if fault=='inactive_fields':
            for k in ('potential','auxiliary','deformation'):fields[k][:]=np.nan
        if fault=='bad_mask':fields['mask'][-1]=2
        if fault=='bad_point':fields['points'][-1]=0
        if fault=='bad_atom':fields['atoms'][-1]=source['atoms']+1
        if fault=='gradient':fields['gradient'][0,-1]=np.nan
        if fault=='displacement':fields['displacement'][0,-1]=np.nan
        if item['initial_status']!=(7 if fault=='sticky' else 0):issues.append('Initial status differs')
        for k,value in fields.items():
            if not np.array_equal(value,arrays[name+'_'+k],equal_nan=True):issues.append('Retained input differs: '+k)
        if issues:errors.extend(name+': '+v for v in issues)
        for captured in (False,True):
            for cycles in ((1,3) if fault is None else (3,)):expected.add((name,captured,cycles))
    seen=set()
    for r in report['checks']:
        name,captured,cycles=(r[k] for k in ('input','captured','cycles'));key=(name,captured,cycles);issues=[]
        if key in seen:issues.append('Duplicate execution')
        seen.add(key);item=input_map[name];src=fixtures[item['fixture']];f=item['fault'];n=item['entries'];label=item['fixture']
        status=1 if f in ('bad_mask','bad_point','bad_atom') else 2 if f in ('gradient','displacement') else 7 if f=='sticky' else 0
        if (r['native_return']!=0 or r['status']!=status or r['expected_status']!=status or r['stage']!=4 or r['graph_nodes']!=int(captured) or r['forbidden_nodes']!=0):issues.append('Native status/graph failed')
        upload=60*n+32+8*src['points']*(src['nsd']+2)+8*(3*src['atoms']+9)+4
        if r['staged']!=upload or r['allocated']!=upload or r['exported']!=8*(3*src['atoms']+9)+4:issues.append('Traffic differs')
        scenario=name+f'_captured{int(captured)}_cycles{cycles}';actual=arrays[scenario+'_result'];error=None
        if f in (None,'all_inactive','inactive_fields','sticky'):
            initial=np.r_[data[label+'_initial_force'].ravel(order='F'),data[label+'_initial_stress'].ravel(order='F')]
            wanted=initial+cycles*(data[label+'_result']-initial) if f is None else initial
            if actual.shape!=wanted.shape or not np.isfinite(actual).all():issues.append('Nonfinite/wrong output')
            else:
                error=float(np.max(np.abs(actual-wanted)))
                if error>=1e-12 or error!=r['max_absolute_error']:issues.append('Pristine arithmetic comparison failed')
        elif r['max_absolute_error'] is not None:issues.append('Invalid-data scenario claims numerical comparison')
        if r['scenario']!=scenario or r['passed'] is not True:issues.append('Recorded scenario failed')
        checks.append(dict(scenario=scenario,passed=not issues,errors=issues,max_absolute_error=error))
    if set(input_map)!=names or len(input_map)!=69 or seen!=expected or len(checks)!=180 or len(arrays.files)!=870:errors.append('Required coverage differs')
    evidence=(path,oracle,folder/ref['arrays_file'],folder/ref['wrapper_file'],folder/report['output_file'])
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
                evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence},verifier_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'atomic-masked-force-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'atomic-masked-force-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Masked atomic force audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
