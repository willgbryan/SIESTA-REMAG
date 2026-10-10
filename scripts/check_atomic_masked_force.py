#!/usr/bin/env python3
"""Qualify masked core/neutral contraction using padded pristine upstream fixtures."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from check_atomic_grid_force_formula import run as build_oracle
from check_atomic_grid_force_gpu import Problem,Info

SIZES=(1,127,128,129,257,258,131073)
FAULTS=('all_inactive','bad_mask','bad_point','bad_atom','gradient','displacement','sticky','inactive_fields')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(upstream,build,library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);oracle=output.with_name('reference.json')
    build_oracle(upstream,build,oracle);source=json.loads(oracle.read_text());data=np.load(oracle.with_suffix('.npz'))
    fn=ct.CDLL(str(library.resolve())).remag_check_atomic_masked_force
    fn.argtypes=[ct.c_int]*6+[ct.POINTER(Problem)]+[ct.c_void_p]*10+[ct.POINTER(ct.c_int),ct.POINTER(Info),ct.c_void_p,ct.c_int];fn.restype=ct.c_int
    checks=[];retained={};inputs=[]
    for kind,nsd in ((1,1),(1,2),(2,1)):
        for n in SIZES:
            m=1 if n<257 else 257
            row=next(r for r in source['checks'] if r['kind']==kind and r['nsd']==nsd and r['entries']==m and r['forces'] and r['stress'])
            label=row['fixture'];slots=np.linspace(0,n-1,m,dtype=np.int32) if m>1 else np.array([n-1],np.int32)
            for fault in (None,*(FAULTS if n in (129,131073) else ())):
                name=f'kind{kind}_nsd{nsd}_n{n}_{fault}'
                mask=np.zeros(n,np.int32);mask[slots]=1;points=np.full(n,-1,np.int32);atoms=points.copy()
                gradient=np.full((3,n),np.nan,order='F');displacement=np.full((3,n),np.inf,order='F')
                points[slots]=data[label+'_points'];atoms[slots]=data[label+'_atoms'];gradient[:,slots]=data[label+'_gradient'];displacement[:,slots]=data[label+'_displacement']
                arrays=[np.array(data[label+'_'+key],order='F',copy=True) for key in ('potential','auxiliary','deformation','initial_force','initial_stress')]
                initial=7 if fault=='sticky' else 0;status_expected=initial
                if fault=='all_inactive' or fault=='inactive_fields':mask[:]=0
                if fault=='inactive_fields':
                    for a in arrays[:3]:a[:]=np.nan
                if fault=='bad_mask':mask[-1]=2;status_expected=1
                if fault=='bad_point':points[-1]=0;status_expected=1
                if fault=='bad_atom':atoms[-1]=row['atoms']+1;status_expected=1
                if fault=='gradient':gradient[0,-1]=np.nan;status_expected=2
                if fault=='displacement':displacement[0,-1]=np.nan;status_expected=2
                fields=dict(mask=mask,points=points,atoms=atoms,gradient=gradient,displacement=displacement,
                    **dict(zip(('potential','auxiliary','deformation','initial_force','initial_stress'),arrays)))
                for key,value in fields.items():retained[name+'_'+key]=value
                inputs.append(dict(name=name,fixture=label,entries=n,slots=slots.tolist(),fault=fault,initial_status=initial))
                for captured in (False,True):
                    for cycles in ((1,3) if fault is None else (3,)):
                        p=Problem(row['dvol'],row['volume'],kind,nsd,row['forces'],row['stress']);result=np.full(3*row['atoms']+9,np.nan);status=ct.c_int(-1);info=Info()
                        code=fn(row['points'],n,row['atoms'],nsd,int(captured),cycles,ct.byref(p),points.ctypes.data,atoms.ctypes.data,
                            gradient.ctypes.data,displacement.ctypes.data,*[a.ctypes.data for a in arrays],result.ctypes.data,ct.byref(status),ct.byref(info),mask.ctypes.data,initial)
                        staged=60*n+32+8*row['points']*(nsd+2)+8*(3*row['atoms']+9)+4
                        valid=(code==0 and status.value==status_expected and info.stage==4 and info.graph_nodes==int(captured) and info.forbidden_nodes==0 and
                               info.staged==staged and info.allocated==staged and info.exported==8*(3*row['atoms']+9)+4)
                        error=None
                        if fault in (None,'all_inactive','inactive_fields','sticky'):
                            base=np.r_[arrays[-2].ravel(order='F'),arrays[-1].ravel(order='F')]
                            expected=base+cycles*(data[label+'_result']-base) if fault is None else base
                            error=float(np.max(np.abs(result-expected)));valid=valid and np.isfinite(result).all() and error<1e-12
                        key=name+f'_captured{int(captured)}_cycles{cycles}';retained[key+'_result']=result
                        checks.append(dict(scenario=key,input=name,captured=captured,cycles=cycles,native_return=code,status=status.value,expected_status=status_expected,
                            max_absolute_error=error,passed=bool(valid),**{k:getattr(info,k) for k,_ in info._fields_}))
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
    report=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,inputs=inputs,oracle_file=oracle.name,oracle_sha256=sha(oracle),
        oracle_arrays_sha256=sha(oracle.with_suffix('.npz')),native_library_sha256=sha(library),checker_sha256=sha(Path(__file__)),output_file=artifact.name,output_sha256=sha(artifact))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Masked atomic force:',len(checks),'passed=',report['passed'])
    if not report['passed']:print([r for r in checks if not r['passed']]);raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.library,a.output)
