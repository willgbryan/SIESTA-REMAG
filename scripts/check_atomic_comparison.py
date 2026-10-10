#!/usr/bin/env python3
"""Qualify full atomic selection, index and derivative comparison on retained arrays."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from check_atomic_basis_gpu import Info

COUNTERS=('points','selected','reference_selected','compared','index_compared','mismatches','nonfinite','mask_mismatches','index_mismatches')
SIZES=(1,127,128,129,4097,131073)
FAULTS=(None,'phi_small','phi_big','gradient_small','gradient_big','displacement','nan_actual','inf_reference','sticky','signed_zero',
        'mask_actual','mask_reference','invalid_mask','point','atom','inactive_phi','inactive_gradient','inactive_ignored')
class Comparison(ct.Structure):
    _fields_=[(k,ct.c_ulonglong) for k in COUNTERS]+[(k,ct.c_double*3) for k in ('absolute','scaled')]


def expected(a,ai,b,bi,n,initial):
    absolute=[];scaled=[];bad=nonfinite=0;mask=bi[:n]==1;m=int(mask.sum())
    for k,(lo,hi) in enumerate(((0,n),(n,4*n),(4*n,7*n))):
        aa=a[lo:hi];bb=b[lo:hi]
        if k==2:aa=aa.reshape(n,3)[mask];bb=bb.reshape(n,3)[mask]
        with np.errstate(invalid='ignore',over='ignore'):delta=np.abs(aa-bb)
        finite=np.isfinite(aa)&np.isfinite(bb)&np.isfinite(delta);nonfinite+=int((~finite).sum());dd=delta[finite]
        vv=dd if k==2 else dd/(1e-10+1e-10*np.abs(bb[finite]))
        bad+=int((vv!=0. if k==2 else vv>1.).sum());absolute.append(float(dd.max(initial=0.)));scaled.append(float(vv.max(initial=0.)))
    masks=int(((ai[:n]!=bi[:n])|~np.isin(ai[:n],[0,1])|~np.isin(bi[:n],[0,1])).sum())
    indices=int((ai[n:].reshape(2,n)[:,mask]!=bi[n:].reshape(2,n)[:,mask]).sum())
    return dict(points=n,selected=int((ai[:n]==1).sum()),reference_selected=m,compared=4*n+3*m,index_compared=2*m,
                mismatches=bad,nonfinite=nonfinite,mask_mismatches=masks,index_mismatches=indices,absolute=absolute,scaled=scaled,
                status=max(initial,3 if bad or masks or indices else 0,2 if nonfinite else 0))


def run(library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    fn=ct.CDLL(str(library.resolve())).remag_check_atomic_comparison
    fn.argtypes=[ct.c_int]*3+[ct.c_void_p]*4+[ct.POINTER(Comparison),ct.POINTER(ct.c_int),ct.POINTER(Info)];fn.restype=ct.c_int
    rng=np.random.default_rng(62791);checks=[];arrays={}
    for n in SIZES:
        reference=rng.normal(size=7*n);indices=np.concatenate(((np.arange(n)%3!=1).astype('i'),np.arange(n,dtype='i')+1,np.ones(n,dtype='i')))
        # Keep the last point selected, including the capped launch tail.
        indices[n-1]=1
        for fault in FAULTS if n in (129,131073) else (None,):
            a=reference.copy();b=reference.copy();ai=indices.copy();bi=indices.copy();initial=7 if fault=='sticky' else 0
            if fault in ('phi_small','phi_big','gradient_small','gradient_big'):
                j=n-1 if fault.startswith('phi') else 4*n-1;a[j]+=(.5 if fault.endswith('small') else 2.)*(1e-10+1e-10*abs(b[j]))
            elif fault=='displacement':a[-1]=np.nextafter(b[-1],np.inf)
            elif fault=='nan_actual':a[-1]=np.nan
            elif fault=='inf_reference':b[n-1]=np.inf
            elif fault=='signed_zero':a[-1]=-0.;b[-1]=0.
            elif fault=='mask_actual':ai[n-1]=0
            elif fault=='mask_reference':bi[n-1]=0
            elif fault=='invalid_mask':ai[n-1]=bi[n-1]=2
            elif fault=='point':ai[2*n-1]+=1
            elif fault=='atom':ai[3*n-1]+=1
            elif fault=='inactive_phi':a[1]+=1.
            elif fault=='inactive_gradient':a[n+3]+=1.
            elif fault=='inactive_ignored':a[4*n+3]=np.nan;ai[n+1]=-1;ai[2*n+1]=-1
            wanted=expected(a,ai,b,bi,n,initial)
            for captured in (False,True):
                info=Info();value=Comparison();status=ct.c_int(-1)
                code=fn(n,int(captured),initial,a.ctypes.data,ai.ctypes.data,b.ctypes.data,bi.ctypes.data,ct.byref(value),ct.byref(status),ct.byref(info))
                actual={k:getattr(value,k) for k in COUNTERS};actual.update(absolute=list(value.absolute),scaled=list(value.scaled),status=status.value)
                name=f'n{n}_captured{int(captured)}_'+str(fault)
                valid=(code==0 and actual==wanted and info.stage==4 and info.graph_nodes==int(captured) and info.forbidden_nodes==0 and
                       info.staged==136*n+124 and info.allocated==info.staged and info.exported==124 and info.poisoned==0)
                checks.append(dict(scenario=name,n=n,captured=captured,fault=fault,initial_status=initial,native_return=code,
                                   actual=actual,expected=wanted,passed=valid,**{k:getattr(info,k) for k,_ in info._fields_}))
                for key,v in (('actual',a),('reference',b),('actual_indices',ai),('reference_indices',bi)):arrays[name+'_'+key]=v
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),
                native_library_sha256=sha(library),output_file=archive.name,output_sha256=sha(archive))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Atomic comparison:',len(checks),'passed=',report['passed'])
    if not report['passed']:print([r for r in checks if not r['passed']]);raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.library,a.output)
