#!/usr/bin/env python3
"""Qualify complete interleaved two-center value/gradient comparisons."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from check_two_center_gpu import Info

SIZES=(1,63,64,65,257,131073)
FAULTS=(None,'value_small','value_big','gradient_small','gradient_big','nan_actual','inf_reference','overflow','mixed','sticky','signed_zero')
class Comparison(ct.Structure):
    _fields_=[(k,ct.c_ulonglong) for k in ('compared','mismatches','nonfinite')]+[(k,ct.c_double*2) for k in ('absolute','scaled')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def expected(a,b,initial):
    absolute=[];scaled=[];bad=nf=0
    for aa,bb in ((a[:,0],b[:,0]),(a[:,1:],b[:,1:])):
        with np.errstate(invalid='ignore',over='ignore'):delta=np.abs(aa-bb)
        finite=np.isfinite(aa)&np.isfinite(bb)&np.isfinite(delta);nf+=int((~finite).sum())
        dd=delta[finite];ratio=dd/(1e-10+1e-10*np.abs(bb[finite]));bad+=int((ratio>1).sum())
        absolute.append(float(dd.max(initial=0.)));scaled.append(float(ratio.max(initial=0.)))
    return dict(compared=a.size,mismatches=bad,nonfinite=nf,absolute=absolute,scaled=scaled,status=max(initial,3 if bad else 0,2 if nf else 0))
def run(library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    fn=ct.CDLL(str(library.resolve())).remag_check_two_center_comparison
    fn.argtypes=[ct.c_int]*3+[ct.c_void_p]*2+[ct.POINTER(Comparison),ct.POINTER(ct.c_int),ct.POINTER(Info)];fn.restype=ct.c_int
    rng=np.random.default_rng(81062);checks=[];arrays={}
    for n in SIZES:
        ref=rng.normal(size=(n,4))
        for fault in FAULTS if n in (65,131073) else (None,):
            a=ref.copy();b=ref.copy();initial=7 if fault=='sticky' else 0
            if fault and fault.endswith(('small','big')):
                j=0 if fault.startswith('value') else 3;a[-1,j]+=(.5 if fault.endswith('small') else 2.)*(1e-10+1e-10*abs(b[-1,j]))
            elif fault=='nan_actual':a[-1,3]=np.nan
            elif fault=='inf_reference':b[-1,0]=np.inf
            elif fault=='overflow':a[-1,3]=np.finfo('d').max;b[-1,3]=-np.finfo('d').max
            elif fault=='mixed':a[-1,3]=np.nan;a[-1,0]+=1.
            elif fault=='signed_zero':a[-1,3]=-0.;b[-1,3]=0.
            wanted=expected(a,b,initial)
            for captured in (False,True):
                info=Info();value=Comparison();status=ct.c_int(-1)
                code=fn(n,int(captured),initial,a.ctypes.data,b.ctypes.data,ct.byref(value),ct.byref(status),ct.byref(info))
                actual={k:getattr(value,k) for k in ('compared','mismatches','nonfinite')};actual.update(absolute=list(value.absolute),scaled=list(value.scaled),status=status.value)
                name=f'n{n}_captured{int(captured)}_{fault}'
                valid=(code==0 and actual==wanted and info.stage==4 and info.graph_nodes==int(captured) and info.forbidden_nodes==0 and
                       info.staged==64*n+60 and info.allocated==info.staged and info.exported==60 and info.poisoned==0)
                checks.append(dict(scenario=name,n=n,captured=captured,fault=fault,initial_status=initial,native_return=code,actual=actual,expected=wanted,passed=valid,
                                   **{k:getattr(info,k) for k,_ in info._fields_}))
                arrays[name+'_actual']=a;arrays[name+'_reference']=b
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays)
    report=dict(scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_library_sha256=sha(library),output_file=archive.name,output_sha256=sha(archive))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Two-center comparison:',len(checks),'passed=',report['passed'])
    if not report['passed']:print([r for r in checks if not r['passed']]);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.library,a.output)
