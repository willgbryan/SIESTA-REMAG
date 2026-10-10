#!/usr/bin/env python3
"""Qualify complete device derivative comparison before using compact shadows."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from check_orbital_basis_gpu import Info


class Comparison(ct.Structure):
    _fields_ = [(n,ct.c_ulonglong) for n in ('compared','mismatches','nonfinite')]+[(n,ct.c_double*3) for n in ('absolute','scaled')]


def expected(a,b,n,initial):
    absolute=[];scaled=[];bad=0;nonfinite=0
    for i,(start,end) in enumerate(((0,n),(n,4*n),(4*n,7*n))):
        aa=a[start:end];bb=b[start:end]
        with np.errstate(invalid='ignore',over='ignore'):delta=np.abs(aa-bb)
        finite=np.isfinite(aa)&np.isfinite(bb)&np.isfinite(delta)
        nonfinite+=int(np.count_nonzero(~finite));dd=delta[finite];rr=bb[finite]
        vv=dd if i==2 else dd/(1e-10+1e-10*np.abs(rr))
        bad+=int(np.count_nonzero(vv!=0. if i==2 else vv>1.))
        absolute.append(float(dd.max(initial=0.)));scaled.append(float(vv.max(initial=0.)))
    return dict(compared=7*n,mismatches=bad,nonfinite=nonfinite,absolute=absolute,scaled=scaled,
                status=max(initial,3 if bad else 0,2 if nonfinite else 0))


def run(library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    fn=ct.CDLL(str(library.resolve())).remag_check_orbital_comparison
    fn.argtypes=[ct.c_int]*3+[ct.c_void_p]*2+[ct.POINTER(Comparison),ct.POINTER(ct.c_int),ct.POINTER(Info)];fn.restype=ct.c_int
    rng=np.random.default_rng(66235);checks=[];arrays={}
    for n in (1,127,128,257,4097,65537):
        reference=rng.normal(size=7*n)
        faults=(None,) if n not in (257,65537) else (None,'phi_small','phi_big','gradient_small','gradient_big','displacement','nan_actual','inf_reference','sticky','signed_zero')
        for fault in faults:
            a=reference.copy();b=reference.copy();initial=7 if fault=='sticky' else 0
            if fault in ('phi_small','phi_big','gradient_small','gradient_big'):
                j=n-1 if fault.startswith('phi') else 4*n-1
                a[j]+=((.5 if fault.endswith('small') else 2.)*(1e-10+1e-10*abs(b[j])))
            elif fault=='displacement':a[-1]=np.nextafter(b[-1],np.inf)
            elif fault=='nan_actual':a[-1]=np.nan
            elif fault=='inf_reference':b[n-1]=np.inf
            elif fault=='signed_zero':a[-1]=-0.;b[-1]=0.
            wanted=expected(a,b,n,initial)
            for captured in (False,True):
                info=Info();value=Comparison();status=ct.c_int(-1)
                code=fn(n,int(captured),initial,a.ctypes.data,b.ctypes.data,ct.byref(value),ct.byref(status),ct.byref(info))
                actual={k:getattr(value,k) for k in ('compared','mismatches','nonfinite')}
                actual.update(absolute=list(value.absolute),scaled=list(value.scaled),status=status.value)
                name=f'n{n}_captured{int(captured)}_'+str(fault)
                valid=(code==0 and actual==wanted and info.stage==4 and info.graph_nodes==int(captured) and info.forbidden_nodes==0 and
                    info.staged==112*n+76 and info.allocated==info.staged and info.exported==76)
                checks.append(dict(scenario=name,n=n,captured=captured,fault=fault,initial_status=initial,native_return=code,
                    actual=actual,expected=wanted,passed=valid,**{k:getattr(info,k) for k,_ in info._fields_}))
                arrays[name+'_actual']=a;arrays[name+'_reference']=b
    archive=output.with_suffix('.npz');np.savez_compressed(archive,**arrays)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),
        native_library_sha256=sha(library),output_file=archive.name,output_sha256=sha(archive))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Orbital comparison:',len(checks),'passed=',report['passed'])
    if not report['passed']:
        print([r for r in checks if not r['passed']]);raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.library,a.output)
