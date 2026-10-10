#!/usr/bin/env python3
"""Exercise shared GPU force accumulation, ownership guards and sticky failure.

Tiny analytic fixtures combine the separately upstream-qualified core, neutral
and dfscf kernels. This checks ownership/accumulation, not material geometry.
Accumulator staging counters exclude this check's independent fixture setup.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np


class Info(ct.Structure):
    _fields_=[('accumulator_staged_bytes',ct.c_size_t)]+[(name,ct.c_int) for name in ('status','graphs','rejections','consumer_mask')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    f=ct.CDLL(str(library.resolve())).remag_check_final_force
    f.argtypes=[ct.c_int]*5+[ct.c_void_p,ct.POINTER(Info)];f.restype=ct.c_int
    checks=[];arrays={}
    scenarios=[(nh,core,0,force,stress) for nh in (1,2,4,8) for core in (0,1) for force,stress in ((1,1),(1,0),(0,1))]
    scenarios += [(8,core,mode,1,1) for core in (0,1) for mode in range(1,10)]
    for nh,core,mode,force,stress in scenarios:
        actual=np.full(12,np.nan);info=Info();code=f(nh,core,mode,force,stress,actual.ctypes.data,ct.byref(info))
        ng=min(nh,4);nsd=min(nh,2);v=np.array([1.2,1.8,.2,.3]);d=np.array([.4,.3,.2,-.1,.07,.08,.12,.14])[:ng].copy()
        if nh==8:d[2:]=[.16,.02]
        weights=v[:ng].copy()
        if ng==4:weights[2:]*=2
        contribution=np.zeros(3)
        if core:contribution+=.1*(v[:nsd].mean()-.6)*np.array([.2,-.3,.4])
        if mode!=9:
            contribution+=.1*.8*np.array([-.2,.5,-.1])
            contribution+=2*.1*(.7*np.dot(d,weights)-.7*.3*.6)*np.array([.3,.2,-.4])
        initial=np.r_[[.1,-.2,.3],np.arange(1,10)*.01];expected=initial.copy()
        if force:expected[:3]+=contribution
        if stress:expected[3:]+=np.outer([.4,.2,-.3],contribution).ravel(order='F')/2
        error=float(np.max(np.abs(actual-expected)))
        passed=(code==0 and info.status==(2 if mode==9 else 0) and info.graphs==core+2 and
                info.rejections==(1 if 1<=mode<=8 else 0) and info.consumer_mask==6+core and
                info.accumulator_staged_bytes==100+88*(core+1) and np.isfinite(actual).all() and error<1e-12)
        label=f'spin{nh}_core{core}_mode{mode}_force{force}_stress{stress}'
        arrays[label+'_actual']=actual;arrays[label+'_expected']=expected
        checks.append(dict(fixture=label,spins=nh,partial_core=core,mode=mode,forces=force,stress=stress,
            native_return=code,**{k:getattr(info,k) for k,_ in info._fields_},max_absolute_error=error,passed=bool(passed)))
    data=output.with_suffix('.npz');np.savez_compressed(data,**arrays)
    report=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,
        checker_sha256=sha(Path(__file__)),native_library_sha256=sha(library),output_file=data.name,output_sha256=sha(data))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Shared final-force scenarios:',len(checks),'passed=',report['passed'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.library,args.output)
