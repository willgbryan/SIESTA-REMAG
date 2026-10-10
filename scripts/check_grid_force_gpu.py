#!/usr/bin/env python3
"""Check the device-only dfscf contraction against retained pinned arithmetic.

Synthetic local geometry only; this does not qualify periodic sparse lookup,
basis generation, other force terms, complete material forces or residency.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
from check_grid_force_formula import run as build_oracle


class Problem(ct.Structure):
    _fields_=[('dvol',ct.c_double),('volume',ct.c_double),('forces',ct.c_int),('stress',ct.c_int)]


class Info(ct.Structure):
    _fields_=[(name,ct.c_size_t) for name in ('allocated','staged','exported')]+[
        (name,ct.c_int) for name in ('graph_nodes','forbidden_nodes','stage')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(upstream,build,library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    oracle=output.with_name('reference.json');build_oracle(upstream,build,oracle)
    source=json.loads(oracle.read_text());data=np.load(oracle.with_suffix('.npz'))
    f=ct.CDLL(str(library.resolve())).remag_check_grid_force
    f.argtypes=[ct.c_int]*6+[ct.POINTER(Problem)]+[ct.c_void_p]*11+[ct.POINTER(ct.c_int),ct.POINTER(Info)];f.restype=ct.c_int
    checks=[];retained={}
    names=('density','atomic_occupation','phi','gradient','displacement','potential','auxiliary','initial_force','initial_stress')
    def evaluate(row,captured,cycles,fault=None):
        label=row['fixture'];atom=data[label+'_atom'].copy()
        arrays=[np.asfortranarray(data[label+'_'+name]).copy(order='F') for name in names]
        p=Problem(row['dvol'],row['volume'],row['forces'],row['stress']);expected_status=0
        if fault=='volume':p.volume=0.;expected_status=1
        elif fault=='flags':p.forces=2;expected_status=1
        elif fault=='atom':atom[0]=0;expected_status=1
        elif fault=='density':arrays[0].flat[0]=np.nan;expected_status=2
        elif fault=='phi':arrays[2].flat[0]=np.nan;expected_status=2
        elif fault=='gradient':arrays[3].flat[0]=np.nan;expected_status=2
        result=np.full(3*row['atoms']+9,np.nan);status=ct.c_int(-1);info=Info()
        code=f(row['nc'],row['nsp'],row['spins'],row['atoms'],int(captured),cycles,ct.byref(p),atom.ctypes.data,
            *[a.ctypes.data for a in arrays],result.ctypes.data,ct.byref(status),ct.byref(info))
        name=f'{label}_captured{int(captured)}_cycles{cycles}'+('_'+fault if fault else '')
        record=dict(fixture=label,captured=captured,cycles=cycles,fault=fault,native_return=code,status=status.value,
            expected_status=expected_status,**{key:getattr(info,key) for key,_ in info._fields_})
        valid=code==0 and status.value==expected_status and info.stage==4 and info.forbidden_nodes==0 and (not captured or info.graph_nodes==1)
        if expected_status==0:
            initial=np.r_[arrays[-2].ravel(order='F'),arrays[-1]]
            expected=initial+cycles*(data[label+'_result']-initial)
            error=float(np.max(np.abs(expected-result)))
            record['max_absolute_error']=error;valid=valid and np.isfinite(result).all() and error<1.e-12
        record['passed']=bool(valid);checks.append(record);retained[name]=result
    for row in source['checks']:
        for captured in (False,True):
            for cycles in (1,3):evaluate(row,captured,cycles)
    row=next(r for r in source['checks'] if r['spins']==8 and r['nc']==17 and r['forces'] and r['stress'])
    for fault in ('volume','flags','atom','density','phi','gradient'):
        for captured in (False,True):evaluate(row,captured,3,fault)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
    report=dict(scope=__doc__,checks=checks,passed=all(c['passed'] for c in checks),
        oracle_file=oracle.name,oracle_sha256=sha(oracle),oracle_arrays_sha256=sha(oracle.with_suffix('.npz')),
        native_library_sha256=sha(library),checker_sha256=sha(Path(__file__)),output_file=artifact.name,output_sha256=sha(artifact))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Grid-force CUDA scenarios:',len(checks),'passed=',report['passed'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.library,a.output)
