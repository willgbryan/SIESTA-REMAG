#!/usr/bin/env python3
"""GPU S/T table evaluator vs pristine get_matel, including captured replay and faults.

This is an isolated primitive. Material tables, neighbor topology and final-force
consumers are not integrated. Retains full GPU arrays and every executed input.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
from check_two_center import run as build_oracle,sha

ATOL=RTOL=1e-10
FIELDS=('ind','indffy','ilm','indffr','cutoffs','radial','coefficients','pairs','xyz')
INDEX_FAULTS=('pair_low','pair_high','harmonic_low','harmonic_high','term_low','term_high','term_order','angular','radial_index','cutoff','base')
VALUE_FAULTS=('xyz','radial','coefficient','range','cutoff_overflow','radius_overflow','result_overflow')
HOST_FAULTS=('delta','pi','bounds')
class Info(ct.Structure):
    _fields_=[(k,ct.c_size_t) for k in ('allocated','staged','exported','poisoned')]+[(k,ct.c_int) for k in ('graph_nodes','forbidden_nodes','stage')]

def descriptor(d):
    _,nr,ny,rows,cols,b1,b2,nf,_=map(int,d['config'])
    return np.array([nf,nr,ny,rows,cols,b1,b2,*d['bounds']],np.int32)

def inputs(data,label,variant):
    d={k.removeprefix(label+'_'):data[k].copy(order='C') for k in data.files if k.startswith(label+'_')}
    config=descriptor(d);delta=float(d['delta'][0]);pi=float(np.pi);wanted=0;initial=0
    if variant=='stride':
        d['pairs']=np.resize(d['pairs'],(131073,2));d['xyz']=np.resize(d['xyz'],(131073,3));d['expected']=np.resize(d['expected'],(131073,4))
    if variant=='empty':
        config[1:3]=0;d['radial']=np.empty((0,2,1025));d['ilm']=np.empty(0,np.int32);d['indffr']=np.empty(0,np.int32)
        d['coefficients']=np.empty(0);d['indffy'][:]=1;d['expected'][:]=0.
    if variant=='sticky':wanted=initial=7
    if variant in (*INDEX_FAULTS,*VALUE_FAULTS):
        d['pairs'][:]=[1,1];d['xyz'][:]=[.3,.4,.5];wanted=1 if variant in INDEX_FAULTS else 2;d['expected'][:]=0.
        if variant=='pair_low':d['pairs'][:,0]=0
        elif variant=='pair_high':d['pairs'][:,1]=config[0]+1
        elif variant=='harmonic_low':d['ind'][0]=0
        elif variant=='harmonic_high':d['ind'][1]=np.iinfo(np.int32).max
        elif variant=='term_low':d['indffy'][:]=0
        elif variant=='term_high':d['indffy'][:]=config[2]+2
        elif variant=='term_order':d['indffy'][:]=1;d['indffy'][0]=2
        elif variant=='angular':d['ilm'][:]=122
        elif variant=='radial_index':d['indffr'][:]=0
        elif variant=='cutoff':d['cutoffs'][:]=-1.
        elif variant=='base':config[5]=np.iinfo(np.int32).max
        elif variant=='xyz':d['xyz'][:]=np.nan
        elif variant=='radial':d['radial'][:]=np.nan
        elif variant=='coefficient':d['coefficients'][:]=np.inf
        elif variant=='range':delta=1e-6
        elif variant=='cutoff_overflow':d['cutoffs'][:]=1e308
        elif variant=='radius_overflow':d['xyz'][:]=1e308
        elif variant=='result_overflow':d['radial'][:]=1e308;d['coefficients'][:]=1e308
    if variant in HOST_FAULTS:
        wanted=-1
        if variant=='delta':delta=0.
        if variant=='pi':pi=np.nan
        if variant=='bounds':config[7]=0
    return d,config,delta,pi,initial,wanted

def run(upstream,build,library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);reference=output.with_name('reference.json')
    build_oracle(upstream,build,reference);oracle=json.loads(reference.read_text());data=np.load(reference.with_suffix('.npz'))
    lib=ct.CDLL(str(library.resolve()));fn=lib.remag_check_two_center
    fn.argtypes=[ct.c_void_p,ct.c_double,ct.c_double]+[ct.c_int]*4+[ct.c_void_p]*10+[ct.POINTER(ct.c_int),ct.POINTER(Info)];fn.restype=ct.c_int
    checks=[];retained={};raw_inputs={}
    plans=[(r['fixture'],'ordinary',c,cycles) for r in oracle['checks'] for c in (False,True) for cycles in (1,3)]
    plans += [('spd_T',v,c,3) for v in (*INDEX_FAULTS,*VALUE_FAULTS,'sticky','empty') for c in (False,True)]
    plans += [('high_T','stride',c,cycles) for c in (False,True) for cycles in (1,3)]
    plans += [('spd_T',v,False,1) for v in HOST_FAULTS]
    for label,variant,captured,cycles in plans:
        d,config,delta,pi,initial,wanted=inputs(data,label,variant);arrays=[d[k] for k in FIELDS];n=len(d['pairs'])
        name=f'{label}_{variant}_captured{int(captured)}_cycles{cycles}';key=f'{label}_{variant}'
        for k,v in dict(zip(FIELDS,arrays),config=config,parameters=np.array([delta,pi]),expected=d['expected']).items():raw_inputs[key+'_'+k]=v
        result=np.full((n,4),np.nan);status=ct.c_int(-1);info=Info()
        native=fn(config.ctypes.data,delta,pi,n,int(captured),cycles,initial,*[a.ctypes.data for a in arrays],result.ctypes.data,ct.byref(status),ct.byref(info))
        staged=sum(a.nbytes for a in arrays)+4;allocated=staged+result.nbytes+sum(a.dtype.itemsize for a in arrays if a.size==0)
        host=variant in HOST_FAULTS
        passed=(info.staged==staged and info.allocated==allocated and info.forbidden_nodes==0)
        passed=passed and ((native==-1 and info.stage==3 and info.exported==0 and info.graph_nodes==0 and info.poisoned==result.nbytes) if host else
            (native==0 and status.value==wanted and info.stage==4 and info.exported==result.nbytes+4 and info.graph_nodes==int(captured) and info.poisoned==cycles*result.nbytes))
        row=dict(scenario=name,fixture=label,variant=variant,input_key=key,queries=n,captured=captured,cycles=cycles,native_return=native,status=status.value,
            expected_status=wanted,expected_staged=staged,expected_allocated=allocated,**{k:getattr(info,k) for k,_ in info._fields_})
        if not host:
            error=np.abs(result-d['expected']);scaled=error/(ATOL+RTOL*np.abs(d['expected']))
            row.update(max_absolute_error=float(np.max(error)),max_scaled_error=float(np.max(scaled)))
            passed=passed and np.isfinite(result).all() and np.all(scaled<=1.)
            if variant in (*INDEX_FAULTS,*VALUE_FAULTS,'empty'):passed=passed and np.all(result==0.)
        row['passed']=bool(passed);checks.append(row);retained[name]=result
        if not passed:print('FAILED',row,flush=True)
    outputs=output.with_suffix('.npz');np.savez_compressed(outputs,**retained);source=output.with_name('inputs.npz');np.savez_compressed(source,**raw_inputs)
    report=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,atol=ATOL,rtol=RTOL,
        oracle_file=reference.name,oracle_sha256=sha(reference),oracle_arrays_sha256=sha(reference.with_suffix('.npz')),
        inputs_file=source.name,inputs_sha256=sha(source),output_file=outputs.name,output_sha256=sha(outputs),
        native_library_sha256=sha(library),checker_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Two-center GPU:',len(checks),'passed=',report['passed'],flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('upstream','build','library','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.library,a.output)
