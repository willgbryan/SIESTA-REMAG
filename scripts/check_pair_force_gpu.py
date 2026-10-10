#!/usr/bin/env python3
"""Qualify GPU kinetic/overlap contractions against literal pristine Fortran."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import numpy as np
from check_pair_force import run as oracle,sha,FIELDS

FAULTS=('volume','volume_nan','kind','row','column','atom_a','atom_b','offset','count','list','gradient','xyz','density','value_nan','unused_spin',
        'arithmetic_overflow','force_overflow','energy_overflow','sticky','tail_row','tail_gradient')
HOST=('spinor','capacity','components')
class Problem(ct.Structure):_fields_=[('volume',ct.c_double),('kind',ct.c_int),('padding',ct.c_int)]
class Info(ct.Structure):_fields_=[(k,ct.c_size_t) for k in ('allocated','staged','exported')]+[(k,ct.c_int) for k in ('graph_nodes','forbidden_nodes','stage')]

def inputs(data,label,variant):
    d={k:data[label+'_'+k].copy(order='C') for k in FIELDS};config=data[label+'_config'].copy();kind,volume=data[label+'_parameters'];kind=int(kind)
    wanted=0;initial=0;expected=data[label+'_expected'].copy();stage=4
    if variant in ('volume','volume_nan','kind','row','column','atom_a','atom_b','offset','count','list'):
        wanted=1;expected=d['initial'].copy()
        if variant=='volume':volume=0.
        elif variant=='volume_nan':volume=np.nan
        elif variant=='kind':kind=0
        elif variant=='row':d['qr'][:]=np.iinfo('i').min
        elif variant=='column':d['qc'][:]=config[3]+1
        elif variant=='atom_a':d['atoms'][:,0]=0
        elif variant=='atom_b':d['atoms'][:,1]=np.iinfo('i').min
        elif variant=='offset':d['ptr'][:]=-1
        elif variant=='count':d['num'][:]=config[4]+1
        elif variant=='list':d['list'][:]=config[3]+1
    elif variant in ('gradient','xyz','density','arithmetic_overflow'):
        wanted=2;expected=d['initial'].copy()
        if variant=='gradient':d['integrals'][:,1:]=np.nan
        elif variant=='xyz':d['xyz'][:]=np.inf
        elif variant=='density':d['density'][:]=np.nan
        else:
            d['qr'][:]=1;d['qc'][:]=1;d['density'][:]=np.finfo('d').max;d['integrals'][:,1:]=np.finfo('d').max
    elif variant=='value_nan':
        d['integrals'][:,0]=np.nan
        if kind==1:wanted=2;expected=d['initial'].copy()
    elif variant=='unused_spin':d['density'][config[6]:]=np.nan
    elif variant in ('force_overflow','energy_overflow'):
        d['qr'][:]=1;d['qc'][:]=1;d['atoms'][:]=[1,2];d['density'][:]=0.;d['density'][0,1:3]=.75;d['xyz'][:]=0.;d['integrals'][:]=0.
        expected=d['initial'].copy()
        if variant=='force_overflow':
            d['integrals'][:,1:]=np.finfo('d').max/2;expected[1:4]=np.inf if kind==1 else -np.inf;expected[4:7]=-expected[1:4];wanted=2
        else:
            d['integrals'][:,0]=np.finfo('d').max/2
            if kind==1:expected[0]=np.inf;wanted=2
    elif variant=='sticky':wanted=initial=7
    elif variant in ('tail_row','tail_gradient'):
        expected=data[label+'_without_last'].copy();wanted=1 if variant=='tail_row' else 2
        if variant=='tail_row':d['qr'][-1]=0
        else:d['integrals'][-1,3]=np.nan
    elif variant in HOST:
        stage=3 if variant=='spinor' else 0
        if variant=='spinor':config[6]=0
        elif variant=='capacity':config[4]=0
        else:config[5]=1
    return d,config,kind,float(volume),expected,wanted,initial,stage

def run(upstream,build,library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);ref=output.parent/'reference.json';oracle(upstream,build,ref)
    manifest=json.loads(ref.read_text());data=np.load(ref.with_suffix('.npz'));fn=ct.CDLL(str(library.resolve())).remag_check_pair_force
    fn.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.POINTER(Problem)]+[ct.c_void_p]*11+[ct.POINTER(ct.c_int),ct.POINTER(Info)];fn.restype=ct.c_int
    checks=[];arrays={};executed={}
    for row in manifest['checks']:
        label=row['fixture'];variants=['normal']
        if row['components']==8 and row['queries']==257 and row['cycles']==1:variants.extend((*FAULTS,*HOST))
        if row['components']==8 and row['queries']==131073 and row['cycles']==1:variants.extend(('tail_row','tail_gradient'))
        for variant in variants:
            d,c,kind,volume,expected,wanted,initial,stage=inputs(data,label,variant);key=label+'_'+variant
            for name,value in d.items():executed[key+'_'+name]=value
            executed[key+'_config']=c;executed[key+'_parameters']=np.array([kind,volume,initial]);executed[key+'_expected']=expected
            for captured in ((False,) if variant in HOST else (False,True)):
                name=key+f'_captured{int(captured)}';out=np.full(expected.shape,np.nan);code=ct.c_int(-9);info=Info();problem=Problem(volume,kind,0)
                result=fn(c.ctypes.data,int(captured),initial,ct.byref(problem),*[d[k].ctypes.data for k in FIELDS],out.ctypes.data,ct.byref(code),ct.byref(info))
                n,na,nr,nc,cap,components,spinor,cycles=map(int,c);staged=72*n+8*nr+4*cap+8*cap*components+24*na+100
                if variant in HOST:
                    valid=result==-1 and info.stage==stage and info.exported==0 and code.value==-9 and np.isnan(out).all();error=None
                    if stage==0:valid=valid and info.staged==0 and info.allocated==0
                else:
                    finite=np.isfinite(expected).all();error=float(np.max(np.abs(out-expected))) if finite else None
                    agree=bool(np.isfinite(out).all() and np.all(np.abs(out-expected)<=1e-10+1e-10*np.abs(expected))) if finite else np.array_equal(out,expected)
                    valid=(result==0 and code.value==wanted and info.stage==4 and info.graph_nodes==int(captured) and info.forbidden_nodes==0 and agree and
                           info.staged==staged and info.allocated==staged+(28 if n==0 else 0) and info.exported==8*(3*na+10)+4)
                arrays[name]=out
                checks.append(dict(scenario=name,fixture=label,variant=variant,inputs_key=key,captured=captured,native_return=result,status=code.value,expected_status=wanted,
                                   initial_status=initial,max_absolute_error=error,passed=bool(valid),**{k:getattr(info,k) for k,_ in info._fields_}))
    output_arrays=output.with_suffix('.npz');np.savez_compressed(output_arrays,**arrays);input_arrays=output.parent/'inputs.npz';np.savez_compressed(input_arrays,**executed)
    report=dict(scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_library_sha256=sha(library),
                output_file=output_arrays.name,output_sha256=sha(output_arrays),inputs_file=input_arrays.name,inputs_sha256=sha(input_arrays),
                oracle_file=ref.name,oracle_sha256=sha(ref),oracle_arrays_sha256=sha(ref.with_suffix('.npz')))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Pair-force CUDA:',len(checks),'passed=',report['passed'])
    if not report['passed']:print([r for r in checks if not r['passed']]);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream,a.build,a.library,a.output)
