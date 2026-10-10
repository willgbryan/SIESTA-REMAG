#!/usr/bin/env python3
"""Check derived GPU populations against pristine spnvec and upstream reduction order.

Fixtures include empty atoms, interleaved species, signed collinear spin and a
second capped-grid stride. Staging/checking is outside the two-kernel capture.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np


class Problem(ct.Structure):
    _fields_=[(k,ct.c_int) for k in ('rows','atoms','species','components')]


class Info(ct.Structure):
    _fields_=[(k,ct.c_size_t) for k in ('allocated','staged','exported','poisoned')]+[(k,ct.c_int) for k in ('graph_nodes','forbidden_nodes','stage')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(upstream,build,library,output,reference_source):
    output=output.resolve();folder=output.parent;folder.mkdir(parents=True,exist_ok=True)
    upstream,build=upstream.resolve(),build.resolve()
    text=(upstream/'Src/mulliken.F').read_text();spin=text[text.index('      subroutine spnvec('):]
    (folder/'spnvec-reference.F').write_text(spin)
    (folder/'reference.f90').write_bytes(reference_source.read_bytes())
    compiler=shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
    rng=np.random.default_rng(9826);arrays={};checks=[]
    fn=ct.CDLL(str(library.resolve())).remag_check_population_observables
    fn.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.POINTER(Problem)]+[ct.c_void_p]*6+[ct.POINTER(ct.c_int),ct.POINTER(Info)]
    fn.restype=ct.c_int
    with tempfile.TemporaryDirectory(prefix='remag-derived-population-') as temporary:
        tmp=Path(temporary);executable=tmp/'reference'
        subprocess.run([compiler,'-O2','-ffixed-line-length-none','-I'+str(build/'Src'),str(folder/'reference.f90'),str(folder/'spnvec-reference.F'),'-o',str(executable)],check=True)
        for components,mode,rows in [(c,m,9) for c in (1,2,4) for m in ('ordinary','zero','negative_spin')]+[(4,'ordinary',262145)]:
            config=np.array([rows,4,3,components],np.int32);ids=np.array([2,1,3,1],np.int32);z=np.array([6.,8.,10.])
            lasto=[0,rows//4,rows//4,2*rows//3,rows]
            orb=rng.normal(.1,.3,(rows,components))
            if mode=='zero':orb[:]=0
            if mode=='negative_spin' and components>=2:orb[:,0]=.1;orb[:,1]=.7
            atom=np.array([orb[a:b].sum(axis=0) for a,b in zip(lasto[:-1],lasto[1:])]);total=orb.sum(axis=0)
            (tmp/'input.bin').write_bytes(b''.join(a.tobytes() for a in (config,ids,z,orb,atom,total)))
            subprocess.run([str(executable),str(tmp/'input.bin'),str(tmp/'output.bin')],check=True)
            reference=np.fromfile(tmp/'output.bin',dtype='f8')
            label=f'components{components}_{mode}_rows{rows}'
            variants=['normal']
            if components==4 and mode=='ordinary':variants+=['nonfinite_orbital']
            if components==4 and mode=='ordinary' and rows==9:variants+=['problem_shape','bad_species','nonfinite_valence','net_overflow','sticky_status']
            for variant in variants:
                p=Problem(*config);values=dict(ids=ids.copy(),valence=z.copy(),orbital=orb.copy(),atomic=atom.copy(),total=total.copy())
                initial=0;wanted=0
                if variant=='problem_shape':p.rows+=1;wanted=1
                if variant=='bad_species':values['ids'][-1]=4;wanted=1
                if variant=='nonfinite_orbital':values['orbital'][-1,0]=np.nan;wanted=2
                if variant=='nonfinite_valence':values['valence'][2]=np.nan;wanted=2
                if variant=='net_overflow':values['valence'][0]=np.finfo('d').max;values['atomic'][1,0]=-1.e308;wanted=2
                if variant=='sticky_status':initial=7;wanted=7
                key=label+'_'+variant
                arrays[key+'_config']=config;arrays[key+'_problem']=np.array([p.rows,p.atoms,p.species,p.components],np.int32)
                arrays[key+'_reference']=reference
                for name,value in values.items():arrays[key+'_'+name]=value
                for capture in (0,1):
                    out=np.full(reference.shape,np.nan);status=ct.c_int(-9);info=Info()
                    result=fn(config.ctypes.data,capture,initial,ct.byref(p),*[values[n].ctypes.data for n in ('ids','valence','orbital','atomic','total')],out.ctypes.data,ct.byref(status),ct.byref(info))
                    staged=16+4*4+8*3+8*components*(rows+4+1)+4
                    good=result==0 and status.value==wanted and info.stage==4 and info.graph_nodes==2*capture and info.forbidden_nodes==0
                    good=good and info.staged==staged and info.allocated==staged+8*out.size and info.exported==8*out.size+4 and info.poisoned==24*out.size
                    error=None
                    if wanted==0:
                        error=float(np.max(np.abs(out-reference)));good=good and np.isfinite(out).all() and np.allclose(out,reference,atol=1e-12,rtol=1e-12)
                    if variant in ('problem_shape','sticky_status'):good=good and np.all(out==0)
                    name=key+f'_capture{capture}';arrays[name+'_output']=out
                    checks.append(dict(fixture=key,capture=capture,initial_status=initial,status=status.value,expected_status=wanted,native_return=result,
                                       inventory={n:int(getattr(info,n)) for n,_ in Info._fields_},max_absolute_error=error,passed=bool(good)))
    np.savez_compressed(folder/'arrays.npz',**arrays)
    report=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_library_sha256=sha(library),
                upstream_mulliken_sha256=sha(upstream/'Src/mulliken.F'),evidence_sha256={name:sha(folder/name) for name in ('arrays.npz','spnvec-reference.F','reference.f90')})
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Derived population observables:',len(checks),'passed=',report['passed'],flush=True)
    if not report['passed']:print([c for c in checks if not c['passed']]);raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','library','output','reference-source'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.upstream,args.build,args.library,args.output,args.reference_source)
