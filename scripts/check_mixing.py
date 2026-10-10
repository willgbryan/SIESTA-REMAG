#!/usr/bin/env python3
"""Retain pristine stable-Pulay sequence oracles before migrating SCF mixing.

The linked upstream mixer is unmodified; an independent NumPy implementation
checks stack rollover and reset semantics. Optional CUDA checks validate the
isolated device plan, not full SCF convergence or other mixer variants.
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
    _fields_=[('size',ct.c_int),('history',ct.c_int),('weight',ct.c_double),
              ('linear_weight',ct.c_double),('condition',ct.c_double),('epoch',ct.c_uint64)]


class Result(ct.Structure):
    _fields_=[(name,ct.c_int) for name in ('status','iteration','history_size','method','rank','oldest','next','active')]+[
        ('gram_max',ct.c_double),('projection_max',ct.c_double),
        ('inverse_first',ct.c_double),('coefficient_first',ct.c_double)]


class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_uint64),('graph_nodes',ct.c_int),('forbidden_nodes',ct.c_int),('iterations',ct.c_int)]

WRAPPER='''program reference
 use precision, only: dp
 use m_mixing, only: tMixer,MIX_PULAY,mixer_init,mixers_history_init,mixers_reset,mixing
 use class_Fstack_dData1D, only: n_items
 implicit none
 type(tMixer),pointer :: mixers(:)
 type(tMixer),pointer :: m
 integer :: n,spins,history,cycles,reset_before,fi,fo,k
 real(dp) :: weight,linear_weight
 real(dp),allocatable :: xin(:,:),residual(:,:),mixed(:,:)
 character(2048) :: input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) n,spins,history,cycles,reset_before,weight,linear_weight
 allocate(xin(n,spins),residual(n,spins),mixed(n,spins))
 allocate(mixers(1))
 m=>mixers(1);m%name='reference';m%m=MIX_PULAY;m%v=0;m%w=weight;m%n_hist=history
 call mixer_init(mixers(1));m%rv(1)=linear_weight
 call mixers_history_init(mixers)
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 do k=1,cycles
   if(k==reset_before) call mixers_history_init(mixers)
   read(fi) xin,residual
   call mixing(m,n,spins,xin,residual,mixed)
   write(fo) mixed,real(n_items(m%stack(1)),dp),real(n_items(m%stack(2)),dp)
 enddo
 close(fi);close(fo);call mixers_reset(mixers)
end program
'''


def fixtures():
    rng=np.random.default_rng(20261006)
    for label,n,spins,history,cycles,reset in (
        ('scalar_history2',23,1,2,10,0),
        ('collinear_history6',257,2,6,18,0),
        ('noncollinear_history6',139,4,6,20,0),
        ('soc_history8',61,8,8,24,0),
        ('same_shape_geometry_reset',53,4,6,20,10),
        ('singular_history_svd',17,2,6,16,0),
        ('scaled_small_residual',127,8,8,20,0),
        ('maximum_history16',53,4,16,40,0),
        ('different_initial_weight',29,2,6,12,0)):
        x=rng.normal(size=(cycles,n*spins))*.3
        f=rng.normal(size=x.shape)*.02
        if label=='singular_history_svd':
            # Exact rank-one residual differences force the inverse/SVD
            # fallback after the first one-dimensional history solve.
            f=np.arange(1,cycles+1)[:,None]*np.arange(1,n*spins+1)[None,:]/1024.
        if label=='scaled_small_residual': f*=1.e-10
        yield label,n,spins,history,reset,.05,.13 if label=='different_initial_weight' else .05,x,f


def stable_pulay(x,residual,history,weight,reset_before,linear_weight=None):
    """Independent arithmetic oracle for the supported stable/all-spin mode."""
    dx=[];df=[];previous_f=None;previous_out=None;outputs=[];counts=[];svds=0
    for cycle,(xin,f) in enumerate(zip(x,residual),1):
        if cycle==reset_before:
            dx=[];df=[];previous_f=previous_out=None
        if previous_f is not None:
            # Preserve the original stack arithmetic, including its cancellation.
            dx.append(previous_f-previous_out+xin)
            df.append(f-previous_f)
            dx=dx[-(history-1):];df=df[-(history-1):]
        value=xin+(weight if df or linear_weight is None else linear_weight)*f
        if df:
            differences=np.array(df)
            gram=differences@differences.T
            try:
                inv=np.linalg.inv(gram)
                if np.max(np.abs(gram@inv-np.eye(len(df))))>1.e-4:
                    raise np.linalg.LinAlgError('inverse residual')
            except np.linalg.LinAlgError:
                # These fixtures exercise successful inversion and actual SVD
                # fallback. They do not qualify the simple-inverse fallback.
                inv=np.linalg.pinv(gram,rcond=1.e-8);svds+=1
            coeff=-(inv@(differences@f))
            for c,d,r in zip(coeff,dx,df): value=value+c*(d+weight*r)
        outputs.append(value);counts.append((len(dx)+1,len(df)))
        previous_f=f.copy();previous_out=xin+f
    return np.array(outputs),np.array(counts),svds


def run(upstream,build,output,library=None):
    upstream=upstream.resolve();build=build.resolve();src=build/'Src'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(reference='Unmodified m_mixing stable Pulay, spin components flattened together',
                gpu_validation=None,fixture_script_sha256=sha(Path(__file__)),
                source_sha256={name:sha(upstream/'Src'/name) for name in
                               ('m_mixing.F90','mixer.F','class_Fstack_Data1D.F90','Fstack.T90')},
                reference_library_sha256=sha(src/'libsiesta.a'),checks=[],gpu_checks=[])
    candidate=None
    if library:
        lib=ct.CDLL(str(library.resolve()));candidate=lib.remag_check_pulay
        candidate.argtypes=[ct.POINTER(Problem),ct.c_int,ct.c_int]+[ct.c_void_p]*3+[ct.POINTER(Result),ct.c_int,ct.POINTER(Check)]
        candidate.restype=ct.c_int;report['native_library_sha256']=sha(library)
        probe=lib.remag_check_pulay_coefficient_probe
        probe.argtypes=[ct.c_double,ct.c_double,ct.c_void_p];probe.restype=ct.c_int
        report['reciprocal_probes']=[]
        for gram,projection in ((.018303768517120487,.008114485713359586),(1.,.5),(2.,.125)):
            values=np.zeros(9);assert probe(gram,projection,values.ctypes.data)==0
            print('Pulay reciprocal probe',gram,projection,values.tolist(),flush=True)
            np.testing.assert_allclose(values,[gram,1/gram,1/gram,0.,1/gram,-projection/gram,
                                               0.,1/gram,-projection/gram],atol=1.e-12,rtol=1.e-12)
            report['reciprocal_probes'].append(dict(gram=gram,projection=projection,values=values.tolist(),passed=True))
    def execute(problem,x,f,reset,graph):
        x=np.ascontiguousarray(x);f=np.ascontiguousarray(f);actual=np.full_like(x,123.)
        records=(Result*len(x))();info=Check()
        code=candidate(ct.byref(problem),len(x),reset,x.ctypes.data,f.ctypes.data,actual.ctypes.data,records,int(graph),ct.byref(info))
        assert code==0,(code,graph)
        assert info.iterations==len(x) and info.forbidden_nodes==0
        if graph: assert info.graph_nodes==11
        return actual,records,info
    arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-mixing-') as tmp:
        tmp=Path(tmp);wrapper=tmp/'reference.f90';wrapper.write_text(WRAPPER);exe=tmp/'reference'
        link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
        libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
        flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
        includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
        subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(wrapper),'-o',str(exe),*libraries],cwd=src,check=True)
        for label,n,spins,history,reset,weight,linear_weight,x,f in fixtures():
            inp=tmp/'input';out=tmp/'output';cycles=len(x)
            inp.write_bytes(np.array([n,spins,history,cycles,reset],dtype=np.int32).tobytes()+
                            np.array([weight,linear_weight],dtype=np.float64).tobytes()+
                            b''.join(a.tobytes()+b.tobytes() for a,b in zip(x,f)))
            completed=subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,capture_output=True,text=True)
            if completed.returncode:
                print(completed.stdout,completed.stderr,flush=True);completed.check_returncode()
            raw=np.fromfile(out,dtype=np.float64).reshape((cycles,n*spins+2))
            assert np.isfinite(raw).all(),label
            values,counts,svds=stable_pulay(x,f,history,weight,reset,linear_weight)
            np.testing.assert_array_equal(raw[:,-2:],counts)
            np.testing.assert_allclose(raw[:,:-2],values,atol=1.e-9,rtol=1.e-10)
            fallback=completed.stdout.count('inversion failed, > SVD')
            if label=='singular_history_svd': assert fallback>0 and svds>0
            for name,a in (('input',x),('residual',f),('mixed',raw[:,:-2]),('history_counts',counts)):
                arrays[label+'_'+name]=a
            report['checks'].append(dict(fixture=label,rows=n,spin_components=spins,history=history,cycles=cycles,
                reset_before=reset,weight=weight,linear_weight=linear_weight,upstream_svd_fallback_count=fallback,
                max_absolute_error=float(np.max(np.abs(raw[:,:-2]-values))),input_sha256=sha(inp),
                upstream_log=completed.stdout,passed=True))
            if candidate:
                problem=Problem(n*spins,history,weight,linear_weight,1.e-8,1)
                for graph in (False,True):
                    actual,records,info=execute(problem,x,f,reset,graph)
                    if any(r.status for r in records):
                        failure=output.with_name('pulay-failure.npz')
                        failure.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(failure,input=x,residual=f,expected=raw[:,:-2],actual=actual,
                            records=np.array([[getattr(r,name) for name,_ in Result._fields_] for r in records]))
                        print('Pulay failure',label,'graph',graph,
                              [{name:getattr(r,name) for name,_ in Result._fields_} for r in records],flush=True)
                    assert all(r.status==0 for r in records),(label,[r.status for r in records])
                    np.testing.assert_array_equal([r.history_size for r in records],counts[:,1])
                    np.testing.assert_array_equal([r.iteration for r in records],
                        [k if not reset or k<reset else k-reset+1 for k in range(1,cycles+1)])
                    np.testing.assert_allclose(actual,raw[:,:-2],atol=1.e-9,rtol=1.e-10)
                    report['gpu_checks'].append(dict(fixture=label,graph=graph,iterations=info.iterations,
                        graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,workspace_bytes=info.workspace_bytes,
                        methods=[r.method for r in records],ranks=[r.rank for r in records],
                        gram_max=[r.gram_max for r in records],projection_max=[r.projection_max for r in records],
                        max_absolute_error=float(np.max(np.abs(actual-raw[:,:-2]))),passed=True))
        if candidate:
            # First step fails. The next step must stay poisoned despite valid
            # inputs, then a device-side geometry epoch resets the history.
            x=np.arange(64,dtype=float).reshape((4,16))*.001
            f=np.full_like(x,.02);f[0,0]=np.nan;f[3]=f[2]+np.linspace(.001,.01,16)
            p=Problem(16,6,.05,.05,1.e-8,1)
            for graph in (False,True):
                actual,records,info=execute(p,x,f,3,graph)
                assert [r.status for r in records]==[2,2,0,0]
                np.testing.assert_array_equal(actual[:2],123.)
                expected,_,_=stable_pulay(x[2:],f[2:],6,.05,0)
                np.testing.assert_allclose(actual[2:],expected,atol=1.e-9,rtol=1.e-10)
                report['gpu_checks'].append(dict(fixture='invalid_density_history_reset',graph=graph,
                    statuses=[r.status for r in records],iterations=info.iterations,graph_nodes=info.graph_nodes,
                    forbidden_nodes=info.forbidden_nodes,passed=True))
            for invalid in ('nan_weight','zero_weight','negative_linear_weight','invalid_cutoff'):
                p=Problem(16,6,.05,.05,1.e-8,1)
                if invalid=='nan_weight': p.weight=np.nan
                elif invalid=='zero_weight': p.weight=0.
                elif invalid=='negative_linear_weight': p.linear_weight=-1.
                else: p.condition=1.
                actual,records,info=execute(p,np.zeros((2,16)),np.ones((2,16)),0,True)
                assert all(r.status==1 for r in records)
                np.testing.assert_array_equal(actual,123.)
                report['gpu_checks'].append(dict(fixture=invalid,graph=True,statuses=[r.status for r in records],
                    iterations=info.iterations,graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            report['gpu_validation']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**arrays)
    report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact),passed=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Stable-Pulay reference sequences: {len(report['checks'])}; GPU checks: {len(report['gpu_checks'])}; passed=True; GPU validation={report['gpu_validation']}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path)
    args=p.parse_args();run(args.upstream,args.build,args.output,args.library)
