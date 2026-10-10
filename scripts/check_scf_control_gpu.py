#!/usr/bin/env python3
"""Couple upstream convergence/Pulay oracles inside a bounded device loop.

Scalar and residual producers are pre-staged fixtures, not an electronic SCF
calculation. Fresh workers isolate each graph, including forced-error checks.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import check_mixing as mixing
import check_scf_convergence as convergence
from check_sparse_state import mixing_executable
from check_scf_convergence_gpu import Problem as ConvergenceProblem, Scalars, Result as ConvergenceResult
from check_block_jacobi_gpu import observer_environment, calibrate_observer


class Problem(ct.Structure):
    _fields_=[('convergence',ConvergenceProblem),('mixer',mixing.Problem),('mix_first',ct.c_int),('reserved',ct.c_int)]


class Result(ct.Structure):
    _fields_=[('convergence',ConvergenceResult),('mixer',mixing.Result)]+[
        (name,ct.c_int) for name in ('status','records','mixed','first_mix_bypassed','continue_loop','finished')]+[
        ('reserved',ct.c_int*2)]


class Step(ct.Structure):
    _fields_=[('scalars',Scalars),('control',Result)]


class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_size_t)]+[(name,ct.c_int) for name in ('graph_nodes','forbidden_nodes','iterations','truncated')]


assert (ct.sizeof(Problem),ct.sizeof(Result),ct.sizeof(Step))==(112,128,240)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def raw(value): return np.frombuffer(bytes(value),np.uint8).copy()
def ints(value): return [getattr(value,name) for name,_ in ConvergenceResult._fields_]
def flags(value): return [getattr(value,name) for name in ('status','records','mixed','first_mix_bypassed','continue_loop','finished')]


def worker(config):
    cfg=json.loads(config.read_text());a=np.load(cfg['input']);x=a['x'];f=a['f'];count,n=x.shape
    ps=(Problem*count).from_buffer_copy(a['problems']);ss=(Scalars*count).from_buffer_copy(a['scalars'])
    results=(Result*count)();steps=(Step*count)();info=Check();values=np.full_like(x,321.)
    call=ct.CDLL(cfg['library']).remag_check_scf_control
    call.argtypes=[ct.c_int]*5+[ct.POINTER(Problem),ct.POINTER(Scalars)]+[ct.c_void_p]*3+[
        ct.POINTER(Result),ct.POINTER(Step),ct.POINTER(Check)];call.restype=ct.c_int
    code=call(n,3,cfg['capacity'],count,cfg['launches'],ps,ss,x.ctypes.data,f.ctypes.data,values.ctypes.data,results,steps,ct.byref(info))
    output=Path(cfg['output']);np.savez_compressed(output.with_suffix('.npz'),values=values,results=raw(results),steps=raw(steps))
    report={name:getattr(info,name) for name,_ in info._fields_};report['native_return']=code
    report['control_flags']=[flags(r) for r in results];report['convergence']=[ints(r.convergence) for r in results]
    report['mixer_status']=[r.mixer.status for r in results]
    if cfg['trace']:
        from siesta_remag.cuda_trace import summarize_cuda
        explicit=dict(host_to_device_bytes=count*(ct.sizeof(Problem)+ct.sizeof(Scalars))+x.nbytes+f.nbytes,
                      device_to_host_bytes=values.nbytes+ct.sizeof(results)+ct.sizeof(steps)+8)
        try:
            trace=json.loads((output.parent/'remag-activity.json').read_text())
            report['cuda_trace']=summarize_cuda(trace,dict(total_explicit=explicit,accounting_errors=[]),
                expected_scf_iterations=1,process_succeeded=code==0,allow_unique_graph_scope=True,memory_only=True)
        except Exception as error:report['cuda_trace']=dict(coverage_complete=False,errors=[str(error)])
    output.write_text(json.dumps(report,indent=2)+'\n')


def fixtures(upstream,build,directory):
    oracle=directory/'convergence-reference.json';convergence.run(upstream,build,oracle)
    reference=json.loads(oracle.read_text());a=np.load(directory/reference['arrays_file']);tol=a['tolerances']
    rng=np.random.default_rng(6060618);records=[]
    with tempfile.TemporaryDirectory(prefix='remag-control-oracle-') as temporary:
        tmp=Path(temporary);exe=mixing_executable(build,tmp)
        (directory/'mixing-reference.f90').write_text(mixing.WRAPPER)
        def mixed(x,f):
            inp=tmp/'input';out=tmp/'output';count,n=x.shape
            inp.write_bytes(np.array([n,1,3,count,0],np.int32).tobytes()+np.array([.05,.13]).tobytes()+
                           b''.join(v.tobytes()+r.tobytes() for v,r in zip(x,f)))
            subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,check=True,capture_output=True)
            return np.fromfile(out).reshape(count,n+2)[:,:n]
        def configuration(mask,minimum,maximum,epoch,mix_first,n):
            return Problem(ConvergenceProblem(mask,minimum,maximum,0,(ct.c_double*5)(*tol),epoch),
                           mixing.Problem(n,3,.05,.13,1.e-8,epoch),mix_first,0)
        def save(label,ps,ss,x,f,expected,expected_flags,decisions,launches,capacity):
            path=directory/(label+'.npz')
            np.savez_compressed(path,problems=raw(ps),scalars=raw(ss),x=x,f=f,expected=expected,
                                expected_flags=expected_flags,expected_convergence=decisions)
            records.append(dict(fixture=label,input_file=path.name,input_sha256=sha(path),launches=launches,
                                capacity=capacity,steps=len(x),size=x.shape[1]))
        for case_index,case in enumerate(reference['checks']):
            source=a[case['fixture']+'_values'];wanted=a[case['fixture']+'_decisions'];reset=case['reset_before']
            segments=[(source,wanted)] if not reset else [(source[:reset-1],wanted[:reset-1]),(source[reset-1:],wanted[reset-1:])]
            # Reuse one executable across two explicit epochs, even when shape
            # and settings match. The reset fixture already has two segments.
            if len(segments)==1:segments*=2
            n=(19,38,76,152,257)[case_index%5]
            for mix_first in (0,1):
                all_p=[];all_s=[];all_x=[];all_f=[];values=[];expected_flags=[];decisions=[]
                capacity=max(len(v) for v,_ in segments)
                for epoch,(v,w) in enumerate(segments,1):
                    maximum=len(v);eligible=np.flatnonzero(w[:,1]);length=int(eligible[0]+1) if len(eligible) else maximum
                    x=rng.normal(size=(length,n))*.1;f=rng.normal(size=(length,n))*.02;m=mixed(x,f)
                    for k in range(length):
                        p=configuration(case['criteria_mask'],case['minimum_iterations'],maximum,epoch,mix_first,n)
                        scalar=Scalars();scalar.active=1
                        for j in range(3):scalar.values[6+j]=v[k,j]
                        scalar.values[3]=v[k,3];scalar.values[4]=v[k,4]
                        matched,converged,harris,free,iteration=map(int,w[k]);mix=1-converged
                        more=mix and iteration<maximum;exhausted=mix and iteration==maximum
                        bypass=mix and iteration==1 and not mix_first
                        values.append(x[k]+f[k] if bypass else m[k] if mix else np.full(n,123.))
                        expected_flags.append([0,k+1,mix,int(bypass),int(more),int(not more)])
                        decisions.append([0,iteration,matched,converged,mix,int(more),int(exhausted),harris+2*free])
                        all_p.append(p);all_s.append(scalar)
                    all_x.extend(x);all_f.extend(f)
                save(case['fixture']+f'_mixfirst{mix_first}',(Problem*len(all_p))(*all_p),(Scalars*len(all_s))(*all_s),
                    np.array(all_x),np.array(all_f),np.array(values),np.array(expected_flags),np.array(decisions),2,capacity)
        # Failed producers/mixers must stop the outer loop, stay poisoned on
        # same-epoch replay, then recover under a new explicit epoch.
        for fault in ('scalar_status','scalar_nan','mixer_nan'):
            n=38;count=5;x=rng.normal(size=(count,n))*.1;f=rng.normal(size=(count,n))*.02
            ps=(Problem*count)();ss=(Scalars*count)()
            for k in range(count):
                ps[k]=configuration(7,0,2,1 if k<3 else 2,0,n)
                for j in range(3):ss[k].values[6+j]=1.
                ss[k].active=1
            if fault=='scalar_status':ss[1].status=2
            elif fault=='scalar_nan':ss[1].values[3]=np.nan
            else:f[1,0]=np.nan
            expected=np.full_like(x,123.);expected[0]=x[0]+f[0];expected[3:]=mixed(x[3:],f[3:]);expected[3]=x[3]+f[3]
            status=3 if fault=='mixer_nan' else 2
            ef=np.array([[0,1,1,1,1,0],[status,2,0,0,0,1],[status,2,0,0,0,1],[0,1,1,1,1,0],[0,2,1,0,0,1]])
            failed=[0,2,0,0,1,0,1,3] if fault=='mixer_nan' else [2,1,0,0,0,0,0,0]
            ec=np.array([[0,1,0,0,1,1,0,0],failed,failed,[0,1,0,0,1,1,0,0],[0,2,0,0,1,0,1,3]])
            save(fault,ps,ss,x,f,expected,ef,ec,3,2)
        for fault in ('zero_weight','epoch_mismatch','history_overflow','invalid_first_mix'):
            n=19;count=3;x=rng.normal(size=(count,n))*.1;f=rng.normal(size=(count,n))*.02
            ps=(Problem*count)();ss=(Scalars*count)()
            for k in range(count):
                ps[k]=configuration(7,0,1,1 if k<2 else 2,0,n)
                for j in range(3):ss[k].values[6+j]=1.
                ss[k].active=1
            if fault=='zero_weight':ps[0].mixer.weight=0.
            elif fault=='epoch_mismatch':ps[0].mixer.epoch=2
            elif fault=='history_overflow':ps[0].convergence.maximum=2
            else:ps[0].mix_first=2
            expected=np.full_like(x,123.);expected[2]=x[2]+f[2]
            ef=np.array([[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,0,1]])
            ec=np.array([[0]*8,[0]*8,[0,1,0,0,1,0,1,0]])
            save(fault,ps,ss,x,f,expected,ef,ec,3,1)
    return records,dict(oracle_file=oracle.name,oracle_sha256=sha(oracle),oracle_arrays_sha256=reference['arrays_sha256'],
        mixer_wrapper_sha256=sha(directory/'mixing-reference.f90'),reference_library_sha256=sha(build/'Src/libsiesta.a'))


def run(upstream,build,output,library=None,trace=False):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    cases,provenance=fixtures(upstream.resolve(),build.resolve(),output.parent)
    report=dict(scope='Device convergence/conditional Pulay/bounded history with staged producers; not full SCF',
        fixtures=cases,checks=[],gpu_validation=None,fixture_script_sha256=sha(Path(__file__)),**provenance)
    if not library:
        report['passed']=True;output.write_text(json.dumps(report,indent=2)+'\n');print('Control fixtures:',len(cases));return
    library=library.resolve();observer=library.parent/'libremag_activity.so';report['native_library_sha256']=sha(library)
    if trace:
        report['observer_calibration']=calibrate_observer(output.parent,observer,library.parent/'remag_activity_check',True)
        if not all(c['passed'] for c in report['observer_calibration']):
            report['passed']=False;output.write_text(json.dumps(report,indent=2)+'\n');raise SystemExit('Control observer calibration failed')
    for case in cases:
        folder=output.parent/case['fixture'];folder.mkdir();config=folder/'config.json';result=folder/'result.json'
        config.write_text(json.dumps(dict(input=str(output.parent/case['input_file']),output=str(result),library=str(library),
            trace=trace,capacity=case['capacity'],launches=case['launches']),indent=2)+'\n')
        record=dict(fixture=case['fixture'],numerical_passed=False,residency_passed=None)
        try:
            env=observer_environment(observer,Path(sys.executable),True) if trace else os.environ.copy()
            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],cwd=folder,env=env,
                                capture_output=True,text=True,timeout=60)
            (folder/'stdout.log').write_text(done.stdout+done.stderr);record['process_returncode']=done.returncode
            assert done.returncode==0,done.returncode
            actual=json.loads(result.read_text());record.update(actual)
            assert actual['native_return']==0 and actual['iterations']==case['steps'] and actual['truncated']==0,actual
            assert actual['forbidden_nodes']==0 and actual['graph_nodes']>0
            a=np.load(result.with_suffix('.npz'));expected=np.load(output.parent/case['input_file'])
            np.testing.assert_allclose(a['values'],expected['expected'],atol=1.e-9,rtol=1.e-10)
            np.testing.assert_array_equal(actual['control_flags'],expected['expected_flags'])
            np.testing.assert_array_equal(actual['convergence'],expected['expected_convergence'])
            # Retained diagnostic records must contain the exact decisions
            # and scalar inputs, and remain unchanged on a poisoned replay.
            rs=(Result*case['steps']).from_buffer_copy(a['results']);steps=(Step*case['steps']).from_buffer_copy(a['steps'])
            scalars=(Scalars*case['steps']).from_buffer_copy(expected['scalars'])
            for k,(r,step) in enumerate(zip(rs,steps)):
                if not r.records:
                    assert bytes(step)==bytes(Step())
                    continue
                assert bytes(step.control)==bytes(r)
                source=k-1 if k==2 and case['fixture'] in ('scalar_status','scalar_nan','mixer_nan') else k
                assert bytes(step.scalars)==bytes(scalars[source])
            record.update(numerical_passed=True,max_value_error=float(np.max(np.abs(a['values']-expected['expected']))),
                output_file=str(result.with_suffix('.npz').relative_to(output.parent)),output_sha256=sha(result.with_suffix('.npz')))
            if trace:
                activity=actual['cuda_trace'];scf=activity.get('phases',{}).get('scf',{})
                record['residency_passed']=activity['coverage_complete'] and activity['phase_attribution_verified'] and (
                    scf.get('host_to_device_bytes',0)==0 and scf.get('device_to_host_bytes',0)==0)
                assert record['residency_passed'],activity['errors']
            record['passed']=True
        except Exception as error:record.update(passed=False,error=f'{type(error).__name__}: {error}')
        report['checks'].append(record);report.update(gpu_validation=True,passed=all(c['passed'] for c in report['checks']),
            completed_all_checks=len(report['checks'])==len(cases))
        output.write_text(json.dumps(report,indent=2)+'\n');print(case['fixture'],'passed=',record['passed'],flush=True)
        if sum(not c['numerical_passed'] for c in report['checks'])>=3:raise SystemExit('Stopped after three numerical/setup failures')
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'));p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path);p.add_argument('--library',type=Path);p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    if args.worker:worker(args.worker)
    elif args.output:run(args.upstream,args.build,args.output,args.library,args.trace)
    else:p.error('--output or --worker is required')
