#!/usr/bin/env python3
"""Independent generalized-eigenproblem oracle for the experimental blocked GPU solver.

Each worker has fresh library state and retains numerical failures. The two modes
are host submission of independent solve graphs and a nested device-controlled
loop over the same pre-staged problems. Neither is a production SCF integration.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import cholesky, eigh
from scipy import __version__ as scipy_version


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def lower_hermitian(a):
    result=np.tril(a)+np.tril(a,-1).conj().swapaxes(-1,-2)
    i=np.arange(a.shape[-1]);result[...,i,i]=result[...,i,i].real
    return result


class Result(ct.Structure):
    _fields_=[('status',ct.c_int),('sweeps',ct.c_int),('relative_offdiagonal',ct.c_double),('history',ct.c_double*32)]


class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_size_t),('host_workspace_bytes',ct.c_size_t)]+[
        (x,ct.c_int) for x in ('graph_nodes','forbidden_nodes','iterations','stage')]


def worker(config):
    cfg=json.loads(config.read_text());data=np.load(cfg['input']);h=data['h'];s=data['s'];cycles,n,_=h.shape
    hp=h.swapaxes(-1,-2).copy();sp=s.swapaxes(-1,-2).copy();z=np.full_like(hp,np.nan);w=np.full((cycles,n),np.nan)
    results=(Result*cycles)();info=Check();f=ct.CDLL(cfg['library']).remag_check_block_jacobi
    f.argtypes=[ct.c_int]*5+[ct.c_double]+[ct.c_void_p]*4+[ct.POINTER(Result),ct.POINTER(Check)];f.restype=ct.c_int
    code=f(n,cycles,int(np.iscomplexobj(h)),cfg['maximum'],int(cfg['nested']),2.e-13,
        hp.ctypes.data,sp.ctypes.data,z.ctypes.data,w.ctypes.data,results,ct.byref(info))
    output=Path(cfg['output']);status=np.array([r.status for r in results]);sweeps=np.array([r.sweeps for r in results])
    relative=np.array([r.relative_offdiagonal for r in results]);history=np.array([list(r.history) for r in results])
    np.savez_compressed(output.with_suffix('.npz'),z=z.swapaxes(-1,-2),w=w,status=status,sweeps=sweeps,
                        relative_offdiagonal=relative,history=history)
    report={name:getattr(info,name) for name,_ in info._fields_};report['native_return']=code
    output.write_text(json.dumps(report,indent=2)+'\n')
    if cfg['trace']:
        from siesta_remag.cuda_trace import summarize_cuda
        width=int(os.environ.get("REMAG_BLOCK_WIDTH","64"));padded=width*((n+width-1)//width);rounds=padded//(width//2)-1
        # Relative round maps plus the final inverse map, staged once.
        explicit=dict(host_to_device_bytes=2*hp.nbytes+4+2*h.dtype.itemsize+(rounds+1)*padded*4,
                      device_to_host_bytes=z.nbytes+w.nbytes+ct.sizeof(Result)*cycles+4)
        try:
            trace=json.loads((output.parent/'remag-activity.json').read_text())
            report['cuda_trace']=summarize_cuda(trace,dict(total_explicit=explicit,accounting_errors=[]),
                expected_scf_iterations=1,process_succeeded=code==0,allow_unique_graph_scope=True,
                memory_only=cfg.get('memory_only',False))
        except Exception as error:
            report['cuda_trace']=dict(coverage_complete=False,phase_attribution_verified=False,errors=[str(error)])
    output.write_text(json.dumps(report,indent=2)+'\n')


def fixtures(directory):
    rng=np.random.default_rng(32641152);records=[]
    for n in (18,64,128,129,257,576,1152):
        for complex_values in (False,True):
            def raw():
                a=rng.normal(size=(n,n))
                return a+1j*rng.normal(size=a.shape) if complex_values else a
            condition=1000. if n>=128 else 4.
            if n<=257:
                q,_=np.linalg.qr(raw());base_s=(q*np.geomspace(1.,condition,n))@q.conj().T
            else: base_s=np.diag(np.geomspace(1.,condition,n)).astype(np.complex128 if complex_values else float)
            h=[];s=[];expected=[]
            for cycle in range(3):
                overlap=base_s*(1.+cycle*.05)+np.eye(n)*cycle*.02;overlap=(overlap+overlap.conj().T)*.5
                if n==129:
                    q,_=np.linalg.qr(raw());eig=np.repeat(np.linspace(-1,1,(n+7)//8),8)[:n]
                    if cycle: eig+=np.arange(n)*1.e-10*cycle
                    l=cholesky(overlap,lower=True);matrix=l@((q*eig)@q.conj().T)@l.conj().T
                else:
                    a=raw();matrix=(a+a.conj().T)/(2*np.sqrt(n))+np.diag(np.linspace(-1.,1.,n)*(1.+.2*cycle))
                matrix=(matrix+matrix.conj().T)*.5;h.append(matrix);s.append(overlap)
                expected.append(eigh(matrix,overlap,eigvals_only=True,driver='gvd'))
            label=f'n{n}_complex{int(complex_values)}';path=directory/(label+'.npz')
            np.savez_compressed(path,h=np.array(h),s=np.array(s),reference_w=np.array(expected))
            records.append(dict(fixture=label,n=n,complex=complex_values,cycles=3,overlap_condition=condition,
                spectrum='repeated_and_clustered' if n==129 else 'dense_changing',input_file=path.name,input_sha256=sha(path),
                expected_status=[0,0,0],maximum=20))
    base=np.load(directory/'n128_complex1.npz')
    h=base['h'].copy();s=base['s'].copy();upper=np.triu_indices(128,1)
    h[:,upper[0],upper[1]]=np.nan;s[:,upper[0],upper[1]]=np.nan
    path=directory/'ignored_upper_triangle.npz';np.savez_compressed(path,h=h,s=s,reference_w=base['reference_w'])
    records.append(dict(fixture='ignored_upper_triangle',n=128,complex=True,cycles=3,input_file=path.name,
        input_sha256=sha(path),expected_status=[0,0,0],maximum=20))
    for label,status in (('indefinite_overlap',1),('nonfinite_hamiltonian',2),('sweep_exhaustion',4)):
        h=base['h'].copy();s=base['s'].copy()
        if status==1:s[1,0,0]=-1.
        elif status==2:h[1,0,0]=np.nan
        path=directory/(label+'.npz');np.savez_compressed(path,h=h,s=s,reference_w=base['reference_w'])
        records.append(dict(fixture=label,n=128,complex=True,cycles=3,input_file=path.name,input_sha256=sha(path),
            expected_status=[status]*3 if status==4 else [0,status,0],maximum=1 if status==4 else 20))
    return records


def regression_fixtures(directory,source):
    """Exact, hashed material inputs which must solve successfully."""
    manifest=source/'manifest.json';record=json.loads(manifest.read_text())
    artifact=source/record['arrays_file']
    if sha(artifact)!=record['arrays_sha256']:raise ValueError('Solver regression archive changed')
    fixtures=[]
    with np.load(artifact) as arrays:
        for case in record['cases']:
            label=case['label'];h=arrays[label+'_h'];s=arrays[label+'_s'];w=arrays[label+'_w'];n=case['n']
            if h.shape!=(n,n) or s.shape!=(n,n) or w.shape!=(n,):raise ValueError('Invalid regression dimensions')
            for name,a in (('remag-failed-h.bin',h),('remag-failed-s.bin',s)):
                if hashlib.sha256(a.tobytes(order='F')).hexdigest()!=case['source_sha256'][name]:
                    raise ValueError('Regression no longer matches captured input')
            reference=eigh(lower_hermitian(h),lower_hermitian(s),eigvals_only=True,driver='gvd')
            if np.max(np.abs(w-reference))>=1.e-10:raise ValueError('Retained LAPACK spectrum was not reproduced')
            path=directory/(label+'.npz')
            np.savez_compressed(path,h=np.repeat(h[None],3,axis=0),s=np.repeat(s[None],3,axis=0),reference_w=np.repeat(w[None],3,axis=0))
            fixtures.append(dict(fixture=label,n=n,complex=bool(np.iscomplexobj(h)),cycles=3,
                input_file=path.name,input_sha256=sha(path),expected_status=[0,0,0],maximum=20,
                source_run=case['source_run'],regression_manifest_sha256=sha(manifest)))
    return fixtures


def observer_environment(observer,executable,memory_only=False):
    env=os.environ.copy()
    env.update(REMAG_TRACE_CUDA='1',REMAG_ACTIVITY_EXECUTABLE=str(executable.resolve()),
               REMAG_ACTIVITY_MEMORY_ONLY='1' if memory_only else '0',
               LD_PRELOAD=str(observer),LD_LIBRARY_PATH=str(observer.parent)+':'+env.get('LD_LIBRARY_PATH',''))
    return env


def calibrate_observer(directory,observer,executable,memory_only=False):
    from siesta_remag.cuda_trace import summarize_cuda
    checks=[]
    driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                          capture_output=True,text=True,timeout=10)
    driver_version=driver.stdout.strip() if driver.returncode==0 else None
    for mode in ('ordinary','graph','conditional-scope','conditional'):
        target=directory/('observer-'+mode+'-check');target.mkdir()
        unique_scope=mode=='conditional-scope'
        ambiguous=mode=='conditional'
        kernels=0 if memory_only and mode=='ordinary' else (10 if mode.startswith('conditional') else 2)
        device_copies=4 if mode.startswith('conditional') else (2 if mode=='graph' else 1)
        record=dict(mode=mode,passed=False,expected_coverage_complete=not ambiguous,memory_only=memory_only,
                    driver_version=driver_version)
        if mode.startswith('conditional') and driver_version=='610.57.04':
            # Reproduced with both observers and copy-only collection; avoid
            # repeating known Xid 31 faults. Numerical execution remains valid.
            record.update(skipped=True,process_returncode=None,
                error='Conditional activity profiling is unqualified on driver 610.57.04: retained runs '
                      '20261006T055600Z and 20261006T061806Z fault. No conditional trace was attempted.')
            checks.append(record);(target/'check.json').write_text(json.dumps(record,indent=2)+'\n');continue
        try:
            done=subprocess.run([str(executable)]+(['--'+mode] if mode!='ordinary' else []),cwd=target,
                env=observer_environment(observer,executable,memory_only),capture_output=True,text=True,timeout=60)
            (target/'stdout.log').write_text(done.stdout+done.stderr)
            record['process_returncode']=done.returncode
            raw_trace=json.loads((target/'remag-activity.json').read_text())
            activity=summarize_cuda(raw_trace,
                dict(total_explicit=dict(host_to_device_bytes=32768,device_to_host_bytes=32776),accounting_errors=[]),
                expected_scf_iterations=1 if unique_scope else 2,process_succeeded=done.returncode==0,
                allow_unique_graph_scope=True,memory_only=memory_only)
            phase=activity.get('phases',{})
            # CUDA lowers these conditional D2D nodes into memcpy32_post
            # kernels. Do not invent memcpy bytes for kernel activity. Verify
            # the known four-copy lowering and numerical destination instead.
            conditional=mode.startswith('conditional')
            lowered=conditional and activity.get('total',{}).get('device_to_device_bytes',0)==0 and (
                activity.get('kernels',{}).get('memcpy32_post',{}).get('kernel_count')==4)
            expected_bytes=0 if lowered else 32768*device_copies
            expected_kernels=kernels+4 if lowered else kernels
            record['conditional_d2d_lowered_to_kernels']=bool(lowered)
            valid=done.returncode==0 and (
                phase.get('initialization',{}).get('host_to_device_bytes')==32768 and
                phase.get('post_scf',{}).get('device_to_host_bytes')==32768 and
                phase.get('scf',{}).get('device_to_host_bytes')==8 and
                activity.get('total',{}).get('device_to_device_bytes',0)==expected_bytes and
                activity.get('total',{}).get('kernel_count',0)==expected_kernels and
                activity.get('steady_state',{}).get('device_to_host_bytes',0)==(0 if unique_scope else 4))
            if ambiguous:
                missing=sum(not row[7] and not row[8] and graph[0]>0 for row,graph in zip(
                    raw_trace.get('events',[]),raw_trace.get('event_graphs',[])))
                valid=valid and not activity['coverage_complete'] and not activity['phase_attribution_verified'] and (
                    missing>0 and activity['errors']==[f'unmapped or ambiguous launch phase: {missing}'])
            else:
                valid=valid and activity['coverage_complete'] and activity['phase_attribution_verified'] and (
                    phase.get('scf',{}).get('device_to_device_bytes',0)==expected_bytes and
                    phase.get('scf',{}).get('kernel_count',0)==expected_kernels)
            record.update(passed=valid,activity=activity)
        except Exception as error:
            record['error']=f'{type(error).__name__}: {error}'
            if isinstance(error,subprocess.TimeoutExpired):
                raw=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
                (target/'stdout.log').write_text(raw(error.stdout)+raw(error.stderr))
        checks.append(record)
        (target/'check.json').write_text(json.dumps(checks[-1],indent=2)+'\n')
    return checks


def run(output,library=None,trace=False,observer=None,memory_only=False,regressions=None):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    output.with_name('implementation.py').write_bytes(Path(__file__).read_bytes())
    records=fixtures(output.parent)
    if regressions:records+=regression_fixtures(output.parent,regressions)
    report=dict(scope='Experimental generalized block Jacobi; no production solver, SCF or speedup qualification',
        gpu_validation=None,fixtures=records,checks=[],fixture_script_sha256=sha(Path(__file__)),
        numpy_version=np.__version__,scipy_version=scipy_version,expected_checks=2*len(records))
    if not library:
        report['passed']=True;output.write_text(json.dumps(report,indent=2)+'\n');print('CPU fixture sets:',len(records));return
    library=library.resolve();report['native_library_sha256']=sha(library)
    observer=(observer or library.parent/'libremag_activity.so').resolve()
    if trace:
        report.update(observer=str(observer),observer_sha256=sha(observer),memory_only=memory_only)
        report['observer_calibration']=calibrate_observer(output.parent,observer,library.parent/'remag_activity_check',memory_only)
        if not all(c['passed'] for c in report['observer_calibration']):
            report.update(passed=False,completed_all_checks=False)
            output.write_text(json.dumps(report,indent=2)+'\n')
            raise SystemExit('Observer calibration failed; blocked solver execution was not attempted')
    for fixture in records:
        data=np.load(output.parent/fixture['input_file']);h=data['h'];s=data['s'];expected=data['reference_w'];n=fixture['n']
        for nested in (False,True):
            name=f"{fixture['fixture']}_nested{int(nested)}";directory=output.parent/name;directory.mkdir()
            result_file=directory/'result.json';config=directory/'config.json'
            config.write_text(json.dumps(dict(input=str(output.parent/fixture['input_file']),output=str(result_file),
                library=str(library),nested=nested,maximum=fixture['maximum'],trace=trace,memory_only=memory_only),indent=2)+'\n')
            record=dict(fixture=fixture['fixture'],nested=nested,expected_status=fixture['expected_status'],isolated_process=True,
                        numerical_passed=False,residency_passed=None,measured_regions=1,problems_per_region=3)
            env=os.environ.copy()
            if trace:env=observer_environment(observer,Path(sys.executable),memory_only)
            try:
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],
                    cwd=directory,env=env,capture_output=True,text=True,timeout=180)
                (directory/'stdout.log').write_text(completed.stdout+completed.stderr);record['process_returncode']=completed.returncode
                assert completed.returncode==0,('Worker failed',completed.returncode)
                actual_info=json.loads(result_file.read_text());record.update(actual_info)
                record['output_file']=str(result_file.with_suffix('.npz').relative_to(output.parent))
                record['output_sha256']=sha(result_file.with_suffix('.npz'))
                assert actual_info['native_return']==0 and actual_info['iterations']==3,{
                    k:v for k,v in actual_info.items() if k!='cuda_trace'}
                assert actual_info['forbidden_nodes']==0 and actual_info['host_workspace_bytes']==0,{
                    k:v for k,v in actual_info.items() if k!='cuda_trace'}
                actual=np.load(result_file.with_suffix('.npz'));status=actual['status'];w=actual['w'];z=actual['z']
                assert status.tolist()==fixture['expected_status'],status.tolist()
                valid=status==0
                if np.any(valid):
                    hv,sv,wv,zv=lower_hermitian(h[valid]),lower_hermitian(s[valid]),w[valid],z[valid]
                    residual=np.linalg.norm(hv@zv-(sv@zv)*wv[...,None,:],axis=(-2,-1))/(
                        np.linalg.norm(hv,axis=(-2,-1))*np.linalg.norm(zv,axis=(-2,-1)))
                    orth=np.linalg.norm(zv.conj().swapaxes(-1,-2)@sv@zv-np.eye(n),ord=np.inf,axis=(-2,-1))
                    error=float(np.max(np.abs(wv-expected[valid])))
                    record.update(max_relative_residual=float(np.max(residual)),max_orthogonality_error=float(np.max(orth)),
                                  max_eigenvalue_error=error,sweeps=actual['sweeps'].tolist())
                    assert np.isfinite(wv).all() and np.isfinite(zv).all()
                    assert np.max(residual)<1.e-10 and np.max(orth)<1.e-9 and error<1.e-10,{
                        k:record[k] for k in ('max_relative_residual','max_orthogonality_error','max_eigenvalue_error','sweeps')}
                record['numerical_passed']=True
                if trace:
                    record['residency_passed']=False
                    activity=record['cuda_trace'];scf=activity.get('phases',{}).get('scf',{})
                    assert activity['coverage_complete'] and activity['phase_attribution_verified'],activity['errors']
                    assert scf.get('host_to_device_bytes',0)==0 and scf.get('device_to_host_bytes',0)==0,scf
                    record['residency_passed']=True
                record['passed']=True
            except Exception as error:record.update(passed=False,error=f'{type(error).__name__}: {error}')
            report['checks'].append(record);report.update(gpu_validation=True,passed=(all(c['passed'] for c in report['checks']) and
                all(c['passed'] for c in report.get('observer_calibration',[]))))
            report['completed_all_checks']=len(report['checks'])==report['expected_checks']
            output.write_text(json.dumps(report,indent=2)+'\n');print(name,'passed=',record['passed'],flush=True)
            if sum(not c['numerical_passed'] for c in report['checks'])>=3:
                print('Stopped after three unexpected numerical/setup failures; remaining scenarios are unvalidated.',flush=True)
                raise SystemExit(1)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--library',type=Path);parser.add_argument('--trace',action='store_true')
    parser.add_argument('--observer',type=Path,help='Optional isolated activity observer; calibrated before numerical checks')
    parser.add_argument('--memory-only',action='store_true',help='Collect memcpy/memset activity without kernel instrumentation')
    parser.add_argument('--regressions',type=Path,help='Hashed captured H/S regression fixture directory')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.output:run(args.output,args.library,args.trace,args.observer,args.memory_only,args.regressions)
    else:parser.error('--output or --worker is required')
