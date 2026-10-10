#!/usr/bin/env python3
"""Probe size-64 batched eigensolves for a future blocked GPU solver.

Independent worker processes contain library/capture failures. Inputs change on
device between three iterations. This does not implement a large eigensolver.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh
from scipy import __version__ as scipy_version


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class Check(ct.Structure):
    _fields_=[('device_workspace',ct.c_size_t),('host_workspace',ct.c_size_t)]+[
        (name,ct.c_int) for name in ('graph_nodes','forbidden_nodes','iterations','stage')]


def worker(config):
    settings=json.loads(config.read_text());data=np.load(settings['input'])
    h=data['h'];cycles,batch,n,_=h.shape
    packed=h.swapaxes(-1,-2).copy();vectors=np.full_like(packed,np.nan)
    w=np.full((cycles,batch,n),np.nan);status=np.full((cycles,batch),-999,np.int32)
    info=Check();function=ct.CDLL(settings['library']).remag_check_small_eigensolver
    function.argtypes=[ct.c_int]*6+[ct.c_void_p]*4+[ct.POINTER(Check)];function.restype=ct.c_int
    code=function(n,batch,cycles,int(np.iscomplexobj(h)),int(settings['graph']),settings['workspace_fill'],
                  packed.ctypes.data,vectors.ctypes.data,w.ctypes.data,status.ctypes.data,ct.byref(info))
    output=Path(settings['output']);np.savez_compressed(output.with_suffix('.npz'),
        w=w,z=vectors.swapaxes(-1,-2),status=status)
    result={name:getattr(info,name) for name,_ in info._fields_};result['native_return']=code
    output.write_text(json.dumps(result,indent=2)+'\n')


def run(output,library=None):
    output.parent.mkdir(parents=True,exist_ok=True);output=output.resolve()
    output.with_name('implementation.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(6418);fixtures=[]
    for complex_values in (False,True):
        for batch in (1,2,4,18):
            label=f'complex{int(complex_values)}_batch{batch}'
            h=np.empty((3,batch,64,64),np.complex128 if complex_values else np.float64)
            expected=np.empty((3,batch,64))
            for cycle in range(3):
                for slot in range(batch):
                    raw=rng.normal(size=(64,64))
                    if complex_values: raw=raw+1j*rng.normal(size=raw.shape)
                    matrix=(raw+raw.conj().T)/(2*np.sqrt(64))
                    if batch==4 and slot==0:
                        # Changing eigenspaces with exact repeated eigenvalues.
                        q,_=np.linalg.qr(raw)
                        matrix=(q*np.repeat(np.linspace(-1,1,8),8))@q.conj().T
                    matrix+=np.diag(np.linspace(-.5,.5,64)) if slot else np.eye(64)*(cycle*.1)
                    h[cycle,slot]=(matrix+matrix.conj().T)*.5
                    expected[cycle,slot]=eigh(h[cycle,slot],eigvals_only=True,driver='evd')
            path=output.parent/(label+'.npz');np.savez_compressed(path,h=h,reference_w=expected)
            fixtures.append(dict(fixture=label,n=64,batch=batch,cycles=3,complex=complex_values,
                                 input_file=path.name,input_sha256=sha(path)))
    report=dict(scope='Isolated small standard eigensolver batching/capture; no generalized solver or SCF qualification',
        gpu_validation=None,fixture_script_sha256=sha(Path(__file__)),fixtures=fixtures,checks=[],
        numpy_version=np.__version__,scipy_version=scipy_version,
        workspace_fill_meaning={'-1':'library default','0':'device workspace zeroed before every solve',
                                '255':'device workspace filled with 0xff before every solve'})
    if not library:
        report['passed']=True;output.write_text(json.dumps(report,indent=2)+'\n')
        print('Eight small-eigensolver CPU fixture sets staged; GPU validation=None');return
    library=library.resolve();report['native_library_sha256']=sha(library)
    for fixture in fixtures:
        input_file=output.parent/fixture['input_file'];data=np.load(input_file);h=data['h'];expected=data['reference_w']
        for graph in (False,True):
            for fill in (-1,0,255):
                label=f"{fixture['fixture']}_graph{int(graph)}_fill{fill}"
                result_file=output.parent/(label+'.json');config=output.parent/(label+'-config.json')
                config.write_text(json.dumps(dict(input=str(input_file),output=str(result_file),library=str(library),
                    graph=graph,workspace_fill=fill),indent=2)+'\n')
                record=dict(fixture=fixture['fixture'],graph=graph,workspace_fill=fill,isolated_process=True)
                try:
                    completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],
                        capture_output=True,text=True,timeout=30)
                    (output.parent/(label+'.log')).write_text(completed.stdout+completed.stderr)
                    record['process_returncode']=completed.returncode
                    assert completed.returncode==0,('Worker process failed',completed.returncode)
                    actual_info=json.loads(result_file.read_text());record.update(actual_info)
                    record['output_sha256']=sha(result_file.with_suffix('.npz'))
                    assert actual_info['native_return']==0,actual_info
                    assert actual_info['iterations']==3 and actual_info['forbidden_nodes']==0,actual_info
                    if graph: assert actual_info['graph_nodes']>0,actual_info
                    actual=np.load(result_file.with_suffix('.npz'));w=actual['w'];z=actual['z'];status=actual['status']
                    assert np.all(status==0) and np.isfinite(w).all() and np.isfinite(z).all()
                    residual=np.linalg.norm(h@z-z*w[...,None,:],axis=(-2,-1))/(
                        np.linalg.norm(h,axis=(-2,-1))*np.linalg.norm(z,axis=(-2,-1)))
                    orthogonality=np.linalg.norm(z.conj().swapaxes(-1,-2)@z-np.eye(64),ord=np.inf,axis=(-2,-1))
                    error=np.max(np.abs(w-expected))
                    record.update(max_relative_residual=float(np.max(residual)),
                        max_orthogonality_error=float(np.max(orthogonality)),max_eigenvalue_error=float(error))
                    assert np.max(residual)<1.e-12 and np.max(orthogonality)<1.e-10 and error<1.e-11,record
                    record['passed']=True
                except Exception as error:
                    record.update(passed=False,error=f'{type(error).__name__}: {error}')
                report['checks'].append(record)
                report.update(passed=all(x['passed'] for x in report['checks']),gpu_validation=True)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print(label,'passed=',record['passed'],flush=True)
    print('Small eigensolver checks:',len(report['checks']),'passed=',report['passed'])
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--library',type=Path)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.output: run(args.output,args.library)
    else: parser.error('--output or --worker is required')
