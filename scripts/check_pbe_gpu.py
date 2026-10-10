#!/usr/bin/env python3
"""Compare resident GPU PBE against pristine complete GridXC mesh evaluation."""
import argparse
import ctypes as ct
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from check_pbe import WRAPPER,fixtures

class Problem(ct.Structure):
    _fields_=[('mesh',ct.c_int*3),('spins',ct.c_int),('gradient',ct.c_double*9),('volume',ct.c_double)]

class Result(ct.Structure):
    _fields_=[('energies',ct.c_double*4),('stress',ct.c_double*9),('status',ct.c_int)]

class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_uint64),('graph_nodes',ct.c_int),('forbidden_nodes',ct.c_int),('iterations',ct.c_int)]

def scenarios():
    yield from fixtures()
    for label,shape,spins in (('vacuum_interface',(8,9,10),2),('nearly_unpolarized',(7,8,5),4),
                              ('left_handed_spinor',(9,7,11),4),('larger_collinear',(40,32,28),2)):
        cell=np.asfortranarray([[8.,1.2,-.7],[0.,9.,.5],[0.,0.,10.]])
        xyz=np.indices(shape)/np.array(shape)[:,None,None,None]
        total=.15+.03*np.cos(2*np.pi*xyz[0])+.02*np.sin(2*np.pi*xyz[2])
        d=np.zeros((*shape,spins),order='F');d[...,0]=.7*total;d[...,1]=.3*total
        if label=='vacuum_interface': d[:4,:,:]=1.e-25
        if label=='nearly_unpolarized':
            d[...,:2]=total[...,None]/2
            d[...,2]=1.e-14*np.cos(2*np.pi*xyz[0]);d[...,3]=1.e-14*np.sin(2*np.pi*xyz[1])
        if label=='left_handed_spinor':
            cell[:,0]*=-1
            d[...,2]=.1*total*np.cos(2*np.pi*xyz[0]);d[...,3]=-.1*total*np.sin(2*np.pi*xyz[1])
        yield label,cell,d

def run(upstream,build,library,output):
    build=build.resolve();source=upstream.resolve()/'External/libgridxc/src'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(reference='Unmodified pristine full GridXC cellXC; built-in PBE, irel=0, legacy NC, FP64, serial',
                script_sha256=sha(Path(__file__)),oracle_script_sha256=sha(Path(__file__).with_name('check_pbe.py')),
                reference_source_sha256={p:sha(source/p) for p in ('cellxc.F90','ggaxc.F90','ldaxc.F90','cellsubs.F90')},
                reference_library_sha256=sha(build/'libgridxc/libgridxc.a'),native_library_sha256=sha(library),checks=[])
    with tempfile.TemporaryDirectory(prefix='remag-pbe-gpu-') as tmp:
        tmp=Path(tmp);wrapper=tmp/'reference.f90';wrapper.write_text(WRAPPER)
        compiler=shlex.split((build/'Src/CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())[0]
        subprocess.run([compiler,'-O2','-shared','-fPIC','-I'+str(build/'libgridxc/include'),str(wrapper),
                        str(build/'libgridxc/libgridxc.a'),'-o',str(tmp/'reference.so')],cwd=tmp,check=True)
        reference=ct.CDLL(str(tmp/'reference.so')).remag_reference_pbe
        reference.argtypes=[ct.c_int]*4+[ct.c_void_p]*5;reference.restype=None
        candidate=ct.CDLL(str(library.resolve())).remag_check_pbe
        candidate.argtypes=[ct.POINTER(Problem),ct.c_void_p,ct.c_void_p,ct.POINTER(Result),ct.c_int,ct.c_int,ct.POINTER(Check)]
        candidate.restype=ct.c_int
        def evaluate(cell,d):
            energies=np.zeros(4);stress=np.zeros((3,3),order='F');potential=np.zeros_like(d,order='F')
            reference(*d.shape,cell.ctypes.data,d.ctypes.data,energies.ctypes.data,stress.ctypes.data,potential.ctypes.data)
            assert all(np.isfinite(a).all() for a in (energies,stress,potential))
            return energies,stress,potential
        def problem(cell,d):
            grad=np.linalg.inv(cell).T@np.diag(d.shape[:3])
            return Problem((ct.c_int*3)(*d.shape[:3]),d.shape[-1],(ct.c_double*9)(*grad.ravel(order='F')),abs(np.linalg.det(cell)))
        saved=[]
        for label,cell,d in scenarios():
            cell=np.asfortranarray(cell);d=np.asfortranarray(d);p=problem(cell,d)
            # Two executions share their buffers; the second changes density
            # on device. PBE is nonlinear, so evaluate the changed CPU density.
            expected=evaluate(cell,np.asfortranarray(d*.75))
            for graph in (False,True):
                result=Result();info=Check();v=np.full_like(d,np.nan,order='F')
                code=candidate(ct.byref(p),d.ctypes.data,v.ctypes.data,ct.byref(result),int(graph),0,ct.byref(info))
                assert code==0 and result.status==0,(label,graph,code,result.status)
                assert info.iterations==2 and info.forbidden_nodes==0
                if graph: assert info.graph_nodes>=4
                actual=(np.array(result.energies),np.array(result.stress).reshape((3,3),order='F'),v)
                errors={}
                for name,value,ref in zip(('energies','stress','potential'),actual,expected):
                    np.testing.assert_allclose(value,ref,atol=1.e-9,rtol=1.e-10)
                    errors[name]=float(np.max(np.abs(value-ref)))
                report['checks'].append(dict(fixture=label,shape=d.shape,graph=graph,graph_nodes=info.graph_nodes,
                    forbidden_nodes=info.forbidden_nodes,iterations=info.iterations,workspace_bytes=info.workspace_bytes,
                    max_absolute_errors=errors,status=result.status,passed=True))
            saved.append((p,d))
        original,d=saved[5] # regular noncollinear texture
        for invalid in ('density_nan','density_overflow','gradient_nan','singular_gradient','volume_negative',
                        'volume_nan','shape_mismatch','spin_mismatch'):
            p=Problem.from_buffer_copy(bytes(original));rho=d.copy(order='F');shape_error=0
            if invalid=='density_nan': rho[0,0,0,0]=np.nan
            elif invalid=='density_overflow': rho.fill(1.e308)
            elif invalid=='gradient_nan': p.gradient[0]=np.nan
            elif invalid=='singular_gradient': p.gradient[:]=[0.]*9
            elif invalid=='volume_negative': p.volume=-1.
            elif invalid=='volume_nan': p.volume=np.nan
            elif invalid=='shape_mismatch': shape_error=1
            elif invalid=='spin_mismatch': shape_error=2
            result=Result();info=Check();v=np.zeros_like(rho,order='F')
            code=candidate(ct.byref(p),rho.ctypes.data,v.ctypes.data,ct.byref(result),1,shape_error,ct.byref(info))
            expected_status=1 if shape_error else 2
            assert code==0 and result.status==expected_status,(invalid,code,result.status)
            assert info.forbidden_nodes==0 and info.iterations==2 and info.graph_nodes>=4
            report['checks'].append(dict(fixture=invalid,graph=True,status=result.status,expected_status=expected_status,
                graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,iterations=info.iterations,passed=True))
    report['passed']=all(c['passed'] for c in report['checks'])
    report['scope']='Component numerical/capture checks only; no SCF integration, library activity or end-to-end residency certification'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Upstream PBE fixtures: {len(saved)}; scenarios: {len(report['checks'])}; device checks passed={report['passed']}")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.upstream,args.build,args.library,args.output)
