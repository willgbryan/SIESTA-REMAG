#!/usr/bin/env python3
"""Compare device periodic Poisson with the complete, unmodified SIESTA routine.

The reference executable links the pristine build's libsiesta and dependencies;
its FFT, reciprocal cutoff, energy and stress routines are not replaced by stubs.
Without --library this only compiles/runs CPU fixtures and validates the oracle.
"""
import argparse
import ctypes as ct
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

WRAPPER = '''program remag_poisson_oracle
 use precision, only: dp,grid_p
 use moreMeshSubs, only: initMeshDistr,UNIFORM
 use mesh, only: nsm
 use cellsubs, only: reclat,volcel
 use m_chkgmx, only: chkgmx
 implicit none
 integer :: shape(3),fi,fo
 real(dp) :: cell(3,3),stress(3,3),u,b(3,3),cutoff,k(3),volume
 real(grid_p),allocatable :: rho(:),v(:)
 character(2048) :: input,output
 if (grid_p/=dp) error stop 'Hartree check requires FP64 grid build'
 call get_command_argument(1,input)
 call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) shape,cell
 allocate(rho(product(shape)),v(product(shape)))
 read(fi) rho
 close(fi)
 nsm=1
 call initMeshDistr(UNIFORM,(/1,1,1/),shape,nsm)
 call reclat(cell,b,1)
 cutoff=1.e30_dp;k=0._dp
 call chkgmx(k,b,shape,cutoff)
 volume=volcel(cell)
 call poison(cell,shape(1),shape(2),shape(3),shape,rho,u,v,stress,nsm)
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) b,cutoff,volume,u,stress,v
 close(fo)
end program
'''

class Problem(ct.Structure):
    _fields_=[('mesh',ct.c_int*3),('reciprocal',ct.c_double*9),('cutoff',ct.c_double),('volume',ct.c_double)]

class Result(ct.Structure):
    _fields_=[('energy',ct.c_double),('stress',ct.c_double*9),('status',ct.c_int)]

class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_uint64),('graph_nodes',ct.c_int),('forbidden_nodes',ct.c_int),('iterations',ct.c_int)]

def pointer(array):
    return array.ctypes.data_as(ct.POINTER(ct.c_double))

def fixtures():
    rng=np.random.default_rng(9723)
    for label,shape,cell in (
        ('constant',(6,8,10),np.diag([8.,9.,10.])),
        ('fourier_modes',(10,12,16),np.diag([8.,9.,10.])),
        ('orthogonal_random',(12,10,8),np.diag([9.,10.,11.])),
        ('oblique_mixed_mesh',(9,10,12),np.array([[8.,1.5,-.7],[0.,9.,.6],[0.,0.,11.]])),
        ('left_handed',(5,9,15),np.array([[-7.,.4,.2],[0.,8.,.5],[0.,0.,10.]])),
        ('larger_grid',(40,36,32),np.array([[12.,1.2,-.5],[0.,11.,.3],[0.,0.,10.]]))):
        if label=='constant': rho=np.full(shape,.25)
        elif label=='fourier_modes':
            xyz=np.indices(shape)/np.array(shape)[:,None,None,None]
            rho=.2+.03*np.cos(2*np.pi*(xyz[0]+xyz[1]))+.02*np.sin(2*np.pi*xyz[2])
            rho+=.01*np.cos(2*np.pi*(4*xyz[0]+4*xyz[1]+3*xyz[2]))
        else: rho=.1+rng.normal(scale=.03,size=shape)
        yield label,shape,np.asfortranarray(cell),np.asfortranarray(rho)

def numpy_reference(problem,rho):
    """An independent transform sanity check, not the GPU comparison oracle."""
    shape=tuple(problem.mesh);n=rho.size
    b=np.array(problem.reciprocal).reshape((3,3),order='F')
    indices=np.indices(shape).reshape((3,-1))
    for a,size in enumerate(shape): indices[a,indices[a]>size//2]-=size
    g=b@indices;g2=np.sum(g*g,axis=0)
    weight=np.zeros(n);inside=(g2<problem.cutoff)&(g2>1.e-15);weight[inside]=8*np.pi/g2[inside]
    # Original GPFA forward transform has positive exponent; inverse is scaled.
    spectrum=np.fft.ifftn(rho)*n
    du=weight*np.abs(spectrum.ravel())**2
    energy=.5*du.sum()*problem.volume/n**2
    coefficient=np.zeros(n);coefficient[inside]=2*du[inside]/g2[inside]
    stress=.5*(g*coefficient)@g.T/n**2+np.eye(3)*energy/problem.volume
    potential=np.fft.fftn(spectrum*weight.reshape(shape)).real/n
    return energy,stress,potential

def run(upstream,build,output,library=None):
    upstream=upstream.resolve();build=build.resolve();src=build/'Src'
    paths=['poison.F','fft.F','m_fft_gpfa.F','chkgmx.f','minvec.f','sorting.f','cellsubs.f']
    report=dict(reference='Unmodified pristine SIESTA libsiesta; serial periodic FP64 grid',
                fixture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_source_sha256={name:hashlib.sha256((upstream/'Src'/name).read_bytes()).hexdigest() for name in paths},
                reference_library_sha256=hashlib.sha256((src/'libsiesta.a').read_bytes()).hexdigest(),
                native_library=str(library) if library else None,checks=[])
    with tempfile.TemporaryDirectory(prefix='remag-hartree-reference-') as temporary:
        directory=Path(temporary);wrapper=directory/'reference.f90';wrapper.write_text(WRAPPER)
        executable=directory/'reference'
        link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
        libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
        flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
        includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
        subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(wrapper),'-o',str(executable),*libraries],cwd=src,check=True)
        candidate=None
        if library:
            candidate=ct.CDLL(str(library)).remag_check_hartree
            candidate.argtypes=[ct.POINTER(Problem),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),
                                ct.POINTER(Result),ct.c_int,ct.c_int,ct.POINTER(Check)]
            candidate.restype=ct.c_int
        for label,shape,cell,rho in fixtures():
            inp=directory/'input';out=directory/'output'
            inp.write_bytes(np.array(shape,dtype=np.int32).tobytes()+cell.tobytes(order='F')+rho.tobytes(order='F'))
            reference=subprocess.run([str(executable),str(inp),str(out)],cwd=directory,check=True,capture_output=True,text=True)
            raw=np.fromfile(out,dtype=np.float64)
            if raw.size!=21+rho.size or not np.isfinite(raw).all(): raise AssertionError('Malformed upstream Poisson output')
            problem=Problem((ct.c_int*3)(*shape),(ct.c_double*9)(*raw[:9]),raw[9],raw[10])
            energy=raw[11];stress=raw[12:21].reshape((3,3),order='F');potential=raw[21:].reshape(shape,order='F')
            independent=numpy_reference(problem,rho)
            for actual,expected in zip(independent,(energy,stress,potential)):
                np.testing.assert_allclose(actual,expected,atol=1.e-10,rtol=1.e-11)
            for graph in ((False,True) if candidate else (False,)):
                item=dict(fixture=label,mesh=shape,cutoff=problem.cutoff,graph=graph,reference_energy=float(energy),
                          input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                          reference_stress=stress.tolist(),reference_potential_range=[float(potential.min()),float(potential.max())],
                          numpy_reference_agrees=True,upstream_stdout=reference.stdout)
                if candidate:
                    v=np.zeros_like(rho,order='F');result=Result();info=Check()
                    status=candidate(ct.byref(problem),pointer(rho),pointer(v),ct.byref(result),int(graph),0,ct.byref(info))
                    assert status==0 and result.status==0,(label,graph,status,result.status)
                    # Two executions share the same buffers/plan, with a device-only
                    # density change before the second; reject captured stale work.
                    expected_v=.75*potential;expected_e=.75**2*energy;expected_s=.75**2*stress
                    actual_s=np.array(result.stress).reshape((3,3),order='F')
                    np.testing.assert_allclose(v,expected_v,atol=1.e-10,rtol=1.e-11)
                    np.testing.assert_allclose(result.energy,expected_e,atol=1.e-10,rtol=1.e-11)
                    np.testing.assert_allclose(actual_s,expected_s,atol=1.e-12,rtol=1.e-11)
                    assert info.iterations==2 and info.forbidden_nodes==0
                    if graph: assert info.graph_nodes>0
                    item.update(energy_error=float(abs(result.energy-expected_e)),stress_error=float(np.max(np.abs(actual_s-expected_s))),
                                potential_error=float(np.max(np.abs(v-expected_v))),iterations=info.iterations,
                                graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,workspace_bytes=info.workspace_bytes,passed=True)
                report['checks'].append(item)
        if candidate:
            # Invalid scientific data stays a device error until explicit export.
            for label in ('density_nan','density_overflow','reciprocal_nan','cutoff_nan','cutoff_zero','volume_negative','shape_mismatch'):
                p=Problem.from_buffer_copy(problem);density=rho.copy(order='F')
                if label=='density_nan': density.flat[0]=np.nan
                elif label=='density_overflow': density.fill(1.e308)
                elif label=='reciprocal_nan': p.reciprocal[0]=np.nan
                elif label=='cutoff_nan': p.cutoff=np.nan
                elif label=='cutoff_zero': p.cutoff=0.
                elif label=='volume_negative': p.volume=-1.
                result=Result();info=Check();v=np.zeros_like(density,order='F')
                status=candidate(ct.byref(p),pointer(density),pointer(v),ct.byref(result),1,int(label=='shape_mismatch'),ct.byref(info))
                assert status==0 and result.status==(1 if label=='shape_mismatch' else 2),(label,status,result.status)
                report['checks'].append(dict(fixture=label,graph=True,status=result.status,passed=True))
    report['passed']=all(v.get('passed',False) for v in report['checks']) if candidate else None
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Upstream Hartree fixtures: 6; scenarios: {len(report['checks'])}; device checks passed={report['passed']}")
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args()
    report=run(args.upstream,args.build,args.output,args.library)
    if args.library and not report['passed']: raise SystemExit('Device Hartree regression failed')
