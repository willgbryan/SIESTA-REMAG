#!/usr/bin/env python3
"""Check device grid composition against original reord/Poisson/GridXC routines.

The small oracle connects unmodified numerical routines with the literal
periodic DHSCF density/energy arithmetic. It is not the full DHSCF executable;
end-to-end force, SCF and MAE regression remains a separate required gate.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from check_hartree import Problem as HartreeProblem
from check_pbe_gpu import Problem as PbeProblem,Check
from check_pbe import fixtures as pbe_fixtures

WRAPPER='''program remag_grid_potential_oracle
 use precision, only: dp,grid_p
 use moreMeshSubs, only: initMeshDistr,UNIFORM
 use mesh, only: nsm
 use cellsubs, only: reclat,volcel
 use m_chkgmx, only: chkgmx
 use gridxc, only: gridxc_init,gridxc_setXC,gridxc_cellXC
 implicit none
 integer :: shape(3),spins,npcc,fi,fo,n,i,s,nsd
 real(dp) :: cell(3,3),b(3,3),grad(3,3),step(3,3),cutoff,k(3),volume,dvol,harris,rnsd,total
 real(dp) :: e(9),hs(3,3),xs(3,3),cross,at,sc
 real(grid_p),allocatable :: rho(:,:),atom(:),core(:),neutral(:),vh(:),v(:,:),map(:),deformation(:)
 character(2048) :: input,output
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(newunit=fi,file=trim(input),access='stream',form='unformatted',status='old')
 read(fi) shape,spins,nsm,npcc,cell,harris
 if (grid_p/=dp.or.any(modulo(shape,nsm)/=0)) error stop 'unsupported oracle configuration'
 n=product(shape);allocate(rho(n,spins),atom(n),core(n),neutral(n),vh(n),v(n,spins),map(n),deformation(n))
 read(fi) rho,atom,core,neutral
 close(fi)
 call initMeshDistr(UNIFORM,(/1,1,1/),shape/nsm,nsm)
 call reclat(cell,b,1);cutoff=1.e30_dp;k=0._dp
 call chkgmx(k,b,shape,cutoff);volume=volcel(cell);dvol=volume/n
 do i=1,3
   step(:,i)=cell(:,i)/shape(i)
 enddo
 call reclat(step,grad,0)
 do i=1,n
   map(i)=i-1
 enddo
 call reord(map,map,shape/nsm,nsm,1)
 do s=1,spins
   call reord(rho(:,s),rho(:,s),shape/nsm,nsm,1)
 enddo
 nsd=min(spins,2);rnsd=1._dp/nsd
 do s=1,nsd
   rho(:,s)=rho(:,s)-atom*rnsd
 enddo
 if(nsd==2) then
   do i=1,n
     total=rho(i,1)+rho(i,2);rho(i,2)=rho(i,2)-rho(i,1);rho(i,1)=total
   enddo
 endif
 deformation=rho(:,1)
 call poison(cell,shape(1),shape(2),shape(3),shape,rho(:,1),e(2),vh,hs,nsm)
 cross=0;at=0;sc=0
 do i=1,n
   cross=cross+vh(i)*atom(i)
   at=at+neutral(i)*atom(i)
   sc=sc+neutral(i)*rho(i,1)
   vh(i)=vh(i)+neutral(i)
 enddo
 e(1)=cross*dvol+harris+e(2);e(3)=harris;e(4)=at*dvol;e(5)=e(4)+sc*dvol
 if(nsd==2) then
   do i=1,n
     total=rho(i,1);rho(i,1)=.5_dp*(total-rho(i,2));rho(i,2)=.5_dp*(total+rho(i,2))
   enddo
 endif
 do s=1,nsd
   if(npcc==1) then
     rho(:,s)=rho(:,s)+(core+atom)*rnsd
   else
     rho(:,s)=rho(:,s)+atom*rnsd
   endif
 enddo
 call gridxc_init()
 call gridxc_setXC(1,(/'GGA'/),(/'PBE'/),(/1._dp/),(/1._dp/))
 call gridxc_cellXC(0,cell,shape,0,shape(1)-1,0,shape(2)-1,0,shape(3)-1,spins,rho, &
      e(6),e(7),e(8),e(9),xs,v,keep_input_distribution=.true.)
 do s=1,nsd
   v(:,s)=v(:,s)+vh
 enddo
 do s=1,spins
   call reord(v(:,s),v(:,s),shape/nsm,nsm,-1)
 enddo
 call reord(vh,vh,shape/nsm,nsm,-1)
 open(newunit=fo,file=trim(output),access='stream',form='unformatted',status='replace')
 write(fo) b,cutoff,volume,grad,dvol,e,hs,xs,v,map,vh,deformation
 close(fo)
end program
'''

class Problem(ct.Structure):
    _fields_=[('hartree',HartreeProblem),('pbe',PbeProblem),('harris_energy',ct.c_double),('dvol',ct.c_double),
              ('nsm',ct.c_int),('partial_core',ct.c_int)]

class Result(ct.Structure):
    _fields_=[('energies',ct.c_double*9),('hartree_stress',ct.c_double*9),('xc_stress',ct.c_double*9),('status',ct.c_int)]

def fixtures():
    for label,cell,rho in pbe_fixtures():
        if label=='noncollinear_texture':
            # Poisson's pinned GPFA admits factors 2/3/5, unlike XC alone.
            xyz=np.indices((6,9,12))/np.array([6,9,12])[:,None,None,None]
            total=.12+.02*np.cos(2*np.pi*xyz[0]);moment=.45*total
            theta=.8+.3*np.cos(2*np.pi*xyz[1]);phi=.4+.2*np.sin(2*np.pi*xyz[2])
            rho=np.asfortranarray(np.stack(((total+moment*np.cos(theta))/2,(total-moment*np.cos(theta))/2,
                    moment*np.sin(theta)*np.cos(phi)/2,-moment*np.sin(theta)*np.sin(phi)/2),axis=-1))
        shape=rho.shape[:3];spins=rho.shape[-1]
        nsm=3 if label=='noncollinear_texture' else 2 if all(n%2==0 for n in shape) else 1
        xyz=np.indices(shape)/np.array(shape)[:,None,None,None]
        atom=.1+.01*np.sin(2*np.pi*xyz[0]);core=.01+.002*np.cos(2*np.pi*xyz[2]);neutral=-.3+.07*np.cos(2*np.pi*xyz[1])
        partial_core=int(label in ('smooth_scalar','noncollinear_texture','small_periodic_spinor','fully_polarized'))
        # Independent arithmetic map; the oracle exports its original reord
        # mapping and each fixture verifies this exactly.
        coarse=np.array(shape)//nsm
        x,y,z=np.indices(shape)
        mapping=(x%nsm+nsm*(y%nsm+nsm*(z%nsm)))+nsm**3*(x//nsm+coarse[0]*(y//nsm+coarse[1]*(z//nsm)))
        cluster=np.zeros((rho[...,0].size,spins),order='F')
        cluster[mapping.ravel(order='F')]=rho.reshape((-1,spins),order='F')
        yield label,shape,spins,nsm,partial_core,cell,cluster,np.asfortranarray(atom),np.asfortranarray(core),np.asfortranarray(neutral),mapping

def check_bridge(library,oracle,shape,spins,nsm,cell,atom,core,neutral,harris):
    """Use the production density producer and H consumer, with opaque host grids.

    A one-orbital basis makes both contractions independently calculable; the
    grid potential and energies still come from the original Fortran oracle.
    Exercise same-address geometry changes and a device failure, not just the
    happy path through the ownership state machine.
    """
    lib=ct.CDLL(str(library.resolve()))
    query=lib.remag_flow_needs_setup;prepare=lib.remag_flow_prepare
    query.argtypes=[ct.c_void_p]+[ct.c_int]*3+[ct.c_void_p]*4
    prepare.argtypes=query.argtypes+[ct.c_void_p,ct.c_double,ct.c_void_p]+[ct.c_double]*3
    arm=lib.remag_flow_arm;arm.argtypes=[ct.c_int]*2+[ct.c_void_p]*2
    solve=lib.remag_flow_solve;solve.argtypes=[ct.c_void_p]*5
    end=lib.remag_flow_end;end.argtypes=[]
    phase=lib.remag_phase_enter;phase.argtypes=[ct.c_int]*3
    lib.remag_geometry_changed.argtypes=[];lib.remag_geometry_changed.restype=None
    grid=lib.remag_grid;grid.argtypes=[ct.c_int]*8+[ct.c_void_p]*9+[ct.c_double]
    size=int(np.prod(shape));nsp=nsm**3;points=size//nsp
    mesh=np.array(shape,dtype=np.int32);nsc=np.ones(3,dtype=np.int32)
    ends=np.arange(points+1,dtype=np.int32);orbs=np.ones(points,dtype=np.int32)
    num=cols=np.ones(1,dtype=np.int32);ptr=np.zeros(1,dtype=np.int32)
    phi=np.array(.6+.15*np.sin(np.arange(size)*.13),dtype=np.float32)
    dm=np.array([.08,.04,.01,.005],dtype=np.float64)[:spins].copy()
    rho=np.full((size,spins),17.25,order='F');v=np.full_like(rho,-23.5,order='F')
    foreign=np.zeros_like(rho,order='F');matrix=np.zeros(spins)
    energy=np.full(9,123.);hs=np.full(9,123.);xs=np.full(9,123.)
    results=[energy,hs,xs]
    no_core=np.array([np.nan])  # exactly the one-element no-core host contract
    def setup_args(npcc):
        return [mesh.ctypes.data,spins,nsm,npcc,cell.ctypes.data,atom.ctypes.data,
                (core if npcc else no_core).ctypes.data,neutral.ctypes.data]
    def contract(mode,input,output):
        arrays=[nsc,ends,orbs,num,ptr,cols,phi,input,output]
        # NaN is deliberate: resident mode must read dvol from device config.
        return grid(mode,1,points,nsp,1,1,spins,spins,*(a.ctypes.data for a in arrays),float('nan'))
    def evaluate(r=rho,p=v): return solve(r.ctypes.data,p.ctypes.data,*(a.ctypes.data for a in results))
    def untouched():
        np.testing.assert_array_equal(rho,17.25);np.testing.assert_array_equal(v,-23.5)
    def no_results():
        for a in results: np.testing.assert_array_equal(a,123.)
    environment={k:os.environ.get(k) for k in ('REMAG_RESIDENT_GRID','REMAG_GPU','REMAG_GRID','REMAG_GRID_VERIFY_EPOCH')}
    checks=[];rejects=0
    def rejected(code):
        nonlocal rejects
        assert code<0,code
        rejects+=1
    try:
        os.environ.update({k:'1' for k in environment})
        for cycle,npcc in enumerate((0,0,1)):
            if cycle!=1:
                lib.remag_geometry_changed()
                if cycle==2:
                    cell[0,0]*=1.031;atom*=1.013;neutral*=.97
            assert phase(1,cycle//2,cycle+1)==0
            expected_density=np.asfortranarray((dm[None,:]*phi[:,None].astype(float))*phi[:,None].astype(float))
            raw=oracle(expected_density,current_npcc=npcc)
            args=setup_args(npcc);required=int(cycle!=1)
            assert query(*args)==required
            if required:
                reciprocal=np.ascontiguousarray(raw[:9]);gradient=np.ascontiguousarray(raw[11:20])
                assert prepare(*args,reciprocal.ctypes.data,raw[9],gradient.ctypes.data,raw[10],raw[20],harris)==0
            assert query(*args)==0
            saved=atom.flat[0];atom.flat[0]=saved+.1
            assert query(*args)==-191;rejects+=1
            atom.flat[0]=saved
            for a in results: a.fill(123.)
            rejected(evaluate());no_results()
            assert arm(size,spins,rho.ctypes.data,v.ctypes.data)==0
            rejected(arm(size,spins,rho.ctypes.data,v.ctypes.data))
            rejected(end());rejected(evaluate());no_results()
            rejected(contract(0,dm,foreign))
            assert contract(0,dm,rho)==0
            rejected(contract(0,dm,rho))
            rejected(evaluate(foreign,v));rejected(evaluate(rho,foreign));no_results()
            rho[0,0]+=1.;rejected(evaluate());rho[0,0]-=1.
            v[0,0]+=1.;rejected(evaluate());v[0,0]-=1.
            no_results();untouched()
            assert evaluate()==0
            rejected(evaluate())
            expected_v=raw[48:48+size*spins].reshape((size,spins),order='F')
            seed=np.arange(spins)*.01+.2;matrix[:]=seed
            rejected(contract(1,foreign,matrix))
            assert contract(1,v,matrix)==0
            expected_matrix=seed+np.sum(phi[:,None]*expected_v*phi[:,None],axis=0)*raw[20]
            errors={}
            for name,a,b in [('energies',energy,raw[21:30]),('hartree_stress',hs,raw[30:39]),
                             ('xc_stress',xs,raw[39:48]),('potential_matrix',matrix,expected_matrix)]:
                np.testing.assert_allclose(a,b,atol=1.e-9,rtol=1.e-10)
                errors[name]=float(np.max(np.abs(a-b)))
            rejected(contract(1,v,matrix));assert end()==0;rejected(end());untouched()
            checks.append(dict(fixture='production_grid_ownership',cycle=cycle,partial_core=npcc,
                               geometry_setup=bool(required),max_absolute_errors=errors,passed=True))
            dm*=.83
        # Invalid device density must not export a partially computed result.
        for a in results: a.fill(123.)
        assert arm(size,spins,rho.ctypes.data,v.ctypes.data)==0
        good=dm[0];dm[0]=np.nan
        assert contract(0,dm,rho)==0
        assert evaluate()==-196;rejects+=1
        no_results();untouched();rejected(contract(1,v,matrix));rejected(end())
        # A fresh producer after the rejected computation must recover.
        dm[0]=good
        assert arm(size,spins,rho.ctypes.data,v.ctypes.data)==0
        assert contract(0,dm,rho)==0;assert evaluate()==0
        raw=oracle(np.asfortranarray((dm[None,:]*phi[:,None].astype(float))*phi[:,None].astype(float)),current_npcc=1)
        for a,b in zip(results,(raw[21:30],raw[30:39],raw[39:48])):
            np.testing.assert_allclose(a,b,atol=1.e-9,rtol=1.e-10)
        assert contract(1,v,matrix)==0;assert end()==0;untouched()
        checks.append(dict(fixture='production_grid_failure_recovery',rejected_invalid_operations=rejects,passed=True))
    finally:
        assert phase(2,0,0)==0
        for key,value in environment.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
    return checks

def run(upstream,build,output,library=None,bridge_library=None):
    upstream=upstream.resolve();build=build.resolve();src=build/'Src'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(reference='Original reord, Poisson/GPFA and full GridXC joined by literal periodic DHSCF arithmetic',
                fixture_script_sha256=sha(Path(__file__)),source_sha256={name:sha(upstream/'Src'/name) for name in ('dhscf.F','reord.f','poison.F')},
                reference_library_sha256=sha(src/'libsiesta.a'),gridxc_library_sha256=sha(build/'libgridxc/libgridxc.a'),
                gpu_validation=None,checks=[])
    arrays={}
    with tempfile.TemporaryDirectory(prefix='remag-grid-potential-') as tmp:
        tmp=Path(tmp);wrapper=tmp/'reference.f90';wrapper.write_text(WRAPPER);exe=tmp/'reference'
        link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text())
        libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
        flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
        includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
        subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(wrapper),'-o',str(exe),*libraries],cwd=src,check=True)
        candidate=None
        if library:
            candidate=ct.CDLL(str(library.resolve())).remag_check_grid_potential
            candidate.argtypes=[ct.POINTER(Problem)]+[ct.c_void_p]*7+[ct.POINTER(Result),ct.c_int,ct.c_int,ct.POINTER(Check)]
            candidate.restype=ct.c_int;report['native_library_sha256']=sha(library)
        for label,shape,spins,nsm,npcc,cell,rho,atom,core,neutral,mapping in fixtures():
            harris=.375
            inp=tmp/'input';out=tmp/'output'
            def oracle(electron,*,current_npcc=None):
                partial_core=npcc if current_npcc is None else current_npcc
                inp.write_bytes(np.array([*shape,spins,nsm,partial_core],dtype=np.int32).tobytes()+cell.tobytes(order='F')+
                    np.float64(harris).tobytes()+electron.tobytes(order='F')+atom.tobytes(order='F')+
                    core.tobytes(order='F')+neutral.tobytes(order='F'))
                completed=subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,capture_output=True,text=True)
                if completed.returncode:
                    print(label,completed.stdout,completed.stderr,flush=True)
                    completed.check_returncode()
                raw=np.fromfile(out,dtype=np.float64)
                assert raw.size==48+rho.size+3*atom.size and np.isfinite(raw).all()
                np.testing.assert_array_equal(raw[48+rho.size:48+rho.size+atom.size],mapping.ravel(order='F'))
                return raw
            raw=oracle(np.asfortranarray(rho*.75))
            hp=HartreeProblem((ct.c_int*3)(*shape),(ct.c_double*9)(*raw[:9]),raw[9],raw[10])
            pp=PbeProblem((ct.c_int*3)(*shape),spins,(ct.c_double*9)(*raw[11:20]),raw[10])
            problem=Problem(hp,pp,harris,raw[20],nsm,npcc)
            reference_energies=raw[21:30];reference_hs=raw[30:39];reference_xs=raw[39:48];reference_v=raw[48:48+rho.size].reshape(rho.shape,order='F')
            for name,a in [('cell',cell),('density',rho),('atom',atom),('core',core),('neutral',neutral),('raw_reference',raw)]: arrays[label+'_'+name]=a.copy(order='F')
            for graph in ((False,True) if candidate else (False,)):
                record=dict(fixture=label,shape=shape,spins=spins,nsm=nsm,partial_core=npcc,graph=graph,reordering_exact=True,
                            reference_energies=reference_energies.tolist(),input_sha256=sha(inp))
                if candidate:
                    result=Result();info=Check();v=np.zeros_like(rho,order='F');aux=np.full(atom.size,np.nan);deformation=aux.copy()
                    code=candidate(ct.byref(problem),rho.ctypes.data,atom.ctypes.data,core.ctypes.data,neutral.ctypes.data,
                                   v.ctypes.data,aux.ctypes.data,deformation.ctypes.data,ct.byref(result),int(graph),0,ct.byref(info))
                    assert code==0 and result.status==0,(label,graph,code,result.status)
                    assert info.iterations==2 and info.forbidden_nodes==0
                    if graph: assert info.graph_nodes>=8
                    errors={}
                    for name,a,b in [('energies',np.array(result.energies),reference_energies),('hartree_stress',np.array(result.hartree_stress),reference_hs),
                                     ('xc_stress',np.array(result.xc_stress),reference_xs),('potential',v,reference_v),
                                     ('auxiliary_potential',aux,raw[48+rho.size+atom.size:48+rho.size+2*atom.size]),
                                     ('deformation_density',deformation,raw[-atom.size:])]:
                        np.testing.assert_allclose(a,b,atol=1.e-9,rtol=1.e-10);errors[name]=float(np.max(np.abs(a-b)))
                    record.update(max_absolute_errors=errors,graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,
                                  iterations=info.iterations,workspace_bytes=info.workspace_bytes)
                    for name,value in (('potential',v),('auxiliary',aux),('deformation',deformation),
                                       ('energies',np.array(result.energies)),('hartree_stress',np.array(result.hartree_stress)),
                                       ('xc_stress',np.array(result.xc_stress))):
                        arrays[f'{label}_graph{int(graph)}_{name}']=value.copy(order='F')
                record['passed']=True;report['checks'].append(record)
        if candidate:
            for invalid in ('density_nan','atom_nan','core_nan','potential_nan','nsm_zero','nsm_nondivisor','invalid_partial_core','bad_dvol'):
                p=Problem.from_buffer_copy(bytes(problem));p.partial_core=1
                d=rho.copy(order='F');a=atom.copy(order='F');c=core.copy(order='F');vna=neutral.copy(order='F')
                if invalid=='density_nan': d[0,0]=np.nan
                elif invalid=='atom_nan': a.flat[0]=np.nan
                elif invalid=='core_nan': c.flat[0]=np.nan
                elif invalid=='potential_nan': vna.flat[0]=np.nan
                elif invalid=='nsm_zero': p.nsm=0
                elif invalid=='nsm_nondivisor': p.nsm=5
                elif invalid=='invalid_partial_core': p.partial_core=2
                elif invalid=='bad_dvol': p.dvol=np.nan
                result=Result();info=Check();v=np.zeros_like(d,order='F');aux=np.zeros(a.size);deformation=aux.copy()
                code=candidate(ct.byref(p),d.ctypes.data,a.ctypes.data,c.ctypes.data,vna.ctypes.data,v.ctypes.data,
                               aux.ctypes.data,deformation.ctypes.data,ct.byref(result),1,0,ct.byref(info))
                assert code==0 and result.status==2,(invalid,code,result.status)
                assert info.forbidden_nodes==0 and info.iterations==2
                report['checks'].append(dict(fixture=invalid,graph=True,status=result.status,graph_nodes=info.graph_nodes,
                    forbidden_nodes=info.forbidden_nodes,iterations=info.iterations,passed=True))
            for corrupt_shape in (1,2):
                result=Result();info=Check();v=np.zeros_like(rho,order='F');aux=np.zeros(atom.size);deformation=aux.copy()
                code=candidate(ct.byref(problem),rho.ctypes.data,atom.ctypes.data,core.ctypes.data,neutral.ctypes.data,
                               v.ctypes.data,aux.ctypes.data,deformation.ctypes.data,ct.byref(result),1,corrupt_shape,ct.byref(info))
                assert code==0 and result.status==1,(corrupt_shape,code,result.status)
                assert info.forbidden_nodes==0 and info.iterations==2
                report['checks'].append(dict(fixture='plan_shape_mismatch',corruption=corrupt_shape,graph=True,
                    status=result.status,graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,
                    iterations=info.iterations,passed=True))
            report['gpu_validation']=True
        if bridge_library:
            # The last fixture has a clustered 4x6x8 grid and four spin channels.
            report['bridge_library_sha256']=sha(bridge_library)
            report['checks'].extend(check_bridge(bridge_library,oracle,shape,spins,nsm,cell,atom,core,neutral,harris))
    output.parent.mkdir(parents=True,exist_ok=True)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**arrays)
    report['arrays_file']=artifact.name;report['arrays_sha256']=sha(artifact)
    report['passed']=all(c['passed'] for c in report['checks'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Grid-potential scenarios: {len(report['checks'])}; passed={report['passed']}; GPU validation={report['gpu_validation']}")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'));p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path)
    p.add_argument('--bridge-library',type=Path)
    args=p.parse_args();run(args.upstream,args.build,args.output,args.library,args.bridge_library)
