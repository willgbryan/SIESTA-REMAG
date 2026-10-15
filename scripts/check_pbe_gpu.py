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

def frame_scenarios():
    """Spinor densities for the local frame: n and m = (2 Re D12, -2 Im D12, D11 - D22)."""
    cell=np.asfortranarray([[8.,1.2,-.7],[0.,9.,.5],[0.,0.,10.]])
    for label,shape in (('frame_signed_axis',(9,8,10)),('frame_noncollinear',(8,9,7)),('frame_node_plane',(10,7,8))):
        xyz=np.indices(shape)/np.array(shape)[:,None,None,None]
        n=.15+.03*np.cos(2*np.pi*xyz[0])+.02*np.sin(2*np.pi*xyz[2])
        if label=='frame_signed_axis':
            # Collinear up to sign along a tilted axis; the moment changes sign twice.
            u=np.array([.3,-.4,.866]);u/=np.linalg.norm(u);m=(.08*n*np.cos(2*np.pi*xyz[0]))[...,None]*u
        elif label=='frame_noncollinear':
            phi=2*np.pi*(xyz[0]+xyz[1]);m=np.stack([.07*n*np.cos(phi),.07*n*np.sin(phi),.02*n*np.cos(2*np.pi*xyz[2])],-1)
        else:
            # A plane where |m| vanishes exactly, as in a domain wall.
            u=np.array([0.,0.,1.]);mz=.06*n*np.sin(2*np.pi*xyz[1]);mz[:,0,:]=0.;m=mz[...,None]*u
        d=np.zeros((*shape,4),order='F')
        d[...,0]=(n+m[...,2])/2;d[...,1]=(n-m[...,2])/2;d[...,2]=m[...,0]/2;d[...,3]=-m[...,1]/2
        yield label,cell,d

def frame_decision(d):
    """remag_xc.F90.in's rule: signed if the top eigenvalue of sum m m^T has 99% of its trace."""
    m=np.stack([2*d[...,2],-2*d[...,3],d[...,0]-d[...,1]],-1).reshape(-1,3)
    t=m.T@m;v=m.sum(0)
    if np.linalg.norm(v)==0:v=np.diag(t).copy()
    if np.linalg.norm(v)==0:v=np.array([0.,0.,1.])
    for _ in range(200):
        w=t@v;a=np.linalg.norm(w)
        if a==0:break
        v=w/a
    trace=np.trace(t);signed=trace>0 and v@t@v>.99*trace
    return signed,v

def frame_reference(evaluate,cell,d,signed,u):
    m=np.stack([2*d[...,2],-2*d[...,3],d[...,0]-d[...,1]],-1);n=d[...,0]+d[...,1]
    a=np.linalg.norm(m,axis=-1);e=np.zeros_like(m);big=a>1.e-14
    s=np.where(np.einsum('...k,k->...',m,u)<0,-1.,1.) if signed else np.ones_like(a)
    e[big]=(s[...,None]*m/np.where(big,a,1.)[...,None])[big]
    if signed:e[~big]=u
    along=np.einsum('...k,...k->...',e,m)
    d2=np.asfortranarray(np.stack([.5*(n+along),.5*(n-along)],-1))
    energies,stress,v2=evaluate(cell,d2)
    sumv=.5*(v2[...,0]+v2[...,1]);difv=.5*(v2[...,0]-v2[...,1])
    v=np.asfortranarray(np.stack([sumv+difv*e[...,2],sumv-difv*e[...,2],difv*e[...,0],-difv*e[...,1]],-1))
    return energies,stress,v

def mc_rule(polar,azimuth=-1):
    """mc_directions in remag_xc.F90.in: half-sphere Gauss-Legendre polar nodes times uniform azimuths."""
    na=4*polar if azimuth<0 else azimuth;nt=2*polar;directions=[];weights=[]
    for i in range(1,polar+1):
        x=np.cos(np.pi*(i-.25)/(nt+.5))
        for _ in range(100):
            p0,p1=1.,x
            for j in range(2,nt+1):p0,p1=p1,((2*j-1)*x*p1-(j-1)*p0)/j
            dp1=nt*(x*p1-p0)/(x*x-1.)
            if abs(p1/dp1)<1.e-15:break
            x-=p1/dp1
        w=2./((1.-x*x)*dp1*dp1)
        for j in range(na):
            phi=2*np.pi*j/na;directions.append([np.sqrt(1-x*x)*np.cos(phi),np.sqrt(1-x*x)*np.sin(phi),x]);weights.append(w/na)
    return np.array(directions),np.array(weights)

def mc_reference(evaluate,cell,d,polar,h=1.e-4):
    """multicollinear in remag_xc.F90.in, with pristine GridXC as the collinear functional."""
    m=np.stack([2*d[...,2],-2*d[...,3],d[...,0]-d[...,1]],-1);n=d[...,0]+d[...,1]
    dvol=abs(np.linalg.det(cell))/np.prod(d.shape[:3]);lam=(1.,1.-h,1.-2*h)
    emc=ec=dc=0.;stress=np.zeros((3,3));vn=np.zeros_like(n);vm=np.zeros_like(m)
    for e,w in zip(*mc_rule(polar)):
        s=m@e;calls=[evaluate(cell,np.asfortranarray(np.stack([.5*(n+l*s),.5*(n-l*s)],-1))) for l in lam]
        (en1,st1,v1),(_,st2,v2),(_,st3,v3)=calls
        emc+=w*(en1[0]+en1[1]+dvol*np.sum(.5*(v1[...,0]-v1[...,1])*s));ec+=w*en1[1];dc+=w*en1[3]
        vn0=.5*(v1[...,0]+v1[...,1]);vs0=.5*(v1[...,0]-v1[...,1])
        dvn=(3*vn0-2*(v2[...,0]+v2[...,1])+.5*(v3[...,0]+v3[...,1]))/(2*h)
        dvs=(3*vs0-2*(v2[...,0]-v2[...,1])+.5*(v3[...,0]-v3[...,1]))/(2*h)
        vn+=w*(vn0+dvn);vm+=w*(2*vs0+dvs)[...,None]*e
        stress+=w*(st1+(3*st1-4*st2+st3)/(2*h))
    energies=np.array([emc-ec,ec,emc-dvol*(np.sum(n*vn)+np.sum(m*vm))-dc,dc])
    v=np.asfortranarray(np.stack([vn+vm[...,2],vn-vm[...,2],vm[...,0],-vm[...,1]],-1))
    return energies,stress,v

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
        # Local frame: the same plan, switched before construction; each fixture
        # decides its own axis on the first (unscaled) evaluation.
        lib=ct.CDLL(str(library.resolve()))
        lib.remag_set_xc_frame.argtypes=[ct.c_int];lib.remag_set_xc_frame(1)
        for label,cell,d in [*frame_scenarios(),*[x for x in scenarios() if x[0] in ('nearly_unpolarized','left_handed_spinor')]]:
            cell=np.asfortranarray(cell);d=np.asfortranarray(d);p=problem(cell,d)
            signed,u=frame_decision(d)
            expected=frame_reference(evaluate,cell,np.asfortranarray(d*.75),signed,u)
            for graph in (False,True):
                lib.remag_reset_xc_frame()
                result=Result();info=Check();v=np.full_like(d,np.nan,order='F')
                code=candidate(ct.byref(p),d.ctypes.data,v.ctypes.data,ct.byref(result),int(graph),0,ct.byref(info))
                assert code==0 and result.status==0,(label,graph,code,result.status)
                decided=ct.c_int();signed_device=ct.c_int();axis=(ct.c_double*3)()
                lib.remag_query_xc_frame(ct.byref(decided),ct.byref(signed_device),axis)
                assert decided.value==1 and bool(signed_device.value)==bool(signed),(label,decided.value,signed_device.value,signed)
                if signed:assert abs(abs(np.dot(np.array(axis[:]),u))-1)<1.e-12,(label,axis[:],u)
                actual=(np.array(result.energies),np.array(result.stress).reshape((3,3),order='F'),v)
                errors={}
                for name,value,ref in zip(('energies','stress','potential'),actual,expected):
                    np.testing.assert_allclose(value,ref,atol=1.e-9,rtol=1.e-10)
                    errors[name]=float(np.max(np.abs(value-ref)))
                report['checks'].append(dict(fixture=label,frame='local',signed_axis=bool(signed),shape=d.shape,graph=graph,
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,iterations=info.iterations,
                    workspace_bytes=info.workspace_bytes,max_absolute_errors=errors,status=result.status,passed=True))
        # Multicollinear: 16 and 64 directions. The l stencil divides rounding by
        # 2h = 2e-4, so potentials and stress are compared to 2e-8.
        lib.remag_set_xc_frame(2)
        for label,cell,d in [x for x in frame_scenarios() if x[0]!='frame_node_plane']+\
                [x for x in scenarios() if x[0]=='left_handed_spinor']:
            for polar in (2,4):
                lib.remag_set_xc_multicollinear(polar,-1)
                cell=np.asfortranarray(cell);d=np.asfortranarray(d);p=problem(cell,d)
                expected=mc_reference(evaluate,cell,np.asfortranarray(d*.75),polar)
                for graph in (False,True):
                    result=Result();info=Check();v=np.full_like(d,np.nan,order='F')
                    code=candidate(ct.byref(p),d.ctypes.data,v.ctypes.data,ct.byref(result),int(graph),0,ct.byref(info))
                    assert code==0 and result.status==0,(label,polar,graph,code,result.status)
                    actual=(np.array(result.energies),np.array(result.stress).reshape((3,3),order='F'),v)
                    errors={}
                    for name,value,ref,atol in zip(('energies','stress','potential'),actual,expected,(1.e-9,2.e-8,2.e-8)):
                        np.testing.assert_allclose(value,ref,atol=atol,rtol=1.e-9)
                        errors[name]=float(np.max(np.abs(value-ref)))
                    report['checks'].append(dict(fixture=label,frame='multicollinear',directions=4*polar*polar,shape=d.shape,
                        graph=graph,graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,iterations=info.iterations,
                        workspace_bytes=info.workspace_bytes,max_absolute_errors=errors,status=result.status,passed=True))
        lib.remag_set_xc_frame(0)
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
