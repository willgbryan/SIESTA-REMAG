#!/usr/bin/env python3
"""Compose all conventional SCF producers inside one bounded CUDA graph.

Synthetic sparse/mesh fixtures use SciPy generalized diagonalization and
pristine upstream FD, Poisson/GridXC, scalar, convergence and Pulay oracles.
This validates cycle composition, not production SIESTA handoff or materials.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from scipy.linalg import eigh
import check_fermi as fd
import check_initial_density as initial_density
import check_grid_potential as gp
import check_hamiltonian_seeds as seeds
import check_scf_scalars as scalars
import check_scf_convergence as convergence
import check_sparse_state as sparse
from check_hamiltonian_seeds_gpu import Result as SeedResult
from check_scf_control_gpu import Result as ControlResult,Step,raw,ints,flags
from check_block_jacobi_gpu import observer_environment,calibrate_observer

class Problem(ct.Structure):
    _fields_=[(n,ct.c_int) for n in ('n','kind','nh','nk','maximum','minimum','mix_first','soc','fault','converge')]+[
        ('charge',ct.c_double),('grid',gp.Problem)]
class Result(ct.Structure):
    _fields_=[(n,ct.c_int) for n in ('status','producer_status','stage','block','pass_index','iterations','finished','reserved')]+[
        ('control',ControlResult)]
class Normalization(ct.Structure):
    _fields_=[('charge',ct.c_double),('scale',ct.c_double),('status',ct.c_int)]
class Diagnostic(ct.Structure):
    _fields_=[('step',Step),('fermi',fd.Result),('normalization',Normalization),('hamiltonian',SeedResult),('grid',gp.Result)]
assert ct.sizeof(Diagnostic)==560
class Check(ct.Structure):
    _fields_=[(n,ct.c_size_t) for n in ('workspace_bytes','staged_bytes','exported_bytes')]+[
        (n,ct.c_int) for n in ('graph_nodes','forbidden_nodes','replay_equal')]
assert (ct.sizeof(Problem),ct.sizeof(Result),ct.sizeof(Check))==(272,160,40)
NAMES=('h','d','e','overlap','kinetic','nonlocal_h','onsite','offsite','phi','kpoints','weights','xyz','atom','core','neutral','constants')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def farray(a):return np.asfortranarray(a)

def compile_wrapper(build,tmp,source):
    tmp.mkdir(exist_ok=True);src=build.resolve()/'Src';code=tmp/'reference.f90';code.write_text(source);exe=tmp/'reference'
    link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text());libraries=link[next(i for i,s in enumerate(link) if Path(s).name=='libsiesta.a'):]
    flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
    includes=next(line.split('=',1)[1].strip() for line in flags.splitlines() if line.startswith('Fortran_INCLUDES ='))
    subprocess.run([link[0],'-O2',*shlex.split(includes),'-I'+str(src),str(code),'-o',str(exe),*libraries],cwd=src,check=True)
    return exe

class Oracle:
    def __init__(self,upstream,build,tmp,retained):
        self.tmp=tmp;self.calls=0;self.hashes=[]
        (tmp/'initial').mkdir()
        self.initial=initial_density.build_reference(build,tmp/'initial')
        self.grid=compile_wrapper(build,tmp/'grid',gp.WRAPPER)
        self.conv=compile_wrapper(build,tmp/'convergence',convergence.WRAPPER)
        for name in ('seeds','scalars','mix','fd'):(tmp/name).mkdir()
        self.seed,seed_wrapper,_=seeds.build_reference(upstream,build,tmp/'seeds')
        self.scalar=scalars.build_reference(build,tmp/'scalars');self.mixer=sparse.mixing_executable(build,tmp/'mix')
        f=tmp/'fd';(f/'stubs.f90').write_text(fd.STUBS);(f/'wrapper.f90').write_text(fd.WRAPPER)
        subprocess.run(['gfortran','-O2','-shared','-fPIC','-cpp','-ffixed-line-length-none','stubs.f90',
            str((upstream/'Src/fermid.F').resolve()),'wrapper.f90','-o','reference.so'],cwd=f,check=True)
        self.fermi=ct.CDLL(str(f/'reference.so')).remag_reference_fermi
        self.fermi.argtypes=[ct.c_int]*5+[ct.c_double]*2+[ct.POINTER(ct.c_double)]*5
        self.fermi.restype=None
        wrappers={'grid-reference.f90':gp.WRAPPER,'convergence-reference.f90':convergence.WRAPPER,
            'seed-reference.F':seed_wrapper,'scalar-reference.f90':scalars.WRAPPER,'mix-reference.f90':sparse.mixing_ref.WRAPPER,
            'fd-stubs.f90':fd.STUBS,'fd-wrapper.f90':fd.WRAPPER,'initial-density-reference.f90':initial_density.WRAPPER}
        for name,source in wrappers.items():(retained/name).write_text(source)
        self.wrappers={name:sha(retained/name) for name in wrappers}
    def execute(self,exe,data,dtype=np.float64):
        folder=exe.parent;inp=folder/'input';out=folder/'output';inp.write_bytes(data)
        proc=subprocess.run([str(exe),str(inp),str(out)],cwd=folder,capture_output=True,text=True)
        if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
        self.calls+=1;self.hashes.append(dict(input_sha256=sha(inp),output_sha256=sha(out),oracle=folder.name))
        return np.fromfile(out,dtype=dtype)
    def grid_eval(self,p,a,rho):
        cell=a['cell'];mesh=tuple(p.grid.pbe.mesh)
        return self.execute(self.grid,np.array([*mesh,min(p.nh,4),p.grid.nsm,p.grid.partial_core],np.int32).tobytes()+
            cell.tobytes(order='F')+np.float64(p.grid.harris_energy).tobytes()+rho.tobytes(order='F')+
            b''.join(a[name].tobytes(order='F') for name in ('atom','core','neutral')))
    def decisions(self,p,values):
        seq=np.array(values)[:,[6,7,8,3,4]]
        return self.execute(self.conv,np.array([7,p.minimum,len(values),0],np.int32).tobytes()+
            np.full(5,1.e3 if p.converge else 0.).tobytes()+seq.tobytes(),np.int32).reshape(-1,5)
    def cycle(self,p,a):
        n,nh=p.n,p.nh;cap=n*n+2;eh=min(nh,4);dim=n*(2 if p.kind>=2 else 1);spins=nh if p.kind<2 else 1
        h=a['h'].copy();d=a['d'].copy();e=a['e'].copy();saved=[];residuals=[];values=[];decisions=[];trace=[];diagnostics=[]
        sm=a['overlap'][:n*n].reshape(n,n,order='F');xy=a['xyz'].reshape(n*n,3)
        phi=a['phi'].reshape(-1,n,p.grid.nsm**3).transpose(0,2,1).reshape(-1,n).astype(float)
        effective_mix_first=p.mix_first
        if p.fault==9:
            _,_,effective_mix_first,d,e=initial_density.evaluate(self.initial,self.tmp/'initial','initial',a['overlap'],d,p.charge,p.mix_first)
            dm=d[:n*n,:eh].copy()
            if nh==8:dm[:,2:]=.5*(d[:n*n,2:4]+d[:n*n,6:8])
            rho=farray(np.column_stack([np.einsum('pi,ij,pj->p',phi,dm[:,spin].reshape(n,n),phi) for spin in range(eh)]))
            gr=self.grid_eval(p,a,rho);v=gr[48:48+rho.size].reshape(rho.shape,order='F');increment=np.zeros_like(h)
            for spin in range(eh):increment[:n*n,spin]=(phi.T@(v[:,spin,None]*phi)*p.grid.dvol).ravel()
            if nh==8:increment[:,6:8]=increment[:,2:4]
            _,_,_,h,_=seeds.evaluate(self.seed,self.tmp/'seeds',p.soc,h,d,a['kinetic'],a['nonlocal_h'],a['onsite'],a['offsite'],increment)
        for iteration in range(1,p.maximum+1):
            old_h=h.copy();old_d=d.copy();old_e=e.copy();ws=[];zs=[]
            for k in a['kpoints'].reshape(-1,3):
                phase=np.exp(-1j*(xy@k)).reshape(n,n,order='F') if p.kind%2 else np.ones((n,n))
                blocks=[h[:n*n,j].reshape(n,n,order='F') for j in range(nh)]
                for spin in range(spins):
                    if p.kind<2:hh=blocks[spin]*phase;ss=sm*phase
                    else:
                        hh=np.empty((dim,dim),complex);ss=np.kron(sm*phase,np.eye(2))
                        hh[::2,::2]=(blocks[0]+(1j*blocks[4] if nh==8 else 0))*phase
                        hh[1::2,1::2]=(blocks[1]+(1j*blocks[5] if nh==8 else 0))*phase
                        hh[::2,1::2]=(blocks[2]-1j*blocks[3])*phase
                        hh[1::2,::2]=(blocks[6]+1j*blocks[7] if nh==8 else blocks[2]+1j*blocks[3])*phase
                    np.testing.assert_allclose(hh,hh.conj().T,atol=1.e-12)
                    w,z=eigh(hh,ss);ws.append(w);zs.append(z)
            bank=np.ascontiguousarray(ws);q=np.zeros_like(bank);ef=ct.c_double();entropy=ct.c_double()
            self.fermi(1 if nh==1 or p.kind==2 else 2,spins,p.nk,dim,dim,.03,p.charge,
                fd.pointer(bank),fd.pointer(a['weights']),fd.pointer(q),ct.byref(ef),ct.byref(entropy))
            d=np.zeros((cap,nh));e=np.zeros((cap,eh))
            for job,z in enumerate(zs):
                k=a['kpoints'].reshape(-1,3)[job//spins];weight=np.where(abs(q[job])>1.e-12,q[job],0.)
                for index in range(n*n):
                    row,col=index//n,index%n;ph=np.exp(-1j*np.dot(k,xy[index])) if p.kind%2 else 1.
                    if p.kind<2:
                        val=(z[row]*z[col].conj()*ph).real
                        d[index,job%spins]+=val@weight;e[index,job%spins]+=val@(weight*bank[job])
                    else:
                        aa=z[2*row]*z[2*col].conj()*ph;bb=z[2*row+1]*z[2*col+1].conj()*ph
                        cc=z[2*row]*z[2*col+1].conj()*ph;ff=z[2*row+1]*z[2*col].conj()*ph
                        if nh==4:cc=.5*(cc+ff.conj())
                        v=np.array([aa.real,bb.real,cc.real,-cc.imag,aa.imag,bb.imag,ff.real,ff.imag])
                        d[index]+=v[:nh]@weight;e[index]+=v[:eh]@(weight*bank[job])
            charge=np.sum(d[:,:min(nh,2)]*a['overlap'][:,None]);assert abs(charge-p.charge)<1.e-6,(charge,p.charge)
            d*=p.charge/charge;e*=p.charge/charge
            dm=d[:n*n,:eh].copy()
            if nh==8:dm[:,2:]=.5*(d[:n*n,2:4]+d[:n*n,6:8])
            rho=farray(np.column_stack([np.einsum('pi,ij,pj->p',phi,dm[:,spin].reshape(n,n),phi) for spin in range(eh)]))
            gr=self.grid_eval(p,a,rho);v=gr[48:48+rho.size].reshape(rho.shape,order='F')
            increment=np.zeros_like(h)
            for spin in range(eh):increment[:n*n,spin]=(phi.T@(v[:,spin,None]*phi)*p.grid.dvol).ravel()
            if nh==8:increment[:,6:8]=increment[:,2:4]
            _,_,en,h,_=seeds.evaluate(self.seed,self.tmp/'seeds',p.soc,old_h,d,a['kinetic'],a['nonlocal_h'],a['onsite'],a['offsite'],increment)
            es=a['constants'].copy();es[1:4]=en;es[5]=gr[25];es[6]=gr[24];es[7]=gr[22];es[8]=gr[26]+gr[27];es[9]=entropy.value
            diagnostics.append(np.array([ef.value,entropy.value,q.sum(),charge,p.charge/charge,*en,*gr[21:48]]))
            scalar,_=scalars.evaluate(self.scalar,self.tmp/'scalars',np.full(n,n,np.int32),np.arange(n,dtype=np.int32)*n,
                es,h,old_h,d,old_d,e,old_e,a['overlap']);values.append(scalar)
            dec=self.decisions(p,values)[-1];decisions.append(dec)
            if not dec[1]:
                saved.append(old_h);residuals.append(h-old_h)
                mixed=sparse.mix(self.mixer,self.tmp/'mix',cap,nh,saved,residuals)
                h=old_h+residuals[-1] if iteration==1 and not effective_mix_first else mixed
            trace.append(np.concatenate([x.ravel(order='F') for x in (h,d,e)]))
            if dec[1]:break
        return dict(effective_mix_first=np.array(effective_mix_first),h=h,d=d,e=e,bank=bank,q=q,fermi=np.array([ef.value,entropy.value,p.charge]),
            values=np.array(values),decisions=np.array(decisions),trajectory=np.array(trace),diagnostics=np.array(diagnostics))

def fixtures(oracle,directory,initialization=False,kpoints=2):
    rng=np.random.default_rng(61006741);records=[]
    modes=[(0,1,0),(0,2,0),(1,1,0),(1,2,0),(2,4,0),(3,4,0),(2,8,1),(3,8,1),(2,8,2),(3,8,2)]
    cases=[(*v,6) for v in modes]+[(0,2,0,33),(3,8,2,33)]
    # More k points than solver lanes exercise the device k loop's waves,
    # including a partial final wave; only sampled-k kinds take part.
    if kpoints!=2:cases=[c for c in cases if c[0]%2]
    for kind,nh,soc,n in cases:
        for mixfirst in (0,1):
            cap=n*n+2;eh=min(nh,4);nk=kpoints if kind%2 else 1;nsm=2 if mixfirst else 1
            mesh=(4,4,4);size=int(np.prod(mesh));cell=farray(np.diag([6.,6.5,7.]));charge=4.31*n/6
            hp=gp.HartreeProblem((ct.c_int*3)(*mesh),(ct.c_double*9)(*np.zeros(9)),1.e30,float(np.linalg.det(cell)))
            pp=gp.PbeProblem((ct.c_int*3)(*mesh),eh,(ct.c_double*9)(*np.zeros(9)),hp.volume)
            p=Problem(n,kind,nh,nk,5,3,mixfirst,soc,0,mixfirst,charge,gp.Problem(hp,pp,.025,hp.volume/size,nsm,1))
            def symmetric(scale=1.):
                x=rng.normal(size=(n,n));return scale*(x+x.T)/2
            def flat(m):return np.concatenate([m.ravel(order='F'),[0.,0.]])
            ov=flat(np.eye(n)+symmetric(.015));kin=flat(np.diag(np.linspace(-.4,.6,n))+symmetric(.035));vkb=flat(symmetric(.02))
            on=np.zeros((cap,6));off=np.zeros((cap,4),complex)
            c=rng.normal(size=(n,n))*.008+1j*rng.normal(size=(n,n))*.007
            on[:,0]=flat(c.real);on[:,1]=flat(-c.imag);on[:,4]=flat(c.real.T);on[:,5]=flat(-c.imag.T)
            aa=rng.normal(size=(n,n))*.004;bb=rng.normal(size=(n,n))*.003
            on[:,2]=flat(aa-aa.T);on[:,3]=flat(bb-bb.T)
            off[:,0]=flat(symmetric(.005)+1j*(aa-aa.T));off[:,1]=flat(symmetric(.005)+1j*(bb-bb.T))
            off[:,2]=flat(c);off[:,3]=flat(c.conj().T)
            phi=np.array(rng.normal(size=(size//nsm**3,n,nsm**3))*(.025*np.sqrt(6/n)),dtype=np.float32)
            k=np.array([[.19,-.27,.07],[-.08,.14,.29]])[:nk] if kind%2 else np.zeros((1,3))
            if kind%2 and nk!=2:k=rng.uniform(-.3,.3,size=(nk,3))
            pos=rng.normal(size=(n,3));xyz=np.array([pos[i//n]-pos[i%n] for i in range(n*n)])
            weights=np.array([.37,.63]) if nk==2 else np.array([.5 if kind==2 else 1.])
            if kind%2 and nk!=2:weights=rng.uniform(.5,1.5,size=nk);weights/=weights.sum()
            h=np.zeros((cap,nh));d=np.zeros_like(h);e=np.zeros((cap,eh));constants=np.array([.4,0.,0.,0.,.21,0.,0.,0.,0.,0.,.03,.012])
            a=dict(zip(NAMES,(h,d,e,ov,kin,vkb,on,off,phi,k,weights,xyz,
                np.full(size,.002),np.full(size,.0002),-.1+.003*np.cos(np.arange(size)),constants)),cell=cell)
            raw_grid=oracle.grid_eval(p,a,np.full((size,eh),.004,order='F'))
            p.grid.hartree.reciprocal=(ct.c_double*9)(*raw_grid[:9]);p.grid.hartree.cutoff=raw_grid[9]
            p.grid.pbe.gradient=(ct.c_double*9)(*raw_grid[11:20]);p.grid.dvol=raw_grid[20]
            _,_,_,h,_=seeds.evaluate(oracle.seed,oracle.tmp/'seeds',soc,h,d,kin,vkb,on,off,np.zeros_like(h))
            for spin in range(min(nh,2)):h[:n*n,spin]+=np.eye(n).ravel(order='F')*(.08 if spin==0 else -.06)
            a['h']=h;a={name:farray(v) if v.ndim==2 and name not in ('kpoints','xyz') else np.ascontiguousarray(v) for name,v in a.items()}
            if initialization:
                p.fault=9
                for spin in range(min(nh,2)):
                    a['d'][:n*n,spin]=np.eye(n).ravel(order='F')*(p.charge/(min(nh,2)*n))
                if nh>=4:
                    a['d'][:n*n,2]=np.eye(n).ravel(order='F')*.02
                    if nh==8:a['d'][:,6]=a['d'][:,2]
                q=np.sum(a['d'][:,:min(nh,2)]*a['overlap'][:,None])
                a['d']*=p.charge/q*(1.000001 if mixfirst else 1.02)
                a['e'][:]=.123 # first initialization must zero the complete EDM
            expected=oracle.cycle(p,a);label=f'n{n}_kind{kind}_spin{nh}_soc{soc}_mixfirst{mixfirst}'
            path=directory/(label+'.npz');np.savez_compressed(path,problem=raw(p),**a,**{'expected_'+k:v for k,v in expected.items()})
            records.append(dict(fixture=label,input_file=path.name,input_sha256=sha(path),expected_status=0,steps=len(expected['values'])))
    if initialization or kpoints!=2:return records
    # Bad producers may leave partial H/D/E. Check the stage/status, absence of
    # downstream decisions, sticky terminal replay, and early-stage opacity.
    base=np.load(directory/records[-1]['input_file'])
    for fault,stage,code in ((1,1,1),(2,2,3),(3,3,-134),(4,4,2),(5,5,2),(6,6,2),(7,8,1),(8,8,1)):
        a={name:base[name].copy(order='K') for name in base.files};p=Problem.from_buffer_copy(a['problem']);p.fault=fault
        if fault==1:a['overlap'][0]=-1.
        if fault==4:a['kinetic'][0]=np.nan
        if fault==5:a['atom'][0]=np.nan
        if fault==6:a['constants'][0]=np.nan
        a['problem']=raw(p);label=f'fault{fault}_stage{stage}';path=directory/(label+'.npz');np.savez_compressed(path,**a)
        records.append(dict(fixture=label,input_file=path.name,input_sha256=sha(path),expected_status=stage,producer_status=code,steps=0))
    return records

def worker(config):
    cfg=json.loads(config.read_text());a=np.load(cfg['input']);p=Problem.from_buffer_copy(a['problem']);cap=p.n*p.n+2;eh=min(p.nh,4)
    dim=p.n*(2 if p.kind>=2 else 1);jobs=p.nk*(p.nh if p.kind<2 else 1)
    inputs=[np.array(a[name],copy=True,order='K') for name in NAMES]
    h=np.empty((cap,p.nh),order='F');d=np.empty_like(h,order='F');e=np.empty((cap,eh),order='F')
    bank=np.empty((jobs,dim));q=np.empty_like(bank);f=fd.Result();r=Result();steps=(Step*p.maximum)();diagnostics=(Diagnostic*p.maximum)();info=Check()
    ins=(ct.c_void_p*len(inputs))(*(v.ctypes.data for v in inputs))
    outs=(ct.c_void_p*9)(h.ctypes.data,d.ctypes.data,e.ctypes.data,bank.ctypes.data,q.ctypes.data,ct.addressof(f),ct.addressof(r),ct.addressof(steps),ct.addressof(diagnostics))
    call=ct.CDLL(cfg['library']).remag_check_scf_cycle;call.argtypes=[ct.POINTER(Problem),ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_void_p),ct.POINTER(Check)]
    code=call(ct.byref(p),ins,outs,ct.byref(info));output=Path(cfg['output'])
    np.savez_compressed(output.with_suffix('.npz'),h=h,d=d,e=e,bank=bank,q=q,fermi=np.array([f.fermi,f.entropy,f.charge]),result=raw(r),history=raw(steps),diagnostics=raw(diagnostics))
    report={name:getattr(info,name) for name,_ in info._fields_};report.update(native_return=code,status=r.status,producer_status=r.producer_status,
        stage=r.stage,iterations=r.iterations,finished=r.finished,control=flags(r.control),convergence=ints(r.control.convergence))
    if cfg['trace']:
        from siesta_remag.cuda_trace import summarize_cuda
        trace=json.loads((output.parent/'remag-activity.json').read_text())
        report['cuda_trace']=summarize_cuda(trace,dict(total_explicit=dict(host_to_device_bytes=info.staged_bytes,
            device_to_host_bytes=info.exported_bytes),accounting_errors=[]),expected_scf_iterations=1,
            process_succeeded=code==0,allow_unique_graph_scope=True,memory_only=True)
    output.write_text(json.dumps(report,indent=2)+'\n')

def run(upstream,build,output,library,trace=False,trace_io=False,initialization=False,fixtures_source=None,kpoints=2):
    output.parent.mkdir(parents=True,exist_ok=True);directory=output.parent
    if fixtures_source:
        reference=fixtures_source/'reference.json';report=json.loads(reference.read_text())
        assert report['passed'] and report['reference_fixtures_passed'] and report.get('initialization',False)==initialization
        assert sha(fixtures_source/'fixtures.json')==report['fixtures_sha256']
        cases=report['fixtures'];report['checks']=[]
        report['imported_reference_sha256']=sha(reference)
        for case in cases:
            src=fixtures_source/case['input_file'];assert sha(src)==case['input_sha256']
            shutil.copyfile(src,directory/case['input_file'])
        for name,digest in report['wrappers_sha256'].items():
            src=fixtures_source/name;assert sha(src)==digest;shutil.copyfile(src,directory/name)
    else:
        report=dict(scope='Synthetic composed electronic SCF cycle, not production SIESTA integration or final observables',
            integrated_into_siesta=False,strict_residency_qualified=False,initialization=initialization,fixture_script_sha256=sha(Path(__file__)),
            reference_library_sha256=sha(build/'Src/libsiesta.a'),gridxc_library_sha256=sha(build/'libgridxc/libgridxc.a'),
            source_sha256={name:sha(upstream/'Src'/name) for name in ('fermid.F','setup_hamiltonian.F','dhscf.F','reord.f','poison.F',
                'compute_energies.F90','dm_charge.F90','compute_max_diff.F90','scfconvergence_test.F','m_convergence.f90','mixer.F')},checks=[])
        with tempfile.TemporaryDirectory(prefix='remag-cycle-oracle-') as temporary:
            oracle=Oracle(upstream,build,Path(temporary),directory);cases=fixtures(oracle,directory,initialization,kpoints)
            report.update(wrappers_sha256=oracle.wrappers,oracle_io_hashes=oracle.hashes,oracle_calls=oracle.calls)
    (directory/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
    report.update(fixtures=cases,fixtures_sha256=sha(directory/'fixtures.json'))
    if not library:
        report.update(reference_fixtures_passed=True,passed=True,checks=cases);output.write_text(json.dumps(report,indent=2)+'\n');return
    library=library.resolve();report.update(native_library_sha256=sha(library),production_library_sha256=sha(library.with_name('libremag_gpu.so')))
    if trace:
        checks=calibrate_observer(directory,library.with_name('libremag_activity.so'),library.with_name('remag_activity_check'),True)
        report['observer_calibration']=checks
        if not all(c['passed'] for c in checks):
            report.update(passed=False,completed_all_checks=False);output.write_text(json.dumps(report,indent=2)+'\n');raise SystemExit('Cycle observer calibration failed')
    for case in cases:
        folder=directory/case['fixture'];folder.mkdir();result=folder/'result.json';config=folder/'worker.json'
        config.write_text(json.dumps(dict(input=str((directory/case['input_file']).resolve()),output=str(result.resolve()),library=str(library),trace=trace)))
        record=dict(case,numerical_passed=False,residency_passed=None,filesystem_passed=None)
        try:
            env=observer_environment(library.with_name('libremag_activity.so'),Path(sys.executable),True) if trace else os.environ.copy()
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(config.resolve())]
            if trace_io:
                from siesta_remag.io_trace import SYSCALLS
                tracer=shutil.which('strace')
                if not tracer:raise RuntimeError('Filesystem cycle check requires strace')
                command=[tracer,'-f','-ttt','-T','-yy','-o','syscalls.log','-e','trace='+SYSCALLS,*command]
            done=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,timeout=90)
            (folder/'stdout.log').write_text(done.stdout+done.stderr);record['process_returncode']=done.returncode
            assert done.returncode==0,done.stdout+done.stderr
            actual=json.loads(result.read_text());record.update(actual)
            assert actual['native_return']==0 and actual['forbidden_nodes']==0 and actual['graph_nodes']>0 and actual['replay_equal']==1,actual
            assert actual['status']==case['expected_status'] and actual['iterations']==case['steps'] and actual['finished']==1,actual
            got=np.load(result.with_suffix('.npz'));want=np.load(directory/case['input_file']);p=Problem.from_buffer_copy(want['problem'])
            history=(Step*p.maximum).from_buffer_copy(got['history']);diagnostics=(Diagnostic*p.maximum).from_buffer_copy(got['diagnostics']);errors={}
            if not case['expected_status']:
                for name in ('h','d','e','bank','q','fermi'):
                    np.testing.assert_allclose(got[name],want['expected_'+name],atol=2.e-9,rtol=2.e-10)
                    errors[name]=float(np.max(abs(got[name]-want['expected_'+name])))
                vals=np.array([list(v.scalars.values) for v in history[:case['steps']]])
                np.testing.assert_allclose(vals,want['expected_values'],atol=2.e-9,rtol=2.e-10);errors['scalars']=float(np.max(abs(vals-want['expected_values'])))
                diagnostic_values=[]
                for h,diag in zip(history[:case['steps']],diagnostics[:case['steps']]):
                    assert bytes(diag.step)==bytes(h)
                    assert diag.fermi.status==diag.normalization.status==diag.hamiltonian.status==diag.grid.status==0
                    assert diag.hamiltonian.stage==2
                    diagnostic_values.append([diag.fermi.fermi,diag.fermi.entropy,diag.fermi.charge,
                        diag.normalization.charge,diag.normalization.scale,diag.hamiltonian.kinetic,
                        diag.hamiltonian.nonlocal_energy,diag.hamiltonian.soc_energy,*diag.grid.energies,*diag.grid.hartree_stress,*diag.grid.xc_stress])
                np.testing.assert_allclose(diagnostic_values,want['expected_diagnostics'],atol=2.e-9,rtol=2.e-10)
                errors['diagnostics']=float(np.max(abs(diagnostic_values-want['expected_diagnostics'])))
                for diag in diagnostics[case['steps']:]:assert bytes(diag)==bytes(Diagnostic())
                for index,(h,dec) in enumerate(zip(history,want['expected_decisions'])):
                    iteration=index+1;matched,converged,ch,cf,_=map(int,dec);mix=1-converged;more=mix and iteration<p.maximum
                    np.testing.assert_array_equal(ints(h.control.convergence),[0,iteration,matched,converged,mix,int(more),int(mix and iteration==p.maximum),ch+2*cf])
                    np.testing.assert_array_equal(flags(h.control),[0,iteration,mix,int(mix and iteration==1 and not int(want['expected_effective_mix_first'])),int(more),int(not more)])
            else:
                assert actual['producer_status']==case['producer_status'],actual
                np.testing.assert_array_equal(got['history'],0);np.testing.assert_array_equal(got['diagnostics'],0)
                if p.fault in (1,2,7,8):
                    for name in ('h','d','e'):np.testing.assert_array_equal(got[name],want[name])
            record.update(numerical_passed=True,max_errors=errors,output_file=str(result.with_suffix('.npz').relative_to(directory)),output_sha256=sha(result.with_suffix('.npz')))
            if trace:
                activity=actual['cuda_trace'];phase=activity.get('phases',{}).get('scf',{})
                record['residency_passed']=activity['coverage_complete'] and activity['phase_attribution_verified'] and phase.get('host_to_device_bytes',0)==0 and phase.get('device_to_host_bytes',0)==0
                assert record['residency_passed'],activity['errors']
            if trace_io:
                from siesta_remag.io_trace import summarize_io
                phases=json.loads((folder/'remag-residency.json').read_text())
                regions=[v for v in phases['intervals'] if v['phase']=='scf']
                assert len(regions)==1 and regions[0]['iteration']==1
                with (folder/'syscalls.log').open() as lines:activity=summarize_io(lines,phases,process_succeeded=done.returncode==0)
                activity['scope']='Selected filesystem syscalls and mappings during the complete composed cycle, first iteration through terminal replay'
                record['io_trace']=activity;region=activity.get('phases',{}).get('scf',{})
                record['filesystem_passed']=activity['coverage_complete'] and region.get('filesystem_calls',0)==0
                record['io_raw_sha256']=sha(folder/'syscalls.log');record['phase_raw_sha256']=sha(folder/'remag-residency.json')
                assert record['filesystem_passed'],activity
            record['passed']=True
        except Exception as error:record.update(passed=False,error=f'{type(error).__name__}: {error}')
        report['checks'].append(record);report.update(gpu_validation=True,passed=all(c['passed'] for c in report['checks']),completed_all_checks=len(report['checks'])==len(cases))
        output.write_text(json.dumps(report,indent=2)+'\n');print(case['fixture'],'passed=',record['passed'],record.get('error',''),flush=True)
        if sum(not c['numerical_passed'] for c in report['checks'])>=3:raise SystemExit('Stopped after three cycle numerical/setup failures')
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'));parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--output',type=Path);parser.add_argument('--library',type=Path);parser.add_argument('--trace',action='store_true');parser.add_argument('--trace-io',action='store_true');parser.add_argument('--initialization',action='store_true');parser.add_argument('--fixtures-from',type=Path)
    parser.add_argument('--k-points',type=int,default=2,help='k points of the sampled-k fixtures (at most 8); other values skip Gamma and fault cases');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.output:run(args.upstream.resolve(),args.build.resolve(),args.output.resolve(),args.library,args.trace,args.trace_io,args.initialization,args.fixtures_from,args.k_points)
    else:parser.error('--output or --worker is required')
