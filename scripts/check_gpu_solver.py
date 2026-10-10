#!/usr/bin/env python3
"""Small direct GPU sanity check: generalized residual, overlap orthogonality and CPU eigenvalues."""
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

def check_grid_epochs(lib):
    """Exercise the ownership contract with same-address basis mutations."""
    phase=lib.remag_phase_enter
    phase.argtypes=[ctypes.c_int]*3
    assert phase(1,0,0)==-140
    assert phase(4,0,1)==-140
    assert phase(0,0,0)==0
    lib.remag_geometry_changed()
    nsc=np.ones(3,dtype=np.int32)
    ends=np.array([0,1],dtype=np.int32)
    orbs=num=cols=np.ones(1,dtype=np.int32)
    ptr=np.zeros(1,dtype=np.int32)
    phi=np.array([.5],dtype=np.float32)
    dm=np.array([2.]);rho=np.zeros(1)
    def contract(status=0):
        arrays=[nsc,ends,orbs,num,ptr,cols,phi,dm,rho]
        assert lib.remag_grid(0,1,1,1,1,1,1,1,*(a.ctypes.data for a in arrays),0.)==status
        if status==0:
            np.testing.assert_allclose(rho,dm*phi.astype(float)**2,rtol=1.e-13)
    contract()
    assert phase(1,0,1)==0
    contract()  # advertised epoch avoids a host scan
    phi*=.75
    lib.remag_geometry_changed()
    contract()  # same pointer, changed geometry; must stage new basis
    os.environ['REMAG_GRID_VERIFY_EPOCH']='1'
    try:
        contract()  # full verification accepts unchanged state
        phi*=.5
        contract(-116)  # mutation without invalidation is an error in verify mode
        lib.remag_geometry_changed()
        contract()
    finally:
        os.environ.pop('REMAG_GRID_VERIFY_EPOCH',None)
    os.environ['REMAG_GRID_EPOCH']='0'
    try:
        phi*=.5
        contract()  # conservative content-hash fallback remains available
    finally:
        os.environ.pop('REMAG_GRID_EPOCH',None)
    assert phase(2,0,0)==0
    return dict(kind='geometry_epoch_invalidation',passed=True)

def check_captured_grid(lib,nh,phi,dm,v,seed,rho,increment):
    """Replay fixed periodic topology with changing D/V/H and device volume."""
    class Check(ctypes.Structure):
        _fields_=[('staged_bytes',ctypes.c_size_t),('exported_bytes',ctypes.c_size_t),
                  ('iterations',ctypes.c_int),('graph_nodes',ctypes.c_int),('forbidden_nodes',ctypes.c_int)]
    library=Path(lib._name).with_name('libremag_grid_graph_check.so')
    call=ctypes.CDLL(str(library)).remag_check_grid_graph
    call.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*7
    dscale=np.array([1.,0.,-.75]);vscale=np.array([1.,-.4,1.3]);hscale=np.array([1.,.25,-.1])
    volume=np.array([.2,.37,.1]);ng=min(nh,4);cycles=3
    densities=np.ascontiguousarray(dscale[:,None]*dm.ravel(order='F'))
    potentials=np.ascontiguousarray(vscale[:,None]*v.ravel(order='F'))
    h=np.ascontiguousarray(hscale[:,None]*seed.ravel(order='F'))
    out_rho=np.empty((cycles,ng,2));out_h=np.empty((cycles,nh,2));info=Check()
    args=[densities,potentials,h,volume,out_rho,out_h]
    assert call(cycles,ng,nh,*(a.ctypes.data for a in args),ctypes.byref(info))==0
    assert info.iterations==cycles and info.graph_nodes==6 and info.forbidden_nodes==0
    assert info.staged_bytes==sum(a.nbytes for a in args[:4])+4
    assert info.exported_bytes==out_rho.nbytes+out_h.nbytes+4
    want_rho=dscale[:,None,None]*rho.ravel(order='F').reshape(ng,2)
    want_h=hscale[:,None,None]*seed.T+(vscale*volume/.2)[:,None,None]*increment.T
    np.testing.assert_allclose(out_rho,want_rho,atol=1.e-12,rtol=1.e-12)
    np.testing.assert_allclose(out_h,want_h,atol=1.e-12,rtol=1.e-12)
    # A graph cannot acquire geometry that changed at the same address or
    # belongs to an invalidated epoch. Rejections must leave outputs intact.
    old_verify=os.environ.get('REMAG_GRID_VERIFY_EPOCH');saved_phi=phi[0,0]
    saved_rho=out_rho.copy();saved_h=out_h.copy()
    try:
        os.environ['REMAG_GRID_VERIFY_EPOCH']='1';phi[0,0]+=np.float32(.125)
        assert call(cycles,ng,nh,*(a.ctypes.data for a in args),ctypes.byref(info))==-1
    finally:
        phi[0,0]=saved_phi
        if old_verify is None:os.environ.pop('REMAG_GRID_VERIFY_EPOCH',None)
        else:os.environ['REMAG_GRID_VERIFY_EPOCH']=old_verify
    lib.remag_geometry_changed()
    assert call(cycles,ng,nh,*(a.ctypes.data for a in args),ctypes.byref(info))==-1
    np.testing.assert_array_equal(out_rho,saved_rho);np.testing.assert_array_equal(out_h,saved_h)
    return dict(kind='captured_grid_contractions',spin_components=nh,cycles=cycles,graph_nodes=info.graph_nodes,
                forbidden_nodes=info.forbidden_nodes,staged_bytes=info.staged_bytes,exported_bytes=info.exported_bytes,
                rejected_host_mutation=True,rejected_stale_geometry=True,
                checker_binary_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                max_matrix_error=max(float(np.max(np.abs(out_rho-want_rho))),float(np.max(np.abs(out_h-want_h)))),passed=True)


def check_density_matrices(lib, rng):
    arm,begin,accum,finish=lib.remag_arm_vectors,lib.remag_dm_begin,lib.remag_dm_accumulate,lib.remag_dm_finish
    arm.argtypes=[ctypes.c_int]*2
    begin.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*5
    accum.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*2+[ctypes.c_double,ctypes.c_void_p]
    finish.argtypes=[ctypes.c_void_p]*2
    checks=[]
    n=17
    num=np.full(n,3,dtype=np.int32)
    ptr=np.arange(n,dtype=np.int32)*3
    nnz=n*3+2  # unused sparse capacity must remain zero
    cols=np.ones(nnz,dtype=np.int32)
    for i in range(n): cols[3*i:3*i+3]=[i+1,(i+2)%n+1,n+i+1]
    mapping=np.tile(np.arange(1,n+1,dtype=np.int32),2)
    xyz=np.asfortranarray(rng.normal(size=(3,nnz)))
    modes=((0,2),(1,2),(2,4),(2,8),(3,4),(3,8))
    norm_arm=lib.remag_dm_normalization_arm
    norm_arm.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]+[ctypes.c_double]*2
    class GraphCheck(ctypes.Structure):
        _fields_=[('staged_bytes',ctypes.c_size_t),('exported_bytes',ctypes.c_size_t),
                  ('iterations',ctypes.c_int),('graph_nodes',ctypes.c_int),('forbidden_nodes',ctypes.c_int)]
    class NormalizationResult(ctypes.Structure):
        _fields_=[('charge',ctypes.c_double),('scale',ctypes.c_double),('status',ctypes.c_int)]
    graph_library=Path(lib._name).with_name('libremag_density_graph_check.so')
    graph_check=ctypes.CDLL(str(graph_library)).remag_check_density_graph
    graph_check.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*5+[ctypes.c_double,ctypes.c_void_p]+[ctypes.c_double]*2+[ctypes.c_void_p]*4
    graph_sha=hashlib.sha256(graph_library.read_bytes()).hexdigest()
    for kind,nh,device_normalize in [(k,s,b) for k,s in modes for b in (False,True)]:
        spinor=kind>=2
        eh=4 if spinor else nh
        dim=n*(2 if spinor else 1)
        nev=dim-2
        d=np.zeros((nnz,nh),order='F');e=np.zeros((nnz,eh),order='F')
        expected_d=np.zeros_like(d);expected_e=np.zeros_like(e)
        jobs=[]
        for kindex in range(2 if kind%2 else 1):
            for spin in range(1 if spinor else nh):
                x=rng.normal(size=(dim,dim));y=rng.normal(size=(dim,dim))
                if kind!=0:
                    x=x+1j*rng.normal(size=x.shape);y=y+1j*rng.normal(size=y.shape)
                h=np.asfortranarray((x+x.conj().T)/2);s=np.asfortranarray(y@y.conj().T+np.eye(dim))
                energies,z=eigh(h,s)
                q=rng.uniform(.1,1.,nev);q[::5]=0.;q[-1]=-.3;q[1]=1.e-12
                k=rng.normal(size=3) if kind%2 else np.zeros(3)
                jobs.append((spin+1 if kind==0 else 0,spin,h,s,energies,z,q,k))
        def capture(job,export=False):
            slot,spin,h,s,energies,z,q,k=job
            out=np.full_like(h,np.nan,order='F');w=np.zeros(dim)
            fn=lib.remag_solve_real if kind==0 else lib.remag_solve_complex
            assert arm(slot,int(export))==0
            assert fn(dim,nev,dim,h.ctypes.data,s.ctypes.data,out.ctypes.data,w.ctypes.data)==0
            if export:
                np.testing.assert_allclose(h@out[:,:nev],(s@out[:,:nev])*w[:nev],atol=1.e-11)
            else:
                assert np.isnan(out).all(), 'Resident solve unexpectedly copied eigenvectors to the host'
        if kind==0:
            for job in jobs: capture(job,export=job[1]==1)
        args=[num,ptr,cols,mapping,xyz]
        assert begin(kind,n,nnz,nh,eh,2*n,*(a.ctypes.data for a in args))==0
        for job in jobs:
            slot,spin,h,s,energies,z,q,k=job
            if kind!=0: capture(job)
            assert accum(slot,nev,spin if not spinor else 0,q.ctypes.data,energies.ctypes.data,1.e-10,k.ctypes.data)==0
            # A consumed slot must never silently reuse eigenvectors from another solve.
            assert accum(slot,nev,spin if not spinor else 0,q.ctypes.data,energies.ctypes.data,1.e-10,k.ctypes.data)==-129
            weights=np.where(np.abs(q)>1.e-10,q,0.)
            for row in range(n):
                for p in range(ptr[row],ptr[row]+num[row]):
                    col=mapping[cols[p]-1]-1
                    phase=np.exp(-1j*np.dot(k,xyz[:,p]))
                    if not spinor:
                        v=np.real(z[row,:nev]*np.conj(z[col,:nev])*phase)
                        expected_d[p,spin]+=np.dot(weights,v)
                        expected_e[p,spin]+=np.dot(weights*energies[:nev],v)
                    else:
                        a=z[2*row,:nev]*np.conj(z[2*col,:nev])*phase
                        b=z[2*row+1,:nev]*np.conj(z[2*col+1,:nev])*phase
                        c=z[2*row,:nev]*np.conj(z[2*col+1,:nev])*phase
                        f=z[2*row+1,:nev]*np.conj(z[2*col,:nev])*phase
                        if nh==4: c=.5*(c+np.conj(f))
                        v=np.array([a.real,b.real,c.real,-c.imag,a.imag,b.imag,f.real,f.imag])
                        expected_d[p,:]+=v[:nh]@weights
                        expected_e[p,:]+=v[:eh]@(weights*energies[:nev])
        # The borrowed geometry is shared with production; graph outputs use
        # separate buffers. Every input is staged before capture. Replays vary
        # occupations without any intermediate host export, including failed
        # charge normalization followed by recovery on a later replay.
        scales=np.array([1.,1.005,1.25,0.,1.] if device_normalize else [1.,1.03,1.17])
        cycles=len(scales)
        graph_vectors=np.concatenate([job[5][:,:nev].ravel(order='F') for job in jobs])
        graph_spins=np.array([job[1] for job in jobs],dtype=np.int32)
        graph_k=np.concatenate([job[7] for job in jobs])
        graph_q=np.ascontiguousarray(scales[:,None]*np.concatenate([job[6] for job in jobs])[None,:])
        graph_eigenvalues=np.concatenate([job[4][:nev] for job in jobs])
        overlap=np.ascontiguousarray(expected_d[:,:2].sum(axis=1))
        charge=float(np.sum(expected_d[:,:2]*overlap[:,None]));target=charge*1.00037
        graph_d=np.empty((cycles,nh,nnz));graph_e=np.empty((cycles,eh,nnz))
        graph_results=(NormalizationResult*cycles)();graph_info=GraphCheck()
        inputs=[graph_spins,graph_k,graph_vectors,graph_q,graph_eigenvalues]
        assert graph_check(cycles,len(jobs),nev,*(a.ctypes.data for a in inputs),1.e-10,
                           overlap.ctypes.data if device_normalize else None,target,.01,
                           graph_d.ctypes.data,graph_e.ctypes.data,ctypes.byref(graph_results),ctypes.byref(graph_info))==0
        assert graph_info.iterations==cycles and graph_info.forbidden_nodes==0
        assert graph_info.graph_nodes==7+2*len(jobs)+(3 if device_normalize else 1)
        assert graph_info.staged_bytes==graph_vectors.nbytes+graph_q.nbytes+graph_eigenvalues.nbytes+4+(overlap.nbytes if device_normalize else 0)
        assert graph_info.exported_bytes==graph_d.nbytes+graph_e.nbytes+ctypes.sizeof(graph_results)+4
        matrix_error=0.
        for cycle,scale in enumerate(scales):
            result=graph_results[cycle]
            status=-134 if device_normalize and cycle in (2,3) else 0
            assert result.status==status
            factor=1.00037 if device_normalize and status==0 else scale
            want_d=expected_d.T*factor;want_e=expected_e.T*factor
            np.testing.assert_allclose(graph_d[cycle],want_d,rtol=1.e-11,atol=1.e-12)
            np.testing.assert_allclose(graph_e[cycle],want_e,rtol=1.e-11,atol=1.e-12)
            if device_normalize:
                np.testing.assert_allclose(result.charge,charge*scale,rtol=1.e-12,atol=1.e-13)
                np.testing.assert_allclose(result.scale,1. if status else 1.00037/scale,rtol=1.e-12)
            matrix_error=max(matrix_error,float(np.max(np.abs(graph_d[cycle]-want_d))),float(np.max(np.abs(graph_e[cycle]-want_e))))
        checks.append(dict(kind='captured_density_matrix',mode=kind,spin_components=nh,device_normalization=device_normalize,
                           cycles=cycles,normalization_errors_checked=2 if device_normalize else 0,
                           graph_nodes=graph_info.graph_nodes,forbidden_nodes=graph_info.forbidden_nodes,
                           staged_bytes=graph_info.staged_bytes,exported_bytes=graph_info.exported_bytes,
                           checker_binary_sha256=graph_sha,max_matrix_error=matrix_error,passed=True))
        if device_normalize:
            overlap=np.ascontiguousarray(expected_d[:,:2].sum(axis=1))
            charge=float(np.sum(expected_d[:,:2]*overlap[:,None]))
            target=charge*1.00037
            assert norm_arm(nnz,nh,eh,2,overlap.ctypes.data,target,.01)==0
            assert norm_arm(nnz,nh,eh,2,overlap.ctypes.data,target,.01)==-136
            expected_d*=1.00037;expected_e*=1.00037
        assert finish(d.ctypes.data,e.ctypes.data)==0
        assert lib.remag_dm_normalization_complete()==int(device_normalize)
        assert lib.remag_dm_normalization_complete()==0
        np.testing.assert_allclose(d,expected_d,rtol=1.e-10,atol=1.e-11)
        np.testing.assert_allclose(e,expected_e,rtol=1.e-10,atol=1.e-11)
        normalization=lib.remag_dm_normalize
        normalization.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[ctypes.c_void_p]*2
        factor=1.00037
        d*=factor;e*=factor;expected_d*=factor;expected_e*=factor
        assert normalization(nnz,nh,eh,factor,d.ctypes.data,e.ctypes.data)==0
        # Exercise the actual DM -> grid handoff, then emulate a host-side mixer
        # changing the same array. Exact-content reuse must reject the mutation.
        grid=lib.remag_grid
        grid.argtypes=[ctypes.c_int]*8+[ctypes.c_void_p]*9+[ctypes.c_double]
        nsc=np.array([2,1,1],dtype=np.int32);ends=np.array([0,2],dtype=np.int32)
        orbs=np.array([1,n+1],dtype=np.int32)
        phi=np.asfortranarray(np.array([[.3,.7],[.5,-.2]],dtype=np.float32))
        ng=min(nh,4)
        for change in (0.,.25,-.25):
            d[0,0]+=change
            rho=np.zeros((2,1,ng),order='F');expected=np.zeros_like(rho)
            arrays=[nsc,ends,orbs,num,ptr,cols,phi,d,rho]
            assert grid(0,2*n,1,2,n,nnz,ng,nh,*(a.ctypes.data for a in arrays),0.)==0
            for s in range(ng):
                for a in range(2):
                    for b in range(2):
                        p=0 if a==b else 2
                        value=d[p,s] if nh!=8 or s<2 else .5*(d[p,s]+d[p,s+4])
                        expected[:,0,s]+=value*phi[:,a].astype(float)*phi[:,b].astype(float)
            np.testing.assert_allclose(rho,expected,rtol=1.e-11,atol=1.e-12)
        checks.append(dict(kind='resident_density_matrix',mode=kind,spin_components=nh,device_normalization=device_normalize,
                           density_error=float(np.max(np.abs(d-expected_d))),
                           energy_density_error=float(np.max(np.abs(e-expected_e))),passed=True))
    return checks

def check_bad_normalization(lib):
    # A bad charge must be rejected before exporting or modifying host D/E.
    lib.remag_geometry_changed()
    one=np.ones(1);zero=np.zeros(1);idx=np.ones(1,dtype=np.int32);ptr=np.zeros(1,dtype=np.int32)
    xyz=np.zeros(3);w=np.zeros(1);z=np.zeros(1)
    arm=lib.remag_dm_normalization_arm
    assert arm(1,1,1,1,one.ctypes.data,1.,1.e-5)==0
    assert lib.remag_dm_normalization_complete()==0  # unused request cancels cleanly
    assert lib.remag_arm_vectors(1,0)==0
    assert lib.remag_solve_real(1,1,1,one.ctypes.data,one.ctypes.data,z.ctypes.data,w.ctypes.data)==0
    assert lib.remag_dm_begin(0,1,1,1,1,1,idx.ctypes.data,ptr.ctypes.data,idx.ctypes.data,idx.ctypes.data,xyz.ctypes.data)==0
    assert lib.remag_dm_accumulate(1,1,0,zero.ctypes.data,w.ctypes.data,0.,xyz.ctypes.data)==0
    assert arm(1,1,1,1,one.ctypes.data,1.,1.e-5)==0
    d=np.array([7.]);e=np.array([9.])
    assert lib.remag_dm_finish(d.ctypes.data,e.ctypes.data)==-134
    np.testing.assert_array_equal(d,[7.]);np.testing.assert_array_equal(e,[9.])
    assert lib.remag_dm_normalization_complete()==0
    return dict(kind='bad_charge_rejected_before_export',passed=True)

def check_overlap_epochs(lib):
    one=np.ones(1);idx=np.ones(1,dtype=np.int32);ptr=np.zeros(1,dtype=np.int32)
    xyz=np.zeros(3);w=np.zeros(1);z=np.zeros(1);overlap=np.ones(1)
    d=np.zeros(1);e=np.zeros(1)
    def begin():
        assert lib.remag_arm_vectors(1,0)==0
        assert lib.remag_solve_real(1,1,1,one.ctypes.data,one.ctypes.data,z.ctypes.data,w.ctypes.data)==0
        assert lib.remag_dm_begin(0,1,1,1,1,1,idx.ctypes.data,ptr.ctypes.data,idx.ctypes.data,idx.ctypes.data,xyz.ctypes.data)==0
        assert lib.remag_dm_accumulate(1,1,0,one.ctypes.data,w.ctypes.data,0.,xyz.ctypes.data)==0
        assert lib.remag_dm_normalization_arm(1,1,1,1,overlap.ctypes.data,1.,4.)==0
    lib.remag_geometry_changed()
    begin()
    assert lib.remag_dm_finish(d.ctypes.data,e.ctypes.data)==0
    assert lib.remag_dm_normalization_complete()==1
    np.testing.assert_array_equal(d,[1.])
    os.environ['REMAG_GRID_VERIFY_EPOCH']='1'
    try:
        overlap[0]=2.
        begin()
        d[0]=7.;e[0]=9.
        assert lib.remag_dm_finish(d.ctypes.data,e.ctypes.data)==-132
        np.testing.assert_array_equal(d,[7.]);np.testing.assert_array_equal(e,[9.])
        lib.remag_geometry_changed()
        assert lib.remag_dm_finish(d.ctypes.data,e.ctypes.data)==0
        assert lib.remag_dm_normalization_complete()==1
        np.testing.assert_array_equal(d,[.5]);np.testing.assert_array_equal(e,[.5])
    finally:
        os.environ.pop('REMAG_GRID_VERIFY_EPOCH',None)
    return dict(kind='normalization_overlap_invalidation',passed=True)

def check_occupation_bank(lib, resident=False):
    """Exercise the actual eigensolver -> FD -> sparse D/E ownership path."""
    from scipy.optimize import brentq
    from scipy.special import expit
    prepare,compute,end=lib.remag_fermi_prepare,lib.remag_fermi_compute,lib.remag_fermi_end
    export=lib.remag_fermi_export
    export.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    prepare.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*3+[ctypes.c_double]*2
    compute.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*2+[ctypes.c_double]*2+[ctypes.c_void_p]*3
    saved={k:os.environ.get(k) for k in ('REMAG_OCCUPATIONS','REMAG_DM','REMAG_GRID_VERIFY_EPOCH','REMAG_RESIDENT_SPECTRA')}
    os.environ.update(REMAG_OCCUPATIONS='1',REMAG_DM='1',REMAG_GRID_VERIFY_EPOCH='1')
    os.environ['REMAG_RESIDENT_SPECTRA']='1' if resident else '0'
    n,ne,nspin,nk=8,7,2,2
    shape=(2,nspin,nk,n,ne)
    energies=np.zeros((n,nspin,nk),order='F');q=np.zeros_like(energies)
    weights=np.array([.3,.7]);temperature=.07;charge=5.31
    ef,entropy=np.zeros(1),np.zeros(1)
    num=np.ones(n,dtype=np.int32);ptr=np.arange(n,dtype=np.int32);cols=ptr+1
    xyz=np.zeros((3,n),order='F');k=np.zeros(3)
    overlap=np.eye(n,dtype=np.complex128,order='F')
    scratch=np.zeros(n);z=np.zeros_like(overlap)
    values=np.linspace(-.3,.4,n)
    cutoff=.02 if resident else 0.
    def stage():
        return prepare(*shape,energies.ctypes.data,q.ctypes.data,weights.ctypes.data,temperature,charge)
    def solve(spin,ik,vectors=False,destination=None):
        # Vector-only pass has a scratch spectrum; first-pass eo must persist.
        h=np.asfortranarray(np.diag(values+.015*spin+.021*ik+.007*cycle*(np.arange(n)+1)+
                                   (2. if vectors else 0.)).astype(complex))
        target=destination if destination is not None else scratch if vectors else energies[:,spin,ik]
        if vectors: assert lib.remag_arm_vectors(0,0)==0
        return lib.remag_solve_complex(n,ne if vectors else -ne,n,h.ctypes.data,overlap.ctypes.data,
                                      z.ctypes.data,target.ctypes.data)
    def occupation(output=None):
        output=q if output is None else output
        return compute(*shape,energies.ctypes.data,weights.ctypes.data,temperature,charge,
                       output.ctypes.data,ef.ctypes.data,entropy.ctypes.data)
    def accumulate(spin,ik,output=None):
        output=q[:,spin,ik] if output is None else output
        return lib.remag_dm_accumulate(0,ne,spin,output.ctypes.data,energies[:,spin,ik].ctypes.data,cutoff,k.ctypes.data)
    try:
        lib.remag_geometry_changed()
        assert prepare(3,*shape[1:],energies.ctypes.data,q.ctypes.data,weights.ctypes.data,temperature,charge)==-150
        for cycle in range(3 if resident else 2):
            if cycle==(2 if resident else 1):
                weights[:]=[.4,.6]
                assert stage()==-151  # same-address mutation requires invalidation
                lib.remag_geometry_changed()
            host_e_before=energies.copy(order='F');host_q_before=q.copy(order='F')
            assert stage()==0
            assert stage()==-150  # an active bank cannot be replaced
            assert end()==-156
            assert occupation()==-154  # no producer has filled any block
            assert solve(0,0,destination=scratch)==-101
            for ik in range(nk):
                for spin in range(nspin): assert solve(spin,ik)==0
            assert solve(0,0)==-101  # duplicate producer cannot replace owned E
            other=q.copy(order='F')
            assert occupation(other)==-153
            original_weight=weights[0]
            weights[0]+=.01
            assert occupation()==-151
            weights[0]=original_weight
            assert occupation()==0
            assert occupation()==-153
            first=np.empty_like(energies)
            for ik in range(nk):
                for spin in range(nspin): first[:,spin,ik]=values+.015*spin+.021*ik+.007*cycle*(np.arange(n)+1)
            if resident:
                np.testing.assert_array_equal(energies,host_e_before)
                np.testing.assert_array_equal(q,host_q_before)
                assert end()==-156  # no consumers yet
                assert export(q.size,energies.ctypes.data,q.ctypes.data)==-158
            else:
                np.testing.assert_allclose(energies,first,atol=1.e-13)
            reference_ef=brentq(lambda mu:np.sum(expit((first[:ne]-mu)/-temperature)*weights)-charge,-1.,1.,xtol=1.e-15)
            probability=expit((first[:ne]-reference_ef)/-temperature)
            reference_q=np.zeros_like(q);reference_q[:ne]=probability*weights
            reference_entropy=float(np.sum((-probability*np.log(probability)-(1-probability)*np.log1p(-probability))*weights))
            if not resident: np.testing.assert_allclose(q,reference_q,atol=1.e-10,rtol=1.e-10)
            np.testing.assert_allclose([ef[0],entropy[0]],[reference_ef,reference_entropy],atol=1.e-10,rtol=1.e-10)
            assert lib.remag_dm_begin(1,n,n,nspin,nspin,n,num.ctypes.data,ptr.ctypes.data,cols.ctypes.data,
                                     cols.ctypes.data,xyz.ctypes.data)==0
            for ik in range(nk):
                for spin in range(nspin):
                    if ik==0 and spin==0:
                        assert solve(spin,ik,True)==0
                        assert accumulate(spin,ik,q[:,spin,ik].copy())==-130
                        for array in (q,energies):
                            original=array[0,spin,ik]
                            array[0,spin,ik]+=.1
                            assert solve(spin,ik,True)==0
                            assert accumulate(spin,ik)==-130
                            array[0,spin,ik]=original
                    assert solve(spin,ik,True)==0
                    assert accumulate(spin,ik)==0
                    assert solve(spin,ik,True)==0
                    assert accumulate(spin,ik)==-130  # one-use occupation block
            d=np.zeros((n,nspin),order='F');e=np.zeros_like(d)
            assert lib.remag_dm_finish(d.ctypes.data,e.ctypes.data)==0
            used_q=np.where(np.abs(reference_q)>cutoff,reference_q,0.)
            if resident: assert np.any((reference_q[:ne]>0.)&(reference_q[:ne]<=cutoff))
            np.testing.assert_allclose(d,used_q.sum(axis=2),atol=1.e-10,rtol=1.e-10)
            np.testing.assert_allclose(e,(used_q*first).sum(axis=2),atol=1.e-10,rtol=1.e-10)
            assert end()==0
            assert end()==-156
            if resident:
                np.testing.assert_array_equal(energies,host_e_before)
                np.testing.assert_array_equal(q,host_q_before)
                assert export(q.size-1,energies.ctypes.data,q.ctypes.data)==-158
                assert export(q.size,energies.ctypes.data,other.ctypes.data)==-158
                assert prepare(*shape,energies.ctypes.data,other.ctypes.data,weights.ctypes.data,temperature,charge)==-157
                q[0,0,0]+=.1
                assert stage()==-159
                assert export(q.size,energies.ctypes.data,q.ctypes.data)==-159
                q[0,0,0]=host_q_before[0,0,0]
                if cycle:  # cycle 1 replaces cycle 0 without an intermediate export
                    assert export(q.size,energies.ctypes.data,q.ctypes.data)==0
                    np.testing.assert_allclose(energies,first,atol=1.e-13)
                    np.testing.assert_allclose(q,reference_q,atol=1.e-10,rtol=1.e-10)
                    assert export(q.size,energies.ctypes.data,q.ctypes.data)==-158
    finally:
        for key,value in saved.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
    return dict(kind='resident_spectral_export' if resident else 'resident_occupation_bank',passed=True,geometry_cycles=2,
                checks=['missing_duplicate_foreign_producers','wrong_output_pointer','weight_epoch_invalidation',
                        'host_E_Q_mutation','duplicate_consumer','scratch_spectrum_preserves_first_pass',
                        'FD_charge_entropy','sparse_density_energy_density']+
                       (['opaque_host_arrays','full_band_cutoff','deferred_repeated_iteration_export',
                         'export_shape_owner_and_mutation','incomplete_consumers'] if resident else []))

def check_matrix_assembly(lib, rng):
    """Independent sparse assembly, periodic phases and device-owner failures."""
    prepare,select,end=lib.remag_hs_prepare,lib.remag_hs_select,lib.remag_hs_end
    prepare.argtypes=[ctypes.c_int]*7+[ctypes.c_void_p]*8
    select.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*2
    saved={key:os.environ.get(key) for key in ('REMAG_HS','REMAG_GRID_VERIFY_EPOCH')}
    os.environ['REMAG_HS']='1';os.environ['REMAG_GRID_VERIFY_EPOCH']='1'
    checks=[]
    class GraphInfo(ctypes.Structure):
        _fields_=[('staged_bytes',ctypes.c_size_t),('exported_bytes',ctypes.c_size_t)]+[
            (key,ctypes.c_int) for key in ('iterations','graph_nodes','forbidden_nodes')]
    graph_library=Path(lib._name).with_name('libremag_hamiltonian_graph_check.so')
    graph_check=ctypes.CDLL(str(graph_library)).remag_check_hamiltonian_graph
    graph_check.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3+[ctypes.POINTER(GraphInfo)]
    graph_check.restype=ctypes.c_int
    graph_library_sha=hashlib.sha256(graph_library.read_bytes()).hexdigest()
    try:
        n=7
        nnz=3*n*n+3;ldh=nnz+2
        num=np.full(n,3*n,dtype=np.int32);ptr=np.arange(n,dtype=np.int32)*3*n
        cols=np.zeros(nnz,dtype=np.int32);mapping=np.tile(np.arange(1,n+1,dtype=np.int32),3)
        xyz=np.full((3,nnz),np.nan,order='F')
        translation=np.array([.7,-.3,.2])
        # Scramble periodic contributions, including duplicate folded entries.
        entries=[]
        for col in range(n):
            for offset,encoded in enumerate(rng.permutation(3*n)):
                image,row=divmod(int(encoded),n);index=int(ptr[col])+offset
                cols[index]=image*n+row+1
                xyz[:,index]=translation*(0 if image==0 else 1 if image==1 else -1)
                entries.append((index,row,col,image))
        kpoints=np.asfortranarray([[0.,.31,-.27],[0.,-.17,.43],[0.,.11,.19]])
        for kind,nh in ((0,1),(0,2),(1,1),(1,2),(2,4),(2,8),(3,4),(3,8)):
            spinor=kind>=2;dim=n*(2 if spinor else 1);nspin=1 if spinor else nh
            nk=3 if kind%2 else 1;passes=2 if kind%2 else 1
            h=np.full((ldh,nh),np.nan,order='F');s=np.full(nnz,np.nan)
            # Each real-space image obeys H(-R)=H(R)^dagger and S(-R)=S(R)^T.
            base=rng.normal(size=(n,n));base=(base+base.T)*.01+np.eye(n)*3.
            plus=rng.normal(size=(n,n))*.005
            overlaps=(base,plus,plus.T)
            if not spinor:
                hb=rng.normal(size=(n,n,nh));hb=(hb+hb.transpose(1,0,2))*.2
                hp=rng.normal(size=(n,n,nh))*.1
                components=(hb,hp,hp.transpose(1,0,2))
            elif nh==4:
                hb=rng.normal(size=(n,n,4));hb=(hb+hb.transpose(1,0,2))*.2
                hp=rng.normal(size=(n,n,4))*.1
                components=(hb,hp,hp.transpose(1,0,2))
            else:
                hb=rng.normal(size=(dim,dim))+1j*rng.normal(size=(dim,dim));hb=(hb+hb.conj().T)*.2
                hp=(rng.normal(size=(dim,dim))+1j*rng.normal(size=(dim,dim)))*.1
                def encode(a):
                    return np.stack((a[::2,::2].real,a[1::2,1::2].real,a[::2,1::2].real,
                                     -a[::2,1::2].imag,a[::2,::2].imag,a[1::2,1::2].imag,
                                     a[1::2,::2].real,a[1::2,::2].imag),axis=2)
                components=tuple(encode(a) for a in (hb,hp,hp.conj().T))
            for index,row,col,image in entries:
                h[index]=components[image][row,col];s[index]=overlaps[image][row,col]
            dtype=np.float64 if kind==0 else np.complex128
            host_h=np.full((dim+2,dim),np.nan,dtype=dtype,order='F');host_s=host_h.copy(order='F')
            z=np.zeros_like(host_h);w=np.zeros(dim)
            fn=lib.remag_solve_real if kind==0 else lib.remag_solve_complex
            def stage():
                args=(num,ptr,cols,mapping,xyz,kpoints,h,s)
                return prepare(kind,n,nnz,nh,3*n,nk,ldh,*(a.ctypes.data for a in args))
            def solve(harg=host_h,narg=dim,solver=fn):
                return solver(narg,narg,dim+2,harg.ctypes.data,host_s.ctypes.data,z.ctypes.data,w.ctypes.data)
            max_error=max_residual=max_orthogonality=max_graph_error=0.
            graph_info=GraphInfo();graph_h=graph_s=None;graph_factors=np.array([1.,1.03,1.17])
            lib.remag_geometry_changed()
            for cycle in range(3):
                if cycle:
                    h[:3*n*n]*=1.03  # mutable H must be staged every iteration
                if cycle==2:
                    s[:3*n*n]*=1.07;xyz[:,:3*n*n]*=1.11
                    lib.remag_geometry_changed()
                assert stage()==0
                if cycle==0:
                    # Capture the production launch seam with all geometry
                    # staged once and three changing H inputs on the device.
                    staged=np.ascontiguousarray(graph_factors[:,None]*h.ravel(order='F')[None,:])
                    graph_h=np.full((3,nk,nspin,dim,dim),np.nan,dtype=dtype)
                    graph_s=np.full_like(graph_h,np.nan)
                    assert graph_check(3,staged.ctypes.data,graph_h.ctypes.data,graph_s.ctypes.data,
                                       ctypes.byref(graph_info))==0
                    assert graph_info.iterations==3 and graph_info.forbidden_nodes==0
                    assert graph_info.graph_nodes==2+5*nk*nspin
                    graph_h=graph_h.swapaxes(-1,-2);graph_s=graph_s.swapaxes(-1,-2)
                assert stage()==-170
                assert end()==-175
                assert select(nspin,0,host_h.ctypes.data,host_s.ctypes.data)==-174
                assert select(0,nk,host_h.ctypes.data,host_s.ctypes.data)==-174
                assert solve()==-101  # an active owner requires an explicit selection
                # Bad identities/shapes/types consume selection, without building anything.
                for kwargs in ({'harg':host_h.copy(order='F')},{'narg':dim-1},
                               {'solver':lib.remag_solve_complex if kind==0 else lib.remag_solve_real}):
                    assert select(0,0,host_h.ctypes.data,host_s.ctypes.data)==0
                    assert solve(**kwargs)==-101
                for source in (h,s,cols,xyz if kind%2 else num):
                    old=source.flat[0];source.flat[0]=old+1
                    assert select(0,0,host_h.ctypes.data,host_s.ctypes.data)==0
                    assert solve()==-101
                    source.flat[0]=old
                for ik in range(nk):
                    for spin in range(nspin):
                        expected_h=np.zeros((dim,dim),dtype=dtype);expected_s=np.zeros_like(expected_h)
                        for index,row,col,image in entries:
                            phase=np.exp(-1j*np.dot(kpoints[:,ik],xyz[:,index])) if kind%2 else 1.
                            if not spinor:
                                expected_h[row,col]+=h[index,spin]*phase
                                expected_s[row,col]+=s[index]*phase
                            else:
                                v=h[index]
                                block=np.array([[v[0]+(1j*v[4] if nh==8 else 0),v[2]-1j*v[3]],
                                                [v[6]+1j*v[7] if nh==8 else v[2]+1j*v[3],
                                                 v[1]+(1j*v[5] if nh==8 else 0)]])
                                expected_h[2*row:2*row+2,2*col:2*col+2]+=block*phase
                                expected_s[2*row:2*row+2,2*col:2*col+2]+=np.eye(2)*s[index]*phase
                        np.testing.assert_allclose(expected_h,expected_h.conj().T,atol=1.e-13)
                        np.testing.assert_allclose(expected_s,expected_s.conj().T,atol=1.e-13)
                        if cycle==0:
                            for step,factor in enumerate(graph_factors):
                                np.testing.assert_allclose(graph_h[step,ik,spin],expected_h*factor,atol=1.e-12,rtol=1.e-12)
                                np.testing.assert_allclose(graph_s[step,ik,spin],expected_s,atol=1.e-12,rtol=1.e-12)
                                max_graph_error=max(max_graph_error,float(np.max(np.abs(graph_h[step,ik,spin]-expected_h*factor))),
                                                    float(np.max(np.abs(graph_s[step,ik,spin]-expected_s))))
                        for attempt in range(passes):
                            assert select(spin,ik,host_h.ctypes.data,host_s.ctypes.data)==0
                            assert select(spin,ik,host_h.ctypes.data,host_s.ctypes.data)==-174
                            assert end()==-175
                            assert solve()==0
                            assert np.isnan(host_h).all() and np.isnan(host_s).all()
                            vectors=z[:dim]
                            max_error=max(max_error,float(np.max(np.abs(w-eigh(expected_h,expected_s,eigvals_only=True)))))
                            max_residual=max(max_residual,float(np.linalg.norm(expected_h@vectors-(expected_s@vectors)*w)))
                            max_orthogonality=max(max_orthogonality,float(np.linalg.norm(vectors.conj().T@expected_s@vectors-np.eye(dim))))
                        assert select(spin,ik,host_h.ctypes.data,host_s.ctypes.data)==-174
                assert end()==0
                assert end()==-175
                if cycle==0:
                    old=s[0];s[0]=old+.25
                    assert stage()==-171  # same geometry, undeclared immutable mutation
                    s[0]=old
            assert max(max_error,max_residual,max_orthogonality)<1.e-10
            checks.append(dict(kind='device_matrix_assembly',matrix_kind=kind,spin_components=nh,
                               eigenvalue_error=max_error,residual=max_residual,orthogonality=max_orthogonality,
                               geometry_cycles=2,scf_iterations=3,passed=True))
            checks.append(dict(kind='captured_device_matrix_assembly',matrix_kind=kind,spin_components=nh,
                graph_replays=3,graph_nodes=graph_info.graph_nodes,forbidden_nodes=graph_info.forbidden_nodes,
                staged_h_bytes=graph_info.staged_bytes,final_export_bytes=graph_info.exported_bytes,
                max_matrix_error=max_graph_error,check_library_sha256=graph_library_sha,passed=True))
    finally:
        for key,value in saved.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
    return checks

def check_hartree_bridge(lib,rng):
    """Reusable geometry ownership, mutable density and guarded host exports."""
    from check_hartree import Problem,numpy_reference
    enabled,need,prepare,solve=lib.remag_hartree_enabled,lib.remag_hartree_needs_setup,lib.remag_hartree_prepare,lib.remag_hartree_solve
    need.argtypes=[ctypes.c_void_p]*2
    prepare.argtypes=[ctypes.c_void_p]*3+[ctypes.c_double]*2
    solve.argtypes=[ctypes.c_void_p]*6
    saved={key:os.environ.get(key) for key in ('REMAG_HARTREE','REMAG_GRID_VERIFY_EPOCH')}
    os.environ['REMAG_HARTREE']='1';os.environ['REMAG_GRID_VERIFY_EPOCH']='1'
    try:
        assert enabled()==1
        shape=np.array([6,8,10],dtype=np.int32);cell=np.asfortranarray(np.diag([8.,9.,10.]))
        rho=np.asfortranarray(rng.normal(scale=.1,size=tuple(shape)))
        v=np.full_like(rho,7.,order='F');energy=np.array([11.]);stress=np.full((3,3),13.,order='F')
        def dispatch(out=v):
            return solve(shape.ctypes.data,cell.ctypes.data,rho.ctypes.data,energy.ctypes.data,out.ctypes.data,stress.ctypes.data)
        problem=None
        for cycle in range(3):
            if cycle!=1: lib.remag_geometry_changed()
            if cycle==2: cell[0,0]*=1.1
            assert need(shape.ctypes.data,cell.ctypes.data)==(0 if cycle==1 else 1)
            if cycle!=1:
                assert dispatch()==-184
                reciprocal=np.asfortranarray(2*np.pi*np.linalg.inv(cell).T);volume=float(abs(np.linalg.det(cell)));cutoff=5.
                problem=Problem((ctypes.c_int*3)(*shape),(ctypes.c_double*9)(*reciprocal.ravel(order='F')),cutoff,volume)
                assert prepare(shape.ctypes.data,cell.ctypes.data,reciprocal.ctypes.data,cutoff,volume)==0
            rho*=.81
            expected=numpy_reference(problem,rho)
            assert need(shape.ctypes.data,cell.ctypes.data)==0
            assert dispatch()==0
            np.testing.assert_allclose(energy[0],expected[0],atol=1.e-10,rtol=1.e-11)
            np.testing.assert_allclose(stress,expected[1],atol=1.e-12,rtol=1.e-11)
            np.testing.assert_allclose(v,expected[2],atol=1.e-10,rtol=1.e-11)
            old=cell[0,0];cell[0,0]=old+.1
            assert need(shape.ctypes.data,cell.ctypes.data)==-181
            assert dispatch()==-184
            cell[0,0]=old
        # The original poison contract permits the density/potential to alias.
        expected=numpy_reference(problem,rho)
        assert dispatch(rho)==0
        np.testing.assert_allclose(rho,expected[2],atol=1.e-10,rtol=1.e-11)
        # Invalid device data cannot overwrite a host potential, energy or stress.
        rho.flat[0]=np.nan;v.fill(7.);energy.fill(11.);stress.fill(13.)
        assert dispatch()==-185
        np.testing.assert_array_equal(v,7.);np.testing.assert_array_equal(energy,11.);np.testing.assert_array_equal(stress,13.)
        rho.fill(.1)
        assert dispatch()==0  # device error state is reset on a later valid call
        np.testing.assert_allclose(v,0.,atol=1.e-12)
    finally:
        for key,value in saved.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
    return dict(kind='hartree_bridge_ownership',geometry_cycles=2,calls=5,passed=True)

def main(library, output):
    lib = ctypes.CDLL(library)
    rng = np.random.default_rng(712)
    checks = []
    for complex_values in (False, True):
        fn = lib.remag_solve_complex if complex_values else lib.remag_solve_real
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_int]*3 + [ctypes.c_void_p]*4
        for n in (1, 17, 64):
            x = rng.normal(size=(n,n))
            y = rng.normal(size=(n,n))
            if complex_values:
                x = x + 1j*rng.normal(size=(n,n))
                y = y + 1j*rng.normal(size=(n,n))
            h = np.asfortranarray((x+x.conj().T)/2)
            s = np.asfortranarray(y@y.conj().T+np.eye(n))
            z = np.zeros_like(h, order="F")
            w = np.zeros(n)
            info = fn(n,n,n,h.ctypes.data,s.ctypes.data,z.ctypes.data,w.ctypes.data)
            if info:
                raise RuntimeError(f"GPU returned {info}")
            residual = np.linalg.norm(h@z - (s@z)*w)/max(1.,np.linalg.norm(h))
            orthogonality = np.linalg.norm(z.conj().T@s@z - np.eye(n))
            error = np.max(np.abs(w-eigh(h,s,eigvals_only=True)))
            passed = max(residual,orthogonality,error)<1.e-10
            checks.append(dict(n=n,complex=complex_values,residual=float(residual),
                               orthogonality=float(orthogonality),eigenvalue_error=float(error),passed=bool(passed)))
            if not passed:
                raise AssertionError(checks[-1])
    checks.extend(check_density_matrices(lib,rng))
    grid = lib.remag_grid
    grid.restype = ctypes.c_int
    grid.argtypes = [ctypes.c_int]*8 + [ctypes.c_void_p]*9 + [ctypes.c_double]
    # Two translated cells of a one-orbital cell: verify periodic folding, SOC blocks,
    # additive Hamiltonian updates, density and basis-cache invalidation.
    nsc = np.array([2,1,1], dtype=np.int32)
    ends = np.array([0,2], dtype=np.int32)
    orbs = np.array([1,2], dtype=np.int32)
    num = np.array([2], dtype=np.int32)
    ptr = np.array([0], dtype=np.int32)
    cols = np.array([1,2], dtype=np.int32)
    for nh in (1, 2, 4, 8):
        ng = min(nh,4)
        for scale in (1., .75):
            phi = np.asfortranarray(np.array([[.3,.7],[.5,-.2]],dtype=np.float32)*scale)
            dm = np.asfortranarray(rng.normal(size=(2,nh)))
            rho = np.zeros((2,1,ng),order="F")
            inputs = [nsc,ends,orbs,num,ptr,cols,phi,dm,rho]
            status = grid(0,2,1,2,1,2,ng,nh,*(a.ctypes.data for a in inputs),0.)
            assert status == 0, status
            expected = np.zeros_like(rho)
            for s in range(ng):
                for a in range(2):
                    for b in range(2):
                        index = (b-a)%2
                        d=dm[index,s] if nh!=8 or s<2 else .5*(dm[index,s]+dm[index,s+4])
                        expected[:,0,s] += d*phi[:,a].astype(float)*phi[:,b].astype(float)
            np.testing.assert_allclose(rho,expected,atol=1.e-12,rtol=1.e-12)
            v = np.asfortranarray(rng.normal(size=(2,1,ng)))
            vs = np.asfortranarray(rng.normal(size=(2,nh)))
            seed_vs = vs.copy()
            expected_vs = vs.copy()
            for s in range(ng):
                for a in range(2):
                    for b in range(2):
                        value=.2*np.sum(phi[:,a].astype(float)*v[:,0,s]*phi[:,b].astype(float))
                        expected_vs[(b-a)%2,s]+=value
                        if nh==8 and s>=2: expected_vs[(b-a)%2,s+4]+=value
            inputs = [nsc,ends,orbs,num,ptr,cols,phi,v,vs]
            status = grid(1,2,1,2,1,2,ng,nh,*(a.ctypes.data for a in inputs),.2)
            assert status == 0, status
            np.testing.assert_allclose(vs,expected_vs,atol=1.e-12,rtol=1.e-12)
            checks.append(dict(kind="periodic_grid",spin_components=nh,basis_scale=scale,passed=True))
            captured=check_captured_grid(lib,nh,phi,dm,v,seed_vs,expected,expected_vs-seed_vs)
            captured['basis_scale']=scale;checks.append(captured)
    checks.append(check_grid_epochs(lib))
    checks.append(check_overlap_epochs(lib))
    checks.append(check_occupation_bank(lib))
    checks.append(check_occupation_bank(lib,resident=True))
    checks.extend(check_matrix_assembly(lib,rng))
    checks.append(check_hartree_bridge(lib,rng))
    # This terminal-error scenario intentionally leaves a failed density plan.
    checks.append(check_bad_normalization(lib))
    Path(output).write_text(json.dumps(checks, indent=2)+"\n")
    print(f"All {len(checks)} GPU solver, density-matrix and grid sanity checks passed", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:])
