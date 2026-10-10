#!/usr/bin/env python3
"""CPU feasibility check for a device-controlled blocked eigensolver.

Large cuSOLVER capture failed in the retained probe; 64-by-64 capture passed.
This experiments with parallel disjoint 64-by-64 block rotations. SciPy supplies
the small solves here. No GPU implementation, capture or performance is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cholesky,eigh,solve_triangular


def blocked_eigh(h,s,block=32,tolerance=2.e-13,maximum=20):
    n=len(h);l=cholesky(s,lower=True)
    a=solve_triangular(l,h,lower=True)
    a=solve_triangular(l,a.conj().T,lower=True).conj().T
    a=(a+a.conj().T)*.5
    # Padding must not dilute the physical stopping criterion.
    norm=float(np.linalg.norm(a,'fro')) or 1.
    blocks=2*((n+2*block-1)//(2*block));size=blocks*block
    # Padding stays decoupled and strictly above the physical spectrum.
    bound=np.linalg.norm(a,np.inf)+1.
    padded=np.eye(size,dtype=a.dtype)*bound;padded[:n,:n]=a;a=padded
    vectors=np.eye(size,dtype=a.dtype)
    residuals=[];rotations=0
    for sweep in range(1,maximum+1):
        ring=list(range(blocks))
        for _ in range(blocks-1):
            # These disjoint principal subproblems and transformations can be
            # batched. Sequential application here has the same mathematical
            # block-diagonal left/right transformation.
            for i in range(blocks//2):
                # Keep the sorted local eigenvalues in ascending global block
                # order. Reversing a pair can otherwise undo the spectral
                # partition from preceding rounds and stall this schedule.
                p,q=sorted((ring[i],ring[-1-i]))
                indices=np.concatenate((np.arange(p*block,(p+1)*block),np.arange(q*block,(q+1)*block)))
                _,rotation=eigh(a[np.ix_(indices,indices)],driver='evd')
                a[:,indices]=a[:,indices]@rotation
                a[indices,:]=rotation.conj().T@a[indices,:]
                vectors[:,indices]=vectors[:,indices]@rotation
                rotations+=1
            ring=[ring[0],ring[-1],*ring[1:-1]]
        off=a-np.diag(np.diag(a));relative=np.linalg.norm(off,'fro')/norm
        residuals.append(float(relative))
        if relative<tolerance: break
    else: raise AssertionError(('Block Jacobi did not converge',n,residuals))
    order=np.argsort(a.diagonal().real)[:n]
    w=a.diagonal().real[order]
    z=solve_triangular(l.conj().T,vectors[:n,order],lower=False)
    return w,z,dict(sweeps=sweep,padded_size=size,block_size=block,small_solve_size=2*block,
                    small_solves=rotations,relative_offdiagonal_history=residuals)


def run(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(scope='CPU blocked-Jacobi feasibility only; SciPy small solves; no GPU/capture/performance qualification',
                gpu_validation=None,checks=[],fixture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    rng=np.random.default_rng(64128);retained={}
    for n,complex_values,condition in ((64,False,4.),(128,True,100.),(257,False,1000.),(257,True,1000.)):
        def random_matrix():
            a=rng.normal(size=(n,n))
            return a+1j*rng.normal(size=(n,n)) if complex_values else a
        q,_=np.linalg.qr(random_matrix())
        s=(q*np.geomspace(1.,condition,n))@q.conj().T;s=(s+s.conj().T)*.5
        for step in range(2):
            raw=random_matrix();h=(raw+raw.conj().T)*.5/np.sqrt(n)
            h+=np.diag(np.linspace(-1.,1.,n)*(1.+step*.2))
            label=f'n{n}_complex{int(complex_values)}_step{step}'
            retained[label+'_h']=h;retained[label+'_s']=s
            started=time.monotonic()
            try: w,z,details=blocked_eigh(h,s)
            except AssertionError as error:
                artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
                report.update(passed=False,failed_fixture=label,error=str(error),arrays_file=artifact.name,
                    arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                output.write_text(json.dumps(report,indent=2)+'\n')
                output.with_name('implementation.py').write_bytes(Path(__file__).read_bytes())
                raise
            seconds=time.monotonic()-started;reference=eigh(h,s,eigvals_only=True,driver='gvd')
            residual=np.linalg.norm(h@z-(s@z)*w,'fro')/(np.linalg.norm(h,'fro')*np.linalg.norm(z,'fro'))
            ortho=np.linalg.norm(z.conj().T@s@z-np.eye(n),np.inf)
            error=float(np.max(np.abs(w-reference)))
            assert residual<1.e-10 and ortho<1.e-9 and error<1.e-10,(n,complex_values,residual,ortho,error)
            for name,a in (('h',h),('s',s),('w',w),('z',z),('reference_w',reference)):
                retained[label+'_'+name]=a
            report['checks'].append(dict(fixture=label,n=n,complex=complex_values,overlap_condition=condition,
                generalized_residual=float(residual),overlap_orthogonality=float(ortho),max_eigenvalue_error=error,
                cpu_prototype_seconds=seconds,**details,passed=True))
            print(label,'sweeps',details['sweeps'],'residual',residual,'eigenvalue error',error,flush=True)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
    report.update(passed=True,arrays_file=artifact.name,arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')
    output.with_name('implementation.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
