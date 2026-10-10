#!/usr/bin/env python3
"""Audit the pristine two-center oracle, cutoff behavior, reciprocity and gradients.

Finite differences and reciprocity are consistency checks, not replacements for
pristine get_matel as the numerical reference for the upcoming GPU component.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(path):
    r=json.loads(path.read_text());folder=path.parent;a=np.load(folder/r['arrays_file']);checks=[];errors=[]
    for k in ('arrays','wrapper'):
        if sha(folder/r[k+'_file'])!=r[k+'_sha256']:errors.append(k+' hash differs')
    expected={family+'_'+kind for family in ('s','spd','high','subset') for kind in ('S','T')};seen=set()
    for row in r['checks']:
        label=row['fixture'];seen.add(label);d={k.removeprefix(label+'_'):a[k] for k in a.files if k.startswith(label+'_')};issues=[]
        nr,nf,ny,rows,cols,base1,base2,nreg,nq=map(int,d['config']);pairs=d['pairs'];xyz=d['xyz'];result=d['expected'];bounds=d['bounds']
        if nr!=1024 or nq!=len(pairs) or d['radial'].shape!=(nf,2,1025) or d['indffy'].shape!=(rows*cols+1,):issues.append('Table dimensions differ')
        if d['ind'].shape!=(nreg+1,) or d['indffy'][0]!=1 or d['indffy'][-1]!=ny+1:issues.append('Table bounds differ')
        if (np.any(np.diff(d['ind'])<0) or np.any(np.diff(d['indffy'])<0) or np.any(d['indffr']<1) or np.any(d['indffr']>nf) or
            np.any(d['ilm']<1) or np.any(d['ilm']>121)):issues.append('Invalid table map')
        for key in ('radial','coefficients','expected','xyz','cutoffs'):
            if not np.isfinite(d[key]).all():issues.append('Nonfinite '+key)
        raw=b''.join(d[k].tobytes() for k in ('config','delta','cutoffs','ind','indffy','ilm','indffr','radial','coefficients','expected'))
        if hashlib.sha256(raw).hexdigest()!=row['output_sha256']:issues.append('Raw oracle output hash differs')
        cut=d['cutoffs'][pairs[:,0]-1]+d['cutoffs'][pairs[:,1]-1]
        outside=np.linalg.norm(xyz,axis=1)>cut
        if not np.all(result[outside]==0):issues.append('Outside cutoff is not exact zero')
        fd_errors=[];sym_errors=[];centers={}
        for i,pair in enumerate(pairs):
            if np.array_equal(xyz[i],[.3,.4,.5]):centers[tuple(pair)]=i
        for pair,i in centers.items():
            opposite=centers.get(pair[::-1])
            if opposite is not None:sym_errors.extend(np.abs(result[i]-result[opposite+1]*[1,-1,-1,-1]))
            start=i-6
            for axis in range(3):
                lo=start+31+axis*2;hi=lo+1
                displacement=np.zeros(3);displacement[axis]=1e-5
                if not np.array_equal(pairs[lo],pair) or not np.array_equal(xyz[lo],xyz[i]-displacement) or not np.array_equal(xyz[hi],xyz[i]+displacement):issues.append('Finite difference coverage differs')
                fd=(result[hi,0]-result[lo,0])/(2e-5);fd_errors.append(abs(fd-result[i,axis+1]))
        fd_max=max(fd_errors,default=np.inf);sym_max=max(sym_errors,default=np.inf)
        if fd_max>1e-8:issues.append('Finite difference gradient failed')
        if sym_max>1e-10:issues.append('Reciprocity failed')
        if nq!=(bounds[1]-bounds[0]+1)*(bounds[3]-bounds[2]+1)*37:issues.append('Query coverage differs')
        if label.startswith('subset') and (base1<=0 or base2<=0):issues.append('Nonzero base coverage missing')
        checks.append(dict(fixture=label,passed=not issues,errors=issues,queries=nq,finite_difference_gradients=len(fd_errors),reciprocity_elements=len(sym_errors),
            max_finite_difference_error=fd_max,max_reciprocity_error=sym_max,max_l=int(np.sqrt(d['ilm'].max()-1))))
    if seen!=expected or len(checks)!=8:errors.append('Fixture coverage differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,reference_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);a=p.parse_args();r=verify(a.reference)
    (a.reference.parent/'oracle-verifier.py').write_bytes(Path(__file__).read_bytes());(a.reference.parent/'oracle-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Two-center oracle audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r);raise SystemExit(1)
