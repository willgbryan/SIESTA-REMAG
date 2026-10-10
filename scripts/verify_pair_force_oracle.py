#!/usr/bin/env python3
"""Independently check pinned sparse pair-force fixtures and their exact binary inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def predict(d,config,kind,volume,drop_last=False):
    n,na,nr,nc,cap,components,spinor,cycles=map(int,config);density=d['density'];weights=np.zeros((nr,nc))
    # Keep each repeated sparse entry, then each diagonal spin, in source order.
    for row in range(nr):
        for entry in range(d['ptr'][row],d['ptr'][row]+d['num'][row]):
            col=d['list'][entry]-1
            for spin in range(spinor):weights[row,col]+=density[spin,entry]
    w=weights[d['qr']-1,d['qc']-1];v=d['integrals'].copy()
    if drop_last and n:v[-1]=0.
    df=w[:,None]*v[:,1:]*(1 if kind==1 else -1)
    result=d['initial'].copy();force=result[1:1+3*na].reshape(na,3);stress=result[1+3*na:].reshape(3,3)
    for _ in range(cycles):
        if kind==1:result[0]+=np.sum(w*v[:,0])
        # Separate additive gathers avoid the original Fortran loop structure.
        np.add.at(force,d['atoms'][:,0]-1,df);np.add.at(force,d['atoms'][:,1]-1,-df)
        stress+=df.T@d['xyz']/volume
    return result

def verify(path,upstream=Path('upstream/siesta')):
    report=json.loads(path.read_text());folder=path.parent;archive=folder/report['arrays_file'];data=np.load(archive);checks=[];seen=set();errors=[]
    fields=('qr','qc','atoms','num','ptr','list','integrals','xyz','density','initial')
    for name,digest in report['source_sha256'].items():
        if sha(upstream/'Src'/name)!=digest:errors.append('Pinned source differs: '+name)
    kinetic=(upstream/'Src/kinefsm.F').read_text();overlap=(upstream/'Src/overfsm.f90').read_text();blocks=[]
    for text,start,end in ((kinetic,'               do j = 1,numd(iio)','            endif'),
                           (overlap,'          do j = 1,numd(iio)','          do jn'),
                           (kinetic,'                     Ekin = Ekin + Di(jo) * Tij','                  endif'),
                           (overlap,'                do ix = 1,3','              end if')):
        begin=text.index(start);blocks.append(text[begin:text.index(end,begin)])
    wrapper=(folder/report['wrapper_file']).read_text()
    if hashlib.sha256(''.join(blocks).encode()).hexdigest()!=report['literal_blocks_sha256'] or any(block not in wrapper for block in blocks):
        errors.append('Literal Fortran blocks do not match pinned source')
    for row in report['checks']:
        label=row['fixture'];config=data[label+'_config'];kind,volume=data[label+'_parameters'];kind=int(kind);n,na,nr,nc,cap,components,spinor,cycles=map(int,config)
        seen.add((kind,components,n,cycles));d={k:data[label+'_'+k] for k in fields};issues=[]
        raw=np.r_[config,np.int32(kind)].astype(np.int32).tobytes()+np.array([volume],dtype='d').tobytes()+b''.join(d[k].tobytes(order='C') for k in fields)
        if hashlib.sha256(raw).hexdigest()!=row['input_sha256']:issues.append('Input bytes do not reproduce')
        maximum=0.
        for suffix,drop in (('expected',False),('without_last',True)):
            wanted=predict(d,config,kind,volume,drop);actual=data[label+'_'+suffix];delta=np.abs(actual-wanted);maximum=max(maximum,float(delta.max()))
            if actual.shape!=(3*na+10,) or not np.isfinite(actual).all() or np.any(delta>1e-10+1e-10*np.abs(actual)):issues.append('Independent sparse contraction differs')
        checks.append(dict(fixture=label,passed=not issues,errors=issues,max_absolute_error=maximum))
    expected={(k,c,n,r) for k in (1,4) for c in (1,2,4,8) for n in (0,1,257,*((131073,) if c==8 else ())) for r in (1,3)}
    if seen!=expected or len(checks)!=52:errors.append('Fixture coverage differs')
    if sha(archive)!=report['arrays_sha256'] or sha(folder/report['wrapper_file'])!=report['wrapper_sha256']:errors.append('Oracle content hash differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'));a=p.parse_args();r=verify(a.reference,a.upstream)
    a.reference.with_name('pair-force-oracle-verification.json').write_text(json.dumps(r,indent=2)+'\n');print('Pair-force oracle audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
