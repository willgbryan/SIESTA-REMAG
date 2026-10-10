#!/usr/bin/env python3
"""Independently audit every retained pair-contraction input, output and failure."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_pair_force_oracle import verify as verify_oracle

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)

def evaluate(d,c,kind,volume,initial):
    n,na,nr,nc,cap,components,spinor,cycles=map(int,c);out=d['initial'].copy();status=initial
    if kind not in (1,4) or not np.isfinite(volume) or volume<=0:return out,max(status,1)
    weights=np.zeros((nr,nc));valid_row=np.ones(nr,bool)
    with np.errstate(invalid='ignore',over='ignore'):
        for r in range(nr):
            start=int(d['ptr'][r]);length=int(d['num'][r])
            if start<0 or start>cap or length<0 or length>cap-start:valid_row[r]=False;continue
            columns=d['list'][start:start+length]
            if np.any((columns<1)|(columns>nc)):valid_row[r]=False;continue
            for offset,col in enumerate(columns):
                for spin in range(spinor):weights[r,col-1]+=d['density'][spin,start+offset]
        qr=d['qr'];qc=d['qc'];atoms=d['atoms'];valid=(qr>=1)&(qr<=nr)&(qc>=1)&(qc<=nc)&np.all((atoms>=1)&(atoms<=na),axis=1)
        valid&=valid_row[np.clip(qr.astype('int64')-1,0,nr-1)]
        if (~valid).any():status=max(status,1)
        w=weights[np.clip(qr.astype('int64')-1,0,nr-1),np.clip(qc.astype('int64')-1,0,nc-1)]
        df=w[:,None]*d['integrals'][:,1:]*(1 if kind==1 else -1);ds=df[:,:,None]*d['xyz'][:,None,:]/volume
        de=w*d['integrals'][:,0] if kind==1 else np.zeros(n)
        finite=np.isfinite(w)&np.isfinite(df).all(axis=1)&np.isfinite(ds).all(axis=(1,2))&np.isfinite(de)
        if (valid&~finite).any():status=max(status,2)
        selected=valid&finite;force=out[1:1+3*na].reshape(na,3);stress=out[1+3*na:].reshape(3,3)
        for _ in range(cycles):
            out[0]+=de[selected].sum()
            np.add.at(force,atoms[selected,0]-1,df[selected]);np.add.at(force,atoms[selected,1]-1,-df[selected])
            stress+=ds[selected].sum(axis=0)
        if not np.isfinite(out).all():status=max(status,2)
    return out,status


def declared(data,label,variant):
    fields=('qr','qc','atoms','num','ptr','list','integrals','xyz','density','initial');d={k:data[label+'_'+k].copy() for k in fields};c=data[label+'_config'].copy()
    kind,volume=data[label+'_parameters'];kind=int(kind);initial=7 if variant=='sticky' else 0
    if variant=='volume':volume=0.
    elif variant=='volume_nan':volume=float('nan')
    elif variant=='kind':kind=0
    elif variant=='row':d['qr'].fill(-2147483648)
    elif variant=='column':d['qc'].fill(int(c[3])+1)
    elif variant=='atom_a':d['atoms'][:,0]=0
    elif variant=='atom_b':d['atoms'][:,1]=-2147483648
    elif variant=='offset':d['ptr'].fill(-1)
    elif variant=='count':d['num'].fill(int(c[4])+1)
    elif variant=='list':d['list'].fill(int(c[3])+1)
    elif variant=='gradient':d['integrals'][:,1:]=float('nan')
    elif variant=='xyz':d['xyz'].fill(float('inf'))
    elif variant=='density':d['density'].fill(float('nan'))
    elif variant=='value_nan':d['integrals'][:,0]=float('nan')
    elif variant=='unused_spin':d['density'][int(c[6]):]=float('nan')
    elif variant=='arithmetic_overflow':
        d['qr'].fill(1);d['qc'].fill(1);d['density'].fill(np.finfo(np.float64).max);d['integrals'][:,1:]=np.finfo(np.float64).max
    elif variant in ('force_overflow','energy_overflow'):
        d['qr'].fill(1);d['qc'].fill(1);d['atoms'][:]=[1,2];d['density'].fill(0.);d['density'][0,1:3]=.75
        d['xyz'].fill(0.);d['integrals'].fill(0.)
        d['integrals'][:,slice(1,4) if variant=='force_overflow' else 0]=np.finfo(np.float64).max/2
    elif variant=='tail_row':d['qr'][-1]=0
    elif variant=='tail_gradient':d['integrals'][-1,3]=float('nan')
    elif variant=='spinor':c[6]=0
    elif variant=='capacity':c[4]=0
    elif variant=='components':c[5]=1
    return d,c,kind,float(volume),initial


def verify(root):
    folder=root/'pair-force-check';path=folder/'check.json';report=json.loads(path.read_text());ref=folder/report['oracle_file'];oracle=json.loads(ref.read_text());source=np.load(folder/oracle['arrays_file'])
    inputs=np.load(folder/report['inputs_file']);outputs=np.load(folder/report['output_file']);checks=[];errors=[];verified=set();seen=set();required=set()
    faults=('volume','volume_nan','kind','row','column','atom_a','atom_b','offset','count','list','gradient','xyz','density','value_nan','unused_spin','arithmetic_overflow','force_overflow','energy_overflow','sticky','tail_row','tail_gradient')
    host=('spinor','capacity','components')
    for fixture in oracle['checks']:
        variants=['normal']
        if fixture['components']==8 and fixture['queries']==257 and fixture['cycles']==1:variants+=list(faults)+list(host)
        if fixture['components']==8 and fixture['queries']==131073 and fixture['cycles']==1:variants+=['tail_row','tail_gradient']
        for v in variants:
            for captured in ((False,) if v in host else (False,True)):required.add((fixture['fixture'],v,captured))
    for row in report['checks']:
        label,v,captured=row['fixture'],row['variant'],row['captured'];key=(label,v,captured);issues=[];prefix=label+'_'+v;name=prefix+f'_captured{int(captured)}'
        if key in seen:issues.append('Duplicate scenario')
        seen.add(key);d,c,kind,volume,initial=declared(source,label,v)
        for k,a in {**d,'config':c,'parameters':np.array([kind,volume,initial])}.items():
            field=prefix+'_'+k;verified.add(field)
            if not equal(a,inputs[field]):issues.append('Executed input differs: '+field)
        expected=inputs[prefix+'_expected'];verified.add(prefix+'_expected');actual=outputs[name]
        n,na,nr,nc,cap,components,spinor,cycles=map(int,c);staged=72*n+8*nr+4*cap+8*cap*components+24*na+100
        if v in host:
            stage=3 if v=='spinor' else 0
            if row['native_return']!=-1 or row['stage']!=stage or row['exported']!=0 or row['status']!=-9 or not np.isnan(actual).all():issues.append('Host rejection differs')
            if stage==0 and (row['allocated']!=0 or row['staged']!=0):issues.append('Rejected host shape allocated memory')
            if stage==3 and (row['allocated']!=staged or row['staged']!=staged):issues.append('Host API rejection setup differs')
        else:
            predicted,status=evaluate(d,c,kind,volume,initial)
            finite=np.isfinite(predicted).all()
            for values,label2 in ((actual,'GPU'),(expected,'saved expectation')):
                good=np.isfinite(values).all() and np.all(np.abs(values-predicted)<=1e-10+1e-10*np.abs(predicted)) if finite else np.array_equal(values,predicted)
                if not good:issues.append('Independent contraction differs: '+label2)
            if v in ('normal','sticky','unused_spin') or (v=='value_nan' and kind==4):pristine=source[label+'_expected']
            elif v in ('tail_row','tail_gradient'):pristine=source[label+'_without_last']
            else:pristine=None
            if pristine is not None and (not equal(expected,pristine) or not np.all(np.abs(actual-pristine)<=1e-10+1e-10*np.abs(pristine))):issues.append('Literal Fortran comparison differs')
            maximum=float(np.max(np.abs(actual-expected))) if np.isfinite(expected).all() else None
            if row['max_absolute_error']!=maximum:issues.append('Recorded error differs')
            if row['native_return']!=0 or row['stage']!=4 or row['status']!=status or row['expected_status']!=status:issues.append('Device status/execution differs')
            if row['graph_nodes']!=int(captured) or row['forbidden_nodes']!=0:issues.append('Captured graph differs')
            if row['staged']!=staged or row['allocated']!=staged+(28 if n==0 else 0) or row['exported']!=8*(3*na+10)+4:issues.append('Shape-derived traffic differs')
        if row['initial_status']!=initial or row['scenario']!=name or row['inputs_key']!=prefix or row['passed'] is not True:issues.append('Scenario identity/final verdict differs')
        checks.append(dict(scenario=name,passed=not issues,errors=issues))
    if seen!=required or len(checks)!=202:errors.append('Scenario coverage differs')
    if verified!=set(inputs.files) or set(outputs.files)!={r['scenario'] for r in report['checks']}:errors.append('Retained array coverage differs')
    for name,field in ((report['inputs_file'],'inputs_sha256'),(report['output_file'],'output_sha256'),(report['oracle_file'],'oracle_sha256')):
        if sha(folder/name)!=report[field]:errors.append('Content hash differs: '+name)
    if sha(folder/oracle['arrays_file'])!=report['oracle_arrays_sha256']:errors.append('Pristine input/output hash differs')
    audit=verify_oracle(ref)
    if not audit['passed']:errors.append('Independent pristine oracle check failed')
    (root/'pair-force-oracle-verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,executed_input_arrays=len(verified),report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'pair-force-component-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'pair-force-oracle-verifier.py').write_bytes(Path(__file__).with_name('verify_pair_force_oracle.py').read_bytes())
    (a.run/'pair-force-component-verification.json').write_text(json.dumps(r,indent=2)+'\n');print('Pair-force component audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r['errors'],[c for c in r['checks'] if not c['passed']]);raise SystemExit(1)
