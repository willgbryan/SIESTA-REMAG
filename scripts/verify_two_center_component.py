#!/usr/bin/env python3
"""Independently reconstruct and verify retained two-center GPU queries and outputs.

Audits complete ordinary/captured outputs, sticky/fault behavior, input hashes,
shape-derived transfer totals and strided-tail coverage. No material integration
or independent whole-process CUDA tracing is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    folder=root/'two-center-check';path=folder/'check.json';report=json.loads(path.read_text());refpath=folder/report['oracle_file'];reference=json.loads(refpath.read_text())
    source=np.load(folder/reference['arrays_file']);inputs=np.load(folder/report['inputs_file']);outputs=np.load(folder/report['output_file'])
    index=('pair_low','pair_high','harmonic_low','harmonic_high','term_low','term_high','term_order','angular','radial_index','cutoff','base')
    values=('xyz','radial','coefficient','range','cutoff_overflow','radius_overflow','result_overflow');host=('delta','pi','bounds')
    fields=('ind','indffy','ilm','indffr','cutoffs','radial','coefficients','pairs','xyz');errors=[];checks=[];seen=set();verified=set()
    labels={family+'_'+kind for family in ('s','spd','high','subset') for kind in ('S','T')}
    expected={(s,'ordinary',c,n) for s in labels for c in (False,True) for n in (1,3)}
    expected|={('spd_T',v,c,3) for v in (*index,*values,'sticky','empty') for c in (False,True)}
    expected|={('high_T','stride',c,n) for c in (False,True) for n in (1,3)}
    expected|={('spd_T',v,False,1) for v in host}
    if report['atol']!=1e-10 or report['rtol']!=1e-10:errors.append('Tolerance differs')
    for key in ('inputs','output','oracle'):
        if sha(folder/report[key+'_file'])!=report[key+'_sha256']:errors.append(key+' hash differs')
    if sha(folder/reference['arrays_file'])!=report['oracle_arrays_sha256']:errors.append('Pristine table archive hash differs')
    for row in report['checks']:
        label,v,c,n=row['fixture'],row['variant'],row['captured'],row['cycles'];key=(label,v,c,n);faults=[]
        if key in seen:faults.append('Repeated scenario')
        seen.add(key);name=f'{label}_{v}_captured{int(c)}_cycles{n}';input_key=f'{label}_{v}'
        if row['scenario']!=name or row['input_key']!=input_key:faults.append('Scenario identity differs')
        d={k:source[label+'_'+k].copy(order='C') for k in (*fields,'expected')};cfg=source[label+'_config'];bounds=source[label+'_bounds']
        nr,nf,ny,rows,cols,b1,b2,functions,nq=map(int,cfg)
        config=np.array([functions,nf,ny,rows,cols,b1,b2,*bounds],np.int32);parameters=np.array([source[label+'_delta'][0],np.pi])
        wanted=7 if v=='sticky' else 1 if v in index else 2 if v in values else -1 if v in host else 0
        if v=='stride':
            for k in ('pairs','xyz','expected'):d[k]=np.resize(d[k],(131073,d[k].shape[1]))
        if v=='empty':
            config[1:3]=0;d['radial']=np.empty((0,2,1025));d['ilm']=np.empty(0,np.int32);d['indffr']=np.empty(0,np.int32);d['coefficients']=np.empty(0);d['indffy'][:]=1;d['expected'][:]=0.
        if v in (*index,*values):
            d['pairs'][:]=1;d['xyz'][:]=[.3,.4,.5];d['expected'][:]=0.
            if v=='pair_low':d['pairs'][:,0]=0
            if v=='pair_high':d['pairs'][:,1]=functions+1
            if v=='harmonic_low':d['ind'][0]=0
            if v=='harmonic_high':d['ind'][1]=2147483647
            if v=='term_low':d['indffy'][:]=0
            if v=='term_high':d['indffy'][:]=ny+2
            if v=='term_order':d['indffy'][:]=1;d['indffy'][0]=2
            if v=='angular':d['ilm'][:]=121+1
            if v=='radial_index':d['indffr'][:]=0
            if v=='cutoff':d['cutoffs'][:]=-1
            if v=='base':config[5]=2147483647
            if v=='xyz':d['xyz'][:]=np.nan
            if v=='radial':d['radial'][:]=np.nan
            if v=='coefficient':d['coefficients'][:]=np.inf
            if v=='range':parameters[0]=1e-6
            if v=='cutoff_overflow':d['cutoffs'][:]=1e308
            if v=='radius_overflow':d['xyz'][:]=1e308
            if v=='result_overflow':d['radial'][:]=1e308;d['coefficients'][:]=1e308
        if v=='delta':parameters[0]=0.
        if v=='pi':parameters[1]=np.nan
        if v=='bounds':config[7]=0
        reconstructed=dict(d,config=config,parameters=parameters)
        for k,a in reconstructed.items():
            actual=inputs[input_key+'_'+k]
            if actual.dtype!=a.dtype or actual.shape!=a.shape or not np.array_equal(actual,a,equal_nan=True):faults.append('Executed input differs: '+k)
            verified.add(input_key+'_'+k)
        out=outputs[name];count=len(d['pairs']);bytes_out=32*count;staged=sum(d[k].nbytes for k in fields)+4;allocated=staged+bytes_out+sum(d[k].dtype.itemsize for k in fields if d[k].size==0)
        if (row['queries']!=count or row['staged']!=staged or row['expected_staged']!=staged or row['allocated']!=allocated or row['expected_allocated']!=allocated or row['forbidden_nodes']!=0 or row['expected_status']!=wanted):faults.append('Shape/status accounting differs')
        if out.shape!=(count,4):faults.append('Output coverage differs')
        if v in host:
            if row['native_return']!=-1 or row['stage']!=3 or row['exported']!=0 or row['graph_nodes']!=0 or row['poisoned']!=bytes_out or not np.isnan(out).all():faults.append('Host rejection differs')
        else:
            if row['native_return']!=0 or row['stage']!=4 or row['status']!=wanted or row['graph_nodes']!=int(c) or row['exported']!=bytes_out+4 or row['poisoned']!=n*bytes_out:faults.append('Execution/graph/export differs')
            delta=np.abs(out-d['expected']);scaled=delta/(1e-10+1e-10*np.abs(d['expected']))
            if not np.isfinite(out).all() or np.any(scaled>1.):faults.append('Complete output comparison failed')
            if row['max_absolute_error']!=float(delta.max()) or row['max_scaled_error']!=float(scaled.max()):faults.append('Recorded errors differ')
            if v in (*index,*values,'empty') and not np.all(out==0.):faults.append('Failed/empty query did not produce zero')
        if row['passed'] is not True:faults.append('Recorded scenario failed')
        checks.append(dict(scenario=name,passed=not faults,errors=faults,queries=count,compared_elements=0 if v in host else int(out.size)))
    if seen!=expected or len(checks)!=79:errors.append('Scenario coverage differs')
    if verified!=set(inputs.files) or set(outputs.files)!={r['scenario'] for r in report['checks']}:errors.append('Array coverage differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,executed_input_arrays=len(verified),report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'two-center-component-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'two-center-component-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Two-center component audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r);raise SystemExit(1)
