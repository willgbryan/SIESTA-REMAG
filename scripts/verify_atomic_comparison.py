#!/usr/bin/env python3
"""Independently recompute atomic comparison records from every retained input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root):
    folder=root/'atomic-comparison-check';path=folder/'check.json';report=json.loads(path.read_text());archive=folder/report['output_file']
    arrays=np.load(archive);checks=[];errors=[];seen=set()
    faults=(None,'phi_small','phi_big','gradient_small','gradient_big','displacement','nan_actual','inf_reference','sticky','signed_zero',
            'mask_actual','mask_reference','invalid_mask','point','atom','inactive_phi','inactive_gradient','inactive_ignored')
    expected={(n,f,c) for n in (1,127,128,129,4097,131073) for f in (faults if n in (129,131073) else (None,)) for c in (False,True)}
    for row in report['checks']:
        n,f,c=(row[k] for k in ('n','fault','captured'));key=(n,f,c);issues=[];name=f'n{n}_captured{int(c)}_'+str(f)
        if key in seen:issues.append('Duplicate scenario')
        seen.add(key)
        aa,bb=(arrays[name+'_'+k] for k in ('actual','reference'));ai,bi=(arrays[name+'_'+k] for k in ('actual_indices','reference_indices'))
        if aa.shape!=(7*n,) or bb.shape!=(7*n,) or ai.shape!=(3*n,) or bi.shape!=(3*n,):issues.append('Retained shape differs')
        a=np.column_stack((aa[:n],aa[n:4*n].reshape(n,3),aa[4*n:].reshape(n,3)))
        b=np.column_stack((bb[:n],bb[n:4*n].reshape(n,3),bb[4*n:].reshape(n,3)))
        selected=bi[:n]==1;valid=np.ones((n,7),bool);valid[:,4:]=selected[:,None]
        with np.errstate(invalid='ignore',over='ignore'):delta=np.abs(a-b)
        finite=np.isfinite(a)&np.isfinite(b)&np.isfinite(delta);nf=int((valid&~finite).sum());good=valid&finite
        absolute=[];scaled=[];bad=0
        for lo,hi,exact in ((0,1,False),(1,4,False),(4,7,True)):
            d=delta[:,lo:hi][good[:,lo:hi]];ref=b[:,lo:hi][good[:,lo:hi]];ratio=d if exact else d/(1e-10+1e-10*np.abs(ref))
            absolute.append(float(d.max(initial=0.)));scaled.append(float(ratio.max(initial=0.)));bad+=int((d!=0 if exact else ratio>1).sum())
        masks=int((~((ai[:n]==0)|(ai[:n]==1))|~((bi[:n]==0)|(bi[:n]==1))|(ai[:n]!=bi[:n])).sum())
        indices=int((ai[n:2*n][selected]!=bi[n:2*n][selected]).sum()+(ai[2*n:][selected]!=bi[2*n:][selected]).sum())
        initial=7 if f=='sticky' else 0;m=int(selected.sum())
        wanted=dict(points=n,selected=int((ai[:n]==1).sum()),reference_selected=m,compared=4*n+3*m,index_compared=2*m,
                    mismatches=bad,nonfinite=nf,mask_mismatches=masks,index_mismatches=indices,absolute=absolute,scaled=scaled,
                    status=max(initial,3 if bad+masks+indices else 0,2 if nf else 0))
        if row['actual']!=wanted or row['expected']!=wanted or row['initial_status']!=initial:issues.append('Comparison does not reproduce from arrays')
        if row['native_return']!=0 or row['stage']!=4 or row['graph_nodes']!=int(c) or row['forbidden_nodes']!=0 or row['poisoned']!=0:issues.append('Graph execution differs')
        if row['staged']!=136*n+124 or row['allocated']!=136*n+124 or row['exported']!=124:issues.append('Shape-derived traffic differs')
        if row['scenario']!=name or row['passed'] is not True:issues.append('Scenario failed')
        checks.append(dict(scenario=name,passed=not issues,errors=issues))
    if seen!=expected or len(checks)!=80 or len(arrays.files)!=320:errors.append('Required scenario coverage differs')
    if sha(archive)!=report['output_sha256']:errors.append('Retained array hash differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
                evidence_sha256={str(p.relative_to(root)):sha(p) for p in (path,archive)},verifier_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'atomic-comparison-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'atomic-comparison-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Atomic comparison audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
