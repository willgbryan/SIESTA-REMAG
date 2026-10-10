#!/usr/bin/env python3
"""Independently audit retained two-center comparison inputs and GPU counters."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    folder=root/'two-center-comparison-check';path=folder/'check.json';report=json.loads(path.read_text());archive=folder/report['output_file']
    arrays=np.load(archive);checks=[];errors=[];seen=set()
    faults=(None,'value_small','value_big','gradient_small','gradient_big','nan_actual','inf_reference','overflow','mixed','sticky','signed_zero')
    expected={(n,f,c) for n in (1,63,64,65,257,131073) for f in (faults if n in (65,131073) else (None,)) for c in (False,True)}
    for row in report['checks']:
        n,f,c=(row[k] for k in ('n','fault','captured'));key=(n,f,c);issues=[];name=f'n{n}_captured{int(c)}_{f}'
        if key in seen:issues.append('Duplicate scenario')
        seen.add(key)
        a,b=(arrays[name+'_'+k] for k in ('actual','reference'))
        if a.shape!=(n,4) or b.shape!=(n,4):issues.append('Retained shape differs')
        with np.errstate(invalid='ignore',over='ignore'):delta=np.abs(a-b)
        finite=np.isfinite(a)&np.isfinite(b)&np.isfinite(delta);nf=int((~finite).sum());absolute=[];scaled=[];bad=0
        for columns in ((0,),(1,2,3)):
            d=delta[:,columns][finite[:,columns]];ref=b[:,columns][finite[:,columns]];ratio=d/(1e-10+1e-10*np.abs(ref))
            bad+=int((ratio>1.).sum());absolute.append(float(d.max(initial=0.)));scaled.append(float(ratio.max(initial=0.)))
        initial=7 if f=='sticky' else 0
        wanted=dict(compared=4*n,mismatches=bad,nonfinite=nf,absolute=absolute,scaled=scaled,status=max(initial,3 if bad else 0,2 if nf else 0))
        if row['actual']!=wanted or row['expected']!=wanted or row['initial_status']!=initial:issues.append('Comparison does not reproduce from arrays')
        if row['native_return']!=0 or row['stage']!=4 or row['graph_nodes']!=int(c) or row['forbidden_nodes']!=0 or row['poisoned']!=0:issues.append('Graph execution differs')
        if row['staged']!=64*n+60 or row['allocated']!=64*n+60 or row['exported']!=60:issues.append('Shape-derived traffic differs')
        if row['scenario']!=name or row['passed'] is not True:issues.append('Scenario failed')
        checks.append(dict(scenario=name,passed=not issues,errors=issues))
    if seen!=expected or len(checks)!=52 or len(arrays.files)!=104:errors.append('Required scenario coverage differs')
    if sha(archive)!=report['output_sha256']:errors.append('Retained array hash differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,evidence_sha256={str(p.relative_to(root)):sha(p) for p in (path,archive)},verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'two-center-comparison-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'two-center-comparison-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Two-center comparison audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
