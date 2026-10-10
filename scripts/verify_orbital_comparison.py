#!/usr/bin/env python3
"""Recompute full-array GPU comparator expectations from retained probe inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(root):
    folder=root/'orbital-comparison-check';path=folder/'check.json';report=json.loads(path.read_text())
    arrays=np.load(folder/report['output_file']);errors=[];checks=[];expected=set();seen=set()
    faults=(None,'phi_small','phi_big','gradient_small','gradient_big','displacement','nan_actual','inf_reference','sticky','signed_zero')
    for n in (1,127,128,257,4097,65537):
        for fault in faults if n in (257,65537) else (None,):
            for captured in (False,True):expected.add((n,fault,captured))
    for row in report['checks']:
        n,fault,captured=(row[k] for k in ('n','fault','captured'));key=(n,fault,captured);issues=[]
        if key in seen:issues.append('Duplicate scenario')
        seen.add(key);name=f'n{n}_captured{int(captured)}_'+str(fault)
        if row['scenario']!=name:issues.append('Scenario name differs')
        a=arrays[name+'_actual'];b=arrays[name+'_reference'];nf=bad=0;absolute=[];scaled=[]
        if a.shape!=(7*n,) or b.shape!=a.shape:issues.append('Array shape differs')
        for k,(lo,hi) in enumerate(((0,n),(n,4*n),(4*n,7*n))):
            aa=a[lo:hi];bb=b[lo:hi]
            with np.errstate(invalid='ignore',over='ignore'):delta=np.abs(aa-bb)
            finite=np.isfinite(aa)&np.isfinite(bb)&np.isfinite(delta);nf+=int((~finite).sum())
            delta=delta[finite];scale=delta if k==2 else delta/(1e-10+1e-10*np.abs(bb[finite]))
            absolute.append(float(delta.max(initial=0.)));scaled.append(float(scale.max(initial=0.)))
            bad+=int((scale!=0. if k==2 else scale>1.).sum())
        initial=7 if fault=='sticky' else 0
        wanted=dict(compared=7*n,mismatches=bad,nonfinite=nf,absolute=absolute,scaled=scaled,status=max(initial,3 if bad else 0,2 if nf else 0))
        if row['expected']!=wanted or row['actual']!=wanted or row['initial_status']!=initial:issues.append('Comparator result does not reproduce')
        if row['native_return']!=0 or row['stage']!=4 or row['graph_nodes']!=int(captured) or row['forbidden_nodes']!=0:issues.append('Execution/graph failed')
        if row['staged']!=112*n+76 or row['allocated']!=112*n+76 or row['exported']!=76:issues.append('Shape-derived traffic differs')
        if row['passed'] is not True:issues.append('Recorded scenario failed')
        checks.append(dict(scenario=name,passed=not issues,errors=issues))
    if seen!=expected or len(checks)!=48 or len(arrays.files)!=96:errors.append('Required comparison scenario coverage differs')
    if sha(folder/report['output_file'])!=report['output_sha256']:errors.append('Input array archive hash differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in (path,folder/report['output_file'])},verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();result=verify(a.run)
    (a.run/'orbital-comparison-verifier.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'orbital-comparison-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Orbital comparison audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
