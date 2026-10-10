#!/usr/bin/env python3
"""Reproduce the retained final-grid field comparisons against the pinned oracle.

Component validation only; production handoff and complete final observables
need separate material and ownership checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    path=root/'grid-potential-check.json';report=json.loads(path.read_text());artifact=root/report['arrays_file']
    if sha(artifact)!=report['arrays_sha256']:raise ValueError('Grid oracle/output archive hash mismatch')
    checks=[]
    with np.load(artifact) as arrays:
        for row in report['checks']:
            if 'auxiliary_potential' not in row.get('max_absolute_errors',{}):continue
            label=row['fixture'];prefix=f"{label}_graph{int(row['graph'])}_"
            rho=arrays[label+'_density'];n=arrays[label+'_atom'].size;raw=arrays[label+'_raw_reference'];offset=48+rho.size
            references=dict(potential=raw[48:offset].reshape(rho.shape,order='F'),auxiliary=raw[offset+n:offset+2*n],
                deformation=raw[-n:],energies=raw[21:30],hartree_stress=raw[30:39],xc_stress=raw[39:48])
            errors={};passed=row['passed'] and row['iterations']==2 and row['forbidden_nodes']==0
            for name,reference in references.items():
                actual=arrays[prefix+name]
                if actual.shape!=reference.shape:raise ValueError('Grid field shape mismatch')
                error=float(np.max(np.abs(actual-reference)));errors[name]=error
                recorded_name={'auxiliary':'auxiliary_potential','deformation':'deformation_density'}.get(name,name)
                passed=passed and np.isfinite(actual).all() and error==row['max_absolute_errors'][recorded_name] and np.allclose(actual,reference,atol=1e-9,rtol=1e-10)
            checks.append(dict(fixture=label,graph=row['graph'],passed=bool(passed),max_absolute_errors=errors))
    return dict(scope=__doc__,checks=checks,passed=bool(checks) and report['passed'] and all(c['passed'] for c in checks),
        report_sha256=sha(path),arrays_sha256=sha(artifact),verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'final-grid-fields-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'final-grid-fields-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Final-grid field comparisons:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
