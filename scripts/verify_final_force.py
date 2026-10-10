#!/usr/bin/env python3
"""Reproduce shared final-force fixture arithmetic and ownership coverage.

Checks retained arrays and native status records. Material equivalence and
independent transfer observation are separate qualifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    folder=root/'final-force-check';path=folder/'check.json';report=json.loads(path.read_text())
    output=folder/report['output_file'];data=np.load(output);errors=[];checks=[];seen=set()
    expected={(n,c,0,f,s) for n in (1,2,4,8) for c in (0,1) for f,s in ((1,1),(1,0),(0,1))}
    expected.update((8,c,m,1,1) for c in (0,1) for m in range(1,10))
    names=set()
    for row in report['checks']:
        nh,core,mode,force,stress=(row[k] for k in ('spins','partial_core','mode','forces','stress'))
        key=(nh,core,mode,force,stress);faults=[]
        if key in seen or key not in expected:faults.append('Unexpected/duplicate scenario')
        seen.add(key);label=f'spin{nh}_core{core}_mode{mode}_force{force}_stress{stress}'
        if row['fixture']!=label:faults.append('Fixture label differs')
        # Evaluate each contraction separately, then accumulate the three
        # vector increments and their strain derivatives in Fortran order.
        fields=[1.2,1.8,.2,.3];density=[.4,.3,.2,-.1]
        if nh==8:density[2:]=[.16,.02]
        core_weight=.1*((1.2 if nh==1 else 1.5)-.6)
        neutral_weight=.08
        df_weight=.14*(sum(density[i]*fields[i]*(2 if i>=2 else 1) for i in range(min(nh,4)))-.18)
        increment=np.zeros(3)
        if core:increment+=core_weight*np.array([.2,-.3,.4])
        if mode!=9:increment+=neutral_weight*np.array([-.2,.5,-.1])+df_weight*np.array([.3,.2,-.4])
        wanted=np.r_[[.1,-.2,.3],np.arange(1,10)*.01]
        if force:wanted[:3]+=increment
        if stress:
            for i in range(3):
                for j,dx in enumerate((.4,.2,-.3)):wanted[3+3*i+j]+=dx*increment[i]/2
        names.update((label+'_actual',label+'_expected'))
        actual=data[label+'_actual'];stored=data[label+'_expected']
        error=float(np.max(np.abs(actual-wanted)))
        if (actual.shape!=(12,) or stored.shape!=(12,) or not np.isfinite(actual).all() or
                not np.isfinite(stored).all() or error>=1e-12 or np.max(np.abs(stored-wanted))>=1e-15):
            faults.append('Retained accumulation failed analytic check')
        original_error=float(np.max(np.abs(actual-stored)))
        if row['max_absolute_error']!=original_error:faults.append('Recorded error differs from retained arrays')
        if (row['native_return']!=0 or row['status']!=(2 if mode==9 else 0) or row['graphs']!=core+2 or
                row['consumer_mask']!=6+core or row['rejections']!=int(1<=mode<=8) or
                row['accumulator_staged_bytes']!=100+88*(core+1) or row['passed'] is not True):
            faults.append('Ownership/status/staging record failed')
        checks.append(dict(fixture=label,passed=not faults,errors=faults,max_absolute_error=error))
    if seen!=expected or len(checks)!=42 or set(data.files)!=names:errors.append('Incomplete required scenario coverage')
    if sha(output)!=report['output_sha256'] or report['passed'] is not True:errors.append('Component report/hash failed')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in (path,output)},verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'final-force-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'final-force-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Shared final forces:',len(result['checks']),'scenarios; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
