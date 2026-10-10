#!/usr/bin/env python3
"""Reproduce retained isolated GPU core/neutral arithmetic and scenario coverage.

This checks stored numerical arrays and recorded native statuses. Production
radial derivatives, point mappings and final force ownership remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    folder=root/'atomic-grid-force-check';path=folder/'check.json';report=json.loads(path.read_text())
    oracle=folder/report['oracle_file'];reference=json.loads(oracle.read_text())
    data=np.load(folder/reference['arrays_file']);outputs=np.load(folder/report['output_file'])
    fixtures={r['fixture']:r for r in reference['checks']};errors=[];checks=[];expected=set()
    for label,row in fixtures.items():
        values=[data[label+'_'+name] for name in ('gradient','displacement','potential','auxiliary','deformation','initial_force','initial_stress')]
        raw=(np.array([row['kind'],row['entries'],row['points'],row['atoms'],row['nsd'],row['forces'],row['stress']],np.int32).tobytes()+
             np.array([row['dvol'],row['volume']]).tobytes()+data[label+'_points'].tobytes()+data[label+'_atoms'].tobytes()+
             b''.join(v.tobytes(order='F') for v in values))
        if hashlib.sha256(raw).hexdigest()!=row['input_sha256']:errors.append('Oracle input hash mismatch: '+label)
        for captured in (False,True):
            for cycles in (1,3):expected.add((label,captured,cycles,None))
        if row['entries']==257 and row['forces'] and row['stress'] and (row['kind'],row['nsd']) in ((1,2),(2,1)):
            fields=('potential','auxiliary') if row['kind']==1 else ('deformation',)
            for fault in ('volume','dvol','kind','diagonal_spins','forces','stress','point','atom','gradient','displacement',*fields):
                for captured in (False,True):expected.add((label,captured,3,fault))
    seen=set()
    for row in report['checks']:
        label,captured,cycles,fault=(row[k] for k in ('fixture','captured','cycles','fault'))
        key=(label,captured,cycles,fault);faults=[]
        if key in seen:faults.append('Duplicate scenario')
        seen.add(key)
        if label not in fixtures:
            errors.append('Unknown fixture: '+label);continue
        source=fixtures[label];status=0
        if fault:status=1 if fault in ('volume','dvol','kind','diagonal_spins','forces','stress','point','atom') else 2
        if row['native_return']!=0 or row['status']!=status or row['expected_status']!=status or row['stage']!=4 or row['forbidden_nodes']!=0 or row['graph_nodes']!=int(captured):
            faults.append('Native execution/status/graph record failed')
        upload=56*source['entries']+32+8*source['points']*(source['nsd']+2)+8*(3*source['atoms']+9)+4
        if row['staged']!=upload or row['allocated']!=upload+(24 if source['entries']==0 else 0) or row['exported']!=8*(3*source['atoms']+9)+4:
            faults.append('Shape-derived allocation/transfer sizes disagree')
        name=f'{label}_captured{int(captured)}_cycles{cycles}'+('_'+fault if fault else '')
        error=None
        if name not in outputs:faults.append('Retained output missing')
        elif status==0:
            initial=np.r_[data[label+'_initial_force'].ravel(order='F'),data[label+'_initial_stress'].ravel(order='F')]
            actual=outputs[name];wanted=initial+cycles*(data[label+'_result']-initial)
            if actual.shape!=wanted.shape or not np.isfinite(actual).all():faults.append('Invalid output shape/nonfinite value')
            else:
                error=float(np.max(np.abs(actual-wanted)))
                if error>=1e-12 or error!=row.get('max_absolute_error'):faults.append('GPU arithmetic error failed or cannot be reproduced')
        if row.get('passed') is not True:faults.append('Recorded scenario failed')
        checks.append(dict(scenario=name,passed=not faults,errors=faults,max_absolute_error=error))
    if seen!=expected or len(seen)!=154 or len(fixtures)!=27 or len(outputs.files)!=154:
        errors.append('Required scenario coverage differs')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in (path,oracle,folder/reference['arrays_file'],folder/reference['wrapper_file'],folder/report['output_file'])},
        verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'atomic-grid-component-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'atomic-grid-component-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Atomic-grid component:',len(result['checks']),'scenarios; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
