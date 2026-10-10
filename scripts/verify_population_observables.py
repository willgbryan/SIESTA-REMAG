#!/usr/bin/env python3
"""Reconstruct derived outputs from retained inputs without the GPU checker.

Verifies the two-kernel capture inventory and pristine-source binding. This
is component evidence, not independent whole-process transfer observation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def view(q):
    result=np.zeros((len(q),5));result[:,0]=q[:,0]
    if q.shape[1]>1:result[:,0]+=q[:,1];result[:,1]=result[:,4]=q[:,0]-q[:,1]
    if q.shape[1]==4:result[:,2:4]=2*q[:,2:4];result[:,1]=np.sqrt(np.sum(result[:,2:]**2,axis=1))
    return result


def verify(root):
    folder=root/'population-observables-check';report=read(folder/'check.json');meta=read(root/'metadata.json')
    errors=[];checks=[];seen=set()
    for name in ('source-verification.json','component-evidence-verification.json'):
        if read(root/name).get('passed') is not True:errors.append('Failed prerequisite: '+name)
    if not meta.get('population') or report['checker_sha256']!=meta['population_observables_check_sha256'] or report['passed'] is not True:
        errors.append('Missing population mode/source or failed native checks')
    for name,digest in report['evidence_sha256'].items():
        if sha(folder/name)!=digest:errors.append('Changed evidence: '+name)
    with tarfile.open(root/'implementation.tar.gz') as archive:
        if archive.extractfile('native/population_observables_reference.f90').read()!=(folder/'reference.f90').read_bytes():errors.append('Reference wrapper differs from archived source')
    upstream=Path(__file__).resolve().parents[1]/'upstream/siesta/Src/mulliken.F'
    if sha(upstream)!=report['upstream_mulliken_sha256'] or upstream.read_text().split('      subroutine spnvec(',1)[1]!=(folder/'spnvec-reference.F').read_text().split('      subroutine spnvec(',1)[1]:
        errors.append('Pristine spin routine differs')
    data=np.load(folder/'arrays.npz')
    wanted=set()
    for c,mode,rows in [(c,m,9) for c in (1,2,4) for m in ('ordinary','zero','negative_spin')]+[(4,'ordinary',262145)]:
        variants=['normal']+(['nonfinite_orbital'] if c==4 and mode=='ordinary' else [])
        if c==4 and mode=='ordinary' and rows==9:variants+=['problem_shape','bad_species','nonfinite_valence','net_overflow','sticky_status']
        wanted.update((f'components{c}_{mode}_rows{rows}_{v}',capture) for v in variants for capture in (0,1))
    for row in report['checks']:
        key=row['fixture'];capture=row['capture'];issues=[];identity=(key,capture)
        if identity in seen or identity not in wanted:issues.append('Duplicate/unexpected fixture')
        seen.add(identity)
        rows,atoms,species,components=map(int,data[key+'_config'])
        ids=data[key+'_ids'];z=data[key+'_valence'];orb=data[key+'_orbital'];atom=data[key+'_atomic'];total=data[key+'_total']
        status=row['initial_status'];expected=None
        if not np.array_equal(data[key+'_config'],data[key+'_problem']) or np.any(ids<1) or np.any(ids>species):status=max(status,1)
        elif not status:
            with np.errstate(over='ignore',invalid='ignore'):
                ov=view(orb);av0=view(atom);av=np.column_stack((z[ids-1]-av0[:,0],av0));sv=[];running=np.zeros(components)
                for species_id in range(1,species+1):
                    for ia in np.flatnonzero(ids==species_id):running+=atom[ia]
                    sv.append(view(running[None,:])[0])
                vector=av[:,3:6].sum(axis=0);spin=np.sqrt(np.sum(vector**2)) if components==4 else vector[2]
                summary=np.r_[view(total[None,:])[0],av[:,0].sum(),spin,vector]
                expected=np.r_[ov.ravel(),av.ravel(),np.asarray(sv).ravel(),summary]
            if not np.isfinite(expected).all():status=2
        out=data[key+f'_capture{capture}_output'];size=5*rows+6*atoms+5*species+10
        staged=20+4*atoms+8*species+8*components*(rows+atoms+1)
        inventory=dict(allocated=staged+8*size,staged=staged,exported=8*size+4,poisoned=24*size,graph_nodes=2*capture,forbidden_nodes=0,stage=4)
        if row['inventory']!=inventory or row['native_return'] or row['status']!=status or row['expected_status']!=status or row['passed'] is not True:issues.append('Status/capture/storage evidence differs')
        error=None
        if status==0:
            reference=data[key+'_reference'];error=float(np.max(np.abs(out-expected)))
            if out.size!=size or not np.isfinite(out).all() or not np.allclose(out,expected,atol=1e-12,rtol=1e-12) or not np.allclose(out,reference,atol=1e-12,rtol=1e-12):issues.append('Derived populations differ from reference')
            if abs(float(np.max(np.abs(out-reference)))-row['max_absolute_error'])>1e-15:issues.append('Reported reference discrepancy differs')
        checks.append(dict(fixture=key,capture=capture,passed=not issues,errors=issues,status=status,max_absolute_error=error))
    if seen!=wanted:errors.append('Missing component/spin/fault/stride coverage')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
                verifier_sha256=sha(Path(__file__)),report_sha256=sha(folder/'check.json'),evidence_sha256=report['evidence_sha256'],independent_cuda_trace=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();result=verify(args.run)
    (args.run/'population-observables-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'population-observables-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Derived population audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:print(result['errors'],[c for c in result['checks'] if not c['passed']]);raise SystemExit(1)
