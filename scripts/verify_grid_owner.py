#!/usr/bin/env python3
"""Independently verify geometry-owner count validation and zero preparation copies."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    folder=root/'grid-owner-check';path=folder/'check.json';data=json.loads(path.read_text());checks=[];seen=set()
    expected={'normal','missing','count','shape','stale','restage','ends','orbs','phi','nsc','lease'}
    with tarfile.open(root/'implementation.tar.gz') as archive:
        source=archive.extractfile('scripts/check_grid_owner.py').read()
    if hashlib.sha256(source).hexdigest()!=data['checker_sha256']:
        raise ValueError('Grid owner checker differs from executed archive')
    if b"'topology-lease'" in source:expected.update(('topology','topology-lease'))
    for row in data['checks']:
        name=row['scenario'];errors=[];files={k:folder/row[k+'_file'] for k in ('report','owner','log')}
        if name in seen:errors.append('Duplicate scenario')
        seen.add(name)
        if any(sha(v)!=row[k+'_sha256'] for k,v in files.items()):errors.append('Evidence hash mismatch')
        native=json.loads(files['report'].read_text());raw=json.loads(files['owner'].read_text())
        double=name in ('stale','restage');n=2 if double else 1
        topology=name in ('topology','topology-lease')
        wanted=dict(scenario=name,passed=True,stages=n,epoch=n,entries=2 if name=='restage' else 1,
                    rho=0. if topology else .5625 if name=='restage' else .25,rejections=[] if name in ('normal','topology') else [-230])
        if native!=wanted:errors.append('Native validation/recovery differs')
        if raw['geometry_epochs']!=n or raw['phase_markers_seen'] is not True:errors.append('Execution markers differ')
        post=[i for i in raw['intervals'] if i['phase']=='post_scf']
        if len(post)!=n or any(i['boundaries'] for i in post):errors.append('Force preparation transfers or coverage differ')
        out=sum(t['device_to_host_bytes'] for i in raw['intervals'] for t in i['boundaries'].values())
        if out!=(0 if topology else 8*n):errors.append('Exports differ from initialization-only density fixture')
        if topology and any(v['host_to_device_bytes'] for i in raw['intervals'] for k,v in i['boundaries'].items() if k!='grid_topology_basis'):
            errors.append('Topology preparation uploaded electronic state')
        if row['native_return']!=0 or not row['passed'] or row['errors']:errors.append('Recorded check failed')
        checks.append(dict(scenario=name,passed=not errors,errors=errors))
    errors=[] if seen==expected and len(checks)==len(expected) else ['Scenario coverage differs']
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'grid-owner-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'grid-owner-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Grid owner audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
