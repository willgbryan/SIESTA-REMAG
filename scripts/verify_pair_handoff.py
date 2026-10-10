#!/usr/bin/env python3
"""Independently audit immutable pair/grid leases and metadata mutation probes."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    folder=root/'pair-handoff-check';path=folder/'check.json';data=json.loads(path.read_text());checks=[];seen=set()
    modes={'normal','reuse','missing','num','ptr','list','shape','wrong_epoch','wrong_consumer','overlap_first','grid_first','mode','scf','duplicate','grid_during_pair','reset_active','wrong_complete','duplicate_complete','overlap_early','pair_during_grid','reset_incomplete','reborrow'}
    if data.get('schema_version')==2:modes=modes|{'nl_'+m for m in modes}|{'nl_'+m for m in ('skip_nonlocal','nonlocal_spin','nonlocal_reset','nonlocal_duplicate','nonlocal_grid','nonlocal_complete','nonlocal_reborrow')}
    for row in data['checks']:
        mode,spins=row['scenario'],row['spins'];key=(mode,spins);errors=[]
        if key in seen or mode not in modes or spins not in (1,2,4,8):errors.append('Invalid/duplicate scenario')
        seen.add(key)
        for kind in ('report','log'):
            if sha(folder/row[kind+'_file'])!=row[kind+'_sha256']:errors.append('Evidence hash mismatch')
        actual=json.loads((folder/row['report_file']).read_text());base=mode.removeprefix('nl_');ng=2 if base=='reuse' else 1
        expected=[]
        for g in range(ng):
            expected.extend(([.4+g,.3,.2,.1,.05,.06,.07,.08][:spins]))
            expected.extend(([1.2+g,1.3,1.4,1.5][:min(4,spins)]))
        if actual!=dict(scenario=mode,spins=spins,geometries=ng,rejections=0 if base in ('normal','reuse') else 1,retained_values=expected):errors.append('Retained values or lifecycle rejection differs')
        if row.get('passed') is not True or row.get('native_return')!=0:errors.append('Native probe failed')
        checks.append(dict(scenario=mode,spins=spins,passed=not errors,errors=errors))
    errors=[] if seen=={(m,s) for m in modes for s in (1,2,4,8)} and len(checks)==4*len(modes) else ['Incomplete scenario coverage']
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,report_sha256=sha(path),verifier_sha256=sha(Path(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'pair-handoff-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'pair-handoff-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Pair handoff audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
