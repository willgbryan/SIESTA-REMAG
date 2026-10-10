#!/usr/bin/env python3
"""Check pair/grid ordering, immutable D/E handoff and sparse topology identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCENARIOS=('normal','reuse','missing','num','ptr','list','shape','wrong_epoch','wrong_consumer','overlap_first','grid_first','mode','scf','duplicate','grid_during_pair','reset_active','wrong_complete','duplicate_complete','overlap_early','pair_during_grid','reset_incomplete','reborrow')
SCENARIOS=SCENARIOS+tuple('nl_'+s for s in SCENARIOS)+tuple('nl_'+s for s in ('skip_nonlocal','nonlocal_spin','nonlocal_reset','nonlocal_duplicate','nonlocal_grid','nonlocal_complete','nonlocal_reborrow'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(executable,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for spins in (1,2,4,8):
        for scenario in SCENARIOS:
            folder=output.parent/f'{scenario}-spins{spins}';folder.mkdir();report=folder/'owner.json';log=folder/'worker.log'
            with log.open('w') as stream:code=subprocess.run([str(executable.resolve()),scenario,str(spins),str(report)],cwd=folder,stdout=stream,stderr=subprocess.STDOUT,timeout=30).returncode
            actual=json.loads(report.read_text()) if report.is_file() else {};mode=scenario.removeprefix('nl_');geometries=2 if mode=='reuse' else 1;values=[]
            for g in range(geometries):values.extend([.4+g,.3,.2,.1,.05,.06,.07,.08][:spins]+[1.2+g,1.3,1.4,1.5][:min(spins,4)])
            expected=dict(scenario=scenario,spins=spins,geometries=geometries,rejections=0 if mode in ('normal','reuse') else 1,retained_values=values)
            checks.append(dict(scenario=scenario,spins=spins,passed=code==0 and actual==expected,native_return=code,
                report_file=str(report.relative_to(output.parent)),report_sha256=sha(report) if report.is_file() else None,
                log_file=str(log.relative_to(output.parent)),log_sha256=sha(log)))
    result=dict(schema_version=2,scope=__doc__,passed=all(r['passed'] for r in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable))
    output.write_text(json.dumps(result,indent=2)+'\n');print('Pair handoff:',len(checks),'passed=',result['passed'])
    if not result['passed']:print([r for r in checks if not r['passed']]);raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.executable,a.output)
