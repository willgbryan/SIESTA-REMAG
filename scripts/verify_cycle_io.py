#!/usr/bin/env python3
"""Reparse and hash selected filesystem activity for complete composed cycles.

This covers every observed filesystem syscall in each whole-cycle region,
including its first accepted iteration and terminal replay. Mappings are
reported separately. Page faults, asynchronous writeback, production SIESTA
integration and performance are outside this qualification.
"""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
import tarfile
from siesta_remag.io_trace import summarize_io


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(root):
    parser=Path(inspect.getsourcefile(summarize_io))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        with archive.extractfile('src/siesta_remag/io_trace.py') as f:archived=hashlib.file_digest(f,'sha256').hexdigest()
    if archived!=sha(parser):raise ValueError('Current IO parser differs from archived source')
    directory=root/'scf-cycle-io-check';report_path=directory/'check.json';report=json.loads(report_path.read_text());checks=[]
    for record in report['checks']:
        output=directory/record['output_file'];folder=output.parent;raw=folder/'syscalls.log';phase=folder/'remag-residency.json'
        timestamps=json.loads(phase.read_text());regions=[v for v in timestamps['intervals'] if v['phase']=='scf']
        with raw.open() as lines:actual=summarize_io(lines,timestamps,process_succeeded=record['process_returncode']==0)
        actual['scope']='Selected filesystem syscalls and mappings during the complete composed cycle, first iteration through terminal replay'
        same=actual==record['io_trace'];region=actual.get('phases',{}).get('scf',{})
        valid=record['passed'] and same and sha(raw)==record['io_raw_sha256'] and sha(phase)==record['phase_raw_sha256'] and (
            len(regions)==1 and regions[0]['iteration']==1 and actual['coverage_complete'] and region.get('filesystem_calls',0)==0)
        checks.append(dict(fixture=record['fixture'],passed=valid,original_summary_reproduced=same,
            raw_trace=str(raw.relative_to(root)),raw_trace_sha256=sha(raw),phase_trace=str(phase.relative_to(root)),
            phase_trace_sha256=sha(phase),device_iterations=record['iterations'],region=region,errors=actual['errors']))
    return dict(scope=__doc__,parser_sha256=archived,verifier_sha256=sha(Path(__file__)),report_sha256=sha(report_path),
        checks=checks,passed=bool(checks) and report['passed'] and report['completed_all_checks'] and all(c['passed'] for c in checks),
        strict_no_io_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();result=verify(args.run)
    (args.run/'cycle-io-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'cycle-io-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Composed-cycle filesystem evidence:',len(result['checks']),'traces; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
