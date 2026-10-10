#!/usr/bin/env python3
"""Reproduce initial-state ownership probes from retained arrays and raw counters."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    meta=json.loads((root/'metadata.json').read_text());path=root/'initial-state-check';report=json.loads((path/'check.json').read_text())
    checker=Path(__file__).with_name('check_initial_state.py')
    assert sha(checker)==meta['initial_state_check_sha256']==report['checker_sha256']
    assert report['native_exec_sha256']==meta['native_binary_sha256']['/opt/candidate/remag-native/remag_initial_state_check']
    with tarfile.open(root/'implementation.tar.gz') as archive:
        assert hashlib.sha256(archive.extractfile('scripts/check_initial_state.py').read()).hexdigest()==sha(checker)
    spec=importlib.util.spec_from_file_location('initial_owner_check',checker);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected={(s,r) for s in (1,2,4,8) for r in range(4)};checks=[];seen=set()
    assert report['passed'] and len(report['checks'])==16
    for check in report['checks']:
        spins,requests=check['spins'],check['requests'];assert (spins,requests) not in seen;seen.add((spins,requests))
        for name,digest in check['evidence_sha256'].items():assert sha(path/name)==digest,name
        row=module.validate(path/f'spin{spins}_requests{requests}',spins,requests)
        assert all(check[k]==v for k,v in row.items())
        checks.append(row)
    assert seen==expected
    return dict(scope=__doc__,passed=True,checks=checks,rejections=sum(c['rejections'] for c in checks),
        verifier_sha256=sha(Path(__file__)),evidence_sha256={'initial-state-check/check.json':sha(path/'check.json'),
        'metadata.json':sha(root/'metadata.json')},strict_residency_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    report=verify(args.run)
    (args.run/'initial-state-component-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'initial-state-component-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Initial-state raw probes:',len(report['checks']),'passed')
