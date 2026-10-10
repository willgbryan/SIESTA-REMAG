#!/usr/bin/env python3
"""Check final-only eigenvalue exports against raw phases and scientific output."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.final_spectrum import record_errors


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    errors, checks, hashes = [], [], {}
    meta=read(root/'metadata.json')
    if meta.get('final_spectra') is not True:
        errors.append('Batch did not request final spectrum ownership')
    for name in ('source-verification.json', 'scientific-recheck.json',
                 'device-cycle-verification.json', 'integral-force-material-verification.json'):
        path=root/name
        if not path.is_file() or read(path).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():hashes[name]=sha(path)
    module=Path(inspect.getsourcefile(record_errors))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        archived=archive.extractfile('src/siesta_remag/final_spectrum.py').read()
    if hashlib.sha256(archived).hexdigest()!=sha(module):
        errors.append('Spectrum validator differs from executed source')
    ledger=read(root/'ledger.json')
    seen=set()
    for row in ledger:
        relative=row.get('candidate_path',row['benchmark']+'/candidate')
        folder=root/relative;issues=[]
        if relative in seen:issues.append('Duplicate ledger candidate')
        seen.add(relative)
        result=read(folder/'result.json')
        cycle=read(folder/'remag-device-scf.json')['cycles']
        native=read(folder/'remag-final-spectrum.json')
        forces=[read(p) for p in sorted(folder.glob('remag-integral-force-*.json'))]
        if native.get('schema')!=1:issues.append('Missing final spectrum schema')
        records=native.get('cycles',[])
        issues.extend(record_errors(records,cycle,forces))
        if result.get('valid') is not True or not row.get('passed'):
            issues.append('Invalid or scientifically inequivalent calculation')
        if result.get('runtime_settings',{}).get('REMAG_FINAL_SPECTRA')!='1':
            issues.append('Final spectrum runtime flag missing')
        if result.get('telemetry',{}).get('final_spectrum')!=records:
            issues.append('Recorded spectrum telemetry differs from native journal')
        total=dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
        for interval in read(folder/'remag-residency.json')['intervals']:
            traffic=interval['boundaries'].get('final_spectrum_output')
            if traffic:
                if interval['phase']!='analysis_export':issues.append('Spectrum export outside final analysis')
                for key in total:total[key]+=traffic[key]
        expected=dict(host_to_device_bytes=0,device_to_host_bytes=8*cycle[-1]['terminal']['bank_size'],copy_calls=1)
        if total!=expected:issues.append('Raw copies do not match exactly one final eigenvalue export')
        evidence=[folder/name for name in ('input.fdf','result.json','remag-device-scf.json','remag-final-spectrum.json','remag-residency.json','remag.EIG')]
        evidence+=sorted(folder.glob('remag-integral-force-*.json'))
        checks.append(dict(candidate=relative,passed=not issues,errors=issues,geometries=len(cycle),
            actual_explicit_exports=total,removed_eigenvalue_bytes=sum(8*c['terminal']['bank_size'] for c in cycle[:-1]),
            removed_occupation_bytes=sum(8*c['terminal']['bank_size'] for c in cycle),
            evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
    return dict(passed=bool(checks) and not errors and all(c['passed'] for c in checks),errors=errors,
        scope=__doc__,checks=checks,prerequisite_sha256=hashes,validator_sha256=sha(module),
        verifier_sha256=sha(Path(__file__)),independent_cuda_trace=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    root=parser.parse_args().run;result=verify(root)
    (root/'final-spectrum-verifier.py').write_bytes(Path(__file__).read_bytes())
    (root/'final-spectrum-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Final spectrum:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:
        print(result['errors'],[c for c in result['checks'] if not c['passed']]);raise SystemExit(1)
