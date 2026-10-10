#!/usr/bin/env python3
"""Audit recorded full device derivative comparisons across material geometries.

Actual material derivative arrays stay on device. This checks the complete
comparison counts/status/maxima, lifecycle and transfer records; it does not
independently replay every material derivative. The comparator has a separate
retained-input numerical qualification, and the upstream scientific ledger is
required independently of these records.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.orbital_basis import record_errors


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(root):
    metadata=json.loads((root/'metadata.json').read_text());ledger=json.loads((root/'ledger.json').read_text())
    errors=[];checks=[];seen=set();validator=Path(inspect.getsourcefile(record_errors));hashes={}
    with tarfile.open(root/'implementation.tar.gz') as archive:
        if hashlib.sha256(archive.extractfile('src/siesta_remag/orbital_basis.py').read()).hexdigest()!=sha(validator):
            raise ValueError('Loaded orbital validator differs from archived implementation')
    required=('source-verification.json','component-evidence-verification.json','orbital-basis-component-verification.json',
              'orbital-comparison-verification.json','scientific-recheck.json','grid-force-verification.json','device-cycle-verification.json')
    for name in required:
        p=root/name
        if not p.is_file() or json.loads(p.read_text()).get('passed') is not True:errors.append('Required evidence failed or missing: '+name)
        if p.is_file():hashes[name]=sha(p)
    for path in sorted(root.glob('*/*/result.json')):
        result=json.loads(path.read_text());settings=result.get('runtime_settings',{})
        if settings.get('REMAG_ORBITAL_BASIS_SHADOW')!='1':continue
        folder=path.parent;case=folder.parent.name;role=folder.name;seen.add(case);issues=[]
        owner=folder/'remag-orbital-basis.json';records=json.loads(owner.read_text()).get('cycles',[]) if owner.is_file() else []
        forces=[json.loads(p.read_text()) for p in sorted(folder.glob('remag-grid-force-*.json'))]
        if not records or [r['epoch'] for r in records]!=[r['epoch'] for r in forces]:issues.append('Missing or reordered geometry coverage')
        for record,force in zip(records,forces):issues.extend(record_errors(record,force))
        telemetry=result['telemetry'];cycles=telemetry.get('gpu_device_cycle',{}).get('cycles',[])
        if len(records)!=len(cycles) or records!=telemetry.get('orbital_basis'):issues.append('Owner records differ from cycle/telemetry coverage')
        rows=[r for r in ledger if r.get('candidate_path',r['benchmark']+'/candidate')==case+'/'+role]
        if not result['valid'] or len(rows)!=1 or not rows[0]['passed']:issues.append('Upstream scientific comparison failed')
        traffic=dict(host_to_device_bytes=sum(r['compact_setup_bytes']+r['shadow_setup_bytes'] for r in records),
                     device_to_host_bytes=sum(r['shadow_export_bytes'] for r in records))
        if traffic!=telemetry.get('gpu_orbital_basis') or telemetry.get('residency',{}).get('accounting_errors'):issues.append('Transfer reconciliation failed')
        phase=json.loads((folder/'remag-residency.json').read_text());setup=post_h2d=post_d2h=0
        for interval in phase['intervals']:
            for name,entry in interval['boundaries'].items():
                if name not in ('orbital_basis_setup','orbital_basis_shadow'):continue
                if name=='orbital_basis_setup':
                    setup+=entry['host_to_device_bytes']
                    if interval['phase']!='initialization' or entry['device_to_host_bytes']:issues.append('Basis staging occurred outside initialization')
                else:
                    post_h2d+=entry['host_to_device_bytes'];post_d2h+=entry['device_to_host_bytes']
                    if interval['phase']!='post_scf':issues.append('Shadow traffic occurred outside post-SCF')
        if (setup!=sum(r['compact_setup_bytes'] for r in records) or post_h2d!=72*len(records) or post_d2h!=72*len(records)):
            issues.append('Phase-derived orbital transfer totals differ')
        evidence=[path,owner,folder/'remag-residency.json',*folder.glob('remag-grid-force-*.json')]
        checks.append(dict(case=case,role=role,passed=not issues,errors=issues,geometries=len(records),
            compared=sum(r['compared'] for r in records),traffic=traffic,
            maximum_absolute_error=[max((r['max_absolute_error'][i] for r in records),default=0.) for i in range(3)],
            evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence if p.is_file()}))
    if not metadata.get('orbital_basis_shadow') or seen!=set(metadata['cases']):errors.append('Requested material coverage differs')
    if any(not r['passed'] for r in ledger):errors.append('Scientific ledger failed')
    return dict(scope=__doc__,passed=not errors and bool(checks) and all(c['passed'] for c in checks),errors=errors,checks=checks,
        prerequisite_sha256=hashes,validator_sha256=sha(validator),verifier_sha256=sha(Path(__file__)),
        production_derivative_upload_removed=False,independent_material_array_replay=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();result=verify(a.run)
    (a.run/'orbital-basis-material-verifier.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'orbital-basis-material-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Orbital material shadow audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
