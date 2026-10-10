#!/usr/bin/env python3
"""Audit material atomic derivative selection and values against upstream CPU callbacks.

This validates shadow generation; CPU derivatives still feed force contraction.
Shadow transfers and duplicated computation are excluded from speedup claims.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.atomic_basis import record_errors

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root):
    metadata=json.loads((root/'metadata.json').read_text());ledger=json.loads((root/'ledger.json').read_text());checks=[];errors=[];seen=set()
    validator_hash=sha(Path(inspect.getsourcefile(record_errors)))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        if hashlib.sha256(archive.extractfile('src/siesta_remag/atomic_basis.py').read()).hexdigest()!=validator_hash:raise ValueError('Loaded atomic validator differs from archive')
    for path in sorted(root.glob('*/*/result.json')):
        result=json.loads(path.read_text());runtime=result.get('runtime_settings',{})
        if runtime.get('REMAG_ATOMIC_BASIS_SHADOW')!='1':continue
        folder=path.parent;case=folder.parent.name;role=folder.name;seen.add(case);issues=[];hashes={'result.json':sha(path)}
        raw=folder/'remag-atomic-basis.json';report=json.loads(raw.read_text());hashes[raw.name]=sha(raw);rows=report['cycles']
        forces=[]
        for force in sorted(folder.glob('remag-grid-force-*.json')):forces.append(json.loads(force.read_text()));hashes[force.name]=sha(force)
        telemetry=result['telemetry'];cycles=telemetry.get('gpu_device_cycle',{}).get('cycles',[])
        if not rows or len(rows)!=len(cycles) or [r['epoch'] for r in rows]!=[r['epoch'] for r in forces]:issues.append('Owner does not cover every geometry')
        if report.get('atol')!=1e-10 or report.get('rtol')!=1e-10 or report.get('displacements_exact') is not True:issues.append('Tolerance contract differs')
        for r,f in zip(rows,forces):issues.extend(record_errors(r,f))
        if rows!=telemetry.get('atomic_basis') or forces!=telemetry.get('grid_force'):issues.append('Raw records differ from parsed telemetry')
        traffic=dict(host_to_device_bytes=sum(r['compact_setup_bytes']+sum(t['shadow_setup_bytes'] for t in r['terms']) for r in rows),
                     device_to_host_bytes=sum(t['shadow_export_bytes'] for r in rows for t in r['terms']))
        if traffic!=telemetry.get('gpu_atomic_basis') or telemetry.get('residency',{}).get('accounting_errors'):issues.append('Transfer reconciliation failed')
        scientific=[r for r in ledger if r.get('candidate_path',r['benchmark']+'/candidate')==case+'/'+role]
        if not result['valid'] or len(scientific)!=1 or not scientific[0]['passed']:issues.append('Scientific calculation failed')
        if any(r.get('performance_comparable') is not False or r.get('end_to_end_speedup') is not None for r in scientific):issues.append('Shadow used for speedup claim')
        checks.append(dict(case=case,role=role,passed=not issues,errors=issues,geometries=len(rows),traffic=traffic,
            selection_points=sum(t['comparison']['points'] for r in rows for t in r['terms']),
            selected=sum(t['comparison']['selected'] for r in rows for t in r['terms']),
            compared=sum(t['comparison']['compared'] for r in rows for t in r['terms']),
            index_compared=sum(t['comparison']['index_compared'] for r in rows for t in r['terms']),
            max_absolute=[max(t['comparison']['absolute'][i] for r in rows for t in r['terms']) for i in range(3)],
            evidence_sha256=hashes))
    if not metadata.get('atomic_basis_shadow') or seen!=set(metadata['cases']):errors.append('Shadow missing requested cases')
    if not ledger or any(not r['passed'] for r in ledger):errors.append('Scientific ledger failed')
    mae={p.name:dict(passed=json.loads(p.read_text())['passed'],sha256=sha(p)) for p in root.glob('mae-*.json')}
    for family,label in (('fept2_soc','fept2'),('fept2_soc_gamma','fept2-gamma')):
        if {family+'_x',family+'_z'}<=set(metadata['cases']) and 'mae-'+label+'.json' not in mae:errors.append('Missing directional MAE')
    if any(not m['passed'] for m in mae.values()):errors.append('MAE failed')
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,mae=mae,
        validator_sha256=validator_hash,verifier_sha256=sha(Path(__file__)),implementation_sha256=sha(root/'implementation.tar.gz'),
        ledger_sha256=sha(root/'ledger.json'),metadata_sha256=sha(root/'metadata.json'),production_derivative_bypass_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'atomic-basis-shadow-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'atomic-basis-shadow-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Atomic material shadow:',len(r['checks']),'calculations; passed=',r['passed'])
    if not r['passed']:raise SystemExit(1)
