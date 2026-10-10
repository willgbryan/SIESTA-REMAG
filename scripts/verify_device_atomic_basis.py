#!/usr/bin/env python3
"""Qualify device-owned material derivatives against sequential CPU controls.

Validates raw lifecycle/phase records, eliminated derivative staging and host
storage, graph structure and scientific prerequisites. Explicit application
counters are not an independent CUDA, managed-memory or filesystem trace.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.atomic_basis import record_errors


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def verify(root):
    meta=read(root/'metadata.json');ledger=read(root/'ledger.json');errors=[];checks=[];hashes={}
    validator=Path(inspect.getsourcefile(record_errors))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        if hashlib.sha256(archive.extractfile('src/siesta_remag/atomic_basis.py').read()).hexdigest()!=sha(validator):
            raise ValueError('Loaded atomic validator differs from archived implementation')
    required=('source-verification.json','component-evidence-verification.json','atomic-basis-component-verification.json',
        'atomic-comparison-verification.json','atomic-owner-verification.json','atomic-masked-force-verification.json','scientific-recheck.json',
        'grid-force-verification.json','device-cycle-verification.json','final-grid-verification.json','final-density-verification.json')
    for name in required:
        p=root/name
        if not p.is_file() or read(p).get('passed') is not True:errors.append('Failed/missing prerequisite: '+name)
        if p.is_file():hashes[name]=sha(p)
    if not meta.get('atomic_basis') or not meta.get('atomic_basis_ablation') or meta.get('atomic_basis_shadow'):
        errors.append('Device derivative mode and sequential controls required')
    if any(r.get('passed') is not True for r in ledger):errors.append('Scientific ledger failed')
    seen=set()
    for case in meta['cases']:
        folder=root/case/'candidate';control=root/case/'candidate-without-atomic-basis';issues=[]
        if not all((p/'result.json').is_file() for p in (folder,control)):
            errors.append('Missing material/control: '+case);continue
        seen.add(case);result=read(folder/'result.json');other=read(control/'result.json')
        evidence=[folder/'result.json',control/'result.json'];records=[];forces=[];control_forces=[]
        for p,r,device in ((folder,result,True),(control,other,False)):
            settings=r['runtime_settings'];role=p.name;telemetry=r['telemetry'];owner=p/'remag-atomic-basis.json'
            if settings.get('REMAG_ATOMIC_BASIS')!=str(int(device)) or settings.get('REMAG_ATOMIC_BASIS_SHADOW')!='0':
                issues.append('Runtime derivative mode differs: '+role)
            rows=[v for v in ledger if v.get('candidate_path',v['benchmark']+'/candidate')==case+'/'+role]
            if not r['valid'] or len(rows)!=1 or rows[0].get('passed') is not True:issues.append('Scientific comparison failed: '+role)
            native=[read(v) for v in sorted(p.glob('remag-grid-force-*.json'))]
            evidence.extend(sorted(p.glob('remag-grid-force-*.json')))
            if not native or native!=telemetry.get('grid_force'):issues.append('Missing force records: '+role)
            if device:
                records=read(owner)['cycles'] if owner.is_file() else [];forces=native
                if owner.is_file():evidence.append(owner)
                if not records or [v['epoch'] for v in records]!=[v['epoch'] for v in native]:issues.append('Owner/force epoch coverage differs')
                for row,force in zip(records,native):issues.extend(record_errors(row,force,False))
                cycles=telemetry.get('gpu_device_cycle',{}).get('cycles',[])
                if len(records)!=len(cycles) or records!=telemetry.get('atomic_basis'):issues.append('Owner/SCF telemetry coverage differs')
                expected=dict(host_to_device_bytes=sum(v['compact_setup_bytes'] for v in records),device_to_host_bytes=0)
                if telemetry.get('gpu_atomic_basis')!=expected:issues.append('Atomic aggregate traffic differs')
            else:
                control_forces=native
                if owner.exists() or telemetry.get('atomic_basis') or telemetry.get('gpu_atomic_basis'):
                    issues.append('Control unexpectedly owns device derivatives')
            if telemetry.get('residency',{}).get('accounting_errors'):issues.append('Transfer reconciliation failed: '+role)
            for force in native:
                if force.get('atomic_basis_device')!=device or force.get('atomic_basis_shadow') is not False:issues.append('Force derivative mode differs: '+role)
                for term in force['atomic_terms']:
                    wanted=0 if device else 56*term['entries']
                    if (term.get('graph_nodes')!=(2 if device else 1) or term.get('derivative_upload_bytes')!=wanted or
                        term.get('host_derivative_storage_bytes')!=wanted or term.get('setup_bytes')!=wanted+32):issues.append('Derivative storage/upload/graph differs: '+role)
            phases=p/'remag-residency.json';evidence.append(phases);staging=0
            for interval in read(phases)['intervals']:
                for name,entry in interval['boundaries'].items():
                    if name=='atomic_basis_shadow' and any(entry.values()):issues.append('Shadow transfer in production/control')
                    if name=='atomic_basis_setup':
                        staging+=entry['host_to_device_bytes']
                        if not device or interval['phase']!='initialization' or entry['device_to_host_bytes']:
                            issues.append('Unexpected compact staging boundary: '+role)
            if staging!=(sum(v['compact_setup_bytes'] for v in records) if device else 0):issues.append('Phase-derived staging differs: '+role)
        if len(forces)!=len(control_forces):issues.append('Control geometry coverage differs')
        for force,cpu in zip(forces,control_forces):
            if any(force[k]!=cpu[k] for k in ('epoch','entries','grid_shape','atoms','spins')):issues.append('Control geometry shape differs')
            if cpu['setup_bytes']-force['setup_bytes']!=sum(t['derivative_upload_bytes'] for t in cpu['atomic_terms']):issues.append('Force staging reduction differs')
            if cpu['export_bytes']!=force['export_bytes']:issues.append('Force export size changed')
        direct=root/f'ablation-atomic-basis-{case}.json';evidence.append(direct)
        if not direct.is_file() or read(direct).get('passed') is not True:issues.append('Direct material/control comparison failed')
        removed=sum(t['derivative_upload_bytes'] for v in control_forces for t in v['atomic_terms']);compact=sum(v['compact_setup_bytes'] for v in records)
        checks.append(dict(case=case,passed=not issues,errors=issues,geometries=len(records),
            derivative_upload_bytes_removed=removed,compact_initial_upload_bytes=compact,net_upload_bytes_removed=removed-compact,
            host_derivative_storage_bytes_removed=sum(t['host_derivative_storage_bytes'] for v in control_forces for t in v['atomic_terms']),
            added_atomic_device_to_host_bytes=0,force_exports_bytes=sum(v['export_bytes'] for v in forces),
            evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence if p.is_file()}))
    if seen!=set(meta['cases']):errors.append('Requested material coverage differs')
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(v['passed'] for v in checks),errors=errors,checks=checks,
        prerequisite_sha256=hashes,validator_sha256=sha(validator),verifier_sha256=sha(Path(__file__)),
        production_derivative_upload_removed=True,independent_cuda_trace=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();result=verify(a.run)
    (a.run/'device-atomic-basis-verifier.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'device-atomic-basis-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Device atomic material audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
