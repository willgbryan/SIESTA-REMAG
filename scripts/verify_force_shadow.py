#!/usr/bin/env python3
"""Audit real-material GPU dfscf shadows against unchanged upstream contractions.

This checks recorded FP64 arrays, geometry coverage, post-SCF copy accounting
and the scientific ledger. It does not qualify final GPU ownership or speedup.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    ledger=json.loads((root/'ledger.json').read_text());checks=[]
    for path in sorted(root.glob('*/*/result.json')):
        result=json.loads(path.read_text())
        if result.get('runtime_settings',{}).get('REMAG_FORCE_SHADOW')!='1':continue
        folder=path.parent;rows=[];errors=[];hashes={'result.json':sha(path)}
        case,role=folder.parent.name,folder.name
        scientific=[v for v in ledger if v.get('candidate_path',v['benchmark']+'/candidate')==case+'/'+role]
        if not result['valid'] or len(scientific)!=1 or not scientific[0]['passed']:
            errors.append('Scientific comparison did not pass')
        for raw in sorted(folder.glob('remag-force-shadow-*.json')):
            record=json.loads(raw.read_text());hashes[raw.name]=sha(raw);rows.append(record)
            if record.get('atoms')!=result['benchmark']['atoms'] or not record.get('passed') or record.get('status')!=0 or record.get('graph_nodes')!=1:
                errors.append('Invalid native shadow result')
            if 'grid_shape' in record:
                no,np,nsp,nuo,nnz,cells=record['grid_shape'];nh=record['spins'];ng=min(nh,4);points=np*nsp
                upload=4*no+8*7*record['entries']*nsp+8*nuo+24+8*(3*record['atoms']+9)+4
                resident_fields=record.get('electronic_inputs_resident') is True
                if not resident_fields:upload+=8*(nnz*nh+points*(ng+1))
                retained=8*(nnz*(nh+ng)+points*(ng+2)) if resident_fields else 0
                if record['setup_bytes']!=upload or record['export_bytes']!=8*(3*record['atoms']+9)+8 or record['retained_electronic_bytes']!=retained:
                    errors.append('Shape-derived shadow transfer/retention sizes disagree')
            for name,count,tolerance,unit in (('force',3*result['benchmark']['atoms'],1e-10,'ry_per_bohr'),('stress',9,1e-12,'ry_per_bohr3')):
                a=record.get('reference_'+name,[]);b=record.get('device_'+name,[])
                if len(a)!=count or len(b)!=count or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in a+b):
                    errors.append('Missing/nonfinite '+name+' arrays');continue
                error=max(abs(x-y) for x,y in zip(a,b))
                if error>=tolerance or error!=record.get('max_'+name+'_error_'+unit):
                    errors.append('Recorded '+name+' error fails or cannot be reproduced')
        telemetry=result['telemetry'];cycles=telemetry.get('gpu_device_cycle',{}).get('cycles',[])
        resident=result['runtime_settings'].get('REMAG_FORCE_SHADOW_RESIDENT')=='1'
        if resident and any(r.get('electronic_inputs_resident') is not True or not r.get('retained_electronic_bytes') for r in rows):
            errors.append('Force did not consume retained electronic fields')
        if not rows or (result['runtime_settings'].get('REMAG_DEVICE_SCF')=='1' and len(rows)!=len(cycles)):
            errors.append('Missing geometry coverage')
        if any(a['epoch']>=b['epoch'] for a,b in zip(rows,rows[1:])):errors.append('Geometry epoch did not advance')
        if rows!=telemetry.get('grid_force_shadow'):errors.append('Raw shadow records differ from telemetry')
        traffic=dict(host_to_device_bytes=sum(r['setup_bytes'] for r in rows),device_to_host_bytes=sum(r['export_bytes'] for r in rows))
        if traffic!=telemetry.get('gpu_grid_force_shadow'):errors.append('Shadow copy totals differ from telemetry')
        checks.append(dict(case=case,role=role,passed=not errors,errors=errors,geometries=len(rows),traffic=traffic,
            electronic_inputs_resident=resident,
            max_force_error_ry_per_bohr=max((r['max_force_error_ry_per_bohr'] for r in rows),default=None),
            max_stress_error_ry_per_bohr3=max((r['max_stress_error_ry_per_bohr3'] for r in rows),default=None),
            evidence_sha256=hashes))
    return dict(scope=__doc__,passed=bool(checks) and all(c['passed'] for c in checks),checks=checks,
        ledger_sha256=sha(root/'ledger.json'),implementation_archive_sha256=sha(root/'implementation.tar.gz'),
        verifier_sha256=sha(Path(__file__)),final_force_residency_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'force-shadow-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'force-shadow-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Material force shadows:',len(result['checks']),'calculations; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
