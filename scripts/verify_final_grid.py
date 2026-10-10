#!/usr/bin/env python3
"""Reproduce final force-grid shadow arrays or authoritative handoff coverage.

The 27 grid diagnostics are reused from device-cycle publication. A pinned
final-force Hamiltonian is generated/exported there only for shadows or a
requested H consumer. Legacy exports and density analysis remain separate.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.final_grid import record_errors


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    metadata=json.loads((root/'metadata.json').read_text());checks=[];errors=[];seen=set()
    validator=sha(Path(inspect.getsourcefile(record_errors)))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        with archive.extractfile('src/siesta_remag/final_grid.py') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=validator:raise ValueError('Final-grid validator differs from retained implementation')
    shadow=bool(metadata.get('final_grid_shadow'))
    if not (shadow or metadata.get('final_grid')) or (shadow and metadata.get('final_grid')):errors.append('Invalid requested final-grid mode')
    for p in sorted(root.glob('*/candidate*/result.json')):
        role=p.parent.name;case=p.parent.parent.name;result=json.loads(p.read_text());env=result['runtime_settings'];faults=[]
        expected=role!='candidate-without-final-grid'
        if not expected:
            if env.get('REMAG_FINAL_GRID')=='1' or env.get('REMAG_FINAL_GRID_SHADOW')=='1':errors.append('Control enabled final-grid bypass/shadow')
            continue
        seen.add(case)
        if (env.get('REMAG_FINAL_GRID_SHADOW')=='1')!=shadow or (env.get('REMAG_FINAL_GRID')=='1')==shadow:
            faults.append('Requested final-grid mode differs from runtime')
        path=p.parent/'remag-final-grid.json';hashes={'result.json':sha(p)}
        rows=json.loads(path.read_text())['cycles'] if path.is_file() else []
        if path.is_file():hashes[path.name]=sha(path)
        if not result['valid'] or rows!=result['telemetry'].get('final_grid') or not rows:faults.append('Final-grid publication invalid/missing')
        force=result['telemetry'].get('grid_force',[])
        if [r['epoch'] for r in rows]!=[r['epoch'] for r in force]:faults.append('Final-grid geometry coverage differs')
        for row,grid in zip(rows,force):
            faults.extend(record_errors(row,p.parent,shadow))
            if row['capacity']!=grid['grid_shape'][4] or row['spins']!=grid['spins']:faults.append('Final electronic shape differs')
            if row.get('matrix_file'):hashes[row['matrix_file']]=sha(p.parent/row['matrix_file'])
        if not shadow:
            raw=p.parent/'remag-residency.json';hashes[raw.name]=sha(raw)
            phases=json.loads(raw.read_text())['intervals']
            forbidden={'grid_density_input','grid_potential_input','grid_sparse_input',
                       'hartree_density_input','hartree_potential_output','hartree_result'}
            for phase in phases:
                if phase['phase']!='post_scf':continue
                for name,counts in phase['boundaries'].items():
                    if name in forbidden and any(counts.values()):faults.append('Final-grid electronic round trip remains: '+name)
            for routine in result['profile']['routines']:
                if '/FINAL_HF/' in routine['path'] and routine['name'].lower() in ('rhoofd','vmat','poison','gxc-cellxc'):
                    faults.append('CPU final-grid recomputation timer remains: '+routine['name'])
        checks.append(dict(case=case,role=role,geometries=len(rows),passed=not faults,errors=faults,evidence_sha256=hashes))
    if seen!=set(metadata['cases']):errors.append('Missing requested material coverage')
    gate=root/('integral-force-material-verification.json' if metadata.get('integral_force') else 'grid-force-verification.json')
    if not gate.is_file() or not json.loads(gate.read_text())['passed']:errors.append('Grid-force/scientific audit is missing or failed')
    if metadata.get('final_grid_ablation'):
        for case in metadata['cases']:
            p=root/f'ablation-final-grid-{case}.json'
            if not p.is_file() or not json.loads(p.read_text())['passed']:errors.append('Missing/failed direct final-grid ablation: '+case)
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(c['passed'] for c in checks),checks=checks,errors=errors,
        validator_sha256=validator,verifier_sha256=sha(Path(__file__)),metadata_sha256=sha(root/'metadata.json'),
        force_audit_sha256=sha(gate) if gate.is_file() else None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'final-grid-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'final-grid-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Final force-grid handoff:',len(result['checks']),'calculations; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
