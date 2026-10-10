#!/usr/bin/env python3
"""Reproduce requested density-output reuse, full shadow arrays and phase traffic.

This qualifies the RHO/DRHO boundary only. Requested density downloads/file writes
are explicit; remaining CPU consumers and legacy electronic exports still exist.
"""
import argparse
import hashlib
import inspect
import math
import numpy as np
from siesta_remag.output import read_grid, grid_artifact
from siesta_remag.compare import difference,TOLERANCES
import json
from pathlib import Path
import tarfile
from siesta_remag.final_density import record_errors


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(root):
    meta=json.loads((root/'metadata.json').read_text());errors=[];checks=[];controls=[];seen=set()
    validator=sha(Path(inspect.getsourcefile(record_errors)))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        with archive.extractfile('src/siesta_remag/final_density.py') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=validator:raise ValueError('Density validator differs from frozen source')
        with archive.extractfile('src/siesta_remag/output.py') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=sha(Path(inspect.getsourcefile(read_grid))):raise ValueError('Density reader differs from frozen source')
    shadow=meta.get('final_density_shadow',False)
    if not (shadow or meta.get('final_density')) or (shadow and meta.get('final_density')):
        errors.append('Invalid requested density mode')
    for path in sorted(root.glob('*/candidate*/result.json')):
        folder=path.parent;role=folder.name;case=folder.parent.name;r=json.loads(path.read_text());env=r['runtime_settings'];faults=[]
        if role=='candidate-without-final-density':
            if env.get('REMAG_FINAL_DENSITY')=='1' or env.get('REMAG_FINAL_DENSITY_SHADOW')=='1':faults.append('Density control enabled feature')
            evidence={}
            for suffix in ('RHO','DRHO'):
                source=root/case/'reference'/('remag.'+suffix);output=folder/('remag.'+suffix)
                try:source_artifact,output_artifact=grid_artifact(source),grid_artifact(output)
                except FileNotFoundError:faults.append('Missing raw control '+suffix);continue
                a,x=read_grid(source);b,y=read_grid(output)
                if not np.array_equal(a,b) or not difference(x,y,TOLERANCES['density_components_e_bohr3'])['passed']:
                    faults.append('Control '+suffix+' differs from upstream')
                evidence[suffix]={'reference':sha(source_artifact),'control':sha(output_artifact)}
            controls.append(dict(case=case,passed=not faults,errors=faults,evidence_sha256=evidence))
            continue
        seen.add(case)
        if (env.get('REMAG_FINAL_DENSITY_SHADOW')=='1')!=shadow or (env.get('REMAG_FINAL_DENSITY')=='1')==shadow:
            faults.append('Density runtime mode differs')
        native=folder/'remag-final-density.json';rows=json.loads(native.read_text())['cycles'] if native.is_file() else []
        hashes={'result.json':sha(path)}
        if native.is_file():hashes[native.name]=sha(native)
        if not r['valid'] or not rows or rows!=r['telemetry'].get('final_density'):faults.append('Invalid/missing density records')
        forces=r['telemetry'].get('grid_force',[])
        expected=[(v['epoch'],kind) for v in forces for kind in (1,2)]
        if [(v['epoch'],v['kind']) for v in rows]!=expected:faults.append('Density geometry/kind coverage differs from forces')
        force_map={v['epoch']:v for v in forces}
        for row in rows:
            faults.extend(record_errors(row,folder,shadow));force=force_map.get(row['epoch'])
            if not force or math.prod(row['mesh'])!=force['grid_shape'][1]*force['grid_shape'][2] or row['nsm']**3!=force['grid_shape'][2] or row['spins']!=min(4,force['spins']):
                faults.append('Retained density shape differs from final forces')
            if row.get('array_file'):hashes[row['array_file']]=sha(folder/row['array_file'])
        exported=[v for v in rows if v['exported']]
        for kind,suffix in ((1,'RHO'),(2,'DRHO')):
            selected=[v for v in rows if v['kind']==kind];outputs=[v for v in selected if v['exported']]
            nominal=folder/('remag.'+suffix)
            try:paths=[grid_artifact(nominal)]
            except FileNotFoundError:paths=[]
            if selected and selected[-1]['requested']:
                if len(outputs)!=1 or outputs[0]['epoch']!=selected[-1]['epoch'] or len(paths)!=1:
                    faults.append('Final requested '+suffix+' coverage differs');continue
                output=outputs[0];cell,field=read_grid(nominal);hashes[paths[0].name]=sha(paths[0])
                if list(field.shape)!=[output['spins'],*reversed(output['mesh'])]:faults.append('Output density shape differs: '+suffix)
                ref=root/case/'reference'/('remag.'+suffix)
                try:ref_artifact=grid_artifact(ref)
                except FileNotFoundError:faults.append('Missing upstream '+suffix);continue
                rc,rf=read_grid(ref);hashes['../reference/'+ref_artifact.name]=sha(ref_artifact)
                if not np.array_equal(rc,cell) or not difference(rf,field,TOLERANCES['density_components_e_bohr3'])['passed']:
                    faults.append('Requested '+suffix+' differs from upstream')
                if shadow:
                    cpu=np.fromfile(folder/output['array_file'],dtype=np.float64,count=field.size)
                    if not np.array_equal(cpu.astype(field.dtype),field.reshape(-1)):
                        faults.append('Shadow reference differs from written '+suffix)
            elif outputs or paths:faults.append('Unrequested '+suffix+' export')
        raw=folder/'remag-residency.json';phases=json.loads(raw.read_text())['intervals'];hashes[raw.name]=sha(raw)
        total=dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0)
        forbidden={'grid_density_input','grid_potential_input','grid_sparse_input','grid_output',
                   'hartree_density_input','hartree_potential_output','hartree_result'}
        for phase in phases:
            for name,counters in phase['boundaries'].items():
                if name=='final_density_output':
                    for k,v in counters.items():total[k]+=v
                    if phase['phase']!='analysis_export':faults.append('Density export occurred outside analysis')
                if not shadow and phase['phase']=='analysis_export' and name in forbidden and any(counters.values()):
                    faults.append('Analysis electronic round trip remains: '+name)
        expected=dict(host_to_device_bytes=0,device_to_host_bytes=sum(v['device_to_host_bytes'] for v in rows),copy_calls=len(exported))
        if total!=expected:faults.append('Raw density output accounting differs')
        if not shadow:
            for routine in r['profile']['routines']:
                if '/Analysis/' in routine['path'] and routine['name'].lower() in ('rhoofd','poison','gxc-cellxc','vmat'):
                    faults.append('CPU analysis grid recomputation remains: '+routine['name'])
        checks.append(dict(case=case,role=role,geometries=len(forces),retained_fields=len(rows),exports=len(exported),traffic=total,
                           passed=not faults,errors=faults,evidence_sha256=hashes))
    if seen!=set(meta['cases']):errors.append('Requested cases incomplete')
    gates={}
    for name in ('final-grid-verification.json','scientific-recheck.json'):
        path=root/name
        if not path.is_file() or not json.loads(path.read_text())['passed']:errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():gates[name]=sha(path)
    if meta.get('final_density_ablation'):
        if {r['case'] for r in controls}!=set(meta['cases']):errors.append('Density controls missing')
        for case in meta['cases']:
            path=root/f'ablation-final-density-{case}.json'
            if not path.is_file() or not json.loads(path.read_text())['passed']:errors.append('Missing/failed density ablation: '+case)
            if path.is_file():gates[path.name]=sha(path)
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(v['passed'] for v in checks+controls),checks=checks,controls=controls,errors=errors,
                validator_sha256=validator,verifier_sha256=sha(Path(__file__)),metadata_sha256=sha(root/'metadata.json'),prerequisite_sha256=gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();result=verify(args.run)
    (args.run/'final-density-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'final-density-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Requested density output:',len(result['checks']),'calculations; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
