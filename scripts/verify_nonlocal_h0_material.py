#!/usr/bin/env python3
"""Recheck material H0 matrices, retained integral leases and explicit traffic.

Checks the selected diagnostic or production H0 mode. Production requires a
qualified material matrix shadow and retained device SCF/force leases. This
verifier does not qualify performance or strict memory/I/O residency.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
from siesta_remag.nonlocal_h0 import audit


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(path.read_text())


def verify(root,component,shadow_material=None,query_component=None):
    metadata=read(root/'metadata.json');ledger=read(root/'ledger.json');errors=[];checks=[];hashes={}
    production=metadata.get('nonlocal_h0') is True;poisoned=metadata.get('poison_nonlocal_h0') is True
    assert metadata.get('integral_force') is True
    assert metadata.get('nonlocal_h0_shadow',False) is not production
    shadow_binding=None
    query_binding=None
    if metadata.get('compact_nonlocal_queries',False):
        assert production and query_component is not None,'Compact query execution requires qualified GPU component evidence'
        qualified=read(query_component/'component-verification.json')
        assert qualified['passed'] and qualified['checks']==22 and qualified['allocation_released']
        for name,digest in qualified['evidence_sha256'].items():assert sha(query_component/name)==digest,name
        query_sources=read(query_component/'metadata.json')['source_sha256']
        with tarfile.open(root/'implementation.tar.gz') as archive:
            for name in ('nonlocal_query.hpp','nonlocal_query.cu'):
                assert hashlib.file_digest(archive.extractfile('native/'+name),'sha256').hexdigest()==query_sources['/opt/probe/native/'+name],name
        query_binding=sha(query_component/'component-verification.json')
    if production:
        assert metadata.get('gpu_initialization') is True and metadata.get('device_scf') is True
        assert shadow_material is not None,'Production H0 requires qualified material shadow evidence'
        qualified=read(shadow_material/'qualification-summary.json')
        assert qualified['passed'] and qualified['material_h0_shadow_qualified']
        for name,digest in qualified['evidence_sha256'].items():assert sha(shadow_material/name)==digest,name
        prior=read(shadow_material/'nonlocal-h0-material-verification.json')
        assert prior['passed'] and prior['production_h0'] is False
        prior_meta=read(shadow_material/'metadata.json')
        assert prior_meta['nonlocal_h0_shadow'] is True and set(metadata['cases'])<=set(prior_meta['cases'])
        shadow_binding=sha(shadow_material/'qualification-summary.json')
    source=read(component/'qualification-summary.json')
    assert source['passed'] and source['component_gpu_qualified'] and source['checks']==1352
    for name,digest in source['evidence_sha256'].items():assert sha(component/name)==digest,name
    native=read(component/'nonlocal-h0.json')['native_sha256']
    primitive_sources=[]
    with tarfile.open(root/'implementation.tar.gz') as archive, tarfile.open(component/'implementation.tar.gz') as parent_archive:
        for name in ('nonlocal_h0.hpp','nonlocal_h0.cu','nonlocal_h0_detail.cuh','soc_projector_detail.cuh'):
            stream=archive.extractfile('native/'+name);assert stream is not None
            content=stream.read();digest=hashlib.sha256(content).hexdigest();inline_linkage=False
            if name=='soc_projector_detail.cuh' and digest!=native[name]:
                parent=parent_archive.extractfile('h0/native/'+name).read()
                assert hashlib.sha256(parent).hexdigest()==native[name]
                assert parent.count(b'__device__ ')==7
                assert content==parent.replace(b'__device__ ',b'__device__ inline ')
                inline_linkage=True
            else:assert digest==native[name],name
            primitive_sources.append(dict(file=name,qualified_sha256=native[name],executed_sha256=digest,
                                          inline_linkage_only=inline_linkage))
        path=Path(inspect.getsourcefile(audit));stream=archive.extractfile('src/siesta_remag/'+path.name)
        assert stream is not None and hashlib.file_digest(stream,'sha256').hexdigest()==sha(path)
    for name in ('source-verification.json','component-evidence-verification.json','scientific-recheck.json',
                 'device-cycle-verification.json','integral-force-material-verification.json'):
        assert read(root/name)['passed'],name;hashes[name]=sha(root/name)
    expected={(name,'candidate' if i==1 else 'candidate-repeat'+str(i))
              for name in metadata['cases'] for i in range(1,metadata['candidate_repeats']+1)}
    actual={tuple(Path(row.get('candidate_path',row['benchmark']+'/candidate')).parts) for row in ledger}
    assert actual==expected and len(ledger)==len(expected) and all(row['passed'] for row in ledger)
    for case,role in sorted(expected):
        folder=root/case/role;result=read(folder/'result.json');report=read(folder/'remag-nonlocal-h0.json')
        geometry=read(folder/'remag-nonlocal-geometry.json');trace=read(folder/'remag-residency.json')
        epochs=[r['epoch'] for r in read(folder/'remag-device-scf.json')['cycles']]
        checked=audit(report,folder,epochs,geometry,trace,production=production,cycle_records=read(folder/'remag-device-scf.json')['cycles'],poisoned=poisoned,compact_queries=metadata.get('compact_nonlocal_queries',False));issues=list(checked['errors'])
        telemetry=result['telemetry']
        if not result['valid'] or result['runtime_settings'].get('REMAG_NONLOCAL_H0' if production else 'REMAG_NONLOCAL_H0_SHADOW')!='1':issues.append('H0 diagnostic runtime is missing or invalid')
        if telemetry.get('nonlocal_h0')!=report or telemetry.get('nonlocal_h0_audit')!=checked:issues.append('H0 raw audit differs from recorded telemetry')
        if telemetry.get('gpu_nonlocal_h0')!=checked.get('traffic'):issues.append('H0 recorded transfer totals differ')
        for row in ledger:
            if row.get('candidate_path',row['benchmark']+'/candidate')==case+'/'+role:
                if (not production or poisoned) and (row.get('performance_comparable') is not False or row.get('end_to_end_speedup') is not None):
                    issues.append('H0 shadow incorrectly claims a speedup')
        paths=['result.json','input.fdf','remag-nonlocal-h0.json','remag-nonlocal-geometry.json',
               'remag-device-scf.json','remag-residency.json']+[r['raw_file'] for r in report['cycles'] if r['raw_file']]
        for name in paths:hashes[str((folder/name).relative_to(root))]=sha(folder/name)
        checks.append(dict(case=case,role=role,passed=not issues,errors=issues,geometries=len(epochs),
                           scalar_max_error=None if production else max(r['scalar_max_error'] for r in checked['checks']),
                           soc_max_error=None if production else max(r['soc_max_error'] for r in checked['checks'])))
        errors.extend(issues)
    return dict(scope=__doc__,passed=not errors,errors=errors,checks=checks,production_h0=production,host_matrices_poisoned=poisoned,material_shadow_qualification_sha256=shadow_binding,
                compact_query_qualification_sha256=query_binding,
                speedup_qualified=False,strict_residency_verified=False,independent_cuda_trace=False,
                component_qualification_sha256=sha(component/'qualification-summary.json'),primitive_sources=primitive_sources,
                evidence_sha256=hashes,validator_sha256=sha(Path(inspect.getsourcefile(audit))),
                verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--component',type=Path,required=True)
    p.add_argument('--shadow-material',type=Path);p.add_argument('--query-component',type=Path);a=p.parse_args()
    result=verify(a.run,a.component,a.shadow_material,a.query_component)
    (a.run/'nonlocal-h0-material-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('H0 material ownership:',len(result['checks']),'comparisons; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
