#!/usr/bin/env python3
"""Audit compact material nonlocal shadows, full scalar/complex outputs and transfers.

Recomputes differences from complete CPU/GPU energy, force and stress vectors.
The upstream selection is checked at runtime before and after SCF; this audit
covers its counts/lifetimes, not independent replay of all material geometry.
"""
import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def verify(root):
    meta=read(root/'metadata.json');ledger=read(root/'ledger.json');checks=[];errors=[];hashes={}
    # This first material run omitted the new flag from the runner's explicit
    # runtime-settings whitelist. Bind the alternative mode evidence to the
    # executed archive; never treat a recorded false/None value as enabled.
    with tarfile.open(root/'implementation.tar.gz') as archive:
        mode_sources={name:archive.extractfile(name).read() for name in
            ('modal_app.py','src/siesta_remag/runner.py','native/nonlocal_force_state.cpp')}
    mode_binding=(b'"REMAG_NONLOCAL_FORCE_SHADOW": "1" if nonlocal_force_shadow and role != "reference" else "0"' in mode_sources['modal_app.py'] and
        b"nonlocal_force_shadow=env.get('REMAG_NONLOCAL_FORCE_SHADOW')=='1'" in mode_sources['src/siesta_remag/runner.py'] and
        b"telemetry['nonlocal_force']=report" in mode_sources['src/siesta_remag/runner.py'] and
        b'require(remag_nonlocal_force_shadow_enabled()&&remag_two_center_shadow_enabled()&&remag::pair_force_shadow_enabled()&&!remag::in_scf()' in mode_sources['native/nonlocal_force_state.cpp'])
    mode_hashes={name:hashlib.sha256(data).hexdigest() for name,data in mode_sources.items()}
    expected={'fe2_forces','fe2_relax','fe2_noncollinear_gamma','fept2_soc_gamma_z','fept2_soc_gamma_x','fept2_soc_z','fept2_soc_x','fe16_noncollinear','fe54_collinear'}
    if meta.get('nonlocal_force_shadow') is not True or set(meta['cases'])!=expected or len(ledger)!=9 or any(r.get('passed') is not True for r in ledger):errors.append('Full material/scientific coverage failed')
    required=('source-verification.json','component-evidence-verification.json','scientific-recheck.json','pair-force-material-verification.json',
                 'pair-handoff-verification.json','two-center-material-verification.json','soc-projector-component-verification.json',
                 'nonlocal-force-component-verification.json','grid-force-verification.json','device-cycle-verification.json')
    if 'nonlocal_compact_check_sha256' in meta:
        required+=('nonlocal-compact-check/nonlocal-compact-verification.json',)
    if 'nonlocal_owner_check_sha256' in meta:
        required+=('nonlocal-owner-verification.json',)
    for name in required:
        path=root/name
        if not path.is_file() or read(path).get('passed') is not True:errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():hashes[name]=sha(path)
    for case in [row['benchmark'] for row in ledger]:
        folder=root/case/'candidate';result=read(folder/'result.json');report=read(folder/'remag-nonlocal-force.json');integrals=read(folder/'remag-two-center.json');raw=read(folder/'remag-residency.json');manifest=read(folder/'manifest.json')
        forces=[read(p) for p in sorted(folder.glob('remag-grid-force-*.json'))];epochs=[f['epoch'] for f in forces];rows=report['cycles'];issues=[];soc=manifest['spin']=='spin-orbit';consumer=3 if soc else 2
        if report.get('complete') is not True or [(r['epoch'],r['consumer']) for r in rows]!=[(e,consumer) for e in epochs]:issues.append('Missing or unordered nonlocal contributions')
        settings=result['runtime_settings'];mode_recorded='REMAG_NONLOCAL_FORCE_SHADOW' in settings
        mode_enabled=settings.get('REMAG_NONLOCAL_FORCE_SHADOW')=='1' if mode_recorded else mode_binding and meta.get('nonlocal_force_shadow') is True
        if result.get('valid') is not True or not mode_enabled:issues.append('Failed or unproven material runtime mode')
        if result['telemetry'].get('nonlocal_force')!=report or result['telemetry']['residency']['accounting_errors']:issues.append('Telemetry or transfer accounting differs')
        maxima=[0.,0.,0.];geometry_bytes=uploads=downloads=queries=0;geometry_copies=0
        for row in rows:
            e=row['epoch'];f=next(f for f in forces if f['epoch']==e);t=next(t for t in integrals['cycles'] if (t['epoch'],t['consumer'])==(e,consumer))
            na,nr,nc,cap,nh,n,centers,neighbors,blocks=(row[k] for k in ('atoms','rows','columns','capacity','components','queries','centers','neighbors','blocks'));size=3*na+18
            if (nc,nr,cap)!=(f['grid_shape'][0],f['grid_shape'][3],f['grid_shape'][4]) or na!=f['atoms'] or nh!=f['spins'] or (soc and nh!=8):issues.append('Retained electronic shape differs')
            setup=8*nr+4*cap+28*centers+16*neighbors+24*blocks+32*n
            allocated=setup+8*size+8*cap+(28 if centers==0 else 0)+(16 if neighbors==0 else 0)+(24 if blocks==0 else 0)+(12 if n==0 else 0)
            wanted=dict(status=0,complete=True,passed=True,electronic_upload_bytes=0,graph_nodes=2,geometry_bytes=setup,setup_bytes=8*size,device_allocation_bytes=allocated,export_bytes=8*size)
            if any(row.get(k)!=v for k,v in wanted.items()) or n!=t['queries'] or t['graph_nodes']!=(4 if n else 2) or t['passed'] is not True:issues.append('Compact contraction/integral evidence differs')
            if not math.isfinite(row['volume']) or row['volume']<=0:issues.append('Invalid volume')
            arrays=[row[k] for k in ('initial','reference','actual')]
            if any(len(a)!=size or any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in a) for a in arrays):issues.append('Incomplete/nonfinite output vectors');continue
            initial,reference,actual=arrays
            if initial[:9]!=[0]*9 or (not soc and (actual[1:9]!=[0]*8 or reference[1:9]!=[0]*8)):issues.append('Energy convention changed')
            for i,(a,b) in enumerate(zip(actual,reference)):
                group=0 if i<9 else 1 if i<9+3*na else 2;err=abs(a-b);maxima[group]=max(maxima[group],err)
                tolerance=(1e-10+1e-10*abs(b)) if group==0 else (1e-10 if group==1 else 1e-12)
                if err>tolerance:issues.append('Independent full energy/force/stress comparison failed')
            geometry_bytes+=setup;uploads+=8*size;downloads+=8*size;queries+=n
            geometry_copies+=3+bool(centers)+bool(neighbors)+bool(blocks)+2*bool(n)
        totals={k:dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0) for k in ('nonlocal_geometry','nonlocal_force_shadow')}
        for interval in raw['intervals']:
            for boundary,b in interval['boundaries'].items():
                if boundary not in totals:continue
                if interval['phase']!=('initialization' if boundary=='nonlocal_geometry' else 'post_scf'):issues.append('Nonlocal copy outside permitted phase')
                for k,v in b.items():totals[boundary][k]+=v
        wanted={'nonlocal_geometry':dict(host_to_device_bytes=geometry_bytes,device_to_host_bytes=0,copy_calls=geometry_copies),
                'nonlocal_force_shadow':dict(host_to_device_bytes=uploads,device_to_host_bytes=downloads,copy_calls=2*len(rows))}
        if totals!=wanted:issues.append('Independent shape-derived transfer accounting failed')
        if result['telemetry'].get('gpu_nonlocal_force')!=dict(host_to_device_bytes=geometry_bytes+uploads,device_to_host_bytes=downloads):issues.append('Nonlocal telemetry traffic differs')
        l=next(r for r in ledger if r['benchmark']==case)
        if l.get('performance_comparable') is not False or l.get('end_to_end_speedup') is not None:issues.append('Shadow incorrectly qualifies speedup')
        evidence=[folder/p for p in ('result.json','remag-nonlocal-force.json','remag-two-center.json','remag-residency.json','manifest.json')]+sorted(folder.glob('remag-grid-force-*.json'))
        checks.append(dict(case=case,passed=not issues,errors=issues,runtime_mode_recorded=mode_recorded,runtime_mode_evidence="runtime_settings" if mode_recorded else "archived launcher/runner/guard and complete raw owner",geometries=len(epochs),queries=queries,centers=sum(r['centers'] for r in rows),neighbors=sum(r['neighbors'] for r in rows),blocks=sum(r['blocks'] for r in rows),max_absolute_error=maxima,
                           geometry_setup_bytes=geometry_bytes,diagnostic_setup_bytes=uploads,diagnostic_export_bytes=downloads,evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,prerequisite_sha256=hashes,runtime_mode_source_sha256=mode_hashes,verifier_sha256=sha(Path(__file__)),independent_cuda_trace=False,cpu_nonlocal_outputs_authoritative=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=verify(a.run)
    (a.run/'nonlocal-force-material-verifier.py').write_bytes(Path(__file__).read_bytes());(a.run/'nonlocal-force-material-verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Nonlocal material audit:',len(r['checks']),'passed=',r['passed'])
    if not r['passed']:print(r['errors'],[c for c in r['checks'] if not c['passed']]);raise SystemExit(1)
