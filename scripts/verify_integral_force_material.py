#!/usr/bin/env python3
"""Recheck authoritative material force publication, owners and explicit transfers.

Numerical equivalence is established by the independent raw scientific recheck,
not by the native publication record. This is not a whole-process CUDA/I/O trace.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile

from siesta_remag.integral_force import record_errors
from siesta_remag.orbital_basis import record_errors as orbital_errors
from siesta_remag.atomic_basis import record_errors as atomic_errors


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def verify(root):
    meta=read(root/'metadata.json');ledger=read(root/'ledger.json');checks=[];errors=[];hashes={}
    expected={'fe2_forces','fe2_relax','fe2_noncollinear_gamma','fept2_soc_gamma_z','fept2_soc_gamma_x','fept2_soc_z','fept2_soc_x','fe16_noncollinear','fe54_collinear'}
    from siesta_remag.corpus import ENERGY_EXPORT_CASES, HISTORY_CASES, NO_DENSITY_EXPORT_CASES, NO_MATRIX_EXPORT_CASES
    actual_cases=set(meta['cases'])
    if not expected <= actual_cases <= expected | (set(ENERGY_EXPORT_CASES) | set(HISTORY_CASES) | set(NO_DENSITY_EXPORT_CASES) | set(NO_MATRIX_EXPORT_CASES)):
        errors.append('Missing base material cases or unknown additional cases')
    expected=actual_cases
    wanted_paths={(case,'candidate') for case in expected}
    wanted_paths.update((case,'candidate-repeat'+str(i)) for case in expected for i in range(2,meta.get('candidate_repeats',1)+1))
    if meta.get('vector_cache_ablation'):wanted_paths.update((case,'candidate-without-vector-cache') for case in expected)
    recorded_paths=[tuple(Path(r.get('candidate_path',r['benchmark']+'/candidate')).parts) for r in ledger]
    if (meta.get('integral_force') is not True or set(meta['cases'])!=expected or len(ledger)!=len(wanted_paths) or
            set(recorded_paths)!=wanted_paths or any(r.get('passed') is not True for r in ledger)):
        errors.append('Full material/scientific coverage failed')
    required=('source-verification.json','component-evidence-verification.json','scientific-recheck.json',
              'device-cycle-verification.json','nonlocal-owner-verification.json','nonlocal-force-component-verification.json',
              'soc-projector-component-verification.json','pair-force-component-verification.json','pair-handoff-verification.json',
              'two-center-component-verification.json','two-center-owner-verification.json','atomic-basis-component-verification.json',
              'atomic-owner-verification.json','orbital-basis-component-verification.json','orbital-owner-verification.json',
              'final-force-verification.json','final-grid-fields-verification.json')
    if 'integral_publication_check_sha256' in meta:
        required += ('integral-publication-verification.json',)
    if 'terminal_exports.py' in meta.get('harness_sha256',{}):
        required += ('terminal-export-verification.json',)
    if meta.get('gpu_initialization'):
        required += ('initialization-verification.json',)
    for name in required:
        p=root/name
        if not p.is_file() or read(p).get('passed') is not True:errors.append('Missing/failed prerequisite: '+name)
        if p.is_file():hashes[name]=sha(p)
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for fn in (record_errors,orbital_errors,atomic_errors):
            p=Path(inspect.getsourcefile(fn));data=archive.extractfile('src/siesta_remag/'+p.name).read()
            if hashlib.sha256(data).hexdigest()!=sha(p):raise ValueError('Loaded validator differs from archive: '+p.name)
    for row in ledger:
        folder=root/row.get('candidate_path',row['benchmark']+'/candidate');case=folder.parent.name;role=folder.name
        result=read(folder/'result.json');issues=[]
        r=[read(p) for p in sorted(folder.glob('remag-integral-force-*.json'))]
        g=[read(p) for p in sorted(folder.glob('remag-integral-grid-*.json'))]
        tables=read(folder/'remag-two-center-tables.json');pairs=read(folder/'remag-pair-geometry.json');nonlocal_geometry=read(folder/'remag-nonlocal-geometry.json')
        orbital=read(folder/'remag-orbital-basis.json')['cycles'];atomic=read(folder/'remag-atomic-basis.json')['cycles']
        telemetry=result['telemetry'];epochs=[c['epoch'] for c in telemetry['gpu_device_cycle']['cycles']]
        issues.extend(record_errors(r,g,tables,pairs,nonlocal_geometry,epochs,result['benchmark']['atoms'],result['benchmark']['spin']=='spin-orbit',bool(meta.get('gpu_history')),bool(meta.get('nonlocal_h0_shadow') or meta.get('nonlocal_h0')),compact_queries=meta.get('compact_nonlocal_queries',False)))
        if not result['valid'] or result['runtime_settings'].get('REMAG_INTEGRAL_FORCE')!='1' or telemetry['residency']['accounting_errors']:
            issues.append('Invalid material runtime or explicit accounting')
        if telemetry.get('integral_force')!=r or telemetry.get('grid_force')!=g or telemetry.get('integral_geometry')!={'tables':tables,'pairs':pairs,'nonlocal':nonlocal_geometry}:
            issues.append('Publication/geometry telemetry differs from raw records')
        for name,records in (('orbital',orbital),('atomic',atomic)):
            if [v['epoch'] for v in records]!=epochs or telemetry.get(name+'_basis')!=records:issues.append(name+' basis coverage/telemetry differs')
        for a,b,c in zip(orbital,atomic,g):issues.extend(orbital_errors(a,c,False));issues.extend(atomic_errors(b,c,False))
        raw=read(folder/'remag-residency.json');totals={};initial_grid=[]
        forbidden={'two_center_shadow','pair_force_shadow','nonlocal_force_shadow','orbital_basis_shadow','atomic_basis_shadow'}
        controlled={'pair_geometry','nonlocal_geometry','two_center_setup','final_force_output','grid_flow_geometry'}
        for interval in raw['intervals']:
            phase=interval['phase']
            for name,v in interval['boundaries'].items():
                if name in forbidden and any(v.values()):issues.append('Reference/intermediate transfer remains: '+name)
                if name=='grid_output' and any(v.values()):
                    # Initial rhoofd/vmat still seed the CPU Hamiltonian before
                    # the device cycle. They are not final-force readbacks.
                    # Account for both actual shapes below; reject this same
                    # boundary everywhere outside initialization.
                    if phase!='initialization':issues.append('Intermediate grid output outside initialization')
                    initial_grid.append(dict(phase=phase,geometry=interval['geometry'],traffic=v))
                if name not in controlled or (name=='grid_flow_geometry' and phase!='post_scf'):continue
                expected_phase='post_scf' if name in ('final_force_output','grid_flow_geometry') else 'initialization'
                if phase!=expected_phase:issues.append('Force transfer occurs in unexpected phase: '+name)
                total=totals.setdefault(name,dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0))
                for key,value in v.items():total[key]+=value
        def counter(upload=0,download=0,copies=0):return dict(host_to_device_bytes=upload,device_to_host_bytes=download,copy_calls=copies)
        wanted={
            'pair_geometry':counter(sum(p['setup_bytes'] for p in pairs['cycles']),copies=sum(3+5*bool(p['queries']) for p in pairs['cycles'])),
            'nonlocal_geometry':counter(sum(n['geometry_bytes'] for n in nonlocal_geometry['cycles']),copies=sum(3+bool(n['centers'])+bool(n['neighbors'])+bool(n['blocks'])+(1+bool(n['projector_entries'])+2*bool(n['neighbors']) if meta.get('compact_nonlocal_queries',False) else 2*bool(n['queries'])) for n in nonlocal_geometry['cycles'])),
            'two_center_setup':counter(sum(t['setup_bytes'] for t in tables['tables']),copies=sum(3+3*bool(t['shape'][2])+bool(t['shape'][1]) for t in tables['tables'])),
            'grid_flow_geometry':counter(sum(v['setup_bytes'] for v in g)+sum(c['setup_bytes'] for v in r for c in v['consumers']),copies=sum(7+len(v['atomic_terms']) for v in g)),
            'final_force_output':counter(download=sum(v['export_bytes'] for v in r),copies=2*len(r))}
        if totals!=wanted:issues.append('Shape-derived staging/terminal transfer reconciliation failed')
        device_initialization=bool(meta.get('gpu_initialization'))
        cycles=telemetry['gpu_device_cycle']['cycles']
        if any(bool(c.get('initialization',{}).get('device'))!=device_initialization for c in cycles):
            issues.append('Native initialization owner differs from requested mode')
        initial_expected=[] if device_initialization else [dict(phase='initialization',geometry=c['geometry'],
                               traffic=counter(download=8*(v['grid_shape'][1]*v['grid_shape'][2]*min(v['spins'],4)+
                                                          v['grid_shape'][4]*v['spins']),copies=2))
                          for c,v in zip(cycles,g)]
        if initial_grid!=initial_expected:issues.append('Initial density/Hamiltonian grid staging differs from shapes')
        grid_counts=read(folder/'remag-grid.json')
        if grid_counts.get('topology_only_prepares',0)!=(len(cycles) if device_initialization else 0):
            issues.append('Geometry-only grid staging differs from initialization mode')
        if grid_counts!=telemetry.get('gpu_grid') or grid_counts['device_to_host_bytes']!=sum(v['traffic']['device_to_host_bytes'] for v in initial_expected):
            issues.append('Initial grid exports differ from raw grid counters')
        if telemetry.get('gpu_integral_force')!=dict(host_to_device_bytes=sum(v['host_to_device_bytes'] for v in wanted.values()),device_to_host_bytes=sum(v['device_to_host_bytes'] for v in wanted.values())):
            issues.append('Force transfer summary differs')
        for routine in result['profile']['routines']:
            if '/FINAL_HF/' in routine['path'] and routine['name'].lower() in ('kinefsm','nlefsm','nlefsm_so_off','overfsm'):
                issues.append('CPU final integral replay remains: '+routine['name'])
        evidence=[folder/p for p in ('result.json','remag-residency.json','remag-grid.json','remag-two-center-tables.json','remag-pair-geometry.json','remag-nonlocal-geometry.json','remag-orbital-basis.json','remag-atomic-basis.json')]
        evidence+=list(folder.glob('remag-integral-*.json'))
        checks.append(dict(case=case,role=role,passed=not issues,errors=issues,geometries=len(epochs),traffic=totals,expected_traffic=wanted,
                           initial_grid_exports=initial_grid,expected_initial_grid_exports=initial_expected,
                           evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
                prerequisite_sha256=hashes,verifier_sha256=sha(Path(__file__)),independent_cuda_trace=False,
                remaining_cpu_terms_and_analysis_exports=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();result=verify(args.run)
    (args.run/'integral-force-material-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'integral-force-material-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Integral force material audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:print(result['errors'],[c for c in result['checks'] if not c['passed']]);raise SystemExit(1)
