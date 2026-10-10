#!/usr/bin/env python3
"""Independently recheck retained population ownership and shared publication.

The actual owner probe has one retained D/S row, nonzero atomic/grid forces,
empty integral geometries and deliberate original-buffer poisoning. Nonempty
population arithmetic and full material equivalence have separate audits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENARIOS = {'normal', 'recovery', 'failed_status', 'failed_nonfinite', 'stale', 'failed_population', 'failed_observables'}
RECOVERY = '''bad_population_partition bad_population_species population_read_before_binding population_duplicate_binding
missing_hartree early_grid early_nonlocal early_collect early_publish population_early_evaluate
population_early_publish unrecorded_kinetic changed_initial_force retained_pair nonlocal_unmarked_reuse nonlocal_retained_generator grid_before_hartree duplicate_hartree
reset_before_collection changed_publication_input population_read_before_publication duplicate_collect
reset_before_publication duplicate_publish population_wrong_shape population_wrong_output_size population_read_after_reset'''.split()
FAILED = 'failed_collect retry_collect failed_publish failed_reset replace_failed_state failed_population_read'.split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(root):
    folder = root/'population-owner-check'
    report, metadata = read(folder/'check.json'), read(root/'metadata.json')
    errors, checks, seen, hashes = [], [], set(), {}
    for name in ('source-verification.json', 'component-evidence-verification.json', 'population-component-verification.json'):
        path = root/name
        if not path.is_file() or read(path).get('passed') is not True:
            errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():hashes[name] = sha(path)
    if not metadata.get('population') or report['checker_sha256'] != metadata.get('population_owner_check_sha256') or report.get('passed') is not True:
        errors.append('Population owner mode/source/check failed')
    for row in report['checks']:
        name, spins, core = row['scenario'], row['spins'], row['partial_core']
        key = name, spins, core
        issues = []
        if key in seen or name not in SCENARIOS or spins not in (1,2,4,8) or core not in (0,1):
            issues.append('Unexpected/duplicate owner fixture')
        seen.add(key)
        sub = folder/f'{name}_spin{spins}_core{core}'
        files = [sub/n for n in ('probe.json','remag-residency.json','remag-population.json','worker.log')]
        bindings = {str(p.relative_to(folder)):sha(p) for p in files}
        if bindings != row['evidence_sha256']:issues.append('Actual raw evidence differs')
        probe, raw, owner = [read(p) for p in files[:3]]
        failed = name not in ('normal','recovery')
        count = 2 if name == 'recovery' else 1
        rejected = RECOVERY*2 if name == 'recovery' else FAILED if failed else []
        if (probe['scenario'],probe['spins'],probe['partial_core'],probe['rejections']) != (name,spins,core,rejected):
            issues.append('Required recovery/failure guards missing')
        # Scalar reference independent of the checker implementation. Retained
        # SOC off-diagonal charge uses the Hermitian average of D3/D7 and D4/D8.
        rho = [.4,.3,.5*(.2+.12) if spins == 8 else .2,.5*(-.1+.14) if spins == 8 else -.1][:min(spins,4)]
        potential = [1.2,1.8,.2,.3]
        contraction = sum(q*potential[i]*(2 if i > 1 else 1) for i,q in enumerate(rho))
        core_factor = .1*((1.2 if spins == 1 else 1.5)-.6) if core else 0.
        delta = [core_factor*a+.08*b+.14*(contraction-.18)*c for a,b,c in zip((.2,-.3,.4),(-.2,.5,-.1),(.3,.2,-.4))]
        force = [0.]*10+[v+d for v,d in zip((.1,-.2,.3),delta)]
        force += [.011*(3*i+j+1)+dx*delta[i]/2 for i in range(3) for j,dx in enumerate((.4,.2,-.3))]
        raw_population = [.7*q for q in rho]
        charge=sum(raw_population[:2]);sz=raw_population[0]-raw_population[1] if spins>1 else 0.
        xyz=[2*raw_population[2],2*raw_population[3],sz] if spins>=4 else [0.,0.,sz]
        magnitude=sum(v*v for v in xyz)**.5 if spins>=4 else sz
        view=[charge,magnitude]+xyz;net=2.4-charge
        population=raw_population*3+view+[net]+view+view+view+[net,magnitude]+xyz
        maximum = 0.
        if len(probe['cycles']) != count or len(owner['cycles']) != count or owner.get('schema') != 2:
            issues.append('Missing force/population geometry')
        for index,(cycle,pop) in enumerate(zip(probe['cycles'],owner['cycles'])):
            wanted = dict(epoch=index+1,published=not failed,failed=failed,captured_nodes=6+core,successful_export_bytes=0 if failed else 180)
            if any(cycle.get(k) != v for k,v in wanted.items()):issues.append('Force publication/capture differs')
            wanted = dict(epoch=index+1,atoms=1,rows=1,capacity=1,spins=spins,species=1,setup_bytes=52,retained_topology_bytes=16,
                          workspace_bytes=280+24*len(rho),evaluated=name!='stale',published=not failed,
                          export_bytes=[0]*7 if failed else [8*len(rho)]*3+[40,48,40,80],reads=[0]*7 if failed else [2,1,1,1,1,1,1])
            if any(pop.get(k) != v for k,v in wanted.items()):issues.append('Population state/shape/cache differs')
            for field,wanted in (('output',force),('population',population)):
                actual = np.asarray(cycle[field])
                if failed:
                    if actual.size:issues.append('Rejected publication exposed '+field)
                elif actual.shape != np.shape(wanted) or not np.isfinite(actual).all():
                    issues.append('Missing/nonfinite '+field)
                else:maximum = max(maximum,float(np.max(np.abs(actual-wanted))))
        if maximum >= 1e-12 or row['max_absolute_error'] is None or abs(maximum-row['max_absolute_error']) >= 1e-15:
            issues.append('Independent population/force reconstruction differs')
        def counter(h=0,d=0,n=0):return dict(host_to_device_bytes=h,device_to_host_bytes=d,copy_calls=n)
        totals = {k:counter() for k in ('population_setup','population_output','grid_flow_geometry','final_force_output')}
        for interval in raw['intervals']:
            for boundary,traffic in interval['boundaries'].items():
                if boundary not in totals or interval['phase'] != ('initialization' if boundary=='population_setup' else 'post_scf'):
                    issues.append('Unexpected owner transfer/phase');continue
                for k,v in traffic.items():totals[boundary][k] += v
        downloaded = 0 if name=='stale' else 4 if name in ('failed_status','failed_population','failed_observables') else 180*count
        copies = 0 if name=='stale' else 1 if name in ('failed_status','failed_population','failed_observables') else 2*count
        wanted = dict(population_setup=counter(52*count,0,5*count),population_output=counter(0,0 if failed else (24*len(rho)+208)*count,0 if failed else 7*count),
                      grid_flow_geometry=counter(180*count,0,2*count),final_force_output=counter(0,downloaded,copies))
        if totals != wanted:issues.append('Shape-derived staging/export/retry counts differ')
        if (raw['schema_version'] != 2 or raw['phase_markers_seen'] is not True or raw['geometry_epochs'] != (2 if name=='stale' else count)
                or raw['strict_residency_verified'] is not False or any(raw[k] is not None for k in ('library_transfers','unified_memory_migrations','file_io'))):
            issues.append('Probe accounting scope differs')
        if row['native_return'] or row['passed'] is not True or row['errors']:issues.append('Native worker/check failed')
        checks.append(dict(scenario=name,spins=spins,partial_core=core,passed=not issues,errors=issues,geometries=count,
                           rejections=len(rejected),max_absolute_error=maximum,transfers=totals,evidence_sha256=bindings))
    if len(checks)!=56 or seen != {(n,s,c) for n in SCENARIOS for s in (1,2,4,8) for c in (0,1)}:
        errors.append('Incomplete owner spin/core/scenario coverage')
    return dict(scope=__doc__,passed=not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
                verifier_sha256=sha(Path(__file__)),report_sha256=sha(folder/'check.json'),prerequisite_sha256=hashes,independent_cuda_trace=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'population-owner-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'population-owner-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Population owner audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:
        print(result['errors'],[c for c in result['checks'] if not c['passed']]);raise SystemExit(1)
