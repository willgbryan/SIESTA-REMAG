#!/usr/bin/env python3
"""Check actual population/force ownership and their shared error publication.

The fixture uses one retained density row and empty integral geometries. It
poisons the original D/S/topology after retention. Full arithmetic and material
coverage remain separate requirements. No independent CUDA trace is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

SCENARIOS = ('normal', 'recovery', 'failed_status', 'failed_nonfinite', 'stale', 'failed_population', 'failed_observables')
RECOVERY = '''bad_population_partition bad_population_species population_read_before_binding population_duplicate_binding
missing_hartree early_grid early_nonlocal early_collect early_publish
population_early_evaluate population_early_publish unrecorded_kinetic changed_initial_force retained_pair nonlocal_unmarked_reuse nonlocal_retained_generator
grid_before_hartree duplicate_hartree reset_before_collection changed_publication_input
population_read_before_publication duplicate_collect reset_before_publication duplicate_publish
population_wrong_shape population_wrong_output_size population_read_after_reset'''.split()
FAILED = 'failed_collect retry_collect failed_publish failed_reset replace_failed_state failed_population_read'.split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(folder, scenario, spins, core):
    def read(name):return json.loads((folder/name).read_text())
    probe, raw, owner = (read(n) for n in ('probe.json', 'remag-residency.json', 'remag-population.json'))
    failed = scenario not in ('normal', 'recovery')
    geometries = 2 if scenario == 'recovery' else 1
    errors = []
    if (probe['scenario'], probe['spins'], probe['partial_core'], probe['rejections']) != (
            scenario, spins, core, RECOVERY*2 if scenario == 'recovery' else FAILED if failed else []):
        errors.append('Scenario/rejection coverage differs')
    ng = min(spins, 4)
    density = np.array([.4, .3, .2, -.1])[:ng]
    if spins == 8:density[2:] = [.16, .02]
    q=.7*density
    charge=q[:2].sum();sz=q[0]-q[1] if ng>1 else 0.
    xyz=np.array([2*q[2],2*q[3],sz]) if ng==4 else np.array([0.,0.,sz])
    spin=np.linalg.norm(xyz) if ng==4 else sz
    view=np.r_[charge,spin,xyz];atomic=np.r_[2.4-charge,view]
    expected_population=np.r_[np.tile(q,3),view,atomic,view,view,atomic[0],spin,xyz]
    v = np.array([1.2, 1.8, .2, .3])[:ng]
    weights = v.copy();weights[2:] *= 2
    increment = .08*np.array([-.2, .5, -.1])+.14*(np.dot(density, weights)-.18)*np.array([.3, .2, -.4])
    if core:increment += .1*(v[:min(spins, 2)].mean()-.6)*np.array([.2, -.3, .4])
    expected_force = np.r_[np.zeros(10), np.array([.1, -.2, .3])+increment,
                           np.arange(1, 10)*.011+np.outer([.4, .2, -.3], increment).ravel(order='F')/2]
    if len(probe['cycles']) != geometries or len(owner['cycles']) != geometries or owner.get('schema') != 2:
        errors.append('Population/force geometry coverage differs')
    maximum = 0.
    for i, (row, pop) in enumerate(zip(probe['cycles'], owner['cycles'])):
        expected = dict(epoch=i+1, published=not failed, failed=failed, captured_nodes=6+core,
                        successful_export_bytes=0 if failed else 180)
        if any(row.get(k) != val for k, val in expected.items()):errors.append('Force publication state differs')
        expected = dict(epoch=i+1, atoms=1, rows=1, capacity=1, spins=spins, species=1, setup_bytes=52,
                        retained_topology_bytes=16, workspace_bytes=280+24*ng,
                        evaluated=scenario != 'stale', published=not failed,
                        export_bytes=[0]*7 if failed else [8*ng]*3+[40,48,40,80], reads=[0]*7 if failed else [2,1,1,1,1,1,1])
        if any(pop.get(k) != val for k, val in expected.items()):errors.append('Population ownership/cache/bytes differ')
        for key, wanted in (('population', expected_population), ('output', expected_force)):
            actual = np.asarray(row[key])
            if failed:
                if actual.size:errors.append('Failed shared publication exposed output')
            elif actual.shape != wanted.shape or not np.isfinite(actual).all():
                errors.append('Missing/nonfinite '+key)
            else:maximum = max(maximum, float(np.max(np.abs(actual-wanted))))
    if maximum >= 1e-12:errors.append('Retained population/force arithmetic differs')
    def counter(h=0,d=0,n=0):return dict(host_to_device_bytes=h,device_to_host_bytes=d,copy_calls=n)
    actual = {n:counter() for n in ('population_setup', 'population_output', 'grid_flow_geometry', 'final_force_output')}
    for interval in raw['intervals']:
        for name, values in interval['boundaries'].items():
            if name not in actual or interval['phase'] != ('initialization' if name == 'population_setup' else 'post_scf'):
                errors.append('Unexpected owner copy/phase');continue
            for key,value in values.items():actual[name][key] += value
    exported = 0 if scenario == 'stale' else 4 if scenario in ('failed_status','failed_population','failed_observables') else 180*geometries
    copies = 0 if scenario == 'stale' else 1 if scenario in ('failed_status','failed_population','failed_observables') else 2*geometries
    expected = dict(population_setup=counter(h=52*geometries,n=5*geometries),
                    population_output=counter(d=0 if failed else (24*ng+208)*geometries,n=0 if failed else 7*geometries),
                    grid_flow_geometry=counter(h=180*geometries,n=2*geometries),final_force_output=counter(d=exported,n=copies))
    if actual != expected:errors.append('Shared publication and cached export counts differ')
    return errors,maximum


def run(executable,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for scenario in SCENARIOS:
        for spins in (1,2,4,8):
            for core in (0,1):
                folder=output.parent/f'{scenario}_spin{spins}_core{core}';folder.mkdir()
                with (folder/'worker.log').open('w') as log:
                    code=subprocess.run([str(executable.resolve()),scenario,str(spins),str(core),str(folder/'probe.json')],cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=30).returncode
                names=('probe.json','remag-residency.json','remag-population.json','worker.log')
                errors,error=(['Native probe failed or omitted evidence'],None) if code or any(not (folder/n).is_file() for n in names) else validate(folder,scenario,spins,core)
                checks.append(dict(scenario=scenario,spins=spins,partial_core=core,native_return=code,passed=not errors,errors=errors,max_absolute_error=error,
                                   evidence_sha256={str((folder/n).relative_to(output.parent)):sha(folder/n) for n in names if (folder/n).is_file()}))
    report=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,checker_sha256=sha(Path(__file__)),native_executable_sha256=sha(executable),independent_cuda_trace=False)
    output.write_text(json.dumps(report,indent=2)+'\n');print('Population owner:',len(checks),'passed=',report['passed'],flush=True)
    if not report['passed']:print([c for c in checks if not c['passed']]);raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.executable,args.output)
