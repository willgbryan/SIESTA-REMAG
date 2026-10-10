#!/usr/bin/env python3
"""Bind integrated population consumers to upstream results and retained epochs.

The ordinary raw scientific audit verifies upstream charges/moments and all
other physical gates. This audit checks ownership, shapes and explicit copy
boundaries. Derived values are computed on the GPU; D/E compatibility exports remain.
"""
import argparse
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
import tarfile

from siesta_remag.population import record_errors
from siesta_remag.output import read_eigenvalues


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())



def formatted_population(path):
    """Final requested table, including orbital/species and atomic totals."""
    text=path.read_text().rsplit('mulliken: Atomic and Orbital Populations:',1)
    if len(text)!=2:raise ValueError('Missing requested Mulliken orbital table')
    block=text[1];partial=block.index('Mulliken Atomic Populations:')
    end=re.search(r'^\s*Total[ \t]+[^\n]+$',block[partial:],re.MULTILINE)
    if end is None:raise ValueError('Missing partial-charge total')
    return block[:partial+end.end()].split()


def formatted_errors(reference,candidate):
    a,b=formatted_population(reference),formatted_population(candidate)
    if len(a)!=len(b):return ['Population output layout differs'],None
    maximum=0.;count=0
    # The scientific gates already compare more precise final moments/charges.
    # Table entries add at most one decimal print quantum between two rounded
    # values. Keep the 1e-4 charge/spin gate and account for that print precision.
    for x,y in zip(a,b):
        if re.fullmatch(r'[-+]?\d+\.\d+',x):
            try:
                value=float(y);error=abs(float(x)-value)
                if not math.isfinite(value) or error>1e-4+10.**(-len(x.split('.')[1]))+1e-12:
                    return ['Printed population value differs'],None
            except ValueError:return ['Population numeric output missing'],None
            maximum=max(maximum,error);count+=1
        elif x!=y:return ['Population labels/order differ'],None
    return ([] if count else ['Population numeric table is empty']),dict(values=count,max_printed_absolute_error=maximum)


def verify(root):
    meta,ledger = read(root/'metadata.json'),read(root/'ledger.json')
    errors,checks,hashes = [],[],{}
    required = ('source-verification.json','component-evidence-verification.json','scientific-recheck.json',
                'device-cycle-verification.json','population-component-verification.json','population-owner-verification.json',
                'population-observables-verification.json',
                'integral-force-material-verification.json','terminal-export-verification.json','final-density-verification.json')
    for name in required:
        path=root/name
        if not path.is_file() or read(path).get('passed') is not True:errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():hashes[name]=sha(path)
    cases={'fe2_forces','fe2_relax','fe2_noncollinear_gamma','fept2_soc_gamma_z','fept2_soc_gamma_x',
           'fept2_soc_z','fept2_soc_x','fe16_noncollinear','fe54_collinear'}
    from siesta_remag.corpus import ENERGY_EXPORT_CASES, HISTORY_CASES, NO_DENSITY_EXPORT_CASES, NO_MATRIX_EXPORT_CASES
    actual_cases=set(meta['cases'])
    if not cases <= actual_cases <= cases | (set(ENERGY_EXPORT_CASES) | set(HISTORY_CASES) | set(NO_DENSITY_EXPORT_CASES) | set(NO_MATRIX_EXPORT_CASES)):
        errors.append('Missing base material cases or unknown additional cases')
    cases=actual_cases
    if not meta.get('population') or set(meta['cases'])!=cases or not all(row['passed'] for row in ledger):
        errors.append('Population full material coverage failed')
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for fn in (record_errors,read_eigenvalues):
            path=Path(inspect.getsourcefile(fn))
            if hashlib.sha256(archive.extractfile('src/siesta_remag/'+path.name).read()).hexdigest()!=sha(path):
                raise ValueError('Loaded population validator differs from executed archive')
    wanted={(case,'candidate') for case in cases}
    wanted.update((case,'candidate-repeat'+str(i)) for case in cases for i in range(2,meta.get('candidate_repeats',1)+1))
    if meta.get('vector_cache_ablation'):wanted.update((case,'candidate-without-vector-cache') for case in cases)
    paths=[tuple(Path(row.get('candidate_path',row['benchmark']+'/candidate')).parts) for row in ledger]
    if len(paths)!=len(wanted) or set(paths)!=wanted:errors.append('Missing/duplicate candidate roles')
    for parts in paths:
        folder=root.joinpath(*parts);result=read(folder/'result.json');native=read(folder/'remag-population.json')
        formatting_issues,formatting=formatted_errors(root/parts[0]/'reference/stdout.log',folder/'stdout.log')
        records=native['cycles'];cycles=read(folder/'remag-device-scf.json')['cycles'];raw=read(folder/'remag-residency.json')
        eig=read_eigenvalues(folder/'remag.EIG');issues=record_errors(records,cycles,result['benchmark']['atoms'],eig,len(result['benchmark']['pseudopotentials']))
        issues.extend(formatting_issues)
        if native.get('schema')!=2 or result['runtime_settings'].get('REMAG_POPULATION')!='1' or not result['valid']:
            issues.append('Invalid/missing production population mode')
        if result['telemetry'].get('population')!=records:issues.append('Raw population records differ from result')
        # This corpus explicitly requests final option-1 Mulliken analysis.
        # Raw orbital/atomic exports are only needed for collinear up/down
        # tables; NC/SOC tables use the final derived views directly.
        needed=(0,1,2,4,6) if eig['nspin']<4 else (2,3,4,5,6)
        forbidden=(3,5) if eig['nspin']<4 else (0,1)
        if (not records or any(records[-1]['reads'][k]<1 for k in needed)
                or any(r['reads'][k] for r in records for k in forbidden)):
            issues.append('Requested population views were missing or redundant raw exports occurred')
        def counter(h=0,d=0,n=0):return dict(host_to_device_bytes=h,device_to_host_bytes=d,copy_calls=n)
        totals={'population_setup':counter(),'population_output':counter()};by_geometry={}
        for interval in raw['intervals']:
            for boundary,traffic in interval['boundaries'].items():
                if boundary not in totals:continue
                if interval['phase'] not in (('initialization',) if boundary=='population_setup' else ('post_scf','analysis_export')):
                    issues.append('Population copy inside SCF or unexpected phase')
                for key,value in traffic.items():totals[boundary][key]+=value
                geometry=by_geometry.setdefault(interval['geometry'],{'population_setup':counter(),'population_output':counter()})
                for key,value in traffic.items():geometry[boundary][key]+=value
        wanted_geometry={cycle['geometry']:{'population_setup':counter(h=record['setup_bytes'],n=5),
                          'population_output':counter(d=sum(record['export_bytes']),n=sum(bool(v) for v in record['export_bytes']))}
                         for record,cycle in zip(records,cycles)}
        if by_geometry!=wanted_geometry:issues.append('Per-geometry setup/output shape or copy counts differ')
        summary=dict(host_to_device_bytes=totals['population_setup']['host_to_device_bytes'],
                     device_to_host_bytes=totals['population_output']['device_to_host_bytes'])
        if result['telemetry'].get('gpu_population')!=summary:issues.append('Population aggregate differs')
        files=[folder/name for name in ('result.json','remag-population.json','remag-device-scf.json','remag-residency.json','remag.EIG','stdout.log')]
        files.append(root/parts[0]/'reference/stdout.log')
        checks.append(dict(case=parts[0],role=parts[1],passed=not issues,errors=issues,geometries=len(records),traffic=totals,formatted_output=formatting,
                           topology_device_copy_bytes=sum(r['retained_topology_bytes'] for r in records),
                           requested_reads=[r['reads'] for r in records],evidence_sha256={str(p.relative_to(root)):sha(p) for p in files}))
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
                prerequisite_sha256=hashes,verifier_sha256=sha(Path(__file__)),strict_residency_verified=False,
                compatibility_matrix_exports_remain=True,derived_spin_and_species_cpu=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=verify(args.run)
    (args.run/'population-material-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'population-material-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Population material audit:',len(result['checks']),'passed=',result['passed'])
    if not result['passed']:
        print(result['errors'],[c for c in result['checks'] if not c['passed']]);raise SystemExit(1)
