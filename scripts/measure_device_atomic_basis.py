#!/usr/bin/env python3
"""Retain single-observation timings and boundaries for qualified atomic controls."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def measure(root):
    gate=root/'device-atomic-basis-verification.json';audit=json.loads(gate.read_text())
    if audit['passed'] is not True:raise ValueError('Device atomic qualification must pass before reporting timings')
    rows=[]
    for check in audit['checks']:
        pair={}
        for role in ('candidate','candidate-without-atomic-basis'):
            folder=root/check['case']/role;path=folder/'result.json';r=json.loads(path.read_text())
            if not r['valid'] or any(r['runtime_settings'].get(name)=='1' for name in
                ('REMAG_TRACE_IO','REMAG_TRACE_CUDA','REMAG_ORBITAL_BASIS_SHADOW','REMAG_ATOMIC_BASIS_SHADOW')):raise ValueError('Traced/invalid timing')
            phases=json.loads((folder/'remag-residency.json').read_text())['intervals'];totals={}
            for interval in phases:
                total=totals.setdefault(interval['phase'],dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0))
                for entry in interval['boundaries'].values():
                    for k in total:total[k]+=entry[k]
            routines=r['profile']['routines'];names=('dfscf','remag_atomic_force','remag_atomic_basis_setup','DHSCF4','FINAL_HF','Analysis')
            pair[role]=dict(wall_seconds=r['wall_seconds'],scf_iterations=r['observables']['scf_iterations'],
                routines={name:dict(inclusive_seconds=sum(v['inclusive_seconds'] for v in routines if v['name']==name),
                    calls=sum(v['calls'] for v in routines if v['name']==name)) for name in names},
                phase_traffic=totals,peak_rss_bytes=r['telemetry'].get('peak_rss_bytes'),
                gpu_memory_peak_mib=r['telemetry'].get('gpu_memory_peak_mib'),result_sha256=sha(path),
                phases_sha256=sha(folder/'remag-residency.json'))
        rows.append(dict(case=check['case'],geometries=check['geometries'],measurements=pair,
            derivative_upload_bytes_removed=check['derivative_upload_bytes_removed'],
            compact_initial_upload_bytes=check['compact_initial_upload_bytes'],net_upload_bytes_removed=check['net_upload_bytes_removed']))
    return dict(scope=__doc__,timing_limit='One observation per role; differing SCF counts confound whole-run attribution. The atomic basis setup timer includes both Fortran table packing and native staging; nested routine timers overlap.',
        comparisons=rows,qualification_sha256=sha(gate),verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();result=measure(a.run)
    (a.run/'atomic-measurements.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'atomic-measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in result['comparisons']:
        values=[]
        for r in row['measurements'].values():values.append((round(r['wall_seconds'],3),r['scf_iterations'],r['routines']['dfscf']['inclusive_seconds'],r['routines']['FINAL_HF']['inclusive_seconds']))
        print(row['case'],values,'net upload removed',row['net_upload_bytes_removed'])
