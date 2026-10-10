#!/usr/bin/env python3
"""Recheck production GPU density reuse, per-geometry physics and boundaries.

Initial input staging and explicit status/coefficient/final output are outside
SCF. This does not qualify hidden CUDA transfers or whole-process I/O freedom.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import tarfile
import numpy as np
from siesta_remag.compare import TOLERANCES
from siesta_remag.density_history import record_errors, policy
from siesta_remag.output import parse_output, floats


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory(path, atoms):
    text=path.read_text()
    starts=list(re.finditer(r'^new_DM -- step:\s*(\d+)',text,re.M))
    if not starts or len({m[1] for m in starts})!=len(starts):raise ValueError('Missing/repeated geometry starts')
    steps=[]
    for i,start in enumerate(starts):
        segment=text[start.start():starts[i+1].start() if i+1<len(starts) else len(text)]
        converged=re.findall(r'^SCF cycle converged after (\d+) iterations',segment,re.M)
        if len(converged)!=1:raise ValueError('History geometry did not converge exactly once')
        parsed=parse_output(segment)
        # Intermediate CG steps print a six-component kbar tensor; the final
        # analysis also prints the full tensor. Compare one consistent form
        # per geometry using the unchanged eV/Ang^3 tolerance.
        voigt=re.findall(r'^Stress tensor Voigt\[x,y,z,yz,xz,xy\] \(kbar\):([^\n]+)',segment,re.M)
        if voigt:
            values=floats(voigt[-1])
            if len(values)!=6:raise ValueError('Incomplete intermediate stress tensor')
            parsed['stress_ev_ang3']=(np.asarray(values)/1602.176634).tolist()
        forces=np.asarray(parsed['forces_ev_ang'])
        if forces.shape!=(atoms,3) or not np.isfinite(forces).all():raise ValueError('Missing history geometry forces')
        counts=re.findall(r'Number of DMs in history:\s*(\d+)',segment)
        blocks=re.findall(r'DM extrapolation coefficients:[ \t]*\n((?:[ \t]*\d+[ \t]+[-+\d.]+[ \t]*\n)+)',segment)
        coefficients=[]
        if counts:
            if len(counts)!=1 or len(blocks)!=1:raise ValueError('History coefficient output coverage differs')
            rows=[line.split() for line in blocks[0].splitlines()]
            if [int(row[0]) for row in rows]!=list(range(1,int(counts[0])+1)):raise ValueError('History coefficient order differs')
            coefficients=[float(row[1]) for row in rows]
            if not np.isfinite(coefficients).all():raise ValueError('Nonfinite history coefficients')
        elif blocks:raise ValueError('Coefficients without history reuse')
        steps.append(dict(geometry=int(start[1]),iterations=int(converged[0]),coefficients=coefficients,output=parsed))
    return steps


def compare_trajectories(reference, candidate, records):
    if len(reference)!=len(candidate) or len(candidate)!=len(records):raise ValueError('History trajectory coverage differs')
    maxima={key:0. for key in ('forces_ev_ang','stress_ev_ang3','total_moment_muB','atom_moments_muB','atom_net_charges_e')}
    coefficient_error=0.;nontrivial=0
    for a,b,r in zip(reference,candidate,records):
        if a['geometry']!=b['geometry']:raise ValueError('History geometry order differs')
        n=r['previous_count'] if r['reused'] else 0
        if len(a['coefficients'])!=n or len(b['coefficients'])!=n:raise ValueError('Printed history count differs from native owner')
        if n:
            error=float(np.max(np.abs(np.asarray(a['coefficients'])-b['coefficients'])))
            # Both upstream and candidate print coefficients with five decimals.
            if error>1e-5+1e-12:raise ValueError('History coefficient differs by more than one print quantum')
            coefficient_error=max(coefficient_error,error)
            nontrivial+=int(n==r['depth'] and n>1 and sum(abs(v)>1e-4 for v in a['coefficients'])>1)
        for key in maxima:
            av,bv=np.asarray(a['output'][key]),np.asarray(b['output'][key])
            if av.shape!=bv.shape or av.size==0 or av.dtype.kind not in 'fi' or bv.dtype.kind not in 'fi' or not np.isfinite(av).all() or not np.isfinite(bv).all():
                raise ValueError('Missing/nonfinite per-geometry '+key)
            error=float(np.max(np.abs(av-bv)))
            if error>TOLERANCES[key]:raise ValueError('Per-geometry '+key+' exceeds unchanged scientific tolerance')
            maxima[key]=max(maxima[key],error)
    return dict(maximum_per_geometry_errors=maxima,maximum_printed_coefficient_error=coefficient_error,
        nontrivial_full_depth_uses=nontrivial)


def verify(root):
    meta,ledger=read(root/'metadata.json'),read(root/'ledger.json');checks=[];errors=[];gates={}
    if not meta.get('gpu_history'):errors.append('GPU history was not requested')
    for name in ('source-verification.json','scientific-recheck.json','device-cycle-verification.json','initial-state-verification.json'):
        path=root/name
        if not path.is_file() or read(path).get('passed') is not True:errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():gates[name]=sha(path)
    module=Path(inspect.getsourcefile(record_errors))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        if archive.extractfile('src/siesta_remag/density_history.py').read()!=module.read_bytes():
            raise ValueError('History validator differs from executed source')
    for row in ledger:
        folder=root/row.get('candidate_path',row['benchmark']+'/candidate')
        reference=root/row.get('reference_path',row['benchmark']+'/reference')
        result=read(folder/'result.json');native=read(folder/'remag-density-history.json');records=native['cycles']
        cycles=read(folder/'remag-device-scf.json')['cycles'];initial=read(folder/'remag-initial-state.json')['cycles']
        forces=[read(p) for p in sorted(folder.glob('remag-integral-force-*.json'))]
        issues=record_errors(records,cycles,initial,forces,policy((folder/'input.fdf').read_text()))
        if (not row['passed'] or not result['valid'] or native['schema']!=1 or
                result['runtime_settings'].get('REMAG_GPU_HISTORY')!='1' or result['telemetry'].get('density_history')!=records):
            issues.append('Invalid or mismatched production history evidence')
        physics={}
        try:
            a=trajectory(reference/'stdout.log',result['benchmark']['atoms']);b=trajectory(folder/'stdout.log',result['benchmark']['atoms'])
            physics=compare_trajectories(a,b,records)
            if [r['iterations'] for r in b]!=[c['iterations'] for c in cycles]:raise ValueError('History SCF replay differs')
            if result['benchmark'].get('density_history_depth',1)>1 and physics['nontrivial_full_depth_uses']==0:
                raise ValueError('Requested history trajectory lacks nontrivial full-depth reuse')
        except (ValueError,KeyError,TypeError) as e:issues.append(str(e))
        keys=('host_to_device_bytes','device_to_host_bytes','copy_calls')
        totals={name:dict.fromkeys(keys,0) for name in ('density_history_setup','density_history_output')}
        for interval in read(folder/'remag-residency.json')['intervals']:
            for name,counts in interval['boundaries'].items():
                if name not in totals:continue
                if interval['phase'] not in ('initialization','post_scf'):issues.append('History transfer inside SCF or unknown phase')
                for key in keys:totals[name][key]+=counts[key]
        expected=dict(density_history_setup=dict(zip(keys,(sum(r['setup_bytes'] for r in records),0,sum(r['setup_copies'] for r in records)))),
            density_history_output=dict(zip(keys,(0,sum(r['initial_status_bytes']+r['post_status_bytes']+r['coefficient_export_bytes'] for r in records),
                sum(r['initial_status_copies']+r['post_status_copies']+bool(r['coefficient_export_bytes']) for r in records)))))
        if totals!=expected:issues.append('History boundary counters differ from raw explicit copies')
        paths=[folder/n for n in ('result.json','input.fdf','stdout.log','remag-density-history.json','remag-device-scf.json','remag-initial-state.json','remag-residency.json')]
        paths+=[reference/'stdout.log',*sorted(folder.glob('remag-integral-force-*.json'))]
        checks.append(dict(case=row['benchmark'],role=folder.name,passed=not issues,errors=issues,physics=physics,traffic=totals,
            evidence_sha256={str(p.relative_to(root)):sha(p) for p in paths}))
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
        prerequisite_sha256=gates,verifier_sha256=sha(Path(__file__)),strict_residency_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    report=verify(args.run)
    (args.run/'density-history-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'density-history-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Production density history:',len(report['checks']),'passed=',report['passed'])
    if not report['passed']:print(report['errors'],[r for r in report['checks'] if not r['passed']]);raise SystemExit(1)
