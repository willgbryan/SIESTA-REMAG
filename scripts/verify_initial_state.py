#!/usr/bin/env python3
"""Audit initial population output and the device density lease into production SCF."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import tarfile
from siesta_remag.initial_state import record_errors
from siesta_remag.output import read_eigenvalues


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def initial_outputs(path):
    text=path.read_text();outputs=[]
    starts=list(re.finditer(r'^(?:Initializing Density Matrix\.\.\.|Re-using DM from previous geometries\.\.\.)\s*$',text,re.M))
    for i,start in enumerate(starts):
        segment=text[start.end():starts[i+1].start() if i+1<len(starts) else len(text)]
        header=re.search(r'^\s*iscf\s+Eharris',segment,re.M)
        if header is None:raise ValueError('Missing SCF header after initialization')
        segment=segment[:header.start()]
        spin=re.findall(r'^\s*spin moment:.*$',segment,re.M)
        if len(spin)!=1:raise ValueError('Initial spin output missing or duplicated')
        parts=segment.split('mulliken: Atomic and Orbital Populations:')
        if len(parts)!=2:raise ValueError('Initial Mulliken table missing or duplicated')
        table=parts[1];partial=table.index('Mulliken Atomic Populations:')
        end=re.search(r'^\s*Total[ \t]+[^\n]+$',table[partial:],re.M)
        if end is None:raise ValueError('Initial partial-charge total missing')
        outputs.append((spin[0]+'\n'+table[:partial+end.end()]).split())
    return outputs


def compare_output(a,b):
    if len(a)!=len(b):raise ValueError('Initial population layout differs')
    numeric=0;maximum=0.
    for x,y in zip(a,b):
        if re.fullmatch(r'[-+]?\d+\.\d+',x):
            value=float(y);error=abs(float(x)-value)
            # Existing charge/spin gate plus one print quantum. No change to
            # the separate higher-precision final scientific gates.
            if not math.isfinite(value) or error>1e-4+10.**(-len(x.split('.')[1]))+1e-12:
                raise ValueError('Initial population value differs')
            numeric+=1;maximum=max(maximum,error)
        elif x!=y:raise ValueError('Initial population labels/order differ')
    if not numeric:raise ValueError('Empty initial population output')
    return dict(values=numeric,max_printed_absolute_error=maximum)


def verify(root):
    meta,ledger=read(root/'metadata.json'),read(root/'ledger.json');errors=[];checks=[];gates={}
    for name in ('source-verification.json','scientific-recheck.json','device-cycle-verification.json','initial-state-component-verification.json'):
        path=root/name
        if not path.is_file() or read(path).get('passed') is not True:errors.append('Missing/failed prerequisite: '+name)
        if path.is_file():gates[name]=sha(path)
    if not (meta.get('gpu_initialization') and meta.get('population') and meta.get('initial_populations')):
        errors.append('Initial output validation mode is absent')
    with tarfile.open(root/'implementation.tar.gz') as archive:
        validator=Path(inspect.getsourcefile(record_errors))
        assert hashlib.sha256(archive.extractfile('src/siesta_remag/initial_state.py').read()).hexdigest()==sha(validator)
    for row in ledger:
        folder=root/row.get('candidate_path',row['benchmark']+'/candidate')
        reference=root/row.get('reference_path',row['benchmark']+'/reference')
        result=read(folder/'result.json');native=read(folder/'remag-initial-state.json');cycles=read(folder/'remag-device-scf.json')['cycles']
        histories=read(folder/'remag-density-history.json')['cycles'] if meta.get('gpu_history') else None
        records=native['cycles'];issues=record_errors(records,cycles,result['benchmark']['atoms'],read_eigenvalues(folder/'remag.EIG'),len(result['benchmark']['pseudopotentials']),histories)
        if not row['passed'] or not result['valid'] or native['schema']!=1 or result['telemetry'].get('initial_state')!=records:
            issues.append('Invalid or mismatched initial owner evidence')
        if any(r['requests']!=3 for r in records) or not result['benchmark'].get('initial_population_output'):
            issues.append('Initial spin and Mulliken requests are missing')
        formatting=[]
        try:
            a,b=initial_outputs(reference/'stdout.log'),initial_outputs(folder/'stdout.log')
            if len(a)!=len(cycles) or len(b)!=len(cycles):raise ValueError('Initial population geometry coverage differs')
            formatting=[compare_output(x,y) for x,y in zip(a,b)]
        except (ValueError,IndexError) as e:issues.append(str(e))
        def counter(h=0,d=0,n=0):return dict(host_to_device_bytes=h,device_to_host_bytes=d,copy_calls=n)
        totals={k:counter() for k in ('initial_density_setup','initial_population_setup','initial_population_output')}
        for interval in read(folder/'remag-residency.json')['intervals']:
            for name,counts in interval['boundaries'].items():
                if name not in totals:continue
                if interval['phase']!='initialization':issues.append('Initial density/population transfer outside initialization')
                for key,value in counts.items():totals[name][key]+=value
        expected=dict(initial_density_setup=counter(h=sum(r['density_upload_bytes'] for r in records),n=sum(bool(r['density_upload_bytes']) for r in records)),
            initial_population_setup=counter(h=sum(r['population_setup_bytes'] for r in records),n=8*len(records)),
            initial_population_output=counter(d=sum(r['status_export_bytes']+sum(r['export_bytes']) for r in records),
                n=sum(1+sum(bool(v) for v in r['export_bytes']) for r in records)))
        if totals!=expected:issues.append('Initial staging/requested export copy totals differ')
        files=[folder/n for n in ('result.json','remag-initial-state.json','remag-device-scf.json','remag-residency.json','stdout.log','input.fdf')]+[reference/'stdout.log']
        checks.append(dict(case=row['benchmark'],role=folder.name,passed=not issues,errors=issues,geometries=len(cycles),initial_output=formatting,
            traffic=totals,expected_traffic=expected,evidence_sha256={str(p.relative_to(root)):sha(p) for p in files}))
    return dict(scope=__doc__,passed=bool(checks) and not errors and all(c['passed'] for c in checks),errors=errors,checks=checks,
        prerequisite_sha256=gates,verifier_sha256=sha(Path(__file__)),production_history_integrated=bool(meta.get('gpu_history')),strict_residency_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    report=verify(args.run)
    (args.run/'initial-state-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'initial-state-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Initial device state:',len(report['checks']),'passed=',report['passed'])
    if not report['passed']:print(report['errors'],[r for r in report['checks'] if not r['passed']]);raise SystemExit(1)
