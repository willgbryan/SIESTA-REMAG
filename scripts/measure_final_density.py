#!/usr/bin/env python3
"""Extract same-allocation density-output ablations from raw phase counters/timers."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def measure(root):
    rows=[];files={}
    meta=json.loads((root/'metadata.json').read_text())
    if not meta.get('final_density_ablation'):raise ValueError('A same-allocation density control is required')
    for case in meta['cases']:
        variants={}
        for role in ('candidate','candidate-without-final-density'):
            folder=root/case/role;result=json.loads((folder/'result.json').read_text());trace=json.loads((folder/'remag-residency.json').read_text())
            boundaries={};timers={}
            for interval in trace['intervals']:
                if interval['phase']!='analysis_export':continue
                for name,counts in interval['boundaries'].items():
                    target=boundaries.setdefault(name,dict(host_to_device_bytes=0,device_to_host_bytes=0,copy_calls=0))
                    for key,value in counts.items():target[key]+=value
            for routine in result['profile']['routines']:
                if '/Analysis' in routine['path']:timers[routine['path']]=routine['inclusive_seconds']
            variants[role]=dict(valid=result['valid'],wall_seconds=result['wall_seconds'],iterations=result['observables']['scf_iterations'],
                               analysis_inclusive_seconds=timers,boundaries=boundaries,
                               totals={k:sum(v[k] for v in boundaries.values()) for k in ('host_to_device_bytes','device_to_host_bytes','copy_calls')})
            for name in ('result.json','remag-residency.json','time.json'):
                files[str((folder/name).relative_to(root))]=sha(folder/name)
        ablation=root/f'ablation-final-density-{case}.json';delta=json.loads(ablation.read_text());files[ablation.name]=sha(ablation)
        a=variants['candidate'];b=variants['candidate-without-final-density']
        rows.append(dict(case=case,scientific_passed=delta['passed'],variants=variants,
                         analysis_transfer_reduction={k:b['totals'][k]-a['totals'][k] for k in a['totals']}))
    return dict(scope='Single cold observations with sequential same-allocation density recomputation controls. Inclusive timers overlap. Whole-run timing includes differing SCF work when iterations differ. Explicit copies only; untraced.',
                measurements=rows,evidence_sha256=files,metadata_sha256=sha(root/'metadata.json'),script_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);a=parser.parse_args();d=measure(a.run)
    (a.run/'final-density-measurement.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'final-density-measurement.json').write_text(json.dumps(d,indent=2)+'\n')
    for row in d['measurements']:
        print(row['case'],row['analysis_transfer_reduction'])
