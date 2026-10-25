from pathlib import Path
import sys,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'build/mae-frozen-reference-20261025'),str(ROOT/'build/upstream-gpu-baseline-v109')]
import analyze
from comparison_inputs import verify_convergence
from verify_exports import compare_exports
from siesta_remag.compare import compare
run=ROOT/'results'/sys.argv[1]
load=lambda p:json.loads(p.read_text())
meta=load(run/'metadata.json');plan=meta['plan'];source=ROOT/'results'/plan['source_run']
marker=load(run/'run-attempt.json');fetch=load(run/'fetch-attempt.json')
assert marker['state']=='complete' and fetch['consistent'] and fetch['after']['attempt_id']==marker['attempt_id']
records=[]
for row in load(run/'ledger.json'):
 name,role=row['case'],row['role'];dest=run/name/role;prior=source/name/'energy_only'
 result=load(dest/'result.json');reference=load(prior/'result.json')
 science=compare(reference,result,reference_dir=prior,candidate_dir=dest,electronic_only=True)
 science.update(end_to_end_speedup=None,performance_comparable=False)
 exports=compare_exports(prior,dest)
 convergence=verify_convergence(dest,expected={'DM.Tolerance':1e-9,'SCF.DM.Tolerance':1e-9,'SCF.H.Tolerance':1e-8})
 material=next(s['material'] for s in plan['cases'] if s['case']==name)
 probe=analyze.probe(dest,result,plan['materials'][material]['kpoints'],role=='frozen_spin')
 obs=result['observables'];spin=obs['total_moment_muB']
 report=dict(case=name,role=role,science=science,exports=exports,convergence=convergence,probe=probe,
   full_scf_observables={k:obs.get(k) for k in ['total_energy_ev','free_energy_ev','total_moment_muB','atom_moments_muB']},
   full_scf_spin_magnitude_muB=float(np.linalg.norm(spin)),passed=all([row['valid'],science['passed'],exports['passed'],convergence['passed'],probe['passed']]))
 if material=='fept':
  cpu=source/name/'cpu';control=compare(load(cpu/'result.json'),result,reference_dir=cpu,candidate_dir=dest,electronic_only=True)
  report['historical_pristine_cpu_passed']=control['passed'];report['passed'] &= control['passed']
 records.append(report)
 print(name,role,'passed=',report['passed'],flush=True)
modes={}
for spec in plan['cases']:
 if spec['role']=='frozen_values' and (run/spec['case']/'frozen_values/result.json').exists():
  modes[spec['case']]=analyze.compare_modes(run/spec['case']/'frozen_spin',run/spec['case']/'frozen_values')
matrix=load(run/'local-independent-matrices.json')
summary=load(run/'summary.json');remote=load(run/'current-comparisons.json')
report=dict(run=run.name,attempt_id=marker['attempt_id'],status=summary['status'],stopping_reason=summary['stopping_reason'],
 completed_calculations=len(records),planned_calculations=len(plan['cases']),remote_checks_passed=all(r['passed'] for r in remote),
 local_checks_passed=all(r['passed'] for r in records) and matrix['passed'] and all(r['passed'] for r in modes.values()),
 checks=records,eigenvalues_only=modes,independent_matrices_passed=matrix['passed'],
 eigenvalues_only_error_units='eV, including legacy keys ending _ry',
 physical_accuracy_qualified=False,method_replacement_qualified=False,speedup_over_upstream_gpu=None,
 temperature_k=300,production_k_guard=128,staged_admission_ceiling=32768,capacity_qualified=False)
(run/'local-qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in report.items() if k not in ('checks','eigenvalues_only')})
