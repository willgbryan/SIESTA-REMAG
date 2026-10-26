"""Recheck fetched evidence, including partial runs; never execute calculations."""
from pathlib import Path
import sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent),str(ROOT/'build/upstream-gpu-baseline-v109')]
import analyze
from comparison_inputs import verify_convergence
from verify_exports import compare_exports
from siesta_remag.compare import compare,difference
from siesta_remag.frozen_reference import evidence_errors
run=ROOT/'results'/sys.argv[1]
load=lambda p:json.loads(p.read_text())
save=lambda name,value:(run/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
meta=load(run/'metadata.json');plan=meta['plan'];source=ROOT/'results'/plan['source_run']
prior=ROOT/'results'/plan['prior_diagnostic_run']
marker=load(run/'run-attempt.json');fetch=load(run/'fetch-attempt.json')
assert marker['state']=='complete' and fetch['consistent'] and fetch['after']['attempt_id']==marker['attempt_id']
records=[];results={};probes={};modes={};biases={};directional=[]
for row in load(run/'ledger.json'):
 name,role=row['case'],row['role'];dest=run/name/role
 spec=next(s for s in plan['cases'] if s['case']==name and s['role']==role)
 try:
  result=load(dest/'result.json');results[name,role]=result
  conv=verify_convergence(dest,expected={'DM.Tolerance':1e-9,'SCF.DM.Tolerance':1e-9,'SCF.H.Tolerance':1e-8})
  errors=evidence_errors(result['telemetry'],spec['soc_free'])
  obs=result['observables']
  report=dict(case=name,role=role,valid=result['valid'],convergence=conv,reference_errors=errors,
    scf_observables={k:obs.get(k) for k in ['total_energy_ev','free_energy_ev','total_moment_muB','atom_moments_muB']},
    scf_soc_mode='soc_free_reference' if spec['soc_free'] else 'full_soc',
    spin_magnitude_muB=float(np.linalg.norm(obs['total_moment_muB'])),
    passed=bool(row['valid'] and result['valid'] and conv['passed'] and not errors))
  if not spec['soc_free']:
   refdir=source/name/'energy_only';reference=load(refdir/'result.json')
   science=compare(reference,result,reference_dir=refdir,candidate_dir=dest,electronic_only=True)
   science.update(end_to_end_speedup=None,performance_comparable=False)
   exports=compare_exports(refdir,dest)
   report.update(science=science,exports=exports);report['passed'] &= science['passed'] and exports['passed']
   if spec['material']=='fept':
    cpu=source/name/'cpu';control=compare(load(cpu/'result.json'),result,reference_dir=cpu,candidate_dir=dest,electronic_only=True)
    report['historical_pristine_cpu_passed']=control['passed'];report['passed'] &= control['passed']
  if 'rotation_file' in spec:
   probe=analyze.probe(dest,result,plan['materials'][spec['material']]['kpoints'],not role.endswith('_values'),analyze.input_directions(spec,run/'inputs'))
   report['probe']=probe;report['passed'] &= probe['passed'];probes[name,role]=probe['report']
   if not spec['soc_free']:
    default=analyze.compare_modes(prior/name/'frozen_spin',dest)
    report['default_probe_regression']=default;report['passed'] &= default['passed']
  records.append(report)
 except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
  records.append(dict(case=name,role=role,passed=False,error=str(error)))
 print(name,role,'passed=',records[-1]['passed'],flush=True)
def anchor(name):return results.get((name,'energy_only')) or results.get((name,'frozen_spin'))
for spec in plan['cases']:
 name,role=spec['case'],spec['role']
 if role=='soc_free_values' and (name,role) in results and (name,'soc_free_spin') in results:
  modes[name]=analyze.compare_modes(run/name/'soc_free_spin',run/name/role)
 if role not in ('soc_free_spin','frozen_spin') or (name,role) not in probes:continue
 names=[name]+spec['targets'];anchors=[anchor(n) for n in names]
 if all(anchors):biases[name+'-'+role]=analyze.bias(probes[name,role],anchors)
for material in ('fept','feco'):
 axes=plan['materials'][material]['axes']
 for target in axes[1:]:
  if not anchor(axes[0]) or not anchor(target):continue
  a=anchor(axes[0])['observables'];b=anchor(target)['observables']
  oa=load(source/axes[0]/'energy_only/result.json')['observables'];ob=load(source/target/'energy_only/result.json')['observables']
  for key in ('total_energy_ev','free_energy_ev'):
   directional.append(dict(material=material,target=target,quantity=key,**difference(ob[key]-oa[key],b[key]-a[key],5e-6)))
axes=plan['materials']['feco']['axes'];spread=None
if all((n,'soc_free_spin') in probes for n in axes[:2]):
 spread=analyze.reference_sensitivity(plan['cases'],probes,'feco',axes)
save('local-method-bias.json',biases);save('local-reference-sensitivity.json',spread)
matrix=load(run/'local-independent-matrices.json') if (run/'local-independent-matrices.json').exists() else dict(passed=False)
summary=load(run/'summary.json');remote=load(run/'current-comparisons.json')
report=dict(run=run.name,attempt_id=marker['attempt_id'],status=summary['status'],stopping_reason=summary['stopping_reason'],
 completed_calculations=sum(r['passed'] for r in records),recorded_calculations=len(records),planned_calculations=len(plan['cases']),
 remote_checks_passed=bool(remote) and all(r['passed'] for r in remote),
 local_checks_passed=bool(records) and all(r['passed'] for r in records) and matrix['passed'] and all(r['passed'] for r in modes.values()) and all(r['passed'] for r in directional),
 checks=records,eigenvalues_only=modes,full_scf_directional_checks=directional,independent_matrices_passed=matrix['passed'],
 physical_accuracy_qualified=False,method_replacement_qualified=False,speedup_over_upstream_gpu=None,
 temperature_k=300,production_k_guard=128,staged_admission_ceiling=32768,capacity_qualified=False)
save('local-qualification.json',report)
print({k:v for k,v in report.items() if k not in ('checks','eigenvalues_only','full_scf_directional_checks')})
