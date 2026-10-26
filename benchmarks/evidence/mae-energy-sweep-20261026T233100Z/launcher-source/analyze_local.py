"""Recheck completed/partial downloaded evidence without executing calculations."""
from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent), str(ROOT/'build/upstream-gpu-baseline-v109')]
import analyze
from siesta_remag.output import parse_output, read_eigenvalues, read_geometry

run = ROOT/'results'/sys.argv[1]
metadata = analyze.read(run/'metadata.json'); plan = metadata['plan']
source = ROOT/'results'/plan['source_run']
marker = analyze.read(run/'run-attempt.json'); fetch = analyze.read(run/'fetch-attempt.json')
assert marker['state'] == 'complete' and fetch['consistent'] and fetch['after']['attempt_id'] == marker['attempt_id']
ledger = analyze.read(run/'ledger.json'); remote = analyze.read(run/'current-comparisons.json')
summary = analyze.read(run/'summary.json'); records = []; results = {}; pairs = []
for row in ledger:
    spec = next(s for s in plan['cases'] if s['label'] == row['label'])
    folder = run/row['label']; names,directions = analyze.axes(spec,plan)
    report = dict(label=row['label'],passed=False)
    try:
        if spec['role'] == 'sweep':
            receipt, outputs, paths = analyze.sweep(folder,spec,plan,run/'inputs')
        else:
            result = copy.deepcopy(analyze.read(folder/'result.json'))
            output = parse_output((folder/'stdout.log').read_text(errors='replace'))
            output['eigenvalues'] = read_eigenvalues(folder/'remag.EIG')
            output['geometry'] = read_geometry(folder/'remag.XV')
            result['observables'] = output
            result['valid'] &= output['completed'] and output['converged']
            receipt = analyze.cold_checks(folder,result,directions[0],plan['materials'][spec['material']]['kpoints'])
            outputs,paths = [result],[folder]
        checks=[]; references=[]
        for name,result,path in zip(names,outputs,paths):
            refdir=source/name/'energy_only'; reference=analyze.read(refdir/'result.json'); references.append(reference)
            checks.append(analyze.regression(reference,result,refdir,path))
            if spec['material']=='fept':
                cpu=source/name/'cpu'
                checks.append(analyze.regression(analyze.read(cpu/'result.json'),result,cpu,path))
        directional=analyze.directional(references,outputs)
        report.update(passed=row['passed'] and receipt['passed'] and all(c['passed'] for c in checks+directional),
            execution_checks=receipt,comparisons=checks,directional=directional,
            observables=[{key:r['observables'].get(key) for key in ('total_energy_ev','free_energy_ev','total_moment_muB','scf_iterations')} for r in outputs])
        results[row['label']] = (outputs,paths)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        report['error'] = str(exc)
    records.append(report)
    print(row['label'],'passed=',report['passed'],flush=True)
for paired in analyze.read(run/'paired-timings.json'):
    material,rep=paired['material'],paired['repetition']
    cold=sorted([r for r in ledger if r['material']==material and r['repetition']==rep and r['role']=='cold'], key=lambda r:r['axis_indices'])
    hot=next(r for r in ledger if r['material']==material and r['repetition']==rep and r['role']=='sweep')
    report=dict(material=material,repetition=rep,passed=False)
    try:
        references=[results[r['label']][0][0] for r in cold]
        outputs,paths=results[hot['label']]
        checks=[analyze.regression(ref,result,results[row['label']][1][0],path)
                for ref,result,row,path in zip(references,outputs,cold,paths)]
        directional=analyze.directional(references,outputs)
        report.update(passed=paired['passed'] and all(c['passed'] for c in checks+directional),comparisons=checks,directional=directional)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        report['error']=str(exc)
    pairs.append(report)
report=dict(run=run.name,attempt_id=marker['attempt_id'],status=summary['status'],stopping_reason=summary['stopping_reason'],
    local_checks_passed=bool(records) and all(r['passed'] for r in records) and all(r['passed'] for r in pairs),
    remote_checks_passed=bool(remote) and all(r['passed'] for r in remote),
    planned_executions=len(plan['cases']),recorded_executions=len(ledger),checks=records,paired_checks=pairs,
    physical_accuracy_qualified=False,speedup_over_upstream_gpu=None,production_k_guard=128,staged_admission_ceiling=32768)
analyze.save(run/'local-qualification.json',report)
print({k:v for k,v in report.items() if k not in ('checks','paired_checks')})
