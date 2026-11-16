"""Synthetic study/native transports and explicit imported usage, no services."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime,timedelta

import pytest

from test_research import experiment
from test_research_execution import prepared
from test_research_evaluation import endpoints
from test_discovery import backend
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.experiments import prepare_experiment, run_experiment
from siesta_remag.research.study import register_study, run_study_arm, evaluate_study
from siesta_remag.research.discovery_scores import score_usage, validate_policy


def policy():
    return dict(assessment_policy=dict(format='GPAC-PROPOSAL-POLICY/1',cutoff_angstrom='3',
        minimum_distance_angstrom='1',max_sites=64,max_orbits=64), excluded_findings=['short_distance'],
        controls=[dict(varied='kmesh',tolerance_ev=5e-5,successive_changes=2,transverse_tolerance_ev=1e-5)])


def make_candidate_snapshot(tmp_path, discovery=None):
    bundle=experiment.__wrapped__(tmp_path)
    for arm in ('scalar','physics'):
        save(tmp_path/('arm-'+arm)/'common.json',load(bundle/'structure.json'))
    script=tmp_path/'policy.py'
    script.write_text('''import json,sys
from pathlib import Path
c=json.load(sys.stdin); n=len(c['history']); source=json.load(open(Path(__file__).parent/('arm-'+c['information_profile'])/'common.json'))
if n<2:
 a={'tool':'author_structure','args':{'request':{'op':'inspect','program':source['source']['program']},'output':'candidate-'+str(n)},'rationale':'Retain an explicitly selected exact candidate'}
elif n==2:
 a={'tool':'author_structure','args':{'structure':'candidate-0/structure.json','request':{'op':'physical-isotropic','expected_sha256':source['source']['sha256'],'scale':'101/100'},'output':'candidate-2'},'rationale':'Test a distinct exact intervention'}
else:a={'finish':'Synthetic authoring cohort; no scientific discovery claimed.'}
print(json.dumps(a))
''')
    d=dict(format='REMAG-PAIRED-STUDY/1',objective='Synthetic candidate comparison',
        policy_command=[sys.executable,str(script)],policy_artifacts=['policy.py'],common_inputs=['common.json'],
        limits=dict(max_steps=8,max_seconds=30,allow_cpu_relaxation=False),
        arms={a:dict(workspace='arm-'+a,execution_config=None) for a in ('scalar','physics')},
        evaluation=dict(heldout_sources=['c'*64],observables={'anisotropy.total_energy_ev':dict(unit='eV/cell',resolution=5e-5)},discovery=discovery or policy()))
    register_study(tmp_path,d,'study')
    for arm in ('scalar','physics'):
        result=run_study_arm(tmp_path,'study',arm)
        assert result['status']=='completed',result
    exe=tmp_path/'binary';exe.write_text('injected native transport only')
    export=load(bundle/'structure.json');source=export['source']['sha256'];reports=[]
    for k in [3,4,5]:
        bundle=tmp_path/('validation-input-'+str(k))
        prepare_experiment(export,tmp_path/'pseudos',bundle,mesh=[k]*3)
        result=run_experiment(bundle,tmp_path/('validation-'+str(k)),exe,executable_sha256=sha256(exe),runner=backend([]))
        assert result['status']=='measured',result
        reports.append(dict(report='validation-'+str(k)+'/report.json'))
    evidence=dict(format='REMAG-STUDY-CANDIDATE-EVIDENCE/1',arms={a:{source:dict(report=reports[-1]['report'],controls=dict(kmesh=reports))} for a in ('scalar','physics')})
    return tmp_path,dict(definition=d,evidence=evidence,source=source,reports=reports)


@pytest.fixture(scope='module')
def candidate_snapshot(tmp_path_factory):
    return make_candidate_snapshot(tmp_path_factory.mktemp('candidate-study'))

@pytest.fixture
def candidate_study(candidate_snapshot,tmp_path):
    root,fixture=candidate_snapshot
    shutil.copytree(root,tmp_path,dirs_exist_ok=True)
    return copy.deepcopy(fixture)


def measured_usage(root,reports):
    allocations=[]
    for i,r in enumerate(reports):
        start=datetime.fromisoformat(load(root/r['report'])['started_at'])
        allocations.append(dict(allocation_id='ta-'+str(i),app_id='ap-'+str(i),gpu_type='A100-40GB',gpu_count=1,
            started_at=start.isoformat(),stopped_at=(start+timedelta(hours=1)).isoformat(),report_sha256s=[sha256(root/r['report'])]))
    path=root/'usage-source.json';save(path,dict(format='REMAG-ALLOCATION-MEASUREMENTS/1',provider='synthetic-provider',allocations=allocations))
    return dict(format='REMAG-STUDY-GPU-USAGE/1',records=[dict(arm='shared',phase='validation',source='usage-source.json',source_sha256=sha256(path),row=i) for i in range(len(allocations))])


def test_complete_cohort_density_units_partial_cases_and_imported_efficiency(candidate_study,tmp_path):
    s=candidate_study
    pending=evaluate_study(tmp_path,'study','pending.json')
    for a in ('scalar','physics'):
        rows=pending['independent_candidate_validation']['arms'][a]
        assert rows['authored_attempts']==3 and rows['distinct_exact_sources']==2
        assert rows['pending_validation']==2 and rows['validated_exact_sources']==0
    usage=measured_usage(tmp_path,s['reports'])
    r=evaluate_study(tmp_path,'study','scores.json',discovery_evidence=s['evidence'],usage=usage)
    for a in ('scalar','physics'):
        cohort=r['independent_candidate_validation']['arms'][a]
        assert cohort['validated_exact_sources']==1 and cohort['pending_validation']==1
        row=next(x for x in cohort['rows'] if x['source_sha256']==s['source'])
        assert row['authoring_steps']==[0,1]
        assert row['validated_under_declared_controls'] and row['post_session_run_recorded']
        assert row['controls']['kmesh']['status']=='passed'
        p=row['properties'];assert p['E_anisotropy_ev_atom']==pytest.approx(p['E_anisotropy_ev_cell']/2)
        assert p['E_anisotropy_MJ_m3']==pytest.approx(p['E_anisotropy_ev_cell']/27*160217.6634)
        assert p['spin_polarization_tesla']==pytest.approx(p['spin_moment_muB']/27*11.654064772740647)
        assert row['thermodynamic_stability'] is None and not row['physical_accuracy_qualified']
        assert r['arms'][a]['actual_gpu_hours']==1.5
        assert r['validated_candidates_per_gpu_hour'][a]==pytest.approx(2/3)
        assert r['arms'][a]['actual_billed_usd'] is None and r['arms'][a]['reserved_cents']==0
    assert not r['scientific_advantage_demonstrated'] and not r['empirical_equal_compute_verified']
    assert r['measured_allocation_usage']['same_imported_gpu_time_by_type']
    assert not r['measured_allocation_usage']['provider_authenticated']
    assert r['heldout']['physics']['missing_sources']==['c'*64]


@pytest.mark.parametrize('mutation', ['native','branch','unstable','old_run','legacy_time','missing_control'])
def test_failed_or_incomplete_validation_stays_in_cohort(candidate_study,tmp_path,mutation):
    s=candidate_study;ep=s['reports'][-1];path=tmp_path/ep['report'];r=load(path)
    if mutation=='native':r['status']='held_soc-x';r['stages'][2]['accepted']=False
    elif mutation=='branch':r['stages'][2]['branch']={'passed':False}
    elif mutation=='unstable':
        r['observations']['soc-x']['free_energy_ev']+=.0002;r['observations']['soc-y']['free_energy_ev']+=.0002
    elif mutation=='old_run':r['started_at']='2000-01-01T00:00:00Z'
    elif mutation=='legacy_time':r.pop('started_at')
    else:
        for a in ('scalar','physics'):s['evidence']['arms'][a][s['source']]['controls']={}
    save(path,r)
    scored=evaluate_study(tmp_path,'study','scores.json',discovery_evidence=s['evidence'])
    for a in ('scalar','physics'):
        cohort=scored['independent_candidate_validation']['arms'][a]
        assert cohort['validated_exact_sources']==0 and len(cohort['rows'])==2
        assert not cohort['validated_density_rankings']['E']
        if mutation=='legacy_time':
            row=next(x for x in cohort['rows'] if x['source_sha256']==s['source'])
            assert row['post_session_run_recorded'] is None
            assert row['controls']['kmesh']['status']=='passed'
            assert row['properties'] is not None


@pytest.mark.parametrize('mutation,match',[
    ('source','not authored'),('control','must end'),('repeat','Repeated'),('geometry','geometry differs'),('artifact','binding differs'),('authorship','artifact changed')])
def test_unregistered_or_changed_candidate_evidence_rejected(candidate_study,tmp_path,mutation,match):
    s=candidate_study;v=s['evidence']['arms']['scalar'][s['source']]
    if mutation=='source':s['evidence']['arms']['scalar']['a'*64]=s['evidence']['arms']['scalar'].pop(s['source'])
    elif mutation=='control':v['controls']['kmesh']=s['reports'][:-1]
    elif mutation=='repeat':v['controls']['kmesh']=[s['reports'][-1]]*3
    elif mutation=='geometry':
        p=tmp_path/v['report'];r=load(p);r['structure']['geometry']['cell'][0][0]=999;save(p,r)
    elif mutation=='artifact':(tmp_path/'validation-5/scalar/input.fdf').write_text('changed')
    else:(tmp_path/'arm-scalar/candidate-0/program.xcl').write_text('changed')
    with pytest.raises(ValueError,match=match):evaluate_study(tmp_path,'study','invalid.json',discovery_evidence=s['evidence'])
    assert not (tmp_path/'invalid.json').exists()


def test_geometry_rule_is_separate_from_numerical_validation(candidate_study,tmp_path):
    # The frozen policy is not retroactively replaced just to improve a score.
    d=load(tmp_path/'study/study.json');d['definition']['evaluation']['discovery']['excluded_findings']=['translational_redundancy'];save(tmp_path/'study/study.json',d)
    with pytest.raises(ValueError,match='registration changed'):
        evaluate_study(tmp_path,'study','changed.json',discovery_evidence=candidate_study['evidence'])


def test_cli_keeps_missing_usage_unknown(candidate_study,tmp_path):
    save(tmp_path/'evidence.json',candidate_study['evidence'])
    p=subprocess.run([sys.executable,'-m','siesta_remag.research','study-evaluate','--workspace',str(tmp_path),
        '--study','study','--output','cli.json','--discovery-evidence',str(tmp_path/'evidence.json')],text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    r=json.loads(p.stdout);assert r['validated_candidates_per_gpu_hour'] is None
    assert r['arms']['physics']['actual_gpu_hours'] is None


def usage_arms():
    return {a:dict(session_verified=True,native_app_ids=['ap-0'] if a=='scalar' else [],native_submissions=1 if a=='scalar' else 0,
        native_report_sha256s=['a'*64] if a=='scalar' else [],unresolved_attempts=0) for a in ('scalar','physics')}


@pytest.fixture
def usage(tmp_path):
    raw=dict(format='REMAG-ALLOCATION-MEASUREMENTS/1',provider='synthetic-provider',allocations=[
        dict(allocation_id='ta-0',app_id='ap-0',gpu_type='H100',gpu_count=2,started_at='2026-11-15T00:00:00Z',
            stopped_at='2026-11-15T00:30:00Z',report_sha256s=['a'*64])])
    save(tmp_path/'usage-source.json',raw)
    manifest=dict(format='REMAG-STUDY-GPU-USAGE/1',records=[dict(arm='scalar',phase='search',source='usage-source.json',source_sha256=sha256(tmp_path/'usage-source.json'),row=0)])
    return raw,manifest


def test_usage_counts_devices_preserves_types_and_requires_known_coverage(usage,tmp_path):
    raw,m=usage
    windows={'a'*64:dict(started_at='2026-11-15T00:00:01Z',wall_seconds=20)}
    r=score_usage(tmp_path,usage_arms(),m,report_windows=windows)
    assert r['arms']['scalar']['actual_gpu_hours']==1 and r['arms']['physics']['actual_gpu_hours']==0
    assert r['arms']['scalar']['imported_gpu_hours_by_type']=={'H100':1}
    assert score_usage(tmp_path,usage_arms(),m)['arms']['scalar']['actual_gpu_hours'] is None
    with pytest.raises(ValueError,match='do not contain'):
        score_usage(tmp_path,usage_arms(),m,report_windows={'a'*64:dict(started_at='2026-11-15T00:29:59Z',wall_seconds=20)})
    missing=score_usage(tmp_path,usage_arms(),m,required_reports={'scalar':['b'*64]})
    assert missing['arms']['scalar']['actual_gpu_hours'] is None
    assert missing['arms']['scalar']['missing_report_sha256s']==['b'*64]
    unknown=usage_arms();unknown['scalar']['unresolved_attempts']=1
    assert score_usage(tmp_path,unknown,m)['arms']['scalar']['actual_gpu_hours'] is None
    unknown=usage_arms();unknown['scalar']['native_submissions']=None
    assert score_usage(tmp_path,unknown,m)['arms']['scalar']['actual_gpu_hours'] is None


@pytest.mark.parametrize('mutation,match',[
    ('hash','hash differs'),('duplicate','Duplicate'),('overlap','Overlapping'),('bool','integer GPU'),
    ('active','ISO timestamp'),('negative','positive terminal'),('timezone','timezone'),('escape','escapes'),
    ('shared_search','cannot be shared'),('unknown','Unknown allocation')])
def test_bad_allocation_evidence_rejected(usage,tmp_path,mutation,match):
    raw,m=usage;row=raw['allocations'][0]
    if mutation=='hash':m['records'][0]['source_sha256']='b'*64
    elif mutation=='duplicate':m['records']*=2
    elif mutation=='overlap':
        raw['allocations'].append(row|{'allocation_id':'ta-other'});m['records'].append(m['records'][0]|{'row':1})
    elif mutation=='bool':row['gpu_count']=True
    elif mutation=='active':row['stopped_at']=None
    elif mutation=='negative':row['stopped_at']=row['started_at']
    elif mutation=='timezone':row['started_at']='2026-11-15T00:00:00'
    elif mutation=='escape':m['records'][0]['source']='../outside.json'
    elif mutation=='shared_search':m['records'][0]['arm']='shared'
    else:m['records'][0]['phase']='billing_estimate'
    if mutation not in ('hash','escape'):
        save(tmp_path/'usage-source.json',raw)
        for entry in m['records']:entry['source_sha256']=sha256(tmp_path/'usage-source.json')
    with pytest.raises(ValueError,match=match):score_usage(tmp_path,usage_arms(),m)


def test_validation_policy_is_finite_and_does_not_accept_boolean_pass():
    pytest.importorskip('gpac')
    assert validate_policy(policy())==policy()
    with pytest.raises(ValueError):validate_policy(policy()|{'validated':True})
    p=policy();p['controls'][0]['tolerance_ev']=float('nan')
    with pytest.raises(ValueError):validate_policy(p)
    p=policy();p['controls']*=2
    with pytest.raises(ValueError):validate_policy(p)


def test_shared_initial_and_grading_costs_are_required_before_reporting_efficiency(endpoints,tmp_path):
    from test_research_study import study_fixture
    from siesta_remag.research.study import grade_study
    from siesta_remag.research.tools import ResearchTools
    d=study_fixture(tmp_path)
    ResearchTools(tmp_path).call('stage_scalar_data',dict(
        calculations={'baseline':{'report':'before/report.json'}},output='initial-scalar.json'))
    for a in ('scalar','physics'):
        shutil.copyfile(tmp_path/'initial-scalar.json',tmp_path/('arm-'+a)/'initial-scalar.json')
    d['common_inputs'].append('initial-scalar.json')
    register_study(tmp_path,d,'study')
    for a in ('scalar','physics'):assert run_study_arm(tmp_path,'study',a)['status']=='completed'
    outcomes=dict(format='REMAG-STUDY-OUTCOMES/1',cases={'c'*64:{'anisotropy.total_energy_ev':
        dict(report='after/report.json',stage='anisotropy',observable='total_energy_ev')}})
    grade_study(tmp_path,'study',outcomes)
    r=evaluate_study(tmp_path,'study','missing-usage.json',usage=dict(format='REMAG-STUDY-GPU-USAGE/1',records=[]))
    expected={sha256(tmp_path/(p+'/report.json')) for p in ('before','after')}
    assert set(r['measured_allocation_usage']['arms']['physics']['missing_report_sha256s'])==expected
    assert r['arms']['physics']['actual_gpu_hours'] is None
    usage=measured_usage(tmp_path,[dict(report=p+'/report.json') for p in ('before','after')])
    usage['records'][0]['phase']='initial_data';usage['records'][1]['phase']='grading'
    result=evaluate_study(tmp_path,'study','covered.json',usage=usage)
    assert result['arms']['physics']['actual_gpu_hours']==1
    key='anisotropy.total_energy_ev'
    assert result['paired_prediction_improvement_per_gpu_hour'][key]==pytest.approx(
        -result['paired_prediction_errors'][key]['physics_minus_scalar_mae'])
