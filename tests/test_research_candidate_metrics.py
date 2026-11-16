"""Analytic phase mixtures and local sessions with injected native transport."""
import copy
from collections import Counter
from pathlib import Path
import shutil
import pytest
from test_research_discovery_scores import make_candidate_snapshot,policy
from test_discovery import backend
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load,save
from siesta_remag.research.experiments import prepare_experiment,run_experiment
from siesta_remag.research.series import prepare_series
from siesta_remag.research.study import evaluate_study
from siesta_remag.research.candidate_metrics import reference_mixture,validate_metrics


def test_binary_hull_conservation_competing_phase_and_energy_origin():
    c=Counter({26:1,78:1});refs=[Counter({26:2}),Counter({78:4}),Counter({26:3,78:1})]
    r=reference_mixture(c,[-2,-4,-3.5],refs)
    assert r['atomic_fractions']==pytest.approx([0,1/3,2/3])
    assert r['energy_ev_atom']==pytest.approx(-11/3)
    assert r['maximum_composition_residual']<1e-9
    shifts={26:-3000,78:200}
    shifted=[v+sum(shifts[z]*n for z,n in x.items())/sum(x.values()) for v,x in zip([-2,-4,-3.5],refs)]
    other=reference_mixture(c,shifted,refs)
    assert other['atomic_fractions']==pytest.approx(r['atomic_fractions'])
    assert other['energy_ev_atom']-r['energy_ev_atom']==pytest.approx(-1400)
    assert reference_mixture(c,[-2],[Counter({26:1})]) is None


def test_ternary_mixture_and_nonfinite_rejection():
    refs=[Counter({z:1}) for z in (8,26,78)]
    r=reference_mixture(Counter({8:3,26:1,78:2}),[-1,-2,-4],refs)
    assert r['atomic_fractions']==pytest.approx([.5,1/6,1/3])
    assert r['energy_ev_atom']==pytest.approx(-13/6)
    with pytest.raises(ValueError,match='Nonfinite'):reference_mixture(Counter({8:1}),[float('nan')],[refs[0]])


@pytest.fixture(scope='module')
def metric_snapshot(tmp_path_factory):
    ir=pytest.importorskip('gpac.ir')
    from gpac.physical import export_geometry
    from gpac.candidate_identity import POLICY
    root=tmp_path_factory.mktemp('candidate-metrics')
    ref=export_geometry(ir.author_program(dict(hall=1,gram=[[16,0,0],[0,16,0],[0,0,16]],orbits=[
        dict(z=26,wyckoff_index=0,parameters=[0,0,0]),dict(z=26,wyckoff_index=0,parameters=['1/2']*3)])),length_unit='angstrom')
    design=dict(format='REMAG-RESPONSE-DESIGN/1',parameter='axial',values=['-1/100','0','1/100'],control=dict(kind='axial_strain',axis=2))
    p=policy()|dict(candidate_equivalence=POLICY,stability=dict(stage='scalar',reference_sources=[ref['source']['sha256']],
        maximum_energy_above_reference_hull_ev_atom=.01,maximum_force_ev_ang=.04,maximum_stress_ev_ang3=.001),
        robustness=dict(designs=[design],maximum_anisotropy_change_MJ_m3=.1,maximum_spin_polarization_change_tesla=.1))
    root,s=make_candidate_snapshot(root,p);exe=root/'binary'
    prepare_experiment(ref,root/'pseudos',root/'reference-input',mesh=[5]*3,query=dict(kind='scalar'))
    base=backend([])
    def phase(*args,**kw):
        r=base(*args,**kw)
        for q in ('total_energy_ev','free_energy_ev'):r['observables'][q]+=.2
        save(Path(args[2])/'result.json',r)
        return r
    r=run_experiment(root/'reference-input',root/'reference',exe,executable_sha256=sha256(exe),runner=phase)
    assert r['status']=='measured',r
    s['reference_source']=ref['source']['sha256']
    s['evidence']['reference_phases']={s['reference_source']:dict(report='reference/report.json')}
    series=prepare_series(root/'bundle/structure.json',root/'pseudos',root/'variation',design=design,query=dict(kind='anisotropy'),mesh=[5]*3)
    evidence={}
    for point in series['points']:
        dest='variation-'+point['id']
        r=run_experiment(root/'variation'/point['bundle'],root/dest,exe,executable_sha256=sha256(exe),runner=backend([]))
        assert r['status']=='measured',r
        evidence[point['id']]=dict(report=dest+'/report.json')
    for arm in ('scalar','physics'):
        s['evidence']['arms'][arm][s['source']]['robustness']={'axial':dict(series='variation',reports=evidence)}
    return root,s


@pytest.fixture
def metrics(metric_snapshot,tmp_path):
    root,s=metric_snapshot;shutil.copytree(root,tmp_path,dirs_exist_ok=True)
    return copy.deepcopy(s)


def test_full_registered_metrics_and_cost_coverage(metrics,tmp_path):
    s=metrics;r=evaluate_study(tmp_path,'study','scores.json',discovery_evidence=s['evidence'],usage=dict(format='REMAG-STUDY-GPU-USAGE/1',records=[]))
    for arm in ('scalar','physics'):
        c=r['independent_candidate_validation']['arms'][arm]
        assert c['distinct_under_declared_policy']==2 and c['plausible_groups_under_declared_policy']==2
        assert c['screened_groups_under_declared_policy']==1 and c['crystallographically_distinct_count'] is None
        row=next(x for x in c['rows'] if x['source_sha256']==s['source'])
        assert row['thermodynamic_stability']['observed_static_screen_passed']
        assert row['thermodynamic_stability']['energies']['E']['signed_energy_relative_to_references_ev_atom']==pytest.approx(-.1)
        assert row['structural_robustness']['observed_robustness_passed']
        assert len(row['structural_robustness']['paths']['axial']['points'])==3
        required=set(r['measured_allocation_usage']['arms'][arm]['missing_report_sha256s'])
        assert sha256(tmp_path/'reference/report.json') in required
        assert sha256(tmp_path/'variation-point-002/report.json') in required
        assert r['arms'][arm]['actual_gpu_hours'] is None


@pytest.mark.parametrize('failure',['missing_phase','failed_phase','lower_phase','force','stress','candidate_native','missing_variation','branch','large_response'])
def test_missing_or_failing_metric_retains_cohort_without_credit(metrics,tmp_path,failure):
    s=metrics
    if failure=='missing_phase':s['evidence']['reference_phases']={}
    elif failure in ('failed_phase','lower_phase'):
        p=tmp_path/'reference/report.json';r=load(p)
        if failure=='failed_phase':r['status']='held_scalar';r['stages'][0]['accepted']=False
        else:
            for q in ('total_energy_ev','free_energy_ev'):r['observations']['scalar'][q]-=2
        save(p,r)
    elif failure in ('force','stress','candidate_native'):
        p=tmp_path/'validation-5/report.json';r=load(p)
        if failure=='candidate_native':r['status']='held_scalar';r['stages'][0]['accepted']=False
        else:r['observations']['scalar']['forces_ev_ang' if failure=='force' else 'stress_ev_ang3'][0][0]=1
        save(p,r)
    elif failure=='missing_variation':
        for a in ('scalar','physics'):s['evidence']['arms'][a][s['source']]['robustness']['axial']['reports'].pop('point-002',None)
    else:
        p=tmp_path/'variation-point-002/report.json';r=load(p)
        if failure=='branch':
            r['observations']['soc-z']['total_moment_muB']=[0,0,6]
            r['observations']['soc-z']['atom_moments_muB']=[[0,0,3]]*2
        else:
            for q in ('total_energy_ev','free_energy_ev'):r['observations']['soc-x'][q]+=.1
        save(p,r)
    r=evaluate_study(tmp_path,'study','scores.json',discovery_evidence=s['evidence'],usage=dict(format='REMAG-STUDY-GPU-USAGE/1',records=[]))
    for a in ('scalar','physics'):
        c=r['independent_candidate_validation']['arms'][a]
        assert len(c['rows'])==2 and c['screened_groups_under_declared_policy']==0
        if failure=='candidate_native':
            missing=r['measured_allocation_usage']['arms'][a]['missing_report_sha256s']
            assert sha256(tmp_path/'reference/report.json') in missing
            assert sha256(tmp_path/'variation-point-002/report.json') in missing
            assert sha256(tmp_path/'validation-3/report.json') in missing
            assert sha256(tmp_path/'validation-4/report.json') in missing


@pytest.mark.parametrize('change,match',[('phase_source','Unregistered'),('phase_runtime','runtime differs'),('variation_runtime','runtime'),('variation_point','Unknown structural'),('phase_artifact','binding differs')])
def test_changed_or_unregistered_evidence_rejected(metrics,tmp_path,change,match):
    s=metrics
    if change=='phase_source':s['evidence']['reference_phases']['a'*64]=s['evidence']['reference_phases'].pop(s['reference_source'])
    elif change in ('phase_runtime','variation_runtime'):
        p=tmp_path/('reference/report.json' if change=='phase_runtime' else 'variation-point-002/report.json')
        r=load(p);r['environment']['REMAG_TEST']='changed';save(p,r)
    elif change=='variation_point':s['evidence']['arms']['scalar'][s['source']]['robustness']['axial']['reports']['unknown']={}
    else:(tmp_path/'reference/scalar/input.fdf').write_text('tampered')
    with pytest.raises(ValueError,match=match):evaluate_study(tmp_path,'study','bad.json',discovery_evidence=s['evidence'])
    assert not (tmp_path/'bad.json').exists()


def test_metric_policy_rejects_unbounded_or_global_claims():
    pytest.importorskip('gpac')
    with pytest.raises(ValueError):validate_metrics(dict(candidate_equivalence='globally-unique'))
    with pytest.raises(ValueError):validate_metrics(dict(robustness=dict(designs=[],maximum_anisotropy_change_MJ_m3=1,maximum_spin_polarization_change_tesla=1)))
