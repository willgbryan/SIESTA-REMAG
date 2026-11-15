"""Engineering controls: synthetic observables, no native solver executions."""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from test_research import experiment
from test_discovery import backend
from test_research_inspection import spectrum_fixture
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.research.experiments import run_experiment, verify_experiment
from siesta_remag.research.series import prepare_series, verify_series, analyze_series
from siesta_remag.research.tools import ResearchTools


def design(control=None, values=None):
    return dict(format='REMAG-RESPONSE-DESIGN/1', parameter='q', values=values or ['-1/50','0','1/100'],
                control=control or dict(kind='axial_strain',axis=2))


def prepare(experiment, tmp_path, *, control=None, values=None, query=None, name='series'):
    return prepare_series(experiment/'structure.json',tmp_path/'pseudos',tmp_path/name,
        design=design(control,values),query=query or {'kind':'anisotropy'},mesh=[3,3,3])


@pytest.mark.parametrize('control',[
    {'kind':'axial_strain','axis':2}, {'kind':'isotropic_strain'}, {'kind':'fixed_volume_tetragonal'},
    {'kind':'fractional_displacement','sites':[{'site_index':0,'direction_fractional':['1','0','0']}]},
    {'kind':'coordination_strain','center_site':0,'neighbors':[{'site_index':1,'image':[0,0,0]}]},
    {'kind':'metric_path','rates':{'metric.g11':'2','metric.g33':'-1'}},
])
def test_exact_paths_prepare_only_independent_queries(experiment,tmp_path,control):
    parent=sha256(experiment/'structure.json')
    series=prepare(experiment,tmp_path,control=control)
    assert verify_series(tmp_path/'series')==series
    assert series['baseline_id']=='point-001' and not series['calculation_performed']
    assert len({p['source_sha256'] for p in series['points']})==3
    geometries=[]
    for point in series['points']:
        bundle=tmp_path/'series'/point['bundle'];request=verify_experiment(bundle)
        assert request['kmesh']==[3,3,3] and request['parameter']['exact']==point['q_exact']
        assert request['authorship']['parent_sha256']==load(experiment/'structure.json')['source']['sha256']
        assert sorted(point['parent_to_child'])==[0,1]
        geometries.append(load(bundle/'structure.json')['geometry'])
        assert not (bundle/'report.json').exists()
    if control['kind']=='fixed_volume_tetragonal':
        assert all(np.linalg.det(g['cell'])==pytest.approx(27) for g in geometries)
    if control['kind']=='axial_strain':
        for q,g in zip(design()['values'],geometries):
            assert np.linalg.norm(g['cell'][2])==pytest.approx(3*(1+float(Fraction(q))))
            assert np.linalg.norm(g['cell'][0])==3
    if control['kind']=='fractional_displacement':
        assert any(p['parent_to_child']!=[0,1] for p in series['points'])
    assert sha256(experiment/'structure.json')==parent
    with pytest.raises(FileExistsError):prepare(experiment,tmp_path,control=control)


@pytest.mark.parametrize('change',[
    {'values':['-1/100','1/100','1/50']}, {'values':['-1/100','0','0']}, {'values':['0','-1/100','1/100']},
    {'values':[-.01,0,.01]}, {'values':['-1','0','1']}, {'values':['0','1/100']},
    {'control':{'kind':'axial_strain','axis':True}}, {'control':{'kind':'unregistered'}},
    {'control':{'kind':'metric_path','rates':{'metric.g11':'0'}}},
    {'values':['-1/1000000000000000000000000000000','0','1/1000000000000000000000000000000']},
])
def test_invalid_design_is_rejected_before_output(experiment,tmp_path,change):
    from gpac.errors import ExactInputError
    with pytest.raises((ValueError,TypeError,ExactInputError)):
        prepare_series(experiment/'structure.json',tmp_path/'pseudos',tmp_path/'bad',
            design=design()|change,query={'kind':'scalar'},mesh=[3,3,3])
    assert not (tmp_path/'bad').exists()


def measured(experiment,tmp_path,*,control=None,values=None,fail=None,branch=None):
    series=prepare(experiment,tmp_path,control=control,values=values)
    exe=tmp_path/'binary';exe.write_text('synthetic fixture only')
    evidence={}
    for point in series['points']:
        q=float(Fraction(point['q_exact']));pid=point['id'];mapping=point['parent_to_child']
        base=backend([],fail='soc-x' if pid==fail else None)
        def runner(*args,**kwargs):
            result=base(*args,**kwargs);o=result['observables'];stage=Path(args[2]).name
            k=.0001+.003*q+.02*q*q;f=.00008+.002*q+.01*q*q
            o.update(total_energy_ev=-20+2*q+3*q*q+(k if stage=='soc-x' else 0),
                     free_energy_ev=-20.01+q+2*q*q+(f if stage=='soc-x' else 0))
            forces=np.zeros((2,3));forces[mapping[0],0]=q+4*q*q
            o['forces_ev_ang']=forces.tolist();o['stress_ev_ang3']=(np.eye(3)*2*q).tolist()
            factor=1.3 if pid==branch else 1+q/2
            o['atom_moments_muB']=(np.asarray(o['atom_moments_muB'])*factor).tolist()
            o['total_moment_muB']=(np.asarray(o['total_moment_muB'])*factor).tolist()
            save(Path(args[2])/'result.json',result)
            return result
        root=tmp_path/(pid+'-result')
        run_experiment(tmp_path/'series'/point['bundle'],root,exe,executable_sha256=sha256(exe),runner=runner)
        evidence[pid]={'report':root/'report.json'}
    return series,evidence


def observables():
    return dict(E={'kind':'energy','stage':'scalar','quantity':'total_energy_ev'},
        K={'kind':'anisotropy','stages':['soc-x','soc-z'],'quantity':'total_energy_ev'},
        KF={'kind':'anisotropy','stages':['soc-x','soc-z'],'quantity':'free_energy_ev'},
        force={'kind':'force','stage':'scalar','site':0,'component':0},
        moment={'kind':'moment','stage':'scalar','site':None,'component':'norm'},
        stress={'kind':'stress','stage':'scalar','component':[2,2]})


def test_unequal_step_physical_responses_and_explicit_error_bounds(experiment,tmp_path):
    series,evidence=measured(experiment,tmp_path,control={'kind':'fractional_displacement',
        'sites':[{'site_index':0,'direction_fractional':['1','0','0']}]})
    errors={'K':{p['id']:1e-6 for p in series['points']}}
    r=analyze_series(tmp_path/'series',evidence,observables(),tmp_path/'analysis',error_bounds=errors)
    for key,first,second in [('E',2,6),('K',.003,.04),('KF',.002,.02),('force',1,8),('moment',2,0),('stress',2,0)]:
        o=r['observables'][key];d=o['derivatives'][0]
        assert all(p['status']=='measured' and p['derivative_eligible'] for p in o['points'])
        assert d['status']=='measured' and d['derivative']['first']==pytest.approx(first,abs=1e-10)
        assert d['derivative']['second']==pytest.approx(second,abs=1e-8)
        assert d['derivative']['truncation_error'] is None
        assert (d['derivative']['propagated_error_bound'] is not None)==(key=='K')
    assert r['runtime']['threads']==1
    assert r['observables']['K']['derivatives'][0]['derivative']['propagated_error_bound']['first']==pytest.approx(1e-6*400/3)
    assert not r['scope']['mechanism_verified'] and not r['scope']['calculation_performed']


@pytest.mark.parametrize('mode',['missing','failed','branch'])
def test_incomplete_or_changed_branches_never_interpolate_across_gaps(experiment,tmp_path,mode):
    series,evidence=measured(experiment,tmp_path,values=['-1/50','-1/100','0','1/100','1/50'],
        fail='point-003' if mode=='failed' else None,branch='point-003' if mode=='branch' else None)
    if mode=='missing':del evidence['point-003']
    r=analyze_series(tmp_path/'series',evidence,{'K':observables()['K']},tmp_path/'analysis')['observables']['K']
    assert [d['status'] for d in r['derivatives']]==['measured','inconclusive','inconclusive']
    p=r['points'][3]
    assert p['status']=={'missing':'pending','failed':'failed','branch':'measured'}[mode]
    assert not p['derivative_eligible']
    assert (p['value'] is not None)==(mode=='branch')
    assert len(r['points'])==5


@pytest.mark.parametrize('mutation',['source','mesh','runtime','threads','artifact','acceptance'])
def test_saved_points_must_match_planned_science_and_artifacts(experiment,tmp_path,mutation):
    _,evidence=measured(experiment,tmp_path)
    p=evidence['point-000']['report'];r=load(p)
    if mutation=='source':r['source_sha256']=r['request']['source_sha256']='f'*64
    if mutation=='mesh':r['request']['kmesh']=[5,5,5]
    if mutation=='runtime':r['environment']['DIFFERENT']='1'
    if mutation=='threads':r['threads']=2
    if mutation=='acceptance':r['request']['settings']['branch_net_fraction']=.5
    if mutation=='artifact':(p.parent/'scalar/stdout.log').write_text('changed')
    save(p,r)
    with pytest.raises(ValueError):analyze_series(tmp_path/'series',evidence,observables(),tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()


def test_saved_spectral_response_is_signed_bound_and_common_sampling(experiment,tmp_path):
    _,evidence=measured(experiment,tmp_path)
    obs={'near_fermi':dict(kind='spectral',stage='soc-z',spin=0,metric='projected_energy_moment',
        selection={'atom_indices':[1],'energy_window_ev':[-.2,.4]})}
    for point in verify_series(tmp_path/'series')['points']:
        endpoint=evidence[point['id']];path=endpoint['report'];r=load(path)
        for suffix in ('HSX','KP','EIG'):
            artifact=path.parent/'soc-z'/('remag.'+suffix);artifact.write_text('synthetic '+suffix)
            r['artifacts']['soc-z/'+artifact.name]=sha256(artifact)
        save(path,r)
        spec=spectrum_fixture(tmp_path/(point['id']+'-spectrum'),shift=float(Fraction(point['q_exact'])))
        m=load(spec);m.update({key:r['artifacts']['soc-z/remag.'+suffix] for key,suffix in
            [('hsx_sha256','HSX'),('kp_sha256','KP'),('eig_sha256','EIG')]});save(spec,m)
        endpoint['spectra']={'near_fermi':spec}
    r=analyze_series(tmp_path/'series',evidence,obs,tmp_path/'analysis')['observables']['near_fermi']
    assert r['derivatives'][0]['derivative']['first']==pytest.approx(1.225)
    assert r['points'][0]['delta_from_baseline']==pytest.approx(-.02*1.225)
    spec=evidence['point-000']['spectra']['near_fermi'];m=load(spec);m['hsx_sha256']='f'*64;save(spec,m)
    with pytest.raises(ValueError,match='bound'):analyze_series(tmp_path/'series',evidence,obs,tmp_path/'bad')


def test_missing_spectra_and_unresolved_net_branch_do_not_invent_derivatives(experiment,tmp_path):
    _,evidence=measured(experiment,tmp_path)
    obs={'E':observables()['E'],'window':dict(kind='spectral',stage='soc-z',spin=0,metric='projected_weight',selection={})}
    r=analyze_series(tmp_path/'series',evidence,obs,tmp_path/'missing-spectra')
    assert r['observables']['E']['derivatives'][0]['status']=='measured'
    assert all(p['status']=='pending' and p['value'] is None for p in r['observables']['window']['points'])
    p=evidence['point-001']['report'];report=load(p)
    report['observations']['scalar']['total_moment_muB']=[0,0,0];save(p,report)
    r=analyze_series(tmp_path/'series',evidence,{'E':observables()['E']},tmp_path/'unresolved')['observables']['E']
    assert all(p['status']=='measured' and not p['derivative_eligible'] for p in r['points'])
    assert 'undefined' in r['points'][0]['branch']['stages']['scalar']['reason']
    assert r['derivatives'][0]['status']=='inconclusive'


def test_tools_pending_analysis_cli_and_profile_limits(experiment,tmp_path,capsys):
    from siesta_remag.research.__main__ import main
    tools=ResearchTools(tmp_path,information_profile='scalar')
    p=tools.call('prepare_response_series',dict(structure='bundle/structure.json',pseudos='pseudos',output='series',
        design=design(),query={'kind':'scalar'},mesh=[3,3,3]))
    assert len(p['bundles'])==3 and 'run_prepared_query' not in tools.catalog()
    args=dict(series='series',evidence={},observables={'E':observables()['E']},output='pending')
    r=tools.call('analyze_response_series',args)
    assert all(x['status']=='pending' for x in r['observables']['E']['points'])
    for malformed in (None,[],{'point-000':{'spectra':[]}}):
        with pytest.raises(ValueError):tools.call('analyze_response_series',args|{'evidence':malformed,'output':'bad'})
    with pytest.raises(ValueError,match='scalar profile'):
        tools.call('analyze_response_series',args|{'observables':{'s':dict(kind='spectral',stage='scalar',spin=0,
            metric='projected_weight',selection={})}})
    with pytest.raises(ValueError,match='not requested'):
        tools.call('analyze_response_series',args|{'observables':{'K':observables()['K']},'output':'bad'})
    with pytest.raises(ValueError,match='escapes'):
        tools.call('analyze_response_series',args|{'evidence':{'point-000':{'report':'../escape.json'}}})
    save(tmp_path/'evidence.json',{});save(tmp_path/'observables.json',{'E':observables()['E']})
    assert main(['analyze-response',str(tmp_path/'series'),'--evidence',str(tmp_path/'evidence.json'),
        '--observables',str(tmp_path/'observables.json'),'--output',str(tmp_path/'cli')])==0
    assert load(tmp_path/'cli/response.json')['series_sha256']==p['series']['sha256']
    capsys.readouterr()
    save(tmp_path/'design.json',design());save(tmp_path/'query.json',{'kind':'scalar'})
    assert main(['prepare-response',str(experiment/'structure.json'),'--pseudos',str(tmp_path/'pseudos'),
        '--output',str(tmp_path/'cli-prepared'),'--design',str(tmp_path/'design.json'),'--query',str(tmp_path/'query.json'),
        '--mesh','3','3','3'])==0
    assert len(verify_series(tmp_path/'cli-prepared')['points'])==3
    capsys.readouterr()


def test_series_correspondence_and_scope_are_reproduced(experiment,tmp_path):
    r=prepare(experiment,tmp_path)
    r['points'][0]['parent_to_child']=[1,0];r['sha256']=digest({k:v for k,v in r.items() if k!='sha256'})
    save(tmp_path/'series/series.json',r)
    with pytest.raises(ValueError,match='correspondence'):verify_series(tmp_path/'series')


def test_withheld_saved_result_is_not_available_to_series_policy(experiment,tmp_path):
    series,evidence=measured(experiment,tmp_path)
    tools=ResearchTools(tmp_path,withheld_sources=[series['points'][0]['source_sha256']])
    with pytest.raises(ValueError,match='held-out evaluator'):
        tools.call('analyze_response_series',dict(series='series',evidence={pid:{'report':str(e['report'].relative_to(tmp_path))}
            for pid,e in evidence.items()},observables=observables(),output='forbidden'))
    assert not (tmp_path/'forbidden').exists()


def test_agent_can_prepare_and_inspect_pending_series_without_execution(experiment,tmp_path):
    import sys
    from siesta_remag.research.agent import run_session
    actions=[dict(tool='prepare_response_series',args=dict(structure='bundle/structure.json',pseudos='pseudos',output='series',
        design=design(),query={'kind':'scalar'},mesh=[3,3,3])),
        dict(tool='analyze_response_series',args=dict(series='series',evidence={},observables={'E':observables()['E']},output='analysis'))]
    actions=[a|{'rationale':'Inspect a controlled response without executing calculations'} for a in actions]
    policy=tmp_path/'policy.py'
    policy.write_text('import json,sys\nr=json.load(sys.stdin)\na='+repr(actions)+'\ni=len(r["history"])\nprint(json.dumps(a[i] if i<len(a) else {"finish":"Pending evidence retained; no solver run"}))\n')
    result=run_session(tmp_path,'session',objective='Prepare one response family and inspect missing data',
        policy_command=[sys.executable,str(policy)],max_steps=4,max_seconds=30)
    assert result['status']=='completed',result
    assert load(tmp_path/'analysis/response.json')['observables']['E']['points'][0]['status']=='pending'
    assert not list((tmp_path/'series').rglob('report.json'))
