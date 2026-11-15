import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from test_discovery import backend
from test_research_execution import prepared
from test_research_inspection import spectrum_fixture
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load,save
from siesta_remag.research.evaluation import validate_target,response_observation,prediction_scores
from siesta_remag.research.experiments import run_experiment
from siesta_remag.research.memory import Notebook
from siesta_remag.research.tools import ResearchTools


def target(metric='projected_energy_moment',**kwargs):
    return dict(kind='spectral_response',baseline_source_sha256='a'*64,metric=metric,
                stage='soc-z',spin=0,selection=dict(atom_indices=[1],energy_window_ev=[-.2,.4])) | kwargs


@pytest.fixture
def endpoints(prepared,tmp_path):
    other=tmp_path/'other';shutil.copytree(prepared,other)
    structure=load(other/'structure.json');structure['source']['sha256']='c'*64;structure['sha256']='d'*64
    save(other/'structure.json',structure)
    request=load(other/'experiment.json');request.update(source_sha256='c'*64,export_sha256='d'*64)
    request['files']['structure.json']=sha256(other/'structure.json');save(other/'experiment.json',request)
    results=[]
    for name,bundle,shift in [('before',prepared,0),('after',other,.01)]:
        root=tmp_path/name
        r=run_experiment(bundle,root,tmp_path/'binary',executable_sha256=sha256(tmp_path/'binary'),runner=backend([]))
        # Deliberately synthetic spectral fixtures; no native solver or HSX
        # reconstruction claim. The outcome adapter must bind their hashes.
        for suffix in ('HSX','KP','EIG'):
            p=root/'soc-z'/('remag.'+suffix);p.write_text('synthetic '+suffix)
            r['artifacts'][str(p.relative_to(root))]=sha256(p)
        save(root/'report.json',r)
        spectrum=spectrum_fixture(tmp_path/(name+'-spectrum'),shift=shift)
        m=load(spectrum);m.update({key:r['artifacts']['soc-z/remag.'+suffix] for key,suffix in
            [('hsx_sha256','HSX'),('kp_sha256','KP'),('eig_sha256','EIG')]});save(spectrum,m)
        results.append(dict(report=root/'report.json',spectrum=spectrum))
    return results


def test_response_integrals_preserve_signed_weights_and_energy_reference(endpoints):
    before,after=endpoints
    r=response_observation(target(),before,after,source_sha256='c'*64)
    assert r['value']==pytest.approx(.01225)  # .01 * (.25*.7 + .75*1.4)
    assert r['unit']=='eV/cell' and r['status']=='measured'
    selected=target(selection=dict(atom_indices=[1],k_indices=[0],energy_window_ev=[-.2,0]))
    r=response_observation(selected,before,after,source_sha256='c'*64)
    assert r['value']==pytest.approx(-.00025)
    assert r['evidence']['before']['selected_k_weight']==.25
    for metric in ('projected_weight','projected_occupation'):
        r=response_observation(selected | {'metric':metric},before,after,source_sha256='c'*64)
        assert r['evidence']['before_value']==pytest.approx(-.025)
        assert r['value']==0


def test_spin_degenerate_occupations_are_not_doubled_twice(endpoints):
    for endpoint in endpoints:
        p=endpoint['spectrum'];m=load(p);m.update(spin_components=1,collinear_spin=0);save(p,m)
        report=load(endpoint['report'])
        for suffix in ('HSX','KP','EIG'):
            report['artifacts']['scalar/remag.'+suffix]=report['artifacts'].pop('soc-z/remag.'+suffix)
            shutil.move(endpoint['report'].parent/'soc-z'/('remag.'+suffix),endpoint['report'].parent/'scalar'/('remag.'+suffix))
        save(endpoint['report'],report)
        artifact=p.parent/m['artifact']
        with np.load(artifact) as z:arrays={k:z[k] for k in z.files}
        arrays['occupations']*=2;np.savez(artifact,**arrays);m['artifact_sha256']=sha256(artifact);save(p,m)
    r=response_observation(target('projected_occupation',stage='scalar'),*endpoints,source_sha256='c'*64)
    # Occupied contributions: 2*(-.1*.25 + .5*.2*.75)=.1, not .2.
    assert r['evidence']['before_value']==pytest.approx(.1)


@pytest.mark.parametrize('mutation,match',[
    ('source','endpoints differ'),('runtime','settings or runtime'),('hash','bound to'),
    ('spin','spin channel'),('metadata','metadata differs'),('sampling','sampling differs'),('bytes','binding differs')])
def test_response_rejects_unmatched_or_tampered_evidence(endpoints,mutation,match):
    before,after=endpoints;p=after['report'];r=load(p);m=load(after['spectrum'])
    if mutation=='source':r['source_sha256']=r['request']['source_sha256']='f'*64;save(p,r)
    elif mutation=='runtime':r['environment']={'REMAG_DIFFERENT':'1'};save(p,r)
    elif mutation=='hash':m['hsx_sha256']='f'*64;save(after['spectrum'],m)
    elif mutation=='spin':m['collinear_spin']=1;save(after['spectrum'],m)
    elif mutation=='metadata':m['orbitals'][0]['l']=1;save(after['spectrum'],m)
    elif mutation=='sampling':
        artifact=after['spectrum'].parent/m['artifact']
        with np.load(artifact) as z:arrays={k:z[k] for k in z.files}
        arrays['k_fractional'][1,0]=.25;np.savez(artifact,**arrays);m['artifact_sha256']=sha256(artifact);save(after['spectrum'],m)
    else:(after['spectrum'].parent/m['artifact']).write_bytes(b'changed')
    with pytest.raises(ValueError,match=match):response_observation(target(),before,after,source_sha256='c'*64)


def test_failed_native_endpoint_is_retained_without_spectral_result(endpoints):
    p=endpoints[1]['report'];r=load(p);r['status']='held_experiment';r['stages'][1]['accepted']=False;save(p,r)
    endpoints[1]['spectrum']=Path('/does-not-exist')
    observed=response_observation(target(),*endpoints,source_sha256='c'*64)
    assert observed['status']=='failed' and observed['value'] is None


def test_immutable_multiple_predictions_and_actual_response_ingestion(endpoints,tmp_path):
    t=ResearchTools(tmp_path)
    h=t.call('record_hypothesis',dict(statement='Electronic feature follows perturbation',predictions=['positive spectral moment'],alternatives=['negative response']))
    e=t.call('propose_experiment',dict(hypothesis_id=h,structure_sha256='c'*64,action='structural intervention',rationale='separate explanations'))
    t.call('record_response_prediction',dict(experiment_id=e,observable='Pt-d-moment',target=target(),value=.012,uncertainty=.001,interval_probability=.9))
    t.call('record_response_prediction',dict(experiment_id=e,observable='E-response',target=dict(kind='anisotropy_response',metric='total_energy_ev',baseline_source_sha256='a'*64),value=0))
    with pytest.raises(ValueError,match='already recorded'):
        t.call('record_prediction',dict(experiment_id=e,observable='Pt-d-moment',unit='eV/cell',value=99))
    args=dict(experiment_id=e,observable='Pt-d-moment',**{key:{k:str(v.relative_to(tmp_path)) for k,v in ep.items()} for key,ep in zip(['before','after'],endpoints)})
    t.call('record_response_outcome',args)
    with pytest.raises(ValueError,match='outcome is known'):
        t.call('record_prediction',dict(experiment_id=e,observable='late-free-energy',unit='eV/cell',value=0))
    with pytest.raises(ValueError,match='immutable'):t.call('record_response_outcome',args)
    args['observable']='E-response'
    for key in ('before','after'):args[key].pop('spectrum')
    t.call('record_response_outcome',args)
    r=t.call('evaluate_hypothesis',dict(experiment_id=e))
    assert r['status']=='evaluated' and set(r['observables'])=={'Pt-d-moment','E-response'}
    assert r['observables']['Pt-d-moment']['error']==pytest.approx(.00025)
    scores=t.call('evaluate_predictions',dict(experiment_ids=[e],resolutions={'Pt-d-moment':.0001,'E-response':.00005}))
    assert scores['metrics']['Pt-d-moment']['intervals'][0]['observed_coverage']==1
    assert scores['metrics']['E-response']['direction']['unresolved_observations']==1
    assert not scores['blinding_verified'] and not scores['mechanism_verified']
    # A fresh worker supports these new operations without a service/model key.
    request=dict(op='evaluate_predictions',args=dict(experiment_ids=[e],resolutions={'Pt-d-moment':.0001,'E-response':.00005}))
    proc=subprocess.run([sys.executable,'-m','siesta_remag.research','tools','--workspace',str(tmp_path)],input=json.dumps(request)+'\n',text=True,capture_output=True,check=True)
    assert json.loads(proc.stdout)['result']['metrics']==scores['metrics']
    save(tmp_path/'experiments.json',[e]);save(tmp_path/'resolutions.json',request['args']['resolutions'])
    command=[sys.executable,'-m','siesta_remag.research','prediction-scores','--workspace',str(tmp_path),
        '--experiments',str(tmp_path/'experiments.json'),'--resolutions',str(tmp_path/'resolutions.json'),
        '--output',str(tmp_path/'scores.json')]
    subprocess.run(command,check=True,capture_output=True)
    assert load(tmp_path/'scores.json')['metrics']==scores['metrics']
    assert subprocess.run(command,capture_output=True).returncode!=0


def test_cohort_scores_keep_failures_abstentions_and_pending_predictions(tmp_path):
    n=Notebook(tmp_path/'memory.sqlite');ids=[]
    h=n.record_hypothesis('response',predictions=['positive'],alternatives=['negative'])
    for i,(pred,value,status,radius) in enumerate([(.2,.3,'measured',.15),(-.1,.2,'measured',.05),
            (0,.01,'measured',None),(1,None,'failed',None),(1,None,None,None),(1,None,'inconclusive',None)]):
        e=n.propose_experiment(h,structure_sha256=f'{i:064x}',action='response',rationale='test');ids.append(e)
        n.record_prediction(e,observable='delta',value=pred,unit='eV',uncertainty=radius,interval_probability=.8 if radius else None)
        if status:n.record_outcome(e,observable='delta',value=value,unit='eV',report_sha256='f'*64,status=status)
    result=prediction_scores(n,ids,dict(delta=.05));m=result['metrics']['delta']
    assert (m['prediction_count'],m['measured_count'],m['failed_count'],m['pending_count'],m['inconclusive_count'])==(6,3,1,1,1)
    assert m['mae']==pytest.approx(.41/3)
    assert m['direction']['accuracy_including_abstentions']==.5
    assert m['intervals'][0]['observed_coverage']==.5 and m['intervals'][0]['count']==2
    with pytest.raises(ValueError,match='distinct'):prediction_scores(n,[ids[0],ids[0]],dict(delta=.05))
    with pytest.raises(ValueError,match='resolution'):prediction_scores(n,ids,{})
    e=n.propose_experiment(h,structure_sha256='a'*64,action='response',rationale='different unit')
    n.record_prediction(e,observable='delta',value=1,unit='meV')
    with pytest.raises(ValueError,match='unlike'):prediction_scores(n,ids+[e],dict(delta=.05))
    n.close()


@pytest.mark.parametrize('change',[dict(selection={'limit':1}),dict(spin=1),dict(selection={'l':[True]}),
    dict(metric='unknown'),dict(baseline_source_sha256='bad'),dict(selection={'energy_window_ev':[0,float('nan')]}),dict(extra=1)])
def test_response_definition_cannot_be_ambiguous_or_truncated(change):
    with pytest.raises(ValueError):validate_target(target() | change)


def test_generic_outcome_cannot_impersonate_a_frozen_response(tmp_path):
    n=Notebook(tmp_path/'memory.sqlite');h=n.record_hypothesis('test',predictions=['up'],alternatives=['down'])
    e=n.propose_experiment(h,structure_sha256='c'*64,action='change',rationale='test')
    n.record_prediction(e,observable='response',value=1,unit='eV/cell',target=target())
    n.record_outcome(e,observable='response',value=1,unit='eV/cell',status='measured',report_sha256='b'*64)
    with pytest.raises(ValueError,match='preregistered'):n.evaluate(e)
    n.close()


@pytest.mark.parametrize('radius,nominal',[(None,.8),(.1,True),(.1,1),(.1,0),(.1,float('nan')),(-1,.8)])
def test_interval_requires_valid_radius_and_nominal_probability(tmp_path,radius,nominal):
    n=Notebook(tmp_path/'memory.sqlite');h=n.record_hypothesis('test',predictions=['up'],alternatives=['down'])
    e=n.propose_experiment(h,structure_sha256='c'*64,action='change',rationale='test')
    with pytest.raises(ValueError):n.record_prediction(e,observable='r',value=1,unit='eV',uncertainty=radius,interval_probability=nominal)
    n.close()


def test_unpredicted_failures_and_empty_experiments_remain_in_inventory(tmp_path):
    n=Notebook(tmp_path/'memory.sqlite');h=n.record_hypothesis('test',predictions=['up'],alternatives=['down']);ids=[]
    for i in range(2):ids.append(n.propose_experiment(h,structure_sha256=f'{i:064x}',action='change',rationale='test'))
    n.record_outcome(ids[0],observable='r',value=None,unit='eV',status='failed',report_sha256='a'*64)
    r=prediction_scores(n,ids,{})
    assert r['experiments_without_predictions']==ids and r['metrics']=={}
    assert r['unpredicted_outcomes'][0]['experiment_id']==ids[0]
    n.close()
