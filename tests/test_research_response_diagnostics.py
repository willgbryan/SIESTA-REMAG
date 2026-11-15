"""Analytic diagnostic responses on synthetic files; no native solver calls."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import eigh

from test_research import experiment
from test_research_series import measured
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load, save
from siesta_remag.research.mechanisms import decompose
from siesta_remag.research.orbital_moments import moment_kernels
from siesta_remag.research.response_diagnostics import validate_diagnostic, measure_diagnostic, compare_diagnostics
from siesta_remag.research.series import analyze_series
from siesta_remag.research.evaluation import response_observation
from siesta_remag.research.tools import ResearchTools


def definitions():
    base=dict(stage='soc-z',spin=0,k_indices=[0,5,13])
    return dict(coupling=dict(kind='coupling',metric='frobenius',groups=[{'site':0,'l':0},{'site':1,'l':0}],**base),
        singular=dict(kind='coupling',metric='largest_singular_value',groups=[{'site':0,'l':0},{'site':1,'l':0}],**base),
        bands=dict(kind='band_sum',reference='absolute',**base),
        relative_bands=dict(kind='band_sum',reference='own_fermi',**base),
        orbital=dict(kind='orbital_moment',stage='soc-z',selection={'sites':[0],'l':[1]},component=2))


def fixtures(experiment,tmp_path,*,control=None,branch=None,fail=None):
    series,endpoints=measured(experiment,tmp_path,control=control,branch=branch,fail=fail)
    definitions_by_name=definitions()
    for point in series['points']:
        q=float(Fraction(point['q_exact']));path=endpoints[point['id']]['report'];r=load(path);stage=path.parent/'soc-z'
        from ase.data import chemical_symbols
        species=[chemical_symbols[z] for z in r['structure']['geometry']['atomic_numbers']]
        mapping=point['parent_to_child'];cell=np.array(r['structure']['geometry']['cell'])/.529177210903
        fractional=np.array(list(product([-1/3,0,1/3],repeat=3)))
        cart=fractional @ np.linalg.inv(cell.T/(2*np.pi))
        (stage/'remag.KP').write_text('27\n'+'\n'.join(str(i+1)+' '+' '.join(f'{v:.5E}' for v in [*row,1/27]) for i,row in enumerate(cart))+'\n')
        (stage/'remag.HSX').write_text('Explicit synthetic matrix fixture; not native output')
        (stage/'remag.EIG').write_text('Explicit synthetic eigenvalue binding')
        for symbol in species:(stage/(symbol+'.ion.xml')).write_text('Explicit synthetic radial binding '+symbol)
        r['artifacts'].update({'soc-z/'+p.name:sha256(p) for p in stage.iterdir() if p.is_file()});save(path,r)
        directory=tmp_path/(point['id']+'-diagnostics');directory.mkdir()
        indices=[0,5,13];spatial=np.array([[1,.2+q],[.2+q,1]]);overlap=np.kron(spatial,np.eye(2))
        w,u=eigh(overlap);root=(u*np.sqrt(w))@u.T;t=.2+2*q+3*q*q
        orth=np.array([[-1,0,t,0],[0,-.5,0,t],[t,0,1,0],[0,t,0,1.5]])
        h=root@orth@root
        d=decompose(h,overlap,0,.025)
        arrays={k:np.array([v]*3) for k,v in d.items()}
        recorded=np.array([[float(v) for v in line.split()[1:4]] for line in (stage/'remag.KP').read_text().splitlines()[1:]])
        arrays.update(k_fractional=(recorded@cell.T/(2*np.pi))[indices],k_weights=np.full(3,1/27),radial_functions=np.array([0.,1.,.5,0.]),cell_bohr=cell.ravel(),
            native_eigenvalue_residual_ev=np.zeros((3,4)))
        orbitals=[dict(index=i,atom_index=i,species=symbol,atom_orbital_index=0,n=1,l=0,zeta=1) for i,symbol in enumerate(species)]
        np.savez_compressed(directory/'matrices.npz',**arrays)
        meta=dict(format='REMAG-ELECTRONIC-MATRICES/1',artifact='matrices.npz',artifact_sha256=sha256(directory/'matrices.npz'),
            report_sha256=sha256(path),source_sha256=r['source_sha256'],executable_sha256=r['executable_sha256'],stage='soc-z',
            evidence={k:r['artifacts']['soc-z/'+v] for k,v in [('hsx','remag.HSX'),('kp','remag.KP'),('eig','remag.EIG')]},
            orbitals=orbitals,ion_sha256={s:r['artifacts']['soc-z/'+s+'.ion.xml'] for s in species},radial_layout=[{'synthetic':True}],
            spin_components=8,collinear_spin=None,spin_frame='Cartesian solver frame; interleaved up/down',lattice_gauge='synthetic',
            fermi_ev=0,temperature_ev=.025,k_indices=indices,native_k_count=27,native_max_abs_error_ev=0,occupation_model='synthetic Fermi-Dirac')
        save(directory/'matrices.json',meta)
        orbitals=[dict(index=3*i+j,atom_index=i,species=symbol,atom_orbital_index=j,n=2,l=1,zeta=1,m=j-1) for i,symbol in enumerate(species) for j in range(3)]
        kernels=moment_kernels(orbitals);unit=np.zeros((8,6,6));c=np.array([1.,0.,-1j])/np.sqrt(2);density=np.outer(c,c.conj())
        for parent,atom in enumerate(mapping):unit[4,3*atom:3*atom+3,3*atom:3*atom+3]=density.imag*((1+2*q+3*q*q) if parent==0 else .5)
        values=np.einsum('aij,ij->ia',kernels,unit[4]+unit[5]);atoms=np.array([values[:3].sum(axis=0),values[3:].sum(axis=0)])
        np.savez_compressed(directory/'moments.npz',native_unitcell_density_components=unit,moment_kernels=kernels,orbital_moments_mu_b=values,atom_moments_mu_b=atoms)
        moment=dict(format='REMAG-ORBITAL-MOMENTS/1',artifact='moments.npz',artifact_sha256=sha256(directory/'moments.npz'),
            report_sha256=sha256(path),source_sha256=r['source_sha256'],executable_sha256=r['executable_sha256'],stage='soc-z',
            evidence={k:r['artifacts']['soc-z/'+v] for k,v in [('hsx','remag.HSX'),('dm','remag.DM'),('stdout','stdout.log')]},
            orbitals=orbitals,spin_components=8,frame='Cartesian solver frame',unit='mu_B',harmonic_order='m=-l..l',
            density_convention='native onsite',approximation='same atom/l/zeta, zero cell')
        save(directory/'moments.json',moment)
        endpoints[point['id']]['diagnostics']={name:directory/('moments.json' if name=='orbital' else 'matrices.json') for name in definitions_by_name}
    return series,endpoints


def test_analytic_hybridization_and_orbital_moment_responses(experiment,tmp_path):
    series,evidence=fixtures(experiment,tmp_path)
    r=analyze_series(tmp_path/'series',evidence,definitions(),tmp_path/'analysis')
    for name,first,second in [('coupling',2*np.sqrt(2)/9,6*np.sqrt(2)/9),('singular',2/9,6/9),('orbital',2,6)]:
        o=r['observables'][name];derivative=o['derivatives'][0]['derivative']
        assert derivative['first']==pytest.approx(first,abs=1e-11)
        assert derivative['second']==pytest.approx(second,abs=1e-9)
        assert derivative['propagated_error_bound'] is None and derivative['truncation_error'] is None
        assert all(p['derivative_eligible'] for p in o['points'])
    for point in r['observables']['bands']['points']:
        q=point['q'];t=.2+2*q+3*q*q
        eigen=np.array([-np.sqrt(1+t*t),.5-np.sqrt(1+t*t),np.sqrt(1+t*t),.5+np.sqrt(1+t*t)])
        expected=np.sum(eigen/(1+np.exp(np.clip(eigen/.025,-700,700))))/9
        assert point['value']==pytest.approx(expected,abs=1e-12)
        assert point['evidence']['selected_k_weight']==pytest.approx(1/9)
        assert not point['evidence']['weight_renormalized']
    pair=response_observation(dict(kind='diagnostic_response',baseline_source_sha256=series['points'][1]['source_sha256'],diagnostic=definitions()['coupling']),
        {'report':evidence['point-001']['report'],'diagnostic':evidence['point-001']['diagnostics']['coupling']},
        {'report':evidence['point-002']['report'],'diagnostic':evidence['point-002']['diagnostics']['coupling']},source_sha256=series['points'][2]['source_sha256'])
    assert pair['status']=='measured'
    assert pair['value']==pytest.approx(r['observables']['coupling']['points'][2]['delta_from_baseline'])
    assert pair['evidence']['diagnostic_correspondence']['radial_max_abs_difference']==0


def test_parent_site_mapping_survives_exact_atom_reordering(experiment,tmp_path):
    series,evidence=fixtures(experiment,tmp_path,control={'kind':'fractional_displacement','sites':[{'site_index':0,'direction_fractional':['1','0','0']}]})
    assert any(p['parent_to_child']!=[0,1] for p in series['points'])
    r=analyze_series(tmp_path/'series',evidence,definitions(),tmp_path/'analysis')
    assert r['observables']['orbital']['derivatives'][0]['derivative']['first']==pytest.approx(2)
    assert r['observables']['coupling']['derivatives'][0]['derivative']['first']==pytest.approx(2*np.sqrt(2)/9)


@pytest.mark.parametrize('mode',['missing','branch','failed'])
def test_unavailable_diagnostics_and_native_branch_failures_remain_evidence(experiment,tmp_path,mode):
    series,evidence=fixtures(experiment,tmp_path,branch='point-002' if mode=='branch' else None,fail='point-002' if mode=='failed' else None)
    if mode=='missing':del evidence['point-002']['diagnostics']
    r=analyze_series(tmp_path/'series',evidence,definitions(),tmp_path/'analysis')
    for o in r['observables'].values():
        assert o['points'][2]['status']=={'missing':'pending','branch':'measured','failed':'failed'}[mode]
        assert not o['points'][2]['derivative_eligible'] and o['derivatives'][0]['status']=='inconclusive'
    if mode=='branch':
        target=dict(kind='diagnostic_response',baseline_source_sha256=series['points'][1]['source_sha256'],diagnostic=definitions()['orbital'])
        endpoints=[{'report':evidence[p]['report'],'diagnostic':evidence[p]['diagnostics']['orbital']} for p in ('point-001','point-002')]
        outcome=response_observation(target,*endpoints,source_sha256=series['points'][2]['source_sha256'])
        assert outcome['status']=='failed' and outcome['value'] is None


@pytest.mark.parametrize('change',[{'k_indices':[True]}, {'k_indices':[0]*2}, {'spin':1}, {'metric':'made_up'},
    {'groups':[{'site':0,'l':0}]*2}, {'groups':[{'site':0,'l':5},{'site':1,'l':0}]}, {'extra':0}])
def test_diagnostic_targets_require_explicit_valid_selection(change):
    with pytest.raises(ValueError):validate_diagnostic(definitions()['coupling']|change)


@pytest.mark.parametrize('mutation',['hash','source','report','stage','kp','radial','k_indices','coordinates','weights','convention','orbitals'])
def test_response_binding_radial_and_sampling_mismatches_are_rejected(experiment,tmp_path,mutation):
    _,evidence=fixtures(experiment,tmp_path);p=evidence['point-002']['diagnostics']['coupling'];m=load(p)
    if mutation=='hash':m['artifact_sha256']='a'*64
    elif mutation=='source':m['source_sha256']='a'*64
    elif mutation=='report':m['report_sha256']='a'*64
    elif mutation=='stage':m['stage']='soc-x'
    elif mutation=='kp':m['evidence']['kp']='a'*64
    elif mutation=='k_indices':m['k_indices']=[1,5,13]
    elif mutation=='convention':m['spin_frame']='rotated frame'
    elif mutation=='orbitals':m['orbitals'][0]['n']=2
    else:
        with np.load(p.parent/m['artifact']) as a:arrays={k:a[k] for k in a.files}
        if mutation=='radial':arrays['radial_functions'][1]+=.01
        if mutation=='coordinates':arrays['k_fractional'][0,0]+=.01
        if mutation=='weights':arrays['k_weights'][0]*=2
        np.savez_compressed(p.parent/m['artifact'],**arrays);m['artifact_sha256']=sha256(p.parent/m['artifact'])
    save(p,m)
    with pytest.raises(ValueError):analyze_series(tmp_path/'series',evidence,definitions(),tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()


def test_agent_paths_withheld_sources_and_scalar_information_profile(experiment,tmp_path):
    series,evidence=fixtures(experiment,tmp_path)
    relative={k:{'report':str(v['report'].relative_to(tmp_path)),'diagnostics':{name:str(p.relative_to(tmp_path)) for name,p in v['diagnostics'].items()}} for k,v in evidence.items()}
    args=dict(series='series',evidence=relative,observables=definitions(),output='agent-analysis')
    r=ResearchTools(tmp_path).call('analyze_response_series',args)
    assert r['observables']['orbital']['derivatives'][0]['status']=='measured'
    with pytest.raises(ValueError,match='scalar profile'):ResearchTools(tmp_path,information_profile='scalar').call('analyze_response_series',args|{'output':'held'})
    with pytest.raises(ValueError,match='held-out evaluator'):ResearchTools(tmp_path,withheld_sources=[series['points'][2]['source_sha256']]).call('analyze_response_series',args|{'output':'held'})
    relative['point-002']['diagnostics']['orbital']='../outside.json'
    with pytest.raises(ValueError,match='escapes'):ResearchTools(tmp_path).call('analyze_response_series',args|{'output':'held'})
    assert not (tmp_path/'held').exists()


def test_preregistered_diagnostic_predictions_are_immutable_and_worker_scored(experiment,tmp_path):
    series,evidence=fixtures(experiment,tmp_path);tools=ResearchTools(tmp_path)
    h=tools.call('record_hypothesis',dict(statement='Synthetic coupling and orbital response increase',
        predictions=['positive coupling and orbital change'],alternatives=['opposite response']))
    e=tools.call('propose_experiment',dict(hypothesis_id=h,structure_sha256=series['points'][2]['source_sha256'],
        action='axial perturbation',rationale='measure independent diagnostics'))
    expected=dict(coupling=float((2*.01+3*.01**2)*np.sqrt(2)/9),orbital=2*.01+3*.01**2)
    for name,value in expected.items():
        target=dict(kind='diagnostic_response',baseline_source_sha256=series['points'][1]['source_sha256'],diagnostic=definitions()[name])
        tools.call('record_response_prediction',dict(experiment_id=e,observable=name,target=target,value=value,
            uncertainty=.0001,interval_probability=.9))
    for name in expected:
        args=dict(experiment_id=e,observable=name)
        for key,pid in [('before','point-001'),('after','point-002')]:
            args[key]=dict(report=str(evidence[pid]['report'].relative_to(tmp_path)),
                diagnostic=str(evidence[pid]['diagnostics'][name].relative_to(tmp_path)))
        with pytest.raises(ValueError,match='scalar policy'):
            ResearchTools(tmp_path,information_profile='scalar').call('record_response_outcome',args)
        tools.call('record_response_outcome',args)
        with pytest.raises(ValueError,match='immutable'):tools.call('record_response_outcome',args)
    with pytest.raises(ValueError,match='outcome is known'):
        tools.call('record_response_prediction',dict(experiment_id=e,observable='late',target=target,value=0))
    request=dict(op='evaluate_predictions',args=dict(experiment_ids=[e],resolutions={n:1e-6 for n in expected}))
    proc=subprocess.run([sys.executable,'-m','siesta_remag.research','tools','--workspace',str(tmp_path)],
        input=json.dumps(request)+'\n',capture_output=True,text=True,check=True)
    scores=json.loads(proc.stdout)['result']
    for name in expected:
        assert scores['metrics'][name]['intervals'][0]['observed_coverage']==1
        assert scores['metrics'][name]['direction']['accuracy_including_abstentions']==1
    assert not scores['blinding_verified'] and not scores['mechanism_verified']


def test_energy_origin_shift_preserves_relative_sum_and_off_diagonal_coupling(experiment,tmp_path):
    _,evidence=fixtures(experiment,tmp_path);endpoint=evidence['point-001'];report=load(endpoint['report'])
    path=endpoint['diagnostics']['coupling'];meta=load(path)
    kinds=('bands','relative_bands','coupling','singular')
    before={n:measure_diagnostic(report,endpoint['report'],path,definitions()[n])[0] for n in kinds}
    with np.load(path.parent/meta['artifact']) as z:arrays={k:z[k] for k in z.files}
    shift=1.75
    # Change only the energy origin: H' = H + shift*S and Ef' = Ef + shift.
    arrays['hamiltonian_ev']+=shift*arrays['overlap'];arrays['eigenvalues_ev']+=shift
    meta['fermi_ev']+=shift
    np.savez_compressed(path.parent/meta['artifact'],**arrays)
    meta['artifact_sha256']=sha256(path.parent/meta['artifact']);save(path,meta)
    after={n:measure_diagnostic(report,endpoint['report'],path,definitions()[n])[0] for n in kinds}
    for n in kinds[1:]:assert after[n]==pytest.approx(before[n],abs=1e-12)
    electrons=sum(float(w)*decompose(h,s,meta['fermi_ev'],meta['temperature_ev'])['occupations'].sum()
        for w,h,s in zip(arrays['k_weights'],arrays['hamiltonian_ev'],arrays['overlap']))
    assert after['bands']-before['bands']==pytest.approx(shift*electrons,abs=1e-12)
