"""Finite saved-evidence stability and structural-variation metrics.

The reference phase set and perturbation paths are declared before search.
These are static numerical screens, not thermodynamic/dynamical certification.
"""
from collections import Counter
import math
import re

import numpy as np

from ..corpus import sha256
from ..discovery.common import load
from ..discovery.config import Settings
from .convergence import numerical_configuration
from .series import validate_design, verify_series, _configuration, _branches


def validate_metrics(policy):
    from .discovery_scores import _fields
    from gpac.candidate_identity import POLICY
    if 'candidate_equivalence' in policy and policy['candidate_equivalence'] != POLICY:
        raise ValueError('Unknown declared candidate-equivalence policy')
    if 'stability' in policy:
        p=policy['stability']
        _fields(p,{'stage','reference_sources','maximum_energy_above_reference_hull_ev_atom',
            'maximum_force_ev_ang','maximum_stress_ev_ang3'})
        if p['stage'] not in ('scalar','soc-z'):raise ValueError('Stability uses scalar or soc-z energies')
        sources=p['reference_sources']
        if (not isinstance(sources,list) or not 1<=len(sources)<=64 or len(set(sources))!=len(sources)
                or any(not isinstance(s,str) or not re.fullmatch('[0-9a-f]{64}',s) for s in sources)):
            raise ValueError('Declare 1..64 distinct reference phase sources')
        for k in ('maximum_energy_above_reference_hull_ev_atom','maximum_force_ev_ang','maximum_stress_ev_ang3'):
            if type(p[k]) not in (int,float) or not math.isfinite(p[k]) or p[k]<0:
                raise ValueError('Stability thresholds must be finite and nonnegative')
    if 'robustness' in policy:
        p=policy['robustness'];_fields(p,{'designs','maximum_anisotropy_change_MJ_m3','maximum_spin_polarization_change_tesla'})
        if not isinstance(p['designs'],list) or not 1<=len(p['designs'])<=8:raise ValueError('Declare 1..8 structural variation paths')
        p['designs']=[validate_design(d) for d in p['designs']]
        if len({d['parameter'] for d in p['designs']})!=len(p['designs']):raise ValueError('Variation path names must differ')
        for k in ('maximum_anisotropy_change_MJ_m3','maximum_spin_polarization_change_tesla'):
            if type(p[k]) not in (int,float) or not math.isfinite(p[k]) or p[k]<0:
                raise ValueError('Robustness thresholds must be finite and nonnegative')


def properties(report):
    g=report['structure']['geometry'];volume=float(np.linalg.det(g['cell']));n=len(g['atomic_numbers'])
    if not math.isfinite(volume) or volume<=0 or n<1:raise ValueError('Invalid physical cell')
    obs=report['observations'];moment=np.asarray(obs['soc-z']['total_moment_muB'],float)
    if moment.shape!=(3,) or not np.isfinite(moment).all():raise ValueError('Invalid native moment vector')
    result={}
    for quantity,label in [('total_energy_ev','E'),('free_energy_ev','F')]:
        value=float(obs['soc-x'][quantity]-obs['soc-z'][quantity])
        if not math.isfinite(value):raise ValueError('Nonfinite directional energy')
        result.update({label+'_anisotropy_ev_cell':value,label+'_anisotropy_ev_atom':value/n,
                       label+'_anisotropy_MJ_m3':value/volume*160217.6634})
    result.update(volume_angstrom3=volume,site_count=n,spin_moment_muB=float(np.linalg.norm(moment)),
        spin_polarization_tesla=float(np.linalg.norm(moment))*11.654064772740647/volume,
        moment_scope='Native soc-z total spin moment; not a proven saturation or magnetic-ground-state value')
    return result


def _phase_configuration(report):
    c=numerical_configuration(report)
    for k in ('source_sha256','pseudopotentials','basis_spec'):c.pop(k)
    c['threads']=report.get('threads')
    return c


def _compatible_phases(reports):
    # Chemical references can contain different species. Shared species must
    # retain their pseudopotential and explicit shell definitions exactly.
    common=None;pseudos={};shells={}
    for r in reports:
        c=_phase_configuration(r)
        if common is not None and c!=common:raise ValueError('Reference phase numerical model or runtime differs')
        common=c
        for name,h in r['request']['pseudopotentials'].items():
            if name in pseudos and pseudos[name]!=h:raise ValueError('Reference phase pseudopotentials differ')
            pseudos[name]=h
        for species,basis in (r['request'].get('basis_spec') or {}).get('species',{}).items():
            if species in shells and shells[species]!=basis:raise ValueError('Reference phase basis shells differ')
            shells[species]=basis


def reference_mixture(composition, energies, reference_compositions):
    """Minimum energy/atom mixture at a conserved composition, or no spanning set.

    Energies have already passed report/model validation. The LP is bounded by
    nonnegative atomic fractions summing to one; residuals are rechecked.
    """
    from scipy.optimize import linprog
    species=sorted(set(composition)|{z for c in reference_compositions for z in c})
    if len(species)>8:raise ValueError('At most eight species in a finite stability screen')
    target=np.array([composition.get(z,0)/sum(composition.values()) for z in species])
    matrix=np.array([[c.get(z,0)/sum(c.values()) for c in reference_compositions] for z in species])
    energies=np.asarray(energies,float)
    if not np.isfinite(energies).all():raise ValueError('Nonfinite phase energies')
    # Subtract a common offset before solving to avoid large absolute DFT
    # energies dominating the small formation/decomposition differences.
    offset=float(energies.min())
    result=linprog(energies-offset,A_eq=matrix,b_eq=target,bounds=(0,1),method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':10})
    if result.status==2:return None
    if not result.success:raise ValueError('Reference mixture optimizer did not complete: '+result.message)
    weights=result.x
    if (not np.isfinite(weights).all() or np.min(weights)<-1e-9 or abs(sum(weights)-1)>1e-8
            or np.max(np.abs(matrix@weights-target))>1e-8):raise ValueError('Reference mixture conservation residual exceeds tolerance')
    return dict(energy_ev_atom=float(energies@weights),atomic_fractions=weights.tolist(),
                maximum_composition_residual=float(np.max(np.abs(matrix@weights-target))))


def load_reference_phases(root, policy, references):
    from .study import _path
    from .evidence import verify_report as _verified_report
    from .discovery_scores import _fields,report_interval
    declared=set(policy['reference_sources'])
    if not isinstance(references,dict) or set(references)-declared:raise ValueError('Unregistered reference phase')
    bindings=[];reports=[];rows=[];windows={};stage=policy['stage']
    for source in policy['reference_sources']:
        row=dict(source_sha256=source,status='pending');rows.append(row)
        if source not in references:continue
        ep=references[source];_fields(ep,{'report'},{'collection'});path=_path(root,ep['report'])
        r=_verified_report(path,_path(root,ep['collection']) if 'collection' in ep else None)
        if r['source_sha256']!=source:raise ValueError('Reference phase source differs')
        h=sha256(path);bindings.append(h);windows[h]=report_interval(r)
        healthy=(r['status']=='measured' and r['request'].get('fixed_geometry') is True and
            any(s['stage']==stage and s['accepted'] and s.get('branch',{}).get('passed',True) for s in r['stages']))
        row.update(status='measured' if healthy else 'failed_native',report=ep['report'],report_sha256=h,verification=r['verification'])
        if healthy:reports.append(r)
    return reports,rows,bindings,windows


def assess_stability(root, candidate, policy, references, *, prepared=None):
    reports,rows,bindings,windows=prepared if prepared is not None else load_reference_phases(root,policy,references)
    stage=policy['stage']
    _compatible_phases([candidate,*reports])
    result=dict(reference_phases=rows,stage=stage,reference_set_complete=len(reports)==len(rows),
        observed_static_screen_passed=None,energies=None,
        scope='Static energies relative to the declared finite phase set, plus candidate residual forces/stress. Missing phases, phonons, magnetic-order competition, thermal/configurational entropy and numerical error remain unassessed; no universal thermodynamic stability certificate.')
    obs=candidate['observations'][stage];g=candidate['structure']['geometry'];n=len(g['atomic_numbers'])
    forces=np.asarray(obs['forces_ev_ang'],float);stress=np.asarray(obs['stress_ev_ang3'],float)
    if forces.shape!=(n,3) or stress.shape!=(3,3) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
        raise ValueError('Invalid residual force/stress arrays')
    fmax=float(np.max(np.linalg.norm(forces,axis=1)));smax=float(np.max(np.abs(stress)))
    result['stationarity']=dict(maximum_force_ev_ang=fmax,maximum_absolute_stress_ev_ang3=smax,
        passed=fmax<=policy['maximum_force_ev_ang'] and smax<=policy['maximum_stress_ev_ang3'])
    if result['reference_set_complete']:
        composition=Counter(g['atomic_numbers'])
        compositions=[Counter(r['structure']['geometry']['atomic_numbers']) for r in reports]
        comparisons={}
        for q,label in [('total_energy_ev','E'),('free_energy_ev','F')]:
            value=obs[q]/n
            if not math.isfinite(value):raise ValueError('Nonfinite candidate energy')
            mixture=reference_mixture(composition,[r['observations'][stage][q]/sum(c.values()) for r,c in zip(reports,compositions)],compositions)
            if mixture is None:comparisons[label]=dict(status='reference_set_does_not_span_composition');continue
            delta=value-mixture['energy_ev_atom']
            comparisons[label]=dict(status='measured',candidate_ev_atom=value,reference_mixture=mixture,
                signed_energy_relative_to_references_ev_atom=delta,
                energy_above_candidate_plus_reference_hull_ev_atom=max(0.,delta),
                passed=max(0.,delta)<=policy['maximum_energy_above_reference_hull_ev_atom'])
        result['energies']=comparisons
        if all(v['status']=='measured' for v in comparisons.values()):
            result['observed_static_screen_passed']=result['stationarity']['passed'] and all(v['passed'] for v in comparisons.values())
    return result,bindings,windows


def assess_robustness(root, candidate, policy, evidence, *, baseline_accepted=True):
    from .study import _path
    from .discovery_scores import _fields,_endpoint
    names={d['parameter'] for d in policy['designs']}
    if not isinstance(evidence,dict) or set(evidence)-names:raise ValueError('Unregistered structural variation path')
    base=properties(candidate) if baseline_accepted else None;results={};hashes=[];windows={}
    n=len(candidate['structure']['geometry']['atomic_numbers']);settings=Settings.from_dict(candidate['request']['settings'])
    for design in policy['designs']:
        name=design['parameter'];out=dict(status='pending',observed_robustness_passed=None);results[name]=out
        if name not in evidence:continue
        e=evidence[name];_fields(e,{'series','reports'});directory=_path(root,e['series']);series=verify_series(directory)
        if series['design']!=design or series['parent_source_sha256']!=candidate['source_sha256']:
            raise ValueError('Structural variation design or candidate parent differs')
        if not isinstance(e['reports'],dict) or set(e['reports'])-{p['id'] for p in series['points']}:
            raise ValueError('Unknown structural variation point')
        rows=[]
        for point in series['points']:
            row=dict(id=point['id'],q_exact=point['q_exact'],source_sha256=point['source_sha256'],status='pending');rows.append(row)
            if point['id'] not in e['reports']:continue
            export=load(directory/point['bundle']/'structure.json')
            r,b=_endpoint(root,e['reports'][point['id']],point['source_sha256'],export)
            hashes.append(b['report_sha256']);windows[b['report_sha256']]=b['interval'];row['evidence']=b
            if (_configuration(r['request'])!=_configuration(candidate['request']) or
                    any(r.get(k)!=candidate.get(k) for k in ('executable_sha256','environment','threads'))):
                raise ValueError('Structural robustness uses another numerical configuration or runtime')
            if not b['accepted']:row['status']='failed_native';continue
            if not baseline_accepted:row['status']='baseline_failed';continue
            branches=_branches(r,candidate,point['parent_to_child'],list(range(n)),['scalar','soc-z','soc-x','soc-y'],settings)
            row.update(properties=properties(r),branches=branches,status='measured' if all(c['passed'] for c in branches.values()) else 'failed_branch')
            row['changes']={k:row['properties'][k]-base[k] for k in ('E_anisotropy_MJ_m3','F_anisotropy_MJ_m3','spin_polarization_tesla')}
            row['within_thresholds']=all(abs(row['changes'][k])<=policy['maximum_anisotropy_change_MJ_m3'] for k in ('E_anisotropy_MJ_m3','F_anisotropy_MJ_m3')) and abs(row['changes']['spin_polarization_tesla'])<=policy['maximum_spin_polarization_change_tesla']
        complete=all(r['status']=='measured' for r in rows)
        out.update(status='measured' if complete else 'incomplete',points=rows,series_sha256=series['sha256'],
            observed_robustness_passed=all(r['within_thresholds'] for r in rows) if complete else None)
    return dict(paths=results,baseline_accepted=baseline_accepted,observed_robustness_passed=(all(r['observed_robustness_passed'] for r in results.values())
        if all(r['observed_robustness_passed'] is not None for r in results.values()) else None),
        scope='Finite preregistered exact structural samples relative to the selected candidate; signed E/F anisotropy density and spin polarization. Missing or branch-changing samples remain inconclusive; no unsampled robustness, statistical uncertainty or finite-temperature claim.'),hashes,windows
