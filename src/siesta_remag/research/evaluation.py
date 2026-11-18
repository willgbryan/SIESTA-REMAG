"""Preregistered electronic/anisotropy responses and finite prediction scores.

These are saved-evidence tools. They neither execute a solver nor establish
blinding, numerical qualification, independent samples or a physical mechanism.
"""
from pathlib import Path
import json
import math
import re

import numpy as np

from ..corpus import sha256
from ..discovery.common import digest
from .convergence import numerical_configuration
from .inspection import load_spectrum, _select


def validate_target(target):
    if not isinstance(target,dict):raise ValueError('Expected a response target object')
    if target.get('kind')=='diagnostic_response':
        from .response_diagnostics import validate_diagnostic
        if (set(target)!={'kind','baseline_source_sha256','diagnostic'} or not isinstance(target['baseline_source_sha256'],str)
                or not re.fullmatch('[0-9a-f]{64}',target['baseline_source_sha256'])):
            raise ValueError('Freeze a diagnostic and exact baseline source hash')
        diagnostic,unit=validate_diagnostic(target['diagnostic'])
        return dict(kind='diagnostic_response',baseline_source_sha256=target['baseline_source_sha256'],diagnostic=diagnostic),unit
    common={'kind','baseline_source_sha256','metric'}
    kind=target.get('kind')
    extra={'stage','spin','selection'} if kind=='spectral_response' else set()
    if kind not in ('spectral_response','anisotropy_response') or set(target)!=common|extra:
        raise ValueError('Expected an exact spectral or anisotropy response definition')
    if not isinstance(target['baseline_source_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',target['baseline_source_sha256']):
        raise ValueError('Response needs an exact baseline source hash')
    if kind=='anisotropy_response':
        if target['metric'] not in ('total_energy_ev','free_energy_ev'):
            raise ValueError('Select total or free-energy anisotropy response')
        unit='eV/cell'
    else:
        units={'projected_weight':'states/cell','projected_occupation':'electrons/cell','projected_energy_moment':'eV/cell'}
        if target['metric'] not in units:raise ValueError('Unknown spectral response metric')
        unit=units[target['metric']]
        if not isinstance(target['stage'],str) or not re.fullmatch(r'scalar|soc-[A-Za-z0-9_-]+',target['stage']):
            raise ValueError('Select a scalar or SOC stage')
        if type(target['spin']) is not int or target['spin'] not in ((0,1) if target['stage']=='scalar' else (0,)):
            raise ValueError('Freeze a scalar spin channel or the single SOC spinor channel')
        s=target['selection']
        if not isinstance(s,dict) or set(s)-{'atom_indices','orbital_indices','l','k_indices','energy_window_ev'}:
            raise ValueError('Response selection cannot truncate states or select unknown fields')
        for key,value in s.items():
            if key=='energy_window_ev':
                if (not isinstance(value,list) or len(value)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in value)
                        or not -100<=value[0]<value[1]<=100):raise ValueError('Invalid response energy window')
            elif (not isinstance(value,list) or not value or any(type(x) is not int or x<0 for x in value)
                  or len(set(value))!=len(value)):raise ValueError('Response selections need distinct nonnegative indices')
        target=target | {'selection':{'energy_window_ev':[-3,3]} | s}
    # Detach caller-owned containers before persisting a scientific definition.
    return json.loads(json.dumps(target,allow_nan=False)),unit


def _endpoint(endpoint, target):
    from .evidence import verify_report as _verified_report
    required={'report'} | ({'spectrum'} if target['kind']=='spectral_response' else {'diagnostic'} if target['kind']=='diagnostic_response' else set())
    if not isinstance(endpoint,dict) or not required<=set(endpoint) or set(endpoint)-required-{'collection'}:
        raise ValueError('Response endpoint requires a report and its target-specific spectrum or diagnostic')
    path=Path(endpoint['report']).resolve()
    r=_verified_report(path,Path(endpoint['collection']).resolve() if 'collection' in endpoint else None)
    stages=['soc-z','soc-x'] if target['kind']=='anisotropy_response' else [target['diagnostic']['stage'] if target['kind']=='diagnostic_response' else target['stage']]
    healthy=(r['status']=='measured' and all(any(s['stage']==name and s['accepted'] and s.get('branch',{}).get('passed',True)
                                              for s in r['stages']) for name in stages))
    return r,dict(report_sha256=sha256(path),source_sha256=r['source_sha256'],native_status=r['status'],
                  accepted=healthy,verification=r['verification'])


def _spectral(endpoint, report, target):
    path=Path(endpoint['spectrum']);m,a=load_spectrum(path);stage=target['stage']
    if (m['collinear_spin'] != (target['spin'] if stage=='scalar' else None)
            or m['spin_components'] not in ((1,2) if stage=='scalar' else (4,8))):
        raise ValueError('Spectrum spin channel differs from the prediction')
    for name,key in [('HSX','hsx_sha256'),('KP','kp_sha256'),('EIG','eig_sha256')]:
        expected=report['artifacts'].get(stage+'/remag.'+name)
        if not expected or m.get(key)!=expected:raise ValueError('Spectrum is not bound to the requested native stage')
    orbitals,ks,window,_=_select(m,a,target['selection'])
    relative=a['eigenvalues_ev'][ks]-m['fermi_ev']
    population=a['orbital_mulliken'][ks][:,orbitals,:].sum(axis=1)
    weight=a['k_weights'][ks,None]*population*((relative>=window[0]) & (relative<=window[1]))
    if target['metric']=='projected_occupation':
        # Exported occupations already include twofold degeneracy for nspin=1.
        weight=weight*a['occupations'][ks]
    else:
        weight=weight*(2 if m['spin_components']==1 else 1)
        if target['metric']=='projected_energy_moment':weight=weight*relative
    return float(weight.sum()),m,a,dict(spectrum_sha256=sha256(path),artifact_sha256=m['artifact_sha256'],
        selected_k_weight=float(a['k_weights'][ks].sum()),energy_reference='each endpoint Fermi level',
        collinear_spin=m['collinear_spin'],projection=m['projection'],occupations_model=m['occupations_model'])


def response_observation(target, before, after, *, source_sha256):
    target,unit=validate_target(target)
    a,ea=_endpoint(before,target);b,eb=_endpoint(after,target)
    if a['source_sha256']!=target['baseline_source_sha256'] or b['source_sha256']!=source_sha256:
        raise ValueError('Response endpoints differ from the predicted baseline or intervention source')
    if a['source_sha256']==b['source_sha256']:raise ValueError('A structural response needs distinct endpoint sources')
    configs=[{k:v for k,v in numerical_configuration(r).items() if k!='source_sha256'} for r in (a,b)]
    if configs[0]!=configs[1]:raise ValueError('Response endpoints use different numerical settings or runtime')
    if a['structure']['geometry']['atomic_numbers']!=b['structure']['geometry']['atomic_numbers']:
        raise ValueError('Response requires the same ordered atom species')
    evidence=dict(target=target,before=ea,after=eb,numerical_configuration=configs[0],
        physical_accuracy_qualified=False,mechanism_verified=False,blinding_verified=False)
    if not ea['accepted'] or not eb['accepted']:
        return dict(status='failed',value=None,unit=unit,evidence=evidence)
    if target['kind']=='anisotropy_response':
        q=target['metric']
        va,vb=[r['observations']['soc-x'][q]-r['observations']['soc-z'][q] for r in (a,b)]
        evidence['scope']='Change in self-consistent E[x]-E[z] or F[x]-F[z]; finite native acceptance, not physical qualification'
    elif target['kind']=='diagnostic_response':
        from .response_diagnostics import measure_diagnostic,compare_diagnostics
        from .series import _branches
        from ..discovery.config import Settings
        definition=target['diagnostic']
        va,sa,ca=measure_diagnostic(a,before['report'],before['diagnostic'],definition)
        vb,sb,cb=measure_diagnostic(b,after['report'],after['diagnostic'],definition)
        ea.update(sa);eb.update(sb);evidence['diagnostic_correspondence']=compare_diagnostics(ca,cb)
        mapping=list(range(len(a['structure']['geometry']['atomic_numbers'])))
        checks=_branches(b,a,mapping,mapping,[definition['stage']],Settings.from_dict(a['request']['settings']))
        evidence.update(before_value=va,after_value=vb,branch_checks=checks,
            scope='Response of a declared onsite moment or partial matrix proxy; no unique SOC/MAE attribution or cross-geometry band tracking')
        if not all(c['passed'] for c in checks.values()):return dict(status='failed',value=None,unit=unit,evidence=evidence)
    else:
        va,ma,da,sa=_spectral(before,a,target);vb,mb,db,sb=_spectral(after,b,target)
        for key in ('orbitals','spin_components','collinear_spin','spin_frame','projection','temperature_ev','occupations_model'):
            if ma.get(key)!=mb.get(key):raise ValueError('Response spectral metadata differs: '+key)
        from .sampling import compare_sampling
        evidence['sampling_correspondence']=compare_sampling(da,db,
            dict(report_path=before['report'],report=a,metadata=ma,stage=target['stage']),
            dict(report_path=after['report'],report=b,metadata=mb,stage=target['stage']))
        ea.update(sa);eb.update(sb)
        evidence['scope']='Signed Mulliken window integral on literal ordered atom/orbital indices; no k-weight renormalization, band tracking, radial-basis identity or SOC attribution. Energy moment is not a total energy or MAE.'
    if not all(math.isfinite(v) for v in (va,vb,vb-va)):raise ValueError('Nonfinite response observation')
    evidence.update(before_value=va,after_value=vb)
    return dict(status='measured',value=vb-va,unit=unit,evidence=evidence)


def prediction_scores(notebook, experiment_ids, resolutions):
    """Retain all selected predictions, including failures and missing outcomes.

    Scores group one observable/unit/definition; intervals are symmetric radii
    with optional stated nominal coverage, never inferred error bars.
    """
    if (not isinstance(experiment_ids,list) or not 1<=len(experiment_ids)<=200
            or any(not isinstance(e,str) for e in experiment_ids) or len(set(experiment_ids))!=len(experiment_ids)):
        raise ValueError('Select 1..200 distinct experiment IDs')
    if not isinstance(resolutions,dict) or any(not isinstance(k,str) or type(v) not in (int,float) or not math.isfinite(v) or v<0 for k,v in resolutions.items()):
        raise ValueError('Declare nonnegative response resolutions by observable')
    groups={};rows=[];unpredicted=[];without_predictions=[]
    for eid in experiment_ids:
        experiment=notebook.get(eid)
        if experiment['kind']!='experiment':raise ValueError('Expected an experiment event')
        events=[notebook.get(r[0]) for r in notebook.db.execute('SELECT id FROM events WHERE parent=? ORDER BY rowid',(eid,))]
        predictions={e['payload']['observable']:e for e in events if e['kind']=='prediction'}
        if not predictions:without_predictions.append(eid)
        outcomes={e['payload']['observable']:e for e in events if e['kind']=='outcome'}
        unpredicted.extend(dict(experiment_id=eid,observable=q,outcome_id=e['id']) for q,e in outcomes.items() if q not in predictions)
        for q,p in predictions.items():
            if q not in resolutions:raise ValueError('Every predicted observable needs an explicit resolution')
            data=p['payload'];definition=dict(unit=data['unit'],target=data.get('target'))
            group=groups.setdefault(q,dict(definition=definition,rows=[]))
            if group['definition']!=definition:raise ValueError('Do not pool unlike observable units or response definitions')
            o=outcomes.get(q);evaluation=notebook.evaluate(eid,q)
            row=dict(experiment_id=eid,source_sha256=experiment['payload']['structure_sha256'],observable=q,
                prediction_id=p['id'],prediction_sha256=digest(p),predicted=data['value'],uncertainty=data['uncertainty'],
                interval_probability=data.get('interval_probability'),outcome_id=o['id'] if o else None,
                outcome_sha256=digest(o) if o else None,status=o['payload']['status'] if o else 'pending',
                observed=o['payload']['value'] if o else None,evaluation=evaluation)
            group['rows'].append(row);rows.append(row)
    if set(resolutions)!=set(groups):raise ValueError('Resolution keys must match selected predicted observables')
    metrics={}
    for q,g in groups.items():
        measured=[r for r in g['rows'] if r['evaluation']['status']=='evaluated'];errors=np.array([r['evaluation']['error'] for r in measured])
        scale=max(abs(errors),default=0.)
        normalized=errors/scale if scale else errors
        resolution=resolutions[q];resolved=[r for r in measured if abs(r['observed'])>resolution]
        asserted=[r for r in resolved if abs(r['predicted'])>resolution]
        correct=sum((r['predicted']>0)==(r['observed']>0) for r in asserted)
        intervals=[]
        for nominal in sorted({r['interval_probability'] for r in measured if r['interval_probability'] is not None}):
            selected=[r for r in measured if r['interval_probability']==nominal]
            covered=sum(r['evaluation']['within_predicted_uncertainty'] for r in selected)
            intervals.append(dict(nominal_probability=nominal,count=len(selected),covered=covered,
                observed_coverage=covered/len(selected),coverage_minus_nominal=covered/len(selected)-nominal,
                mean_width=float(np.mean([2*r['uncertainty'] for r in selected]))))
        metrics[q]=dict(**g['definition'],prediction_count=len(g['rows']),measured_count=len(measured),
            failed_count=sum(r['status']=='failed' for r in g['rows']),inconclusive_count=sum(r['status']=='inconclusive' for r in g['rows']),
            pending_count=sum(r['status']=='pending' for r in g['rows']),
            unique_measured_sources=len({r['source_sha256'] for r in measured}),
            mae=float(np.mean(abs(normalized))*scale) if len(errors) else None,rmse=float(np.sqrt(np.mean(normalized**2))*scale) if len(errors) else None,
            bias=float(np.mean(normalized)*scale) if len(errors) else None,resolution=resolution,
            direction=dict(resolved_observations=len(resolved),unresolved_observations=len(measured)-len(resolved),
                asserted_predictions=len(asserted),abstained_predictions=len(resolved)-len(asserted),correct=correct,
                accuracy_including_abstentions=correct/len(resolved) if resolved else None),
            intervals=intervals,uncertainty_without_nominal_count=sum(r['uncertainty'] is not None and r['interval_probability'] is None for r in measured))
    return dict(format='REMAG-PREDICTION-SCORES/1',experiment_ids=experiment_ids,metrics=metrics,rows=rows,
        unpredicted_outcomes=unpredicted,experiments_without_predictions=without_predictions,blinding_verified=False,independent_samples_assessed=False,
        mechanism_verified=False,physical_accuracy_qualified=False,
        scope='Finite scores for the selected recorded predictions; failed/pending cases retained. Selection, leakage, independent holdout, transfer and equal-compute advantage require separate evaluation design.')
