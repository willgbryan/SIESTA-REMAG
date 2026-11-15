"""Observable-specific exploratory calibration from saved matched interventions.

Observed errors are empirical discrepancies on this set, never transferable
uncertainty bounds. This performs no new calculation or parameter search.
"""
from pathlib import Path
import math

from ..corpus import sha256
from ..discovery.common import load
from .convergence import numerical_configuration


def _series(paths):
    rows, fixed = {}, None
    for value in paths:
        path=Path(value); report=load(path)
        if report.get('format')!='REMAG-EXPERIMENT-RESULT/1' or report.get('status')!='measured':
            raise ValueError('Calibration requires completed research reports')
        stages=report['stages']
        if ([s['stage'] for s in stages]!=['scalar','soc-z','soc-x','soc-y']
                or not all(s['accepted'] and s.get('branch',{}).get('passed',True) for s in stages)):
            raise ValueError('Calibration requires accepted directional stages and branches')
        source=report['source_sha256']
        if source!=report['request']['source_sha256'] or source in rows:
            raise ValueError('Source mismatch or repeated calibration structure')
        config=numerical_configuration(report)
        config.pop('source_sha256')
        if fixed is None:fixed=config
        elif config!=fixed:raise ValueError('Each calibration tier must have one numerical configuration')
        for name,h in report['artifacts'].items():
            artifact=(path.parent/name).resolve()
            if not artifact.is_relative_to(path.parent.resolve()) or sha256(artifact)!=h:
                raise ValueError('Calibration artifact binding differs')
        obs=report['observations']
        values={q:obs['soc-x'][q]-obs['soc-z'][q] for q in ('total_energy_ev','free_energy_ev')}
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite calibration energies')
        rows[source]=dict(values=values,report_sha256=sha256(path),wall_seconds=report.get('wall_seconds'))
    return rows,fixed


def calibrate_response(exploratory, reference, *, baseline_source_sha256, resolution_ev):
    """Compare response sign, magnitude and pair ranking at a declared resolution.

    Both tiers need the same sources, one reference structure plus at least two
    interventions. Values within resolution of zero/ties remain unresolved.
    """
    if type(resolution_ev) not in (int,float) or not math.isfinite(resolution_ev) or resolution_ev<=0:
        raise ValueError('Declare a positive finite response resolution in eV/cell')
    if not 3<=len(exploratory)<=100 or not 3<=len(reference)<=100:
        raise ValueError('Calibration needs 3..100 structures per tier')
    cheap,cheap_config=_series(exploratory);dense,dense_config=_series(reference)
    if set(cheap)!=set(dense) or baseline_source_sha256 not in cheap:
        raise ValueError('Calibration requires matching structures and baseline in both tiers')
    # Changes in executable/runtime and numerical settings are explicitly exposed,
    # not mislabeled a one-variable convergence experiment.
    changes={k:dict(exploratory=cheap_config[k],reference=dense_config[k])
             for k in cheap_config if cheap_config[k]!=dense_config[k]}
    output={}
    sign=lambda x: 1 if x>resolution_ev else -1 if x < -resolution_ev else 0
    sources=sorted(cheap)
    for q in ('total_energy_ev','free_energy_ev'):
        responses=[];ranks=[]
        for source in sources:
            if source==baseline_source_sha256:continue
            a=cheap[source]['values'][q]-cheap[baseline_source_sha256]['values'][q]
            b=dense[source]['values'][q]-dense[baseline_source_sha256]['values'][q]
            resolved=bool(sign(a) and sign(b))
            responses.append(dict(source_sha256=source,exploratory_response_ev=a,reference_response_ev=b,
                error_ev=a-b,sign_status=('agree' if sign(a)==sign(b) else 'disagree') if resolved else 'unresolved'))
        for i,a in enumerate(sources):
            for b in sources[i+1:]:
                ca=sign(cheap[b]['values'][q]-cheap[a]['values'][q])
                cb=sign(dense[b]['values'][q]-dense[a]['values'][q])
                ranks.append(dict(before_source_sha256=a,after_source_sha256=b,
                    status=('agree' if ca==cb else 'disagree') if ca and cb else 'unresolved'))
        errors=[abs(r['error_ev']) for r in responses]
        output[q]=dict(responses=responses,mean_abs_response_error_ev=sum(errors)/len(errors),
            max_abs_response_error_ev=max(errors),pair_rankings=ranks,
            resolved_sign_agreements=sum(r['sign_status']=='agree' for r in responses),
            resolved_sign_disagreements=sum(r['sign_status']=='disagree' for r in responses),
            unresolved_signs=sum(r['sign_status']=='unresolved' for r in responses))
    return dict(format='REMAG-EXPLORATORY-CALIBRATION/1',baseline_source_sha256=baseline_source_sha256,
        resolution_ev=resolution_ev,exploratory=cheap,reference=dense,
        exploratory_configuration=cheap_config,reference_configuration=dense_config,
        changed_settings=changes,observables=output,
        scope='Observed response/sign/ranking agreement on these matched structures; no transferable error bound or default selected',
        reference_physical_accuracy_qualified=False,latency_target_demonstrated=False)
