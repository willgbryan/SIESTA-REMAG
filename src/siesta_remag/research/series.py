"""Controlled GPAC perturbation families and saved REMAG response diagnostics.

Preparation creates independently runnable bundles. It never submits, queues,
retries or reserves a calculation. Analysis retains missing and failed points.
"""
from fractions import Fraction
import math
from pathlib import Path
import re

import numpy as np

from ..corpus import sha256
from ..discovery.common import digest, load, save
from ..discovery.config import Settings
from .authoring import _worker, author_candidate, verify_candidate
from .experiments import prepare_experiment, verify_experiment


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value)-set(required)-set(optional):
        raise ValueError('Missing or unknown response-design fields')


def validate_design(design):
    from gpac.exact import parse_exact_scalar, validate_parameter_edits
    _fields(design, {'format', 'parameter', 'values', 'control'})
    if design['format'] != 'REMAG-RESPONSE-DESIGN/1': raise ValueError('Expected REMAG-RESPONSE-DESIGN/1')
    if not isinstance(design['parameter'], str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{0,47}', design['parameter']):
        raise ValueError('Parameter needs a short identifier')
    raw = design['values']
    if not isinstance(raw, list) or not 3 <= len(raw) <= 9: raise ValueError('Select 3..9 exact samples including zero')
    qs = [parse_exact_scalar(q, f'/values/{i}') for i,q in enumerate(raw)]
    if any(abs(q)>Fraction(1,2) for q in qs) or not all(a<b for a,b in zip(qs,qs[1:])) or 0 not in qs:
        raise ValueError('Samples must increase strictly in [-1/2,1/2] and include zero')
    if not np.all(np.diff([float(q) for q in qs])>0): raise ValueError('Distinct exact samples collapse at numerical response precision')
    c = design['control']; kind = c.get('kind') if isinstance(c,dict) else None
    fields = {'axial_strain': {'axis'}, 'isotropic_strain': set(), 'fixed_volume_tetragonal': set(),
              'fractional_displacement': {'sites'}, 'coordination_strain': {'center_site','neighbors'},
              'metric_path': {'rates'}}
    if kind not in fields: raise ValueError('Unknown physical response path')
    _fields(c, {'kind'} | fields[kind]); c=dict(c)
    if kind=='axial_strain' and (type(c['axis']) is not int or c['axis'] not in (0,1,2)):
        raise ValueError('Strain axis must be 0, 1 or 2')
    if kind=='fractional_displacement':
        if not isinstance(c['sites'],list) or not 1<=len(c['sites'])<=256: raise ValueError('Select 1..256 displacement sites')
        rows=[];seen=set()
        for i,r in enumerate(c['sites']):
            _fields(r,{'site_index','direction_fractional'});index=r['site_index']
            if type(index) is not int or not 0<=index<256 or index in seen: raise ValueError('Distinct bounded displacement site indices required')
            seen.add(index)
            v=r['direction_fractional']
            if not isinstance(v,list) or len(v)!=3: raise ValueError('Displacement direction needs three exact components')
            v=[parse_exact_scalar(x,f'/control/sites/{i}/direction_fractional/{j}') for j,x in enumerate(v)]
            if not any(v) or any(abs(x)>1 for x in v): raise ValueError('Nonzero displacement directions must lie in [-1,1]')
            rows.append(dict(site_index=index,direction_fractional=list(map(str,v))))
        c['sites']=rows
    if kind=='coordination_strain':
        from gpac.exact import validate_coordination_control
        checked=validate_coordination_control(dict(center_site=c['center_site'],neighbors=c['neighbors'],scale='1'))
        c.update(center_site=checked['center_site'],neighbors=checked['neighbors'])
    if kind=='metric_path':
        rates=validate_parameter_edits(c['rates'],'/control/rates')
        if not rates or not any(rates.values()) or any(not k.startswith('metric.') for k in rates):
            raise ValueError('Metric path needs nonzero rates of exact metric parameters')
        c['rates']={k:str(v) for k,v in sorted(rates.items())}
    return dict(format=design['format'],parameter=design['parameter'],values=list(map(str,qs)),control=c)


def _request(parent, design, q):
    from gpac.parametric import describe
    c=design['control'];kind=c['kind'];q=Fraction(q)
    r=dict(expected_sha256=parent['source']['sha256'])
    if kind=='axial_strain':r.update(op='physical-strain',axis=c['axis'],amount=str(q))
    elif kind in ('isotropic_strain','fixed_volume_tetragonal'):
        r.update(op='physical-isotropic' if kind=='isotropic_strain' else 'physical-tetragonal',scale=str(1+q))
    elif kind=='fractional_displacement':
        r.update(op='physical-displace',symmetry_policy='explicit-p1',displacements=[dict(site_index=s['site_index'],
            delta_fractional=[str(q*Fraction(x)) for x in s['direction_fractional']]) for s in c['sites']])
    elif kind=='coordination_strain':
        r.update(op='physical-coordination-scale',symmetry_policy='explicit-p1',
            control=dict(center_site=c['center_site'],neighbors=c['neighbors'],scale=str(1+q)))
    else:
        space=describe(parent['source']['program'])
        if any(k not in space['parameters'] for k in c['rates']): raise ValueError('Metric rate is not a free parameter of this parent')
        r.update(op='apply',expected_space_sha256=space['space_sha256'],
            changes={k:str(Fraction(space['parameters'][k])+q*Fraction(rate)) for k,rate in c['rates'].items()})
    return r


def _child(parent, request):
    from gpac import ir
    from gpac.physical import export_geometry
    reply=_worker(request|dict(program=parent['source']['program']))['result']
    child=reply['program'] if request['op']=='apply' else reply['child']
    export=export_geometry(child['program'],length_unit='angstrom')
    a,b=[ir.execute(s['source']['program']) for s in (parent,export)]
    if request['op']=='apply':
        lookup={(s.z,s.x):i for i,s in enumerate(b.sites)}
        mapping=[lookup[(s.z,s.x)] for s in a.sites]  # Metric paths retain exact fractional sites.
    else:
        rows=sorted(reply['correspondence'],key=lambda x:x['parent_index'])
        if [r['parent_index'] for r in rows]!=list(range(len(a.sites))):raise ValueError('Incomplete parent correspondence')
        mapping=[r['child_index'] for r in rows]
    if sorted(mapping)!=list(range(len(b.sites))) or any(a.sites[i].z!=b.sites[j].z for i,j in enumerate(mapping)):
        raise ValueError('Response path must preserve all parent atoms with explicit correspondence')
    return export,mapping


def prepare_series(structure, pseudos, output, *, design, query, settings=None, mesh=None,
                   basis_size='DZP', basis_energy_shift_ry=.02, basis_spec=None, soc_matrix=False):
    from gpac.physical import verify_export
    from .queries import validate_query
    design=validate_design(design);query=validate_query(query)
    if mesh is None: raise ValueError('Response families require one explicit common k mesh')
    output=Path(output)
    if output.exists(): raise FileExistsError('Response series output must be new')
    parent=verify_export(load(structure));settings=settings or Settings()
    prepared=[]
    # Validate the entire path before creating any family artifacts.
    for q in design['values']:
        request=_request(parent,design,q);child,mapping=_child(parent,request)
        prepared.append((q,request,child,mapping))
    if len({r[2]['source']['sha256'] for r in prepared})!=len(prepared): raise ValueError('Distinct parameter values must produce distinct exact sources')
    if len({digest({k:r[2]['geometry'][k] for k in ('cell','fractional','atomic_numbers')}) for r in prepared})!=len(prepared):
        raise ValueError('Exact path collapses to duplicate numerical solver geometries')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'parent.json',parent);save(output/'design.json',design)
    rows=[]
    try:
        for i,(q,request,child,mapping) in enumerate(prepared):
            name=f'point-{i:03d}';candidate=output/name/'candidate'
            author_candidate(request,candidate,structure=output/'parent.json')
            if load(candidate/'structure.json')!=child: raise ValueError('Prepared candidate differs from validated path')
            bundle=output/name/'query'
            experiment=prepare_experiment(child,pseudos,bundle,settings=settings,mesh=mesh,basis_size=basis_size,
                basis_energy_shift_ry=basis_energy_shift_ry,basis_spec=basis_spec,query=query,soc_matrix=soc_matrix,
                candidate=candidate/'candidate.json',parameter=dict(name=design['parameter'],exact=q,unit='dimensionless'))
            rows.append(dict(id=name,q_exact=q,source_sha256=child['source']['sha256'],parent_to_child=mapping,
                candidate=str((candidate/'candidate.json').relative_to(output)),bundle=str(bundle.relative_to(output)),
                experiment_sha256=sha256(bundle/'experiment.json')))
        result=dict(format='REMAG-RESPONSE-SERIES/1',design=design,parent_source_sha256=parent['source']['sha256'],
            parameter_unit='dimensionless',baseline_id=next(r['id'] for r in rows if Fraction(r['q_exact'])==0),points=rows,
            files={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()},
            execution='Each bundle requires a separate explicit invocation; no queue or reservation',
            calculation_performed=False,physical_accuracy_qualified=False)
        result['sha256']=digest(result);save(output/'series.json',result)
        return result
    except Exception as exc:
        save(output/'preparation-failure.json',dict(error=type(exc).__name__,message=str(exc),completed_points=rows,
            calculation_performed=False,retry='Use a new output directory after diagnosis'))
        raise


def verify_series(directory):
    directory=Path(directory).resolve();r=load(directory/'series.json')
    if r.get('format')!='REMAG-RESPONSE-SERIES/1' or r.get('sha256')!=digest({k:v for k,v in r.items() if k!='sha256'}):
        raise ValueError('Response series manifest differs')
    if r['design']!=validate_design(r['design']) or load(directory/'design.json')!=r['design']:
        raise ValueError('Response design differs')
    if (r.get('parameter_unit')!='dimensionless' or r.get('calculation_performed') is not False
            or r.get('physical_accuracy_qualified') is not False):
        raise ValueError('Response preparation scope differs')
    for name,h in r['files'].items():
        p=(directory/name).resolve()
        if not p.is_relative_to(directory) or sha256(p)!=h: raise ValueError('Response series artifact changed: '+name)
    from gpac.physical import verify_export
    parent=verify_export(load(directory/'parent.json'))
    if parent['source']['sha256']!=r['parent_source_sha256'] or len(r['points'])!=len(r['design']['values']):raise ValueError('Response parent or samples differ')
    common=None
    for i,(point,q) in enumerate(zip(r['points'],r['design']['values'])):
        if point['id']!=f'point-{i:03d}' or point['q_exact']!=q or point['bundle']!=point['id']+'/query' or point['candidate']!=point['id']+'/candidate/candidate.json':
            raise ValueError('Response sample identity or paths differ')
        child,mapping=_child(parent,_request(parent,r['design'],q))
        candidate=verify_candidate(directory/point['candidate'])
        request=verify_experiment(directory/point['bundle'])
        config=_configuration(request)
        if (request.get('parameter')!=dict(name=r['design']['parameter'],exact=q,unit='dimensionless')
                or (common is not None and config!=common)):
            raise ValueError('Response parameter or common scientific settings differ')
        common=config
        if (child!=load(directory/point['bundle']/'structure.json') or point['parent_to_child']!=mapping
                or point['source_sha256']!=candidate['source_sha256'] or point['source_sha256']!=request['source_sha256']
                or point['experiment_sha256']!=sha256(directory/point['bundle']/'experiment.json')):
            raise ValueError('Response source, correspondence or query binding differs')
    if r['baseline_id']!=next(p['id'] for p in r['points'] if Fraction(p['q_exact'])==0): raise ValueError('Response baseline differs')
    return r


def validate_observables(observables):
    from .evaluation import validate_target
    if not isinstance(observables,dict) or not 1<=len(observables)<=16: raise ValueError('Select 1..16 named response observables')
    result={}
    for name,raw in observables.items():
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,47}',name): raise ValueError('Observable needs a short identifier')
        kind=raw.get('kind') if isinstance(raw,dict) else None
        from .response_diagnostics import KINDS, validate_diagnostic
        if kind in KINDS:
            o,unit=validate_diagnostic(raw);result[name]=dict(definition=o,unit=unit);continue
        fields={'energy':{'stage','quantity'},'anisotropy':{'stages','quantity'},'force':{'stage','site','component'},
                'moment':{'stage','site','component'},'stress':{'stage','component'},'spectral':{'stage','spin','metric','selection'}}
        if kind not in fields: raise ValueError('Unknown structural-response observable')
        _fields(raw,{'kind'}|fields[kind]);o=dict(raw)
        stages=o['stages'] if kind=='anisotropy' else [o['stage']]
        if (not isinstance(stages,list) or (kind=='anisotropy' and (len(stages)!=2 or stages[0]==stages[1]))
                or any(not isinstance(s,str) or not re.fullmatch(r'scalar|soc-[a-z][a-z0-9_-]{0,31}',s) for s in stages)):
            raise ValueError('Select measured stage labels; anisotropy is the first minus second stage')
        if kind=='anisotropy' and any(not s.startswith('soc-') for s in stages):raise ValueError('Anisotropy requires two SOC stages')
        if kind in ('energy','anisotropy') and o['quantity'] not in ('total_energy_ev','free_energy_ev'):
            raise ValueError('Select a total or free-energy quantity')
        if kind in ('moment','force'):
            if not ((kind=='moment' and o['site'] is None) or (type(o['site']) is int and 0<=o['site']<256)):
                raise ValueError('Select a parent-site index, or null for total moment')
            if not ((type(o['component']) is int and o['component'] in (0,1,2)) or (kind=='moment' and o['component']=='norm')):
                raise ValueError('Select a Cartesian component; moment also accepts norm')
        if kind=='stress' and (not isinstance(o['component'],list) or len(o['component'])!=2
                or any(type(i) is not int or i not in (0,1,2) for i in o['component'])):
            raise ValueError('Stress component needs two Cartesian indices')
        if kind=='spectral':
            t,unit=validate_target({k:v for k,v in o.items() if k!='kind'}|dict(kind='spectral_response',baseline_source_sha256='0'*64))
            o['selection']=t['selection']
        else:unit={'energy':'eV/cell','anisotropy':'eV/cell','force':'eV/angstrom','moment':'muB','stress':'eV/angstrom^3'}[kind]
        result[name]=dict(definition=o,unit=unit)
    return result


def _configuration(request):
    from .queries import validate_query
    keys=('kmesh','basis_size','basis_energy_shift_ry','pseudopotentials')
    physical=('mesh_ry','smearing_k','dm_tolerance','h_tolerance_ev','mixing_weight','mixing_history',
              'axis_tolerance_deg','branch_rms_mub','branch_net_fraction')
    return {k:request[k] for k in keys}|dict(basis_spec=request.get('basis_spec'),
        settings={k:request['settings'][k] for k in physical},query=validate_query(request.get('query',{'kind':'anisotropy'})))


def _axes(report):
    from ase import Atoms
    from ..discovery.inputs import axes
    from .queries import directions
    g=report['structure']['geometry']
    a=Atoms(numbers=g['atomic_numbers'],cell=g['cell'],scaled_positions=g['fractional'],pbc=True)
    return {'scalar':[0,0,1]}|{'soc-'+k:v for k,v in directions(report['request'].get('query',{'kind':'anisotropy'}),axes(a)).items()}


def _healthy(report, stages):
    return report['status']=='measured' and all(any(s['stage']==stage and s['accepted'] and
        s.get('branch',{}).get('passed',True) for s in report['stages']) for stage in stages)


def _branches(report, baseline, mapping, baseline_mapping, stages, settings):
    from ..discovery.review import branch_check
    axes,base_axes=_axes(report),_axes(baseline);checks={}
    for stage in stages:
        obs,base=[dict(r['observations'][stage],atom_moments_muB=np.asarray(r['observations'][stage]['atom_moments_muB'])[m].tolist())
                  for r,m in [(report,mapping),(baseline,baseline_mapping)]]
        if np.linalg.norm(base['total_moment_muB'])<=1e-8:
            checks[stage]=dict(passed=False,reason='Relative net-moment branch test is undefined for this reference; no smooth-branch derivative claimed')
        else:checks[stage]=branch_check(obs,base,axes[stage],base_axes[stage],settings)
    return checks


def _value(report, point, endpoint, definition, name):
    o=definition;kind=o['kind'];mapping=point['parent_to_child'];evidence={};signature=None
    from .response_diagnostics import KINDS, measure_diagnostic
    if kind in KINDS:
        path=endpoint.get('diagnostics',{}).get(name)
        if path is None:return None,dict(status='pending',reason='Requested saved matrix/moment diagnostic is missing'),None
        return measure_diagnostic(report,endpoint['report'],path,o,mapping=mapping)
    if kind in ('energy','anisotropy'):
        stages=o['stages'] if kind=='anisotropy' else [o['stage']]
        values=[report['observations'][s][o['quantity']] for s in stages]
        value=values[0] if kind=='energy' else values[0]-values[1]
    elif kind=='spectral':
        from .evaluation import _spectral
        selection=dict(o['selection'])
        if 'orbital_indices' in selection and mapping!=list(range(len(mapping))):
            raise ValueError('Literal orbital-index selection requires unchanged atom ordering; use parent atom/l selections')
        if 'atom_indices' in selection:
            if any(i>=len(mapping) for i in selection['atom_indices']):raise ValueError('Spectral parent atom is outside this source')
            selection['atom_indices']=[mapping[i] for i in selection['atom_indices']]
        path=endpoint.get('spectra',{}).get(name)
        if path is None:return None,dict(status='pending',reason='Requested saved spectrum is missing'),None
        target={k:v for k,v in o.items() if k!='kind'}|dict(kind='spectral_response',selection=selection)
        value,m,a,evidence=_spectral(dict(spectrum=path),report,target)
        inverse={child:parent for parent,child in enumerate(mapping)}
        orbitals=[]
        for row in m['orbitals']:
            if row['atom_index'] not in inverse:raise ValueError('Spectrum atom is outside source correspondence')
            orbitals.append({k:v for k,v in row.items() if k!='index'}|dict(atom_index=inverse[row['atom_index']]))
        # Compare per-parent orbital inventories without assuming band tracking.
        import json
        signature=dict(metadata={k:m.get(k) for k in ('spin_components','collinear_spin','spin_frame','projection','temperature_ev','occupations_model')},
            orbitals=sorted(json.dumps(r,sort_keys=True) for r in orbitals),k_fractional=a['k_fractional'],k_weights=a['k_weights'],
            native_context=dict(report_path=endpoint['report'],report=report,metadata=m,stage=o['stage']))
    else:
        row=report['observations'][o['stage']]
        if kind=='stress':
            values=np.asarray(row['stress_ev_ang3'],float)
            if values.shape!=(3,3):raise ValueError('Invalid stress tensor')
            value=values[tuple(o['component'])]
        else:
            if o['site'] is None:vector=np.asarray(row['total_moment_muB'],float)
            else:
                if o['site']>=len(mapping):raise ValueError('Parent site is outside this response source')
                values=np.asarray(row['forces_ev_ang'] if kind=='force' else row['atom_moments_muB'],float)
                if values.shape!=(len(mapping),3):raise ValueError('Invalid site-observable array')
                vector=values[mapping[o['site']]]
            if vector.shape!=(3,):raise ValueError('Invalid vector observable')
            value=np.linalg.norm(vector) if o['component']=='norm' else vector[o['component']]
    if not math.isfinite(value):raise ValueError('Nonfinite response value')
    return float(value),evidence,signature


def analyze_series(directory, evidence, observables, output, *, error_bounds=None):
    """Inspect saved outcomes; derivative errors remain unknown unless supplied.

    All parent atom selections use the exact authoring correspondence. Native
    failure, missing points and magnetic branch changes remain in the report.
    """
    from .evidence import verify_report as _verified_report
    from .electronic import finite_difference
    directory=Path(directory).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Response analysis output must be new')
    definitions=validate_observables(observables);series=verify_series(directory)
    points={p['id']:p for p in series['points']};baseline_id=series['baseline_id']
    baseline_request=verify_experiment(directory/points[baseline_id]['bundle'])
    from .queries import directions
    # Validate requested stages even when all outcomes are still pending.
    allowed={'scalar'}|{'soc-'+k for k in directions(baseline_request.get('query',{'kind':'anisotropy'}),
        {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1]})}
    for record in definitions.values():
        o=record['definition'];stages=o['stages'] if o['kind']=='anisotropy' else [o['stage']]
        if not set(stages)<=allowed:raise ValueError('Observable stage was not requested by this response series')
    if not isinstance(evidence,dict) or set(evidence)-set(points):raise ValueError('Evidence must name registered series points')
    error_bounds={} if error_bounds is None else error_bounds
    if not isinstance(error_bounds,dict) or set(error_bounds)-set(definitions):raise ValueError('Unknown error-bound observable')
    for name,bounds in error_bounds.items():
        if not isinstance(bounds,dict) or set(bounds)-set(points) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in bounds.values()):
            raise ValueError('Error bounds must be finite nonnegative values for registered points')
    reports={};bindings={};runtime=None;planned=None
    for pid,point in points.items():
        request=verify_experiment(directory/point['bundle']);config=_configuration(request)
        if planned is not None and config!=planned:raise ValueError('Series queries differ in their scientific settings')
        planned=config
        if pid not in evidence:continue
        endpoint=evidence[pid];_fields(endpoint,{'report'},{'collection','spectra','diagnostics'})
        if 'spectra' in endpoint and (not isinstance(endpoint['spectra'],dict) or any(k not in definitions or definitions[k]['definition']['kind']!='spectral' for k in endpoint['spectra'])):
            raise ValueError('Spectra must be keyed by selected spectral observables')
        from .response_diagnostics import KINDS
        if 'diagnostics' in endpoint and (not isinstance(endpoint['diagnostics'],dict) or any(k not in definitions or definitions[k]['definition']['kind'] not in KINDS for k in endpoint['diagnostics'])):
            raise ValueError('Diagnostics must be keyed by selected matrix/moment observables')
        path=Path(endpoint['report']).resolve()
        r=_verified_report(path,Path(endpoint['collection']).resolve() if 'collection' in endpoint else None)
        if (r['source_sha256']!=point['source_sha256'] or r['structure']!=load(directory/point['bundle']/'structure.json')
                or r['request'].get('fixed_geometry') is not True or _configuration(r['request'])!=config):
            raise ValueError('Saved response source or scientific query differs from the prepared point')
        identity=dict(executable_sha256=r['executable_sha256'],environment=r['environment'],threads=r.get('threads'))
        if runtime is not None and identity!=runtime:raise ValueError('Response points use different executable or runtime environment')
        runtime=identity;reports[pid]=r
        bindings[pid]=dict(report=str(path),report_sha256=sha256(path),source_sha256=r['source_sha256'],
            source_code=r.get('source_code'),verification=r['verification'],native_status=r['status'])
    settings=Settings.from_dict(load(directory/points[baseline_id]['bundle']/'experiment.json')['settings'])
    results={}
    for name,record in definitions.items():
        definition=record['definition'];stages=definition['stages'] if definition['kind']=='anisotropy' else [definition['stage']]
        rows=[];spectral_reference=None
        for point in series['points']:
            pid=point['id'];r=reports.get(pid)
            row=dict(id=pid,q_exact=point['q_exact'],q=float(Fraction(point['q_exact'])),value=None,status='pending',
                source_sha256=point['source_sha256'],error_bound=error_bounds.get(name,{}).get(pid),derivative_eligible=False)
            rows.append(row)
            if r is None:continue
            if not set(stages)<=set(_axes(r)):raise ValueError('Observable stage was not requested by this response series')
            if not _healthy(r,stages):row.update(status='failed',reason='Native stage or within-case branch did not pass');continue
            value,detail,signature=_value(r,point,evidence[pid],definition,name)
            if value is None:row.update(detail);continue
            if signature is not None:
                if spectral_reference is not None:
                    if 'kind' in signature:
                        from .response_diagnostics import compare_diagnostics
                        detail['diagnostic_correspondence']=compare_diagnostics(spectral_reference,signature)
                    else:
                        if any(signature[k]!=spectral_reference[k] for k in ('metadata','orbitals')):raise ValueError('Response spectral metadata or parent orbital inventory differs')
                        from .sampling import compare_sampling
                        detail['sampling_correspondence']=compare_sampling(spectral_reference,signature,
                            spectral_reference['native_context'],signature['native_context'])
                spectral_reference=signature
            row.update(value=value,status='measured',evidence=detail)
            base=reports.get(baseline_id)
            if base is None or not _healthy(base,stages):row['branch']=dict(passed=False,reason='Baseline native branch is unavailable')
            else:
                checks=_branches(r,base,point['parent_to_child'],points[baseline_id]['parent_to_child'],stages,settings)
                row['branch']=dict(passed=all(c['passed'] for c in checks.values()),stages=checks)
            row['derivative_eligible']=row['branch']['passed']
        baseline=next(r for r in rows if r['id']==baseline_id)
        for row in rows:
            row['delta_from_baseline']=row['value']-baseline['value'] if row['value'] is not None and baseline['value'] is not None else None
        derivatives=[]
        for a,b,c in zip(rows,rows[1:],rows[2:]):
            triplet=[a,b,c];entry=dict(at_exact=b['q_exact'],sample_ids=[r['id'] for r in triplet],status='inconclusive',derivative=None)
            if all(r['derivative_eligible'] for r in triplet):
                entry.update(status='measured',derivative=finite_difference(triplet,parameter_unit=series['parameter_unit'],property_unit=record['unit']))
            else:entry['reason']='Missing/failed evidence or unmatched magnetic branch; no interpolation across gaps'
            derivatives.append(entry)
        results[name]=dict(**record,points=rows,derivatives=derivatives)
    result=dict(format='REMAG-STRUCTURAL-RESPONSE/1',series_sha256=series['sha256'],design=series['design'],
        baseline_id=baseline_id,parameter_unit=series['parameter_unit'],observables=results,evidence=bindings,
        scientific_configuration=planned,runtime=runtime,declared_error_bounds=error_bounds,
        scope=dict(geometry='Fixed GPAC-authored geometry; parent atom indices transported by exact lineage',
            vector_frame='Each authored right-handed Cholesky Cartesian frame; no hidden rotation or lattice matching',
            spectral='Signed Mulliken window integrals relative to each Fermi level; no band tracking, radial-basis identity or unique SOC attribution',
            diagnostics='Native onsite orbital moments and partial native-weighted matrix proxies; no unique hybridization/SOC/MAE decomposition or cross-geometry state matching',
            uncertainty='Supplied absolute observable error bounds propagated conservatively; truncation and convergence errors unknown',
            runtime='Recorded executable/environment/thread count must agree; legacy reports may leave thread count unknown',
            physical_accuracy_qualified=False,mechanism_verified=False,calculation_performed=False))
    result['sha256']=digest(result)
    output.mkdir(parents=True,exist_ok=False);save(output/'response.json',result)
    return result
