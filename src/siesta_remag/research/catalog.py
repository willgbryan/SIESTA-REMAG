"""Explicit general-structure catalog for the local research workspace."""
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from ..discovery.common import load
from ..corpus import sha256
from .inspection import load_spectrum


def orbital_moment_view(path, report_sha256, stage, expected_hsx, expected_dm, atom_count):
    from .orbital_moments import load_orbital_moments
    meta, arrays = load_orbital_moments(path)
    if (meta['report_sha256'] != report_sha256 or meta['stage'] != stage
            or meta['evidence']['hsx'] != expected_hsx or meta['evidence']['dm'] != expected_dm):
        raise ValueError('Orbital moments belong to another displayed calculation/density')
    if arrays['atom_moments_mu_b'].shape != (atom_count,3):
        raise ValueError('Orbital moments do not belong to the displayed geometry')
    return dict(orbitals=meta['orbitals'], atom_moments_mu_b=arrays['atom_moments_mu_b'].tolist(),
        orbital_moments_mu_b=arrays['orbital_moments_mu_b'].tolist(),
        total_moment_mu_b=arrays['atom_moments_mu_b'].sum(axis=0).tolist(),
        artifact_sha256=meta['artifact_sha256'], approximation=meta['approximation'], scope=meta['scope'])


def matrix_view(path, report_sha256, stage, expected_hsx, atom_count):
    from .mechanisms import _load_matrices, _groups, coupling_summary
    meta, arrays = _load_matrices(path)
    if (meta['report_sha256'] != report_sha256 or meta['stage'] != stage
            or meta['evidence']['hsx'] != expected_hsx):
        raise ValueError('Matrices belong to another displayed calculation/Hamiltonian')
    if any(type(o['atom_index']) is not int or not 0 <= o['atom_index'] < atom_count for o in meta['orbitals']):
        raise ValueError('Matrix orbital atom does not belong to the displayed geometry')
    groups = _groups(meta['orbitals'], meta['spin_components'] > 2)
    return dict(k_indices=meta['k_indices'], native_k_count=meta['native_k_count'],
        k_fractional=arrays['k_fractional'].tolist(), selected_k_weight=float(arrays['k_weights'].sum()),
        coupling=[coupling_summary(h, groups) for h in arrays['lowdin_hamiltonian_ev']],
        artifact_sha256=meta['artifact_sha256'], native_max_abs_error_ev=meta['native_max_abs_error_ev'],
        scope=meta['scope'])


def _checked_spectrum(path, expected_hsx, atom_count):
    meta,data=load_spectrum(path)
    if meta['hsx_sha256']!=expected_hsx:raise ValueError('Spectrum belongs to another displayed Hamiltonian')
    if not 1<=atom_count<=64 or len(meta['orbitals'])>1024:
        raise ValueError('Interactive spectrum preview is bounded to 64 atoms and 1024 orbitals')
    if any(type(o['atom_index']) is not int or not 0<=o['atom_index']<atom_count for o in meta['orbitals']):
        raise ValueError('Spectrum orbital atom does not belong to the displayed geometry')
    return meta,data


def electronic_view(path, expected_hsx, atom_count, *, include_sampling=True):
    """Bounded preview; full saved sampling can be requested separately."""
    meta,data=_checked_spectrum(path,expected_hsx,atom_count)
    energies=data['eigenvalues_ev']-meta['fermi_ev']
    grid=np.linspace(-3,3,121);sigma=.12
    factor=2 if meta['spin_components']==1 else 1
    atom_pop=[data['orbital_mulliken'][:,[i for i,o in enumerate(meta['orbitals']) if o['atom_index']==a],:].sum(axis=1)
              for a in range(atom_count)]
    groups=[dict(atom_index=i,l=None,density=[]) for i in range(atom_count)]
    for energy in grid:
        kernel=np.exp(-.5*((energy-energies)/sigma)**2)*data['k_weights'][:,None]*factor/(sigma*np.sqrt(2*np.pi))
        for group,pop in zip(groups,atom_pop):group['density'].append(float(np.sum(kernel*pop)))
    candidates=np.argwhere(abs(energies)<=3)
    order=np.argsort(abs(energies[candidates[:,0],candidates[:,1]]),kind='stable')[:100]
    states=[]
    for k,n in candidates[order]:
        states.append(dict(k_index=int(k),band_index=int(n),k_fractional=data['k_fractional'][k].tolist(),
            k_weight=float(data['k_weights'][k]),energy_minus_fermi_ev=float(energies[k,n]),
            occupation=float(data['occupations'][k,n]),orbital_weights=data['orbital_mulliken'][k,:,n].tolist(),
            spin_pauli=data['spin_pauli'][k,n].tolist() if data['spin_pauli'].size else None))
    from .reciprocal import sampled_view
    return dict(energy_ev=grid.tolist(),groups=groups,gaussian_sigma_ev=sigma,states=states,
        sampled=sampled_view(meta,data,atom_count) if include_sampling else dict(available=False,deferred=True),
        orbitals=meta['orbitals'],matching_states=len(candidates),omitted_states=max(0,len(candidates)-len(states)),
        projection=meta['projection'],native_max_abs_error_ev=meta['native_max_abs_error_ev'],
        sampling=meta.get('sampling'),occupations_model=meta['occupations_model'],spin_frame=meta['spin_frame'])


def load_catalog(root):
    manifest=load(Path(root)/'workspace.json')
    if (manifest.get('format')!='REMAG-RESEARCH-WORKSPACE/1' or set(manifest)-{'format','title','structures','research'} or
        not isinstance(manifest.get('title'),str) or not 1<=len(manifest['title'])<=100 or
        not isinstance(manifest.get('structures'),list) or not 1<=len(manifest['structures'])<=64):
        raise ValueError('Expected a titled workspace with 1..64 registered structures')
    ids=set()
    for entry in manifest['structures']:
        if (not isinstance(entry,dict) or set(entry)-{'id','label','structure','calculations','relaxation','candidate'} or
            not {'id','label','structure'}<=set(entry) or not all(isinstance(entry[k],str) and 1<=len(entry[k])<=100 for k in ['id','label']) or entry['id'] in ids):
            raise ValueError('Workspace structures need distinct identifiers, labels and source paths')
        ids.add(entry['id'])
    from .workspace_actions import validate_entries
    validate_entries(manifest.get('research',[]))
    return manifest


def registered_dataset(root, *, include_sampling=True):
    from ase.data import chemical_symbols
    from gpac.physical import verify_export,coordination,describe_dofs
    from .tools import ResearchTools,_verified_report
    from .workspace import comparison_key
    from .relaxation import inspect_relaxation
    from .authoring import verify_candidate
    root=Path(root).resolve();resolver=ResearchTools(root)
    manifest=load_catalog(root);families=[]
    for entry in manifest['structures']:
        export=verify_export(load(resolver.path(entry['structure'])))
        from .material_scope import check_scope
        check_scope(export,resolver.material_scope)
        if len(export['geometry']['site_ids'])>256:raise ValueError('Interactive structures are bounded to 256 sites')
        row=dict(strain=0,exact_strain='0',label=entry['label'],structure=export,structure_path=entry['structure'],
            coordination=coordination(export['geometry'],cutoff_angstrom=3.4),
            parameters=describe_dofs(export['source']['program']),calculations=[],result=None)
        if include_sampling:
            from .newfound import view as synthesis_view
            saved = synthesis_view(root, entry['id'], export['source']['sha256'])
            if saved['status'] != 'empty':
                row['synthesis'] = saved
        if 'candidate' in entry:
            row['candidate']=verify_candidate(resolver.path(entry['candidate']))
            if row['candidate']['source_sha256']!=export['source']['sha256']:
                raise ValueError('Candidate lineage belongs to another displayed structure')
        calculations=entry.get('calculations',[])
        if not isinstance(calculations,list) or len(calculations)>32:raise ValueError('At most 32 calculations per structure')
        for calculation_index,item in enumerate(calculations):
            if not isinstance(item,dict) or set(item)-{'report','spectra','matrices','orbital_moments','collection'} or 'report' not in item:raise ValueError('Invalid calculation registration')
            path=resolver.path(item['report']);report=_verified_report(path,resolver.path(item['collection']) if 'collection' in item else None)
            if report['source_sha256']!=export['source']['sha256']:raise ValueError('Calculation belongs to another registered source')
            request=report['request']
            calc={k:request.get(k) for k in ('settings','kmesh','basis_size','basis_spec','basis_energy_shift_ry','pseudopotentials')}
            calc.update(path=item['report'],status=report['status'],anisotropy=report.get('anisotropy'),
                angular_fit=report.get('angular_fit'),stages=report['stages'],comparison_key=comparison_key(report),
                observations={k:{key:v for key,v in obs.items() if key!='eigenvalues'} for k,obs in report['observations'].items()},
                executable_sha256=report['executable_sha256'],verification=report['verification'],spectra={},matrices={},orbital_moments={},electronic=None)
            if report.get('format') == 'REMAG-DISCOVERY-OBSERVATION/1':
                calc.update(geometry_origin=report['geometry_origin'], evaluated_geometry=report['evaluated_geometry'],
                    relaxation_metrics=report['relaxation_metrics'], native_status=report['native_status'])
            if calc['basis_spec'] is not None:
                from .basis import orbital_count
                calc['basis_orbitals']=orbital_count(calc['basis_spec'],[chemical_symbols[z] for z in export['geometry']['atomic_numbers']])
            spectra=item.get('spectra',{})
            if not isinstance(spectra,dict) or len(spectra)>37:raise ValueError('Invalid registered spectra')
            for stage,spectrum in spectra.items():
                hsx=report['artifacts'].get(stage+'/remag.HSX')
                if stage not in report['observations'] or not hsx:raise ValueError('Spectrum requires a collected stage Hamiltonian')
                spectrum_path=resolver.path(spectrum)
                calc['spectra'][stage]=electronic_view(spectrum_path,hsx,len(export['geometry']['site_ids']),include_sampling=include_sampling)
                if not include_sampling:
                    calc['spectra'][stage]['sampled']['url']='/api/sampling?'+urlencode(dict(
                        structure=entry['id'],calculation=calculation_index,stage=stage,
                        report_sha256=sha256(path),spectrum_sha256=sha256(spectrum_path)))
            if calc['spectra']:calc['electronic']=calc['spectra'].get('soc-z',next(iter(calc['spectra'].values())))
            matrices=item.get('matrices',{})
            if not isinstance(matrices,dict) or len(matrices)>37:raise ValueError('Invalid registered matrix diagnostics')
            for stage,metadata in matrices.items():
                hsx=report['artifacts'].get(stage+'/remag.HSX')
                if stage not in report['observations'] or not hsx:raise ValueError('Matrices require a collected stage Hamiltonian')
                calc['matrices'][stage]=matrix_view(resolver.path(metadata),sha256(path),stage,hsx,len(export['geometry']['site_ids']))
            moments=item.get('orbital_moments',{})
            if not isinstance(moments,dict) or len(moments)>37:raise ValueError('Invalid registered orbital moments')
            for stage,metadata in moments.items():
                hsx=report['artifacts'].get(stage+'/remag.HSX');dm=report['artifacts'].get(stage+'/remag.DM')
                if stage not in report['observations'] or not hsx or not dm:raise ValueError('Orbital moments require native density and Hamiltonian bindings')
                calc['orbital_moments'][stage]=orbital_moment_view(resolver.path(metadata),sha256(path),stage,hsx,dm,len(export['geometry']['site_ids']))
            row['calculations'].append(calc);row['result']=calc
        if 'relaxation' in entry:
            r=entry['relaxation']
            if not isinstance(r,dict) or set(r)!={'bundle','report'}:raise ValueError('Relaxation needs a bundle and report')
            feedback=inspect_relaxation(export,resolver.path(r['bundle']),resolver.path(r['report']))
            row['relaxation']=feedback
            row['observed_coordination']=coordination(feedback['feedback']['observation']['observed'],cutoff_angstrom=3.4)
        families.append(dict(id=entry['id'],label=entry['label'],hint='Registered exact source',mode='structure',rows=[row]))
    from .workspace_actions import research_view
    return dict(format='REMAG-WORKSPACE/1',material=manifest['title'],mode='registered',
        families=families,rows=families[0]['rows'],element_symbols=chemical_symbols,
        research=research_view(root,manifest))


def registered_sampling(root, query):
    """Read one registered spectrum with the same evidence checks as the IDE.

    The URLs bind the previously displayed report/metadata. They cannot select
    arbitrary paths, silently switch evidence after a refresh, or run a solver.
    """
    from gpac.physical import verify_export
    from .tools import ResearchTools,_verified_report
    from .reciprocal import sampled_view
    if (not isinstance(query,dict) or set(query)!={'structure','calculation','stage','report_sha256','spectrum_sha256'}
            or any(not isinstance(v,str) for v in query.values())
            or not query['calculation'].isascii() or not query['calculation'].isdecimal()):
        raise ValueError('Expected exact registered sampling selection')
    root=Path(root).resolve();resolver=ResearchTools(root);manifest=load_catalog(root)
    entry=next((v for v in manifest['structures'] if v['id']==query['structure']),None)
    index=int(query['calculation'])
    if entry is None or not 0<=index<len(entry.get('calculations',[])):
        raise ValueError('Unknown registered calculation')
    item=entry['calculations'][index];stage=query['stage']
    if stage not in item.get('spectra',{}):raise ValueError('Unknown registered spectrum')
    path=resolver.path(item['report']);spectrum=resolver.path(item['spectra'][stage])
    if sha256(path)!=query['report_sha256'] or sha256(spectrum)!=query['spectrum_sha256']:
        raise ValueError('Displayed sampling evidence changed; reload the workspace')
    report=_verified_report(path,resolver.path(item['collection']) if 'collection' in item else None)
    export=verify_export(load(resolver.path(entry['structure'])))
    if report['source_sha256']!=export['source']['sha256']:
        raise ValueError('Sampling belongs to another registered source')
    hsx=report['artifacts'].get(stage+'/remag.HSX')
    if stage not in report['observations'] or not hsx:raise ValueError('Sampling requires an accepted collected stage')
    atoms=len(export['geometry']['site_ids']);meta,arrays=_checked_spectrum(spectrum,hsx,atoms)
    return sampled_view(meta,arrays,atoms)
