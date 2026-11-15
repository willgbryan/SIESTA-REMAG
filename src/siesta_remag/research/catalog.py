"""Explicit general-structure catalog for the local research workspace."""
from pathlib import Path

import numpy as np

from ..discovery.common import load
from ..corpus import sha256
from .inspection import load_spectrum


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


def electronic_view(path, expected_hsx, atom_count):
    """Bounded cached preview; all selected states retain k and orbital labels."""
    meta,data=load_spectrum(path)
    if meta['hsx_sha256']!=expected_hsx:raise ValueError('Spectrum belongs to another displayed Hamiltonian')
    if not 1<=atom_count<=64 or len(meta['orbitals'])>1024:
        raise ValueError('Interactive spectrum preview is bounded to 64 atoms and 1024 orbitals')
    if any(type(o['atom_index']) is not int or not 0<=o['atom_index']<atom_count for o in meta['orbitals']):
        raise ValueError('Spectrum orbital atom does not belong to the displayed geometry')
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
    return dict(energy_ev=grid.tolist(),groups=groups,gaussian_sigma_ev=sigma,states=states,
        orbitals=meta['orbitals'],matching_states=len(candidates),omitted_states=max(0,len(candidates)-len(states)),
        projection=meta['projection'],native_max_abs_error_ev=meta['native_max_abs_error_ev'],
        sampling=meta.get('sampling'),occupations_model=meta['occupations_model'],spin_frame=meta['spin_frame'])


def registered_dataset(root):
    from ase.data import chemical_symbols
    from gpac.physical import verify_export,coordination,describe_dofs
    from .tools import ResearchTools,_verified_report
    from .workspace import comparison_key
    from .relaxation import inspect_relaxation
    root=Path(root).resolve();resolver=ResearchTools(root)
    manifest=load(root/'workspace.json')
    if (manifest.get('format')!='REMAG-RESEARCH-WORKSPACE/1' or set(manifest)-{'format','title','structures'} or
        not isinstance(manifest.get('title'),str) or not 1<=len(manifest['title'])<=100 or
        not isinstance(manifest.get('structures'),list) or not 1<=len(manifest['structures'])<=64):
        raise ValueError('Expected a titled workspace with 1..64 registered structures')
    families=[];ids=set()
    for entry in manifest['structures']:
        if (not isinstance(entry,dict) or set(entry)-{'id','label','structure','calculations','relaxation'} or
            not {'id','label','structure'}<=set(entry) or not all(isinstance(entry[k],str) and 1<=len(entry[k])<=100 for k in ['id','label']) or entry['id'] in ids):
            raise ValueError('Workspace structures need distinct identifiers, labels and source paths')
        ids.add(entry['id']);export=verify_export(load(resolver.path(entry['structure'])))
        if len(export['geometry']['site_ids'])>256:raise ValueError('Interactive structures are bounded to 256 sites')
        row=dict(strain=0,exact_strain='0',label=entry['label'],structure=export,
            coordination=coordination(export['geometry'],cutoff_angstrom=3.4),
            parameters=describe_dofs(export['source']['program']),calculations=[],result=None)
        calculations=entry.get('calculations',[])
        if not isinstance(calculations,list) or len(calculations)>32:raise ValueError('At most 32 calculations per structure')
        for item in calculations:
            if not isinstance(item,dict) or set(item)-{'report','spectra','matrices','collection'} or 'report' not in item:raise ValueError('Invalid calculation registration')
            path=resolver.path(item['report']);report=_verified_report(path,resolver.path(item['collection']) if 'collection' in item else None)
            if report['source_sha256']!=export['source']['sha256']:raise ValueError('Calculation belongs to another registered source')
            request=report['request']
            calc={k:request.get(k) for k in ('settings','kmesh','basis_size','basis_spec','basis_energy_shift_ry','pseudopotentials')}
            calc.update(path=item['report'],status=report['status'],anisotropy=report.get('anisotropy'),
                angular_fit=report.get('angular_fit'),stages=report['stages'],comparison_key=comparison_key(report),
                observations={k:{key:v for key,v in obs.items() if key!='eigenvalues'} for k,obs in report['observations'].items()},
                executable_sha256=report['executable_sha256'],verification=report['verification'],spectra={},matrices={},electronic=None)
            if calc['basis_spec'] is not None:
                from .basis import orbital_count
                calc['basis_orbitals']=orbital_count(calc['basis_spec'],[chemical_symbols[z] for z in export['geometry']['atomic_numbers']])
            spectra=item.get('spectra',{})
            if not isinstance(spectra,dict) or len(spectra)>37:raise ValueError('Invalid registered spectra')
            for stage,spectrum in spectra.items():
                hsx=report['artifacts'].get(stage+'/remag.HSX')
                if stage not in report['observations'] or not hsx:raise ValueError('Spectrum requires a collected stage Hamiltonian')
                calc['spectra'][stage]=electronic_view(resolver.path(spectrum),hsx,len(export['geometry']['site_ids']))
            if calc['spectra']:calc['electronic']=calc['spectra'].get('soc-z',next(iter(calc['spectra'].values())))
            matrices=item.get('matrices',{})
            if not isinstance(matrices,dict) or len(matrices)>37:raise ValueError('Invalid registered matrix diagnostics')
            for stage,metadata in matrices.items():
                hsx=report['artifacts'].get(stage+'/remag.HSX')
                if stage not in report['observations'] or not hsx:raise ValueError('Matrices require a collected stage Hamiltonian')
                calc['matrices'][stage]=matrix_view(resolver.path(metadata),sha256(path),stage,hsx,len(export['geometry']['site_ids']))
            row['calculations'].append(calc);row['result']=calc
        if 'relaxation' in entry:
            r=entry['relaxation']
            if not isinstance(r,dict) or set(r)!={'bundle','report'}:raise ValueError('Relaxation needs a bundle and report')
            feedback=inspect_relaxation(export,resolver.path(r['bundle']),resolver.path(r['report']))
            row['relaxation']=feedback
            row['observed_coordination']=coordination(feedback['feedback']['observation']['observed'],cutoff_angstrom=3.4)
        families.append(dict(id=entry['id'],label=entry['label'],hint='Registered exact source',mode='structure',rows=[row]))
    return dict(format='REMAG-WORKSPACE/1',material=manifest['title'],mode='registered',
        families=families,rows=families[0]['rows'],element_symbols=chemical_symbols)
