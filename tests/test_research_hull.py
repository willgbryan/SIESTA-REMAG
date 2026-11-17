"""Synthetic hull contracts and saved fixtures; no service or solver calls."""
import json
import shutil
from uuid import UUID

import numpy as np
import pytest

pytest.importorskip('jsonschema')
pytest.importorskip('ase')
pytest.importorskip('gpac')
from ase import Atoms
from ase.io import read, write
from jsonschema import ValidationError
from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.research.hull import HULL_ROUTE_PARENT, inspect_ouro_hull
from siesta_remag.research.ouro import prepare_request
from siesta_remag.research.toolhost import ToolHost
from siesta_remag.research.tools import ResearchTools
from test_discovery import bundle, execute
from test_research_relaxation import export


def uid(n):
    return str(UUID(int=n))


@pytest.fixture
def evidence(tmp_path, export):
    g = export['geometry']
    atoms = Atoms(numbers=g['atomic_numbers'], cell=g['cell'], scaled_positions=g['fractional'], pbc=True)
    write(tmp_path/'input.cif', atoms)
    save(tmp_path/'structure.json', export)
    response = dict(formula='Fe', energy_surface='orb-v3-conservative-inf-mpa',
        input_model='orb-v3-conservative-inf-mpa', e_above_hull=.012,
        formation_energy_per_atom=-.1, correction_per_atom=0, predicted_stable=False,
        decomposition={'Fe': 1}, references_by_source={'fixture': 3}, num_reference_entries=3,
        num_missing_references=0, num_user_contributions=0, num_entries_at_composition=3,
        num_relaxed_for_this_run=0, relaxed_cif=None)
    schema = dict(type='object', required=list(response), properties={
        'formula': {'type':'string'}, 'e_above_hull': {'type':'number'},
        'predicted_stable': {'type':'boolean'}})
    route = dict(id=uid(1), parent_id=HULL_ROUTE_PARENT, org_id=uid(2), team_id=uid(3),
        visibility='public', monetization='free', route=dict(path='/crystal-db/ehull', method='POST',
        input_assets={'structure': {'asset_type':'file', 'file_extensions':['cif']}},
        request_body={'content': {'application/json': {'schema': dict(type='object',
            required=['structure'], properties={
                'structure': dict(type='object', required=['url','filename']),
                'energy_surface': dict(type='string', default='orb-v3-conservative-inf-mpa'),
                'allow_unrelaxed': dict(type='boolean', default=False),
                'bypass_cif_health_screen': dict(type='boolean', default=False)})}}},
        responses={'202': {'content': {'application/json': {'schema': schema}}}}))
    prepare_request(route, {'structure': tmp_path/'input.cif'}, tmp_path/'attempt',
                    organization_id=uid(20), team_id=uid(21))
    request = load(tmp_path/'attempt/request.json')
    action = dict(id=uid(30), route_id=uid(1), status='success', org_id=uid(20), response=response,
                  private_handle='do-not-return', output_assets=[])
    save(tmp_path/'attempt/action-submission.json', action | {'status':'queued'})
    save(tmp_path/'attempt/state.json', dict(format='REMAG-OURO-ATTEMPT/1', state='success',
        request_sha256=digest(request), action_id=uid(30), upload_refs={'secret':'do-not-return'}))
    save(tmp_path/'collected/action.json', action)
    save(tmp_path/'collected/manifest.json', dict(format='REMAG-OURO-COLLECTION/1',
        action_id=uid(30), route_id=uid(1), request_sha256=digest(request), service_status='success',
        complete=True, missing_outputs=[], artifacts={'action.json':sha256(tmp_path/'collected/action.json')}, files=[]))
    return dict(attempt=tmp_path/'attempt', collection=tmp_path/'collected',
                structure=tmp_path/'structure.json', geometry={'kind':'authored'})


def change_action(evidence, mutation):
    path=evidence['collection']/'action.json'
    value=load(path); mutation(value); save(path,value)
    path=evidence['collection']/'manifest.json'
    value=load(path); value['artifacts']['action.json']=sha256(evidence['collection']/'action.json'); save(path,value)


def change_request(evidence, mutation):
    path=evidence['attempt']/'request.json'
    value=load(path); mutation(value); value['contract_sha256']=digest(value['contract']); save(path,value)
    for path in (evidence['attempt']/'state.json', evidence['collection']/'manifest.json'):
        item=load(path); item['request_sha256']=digest(value); save(path,item)


def test_saved_hull_is_geometry_specific_and_read_only(evidence, tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda *a,**k: pytest.fail('Network access'))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result=inspect_ouro_hull(**evidence)
    assert result['hull']['energy_above_hull_ev_per_atom']==.012
    assert result['hull']['provider_predicted_stable'] is False
    assert result['geometry']['kind']=='authored'
    assert result['stability']==dict(thermodynamic='provider_model_screen_only',
        mechanical='not_evaluated', dynamical='not_evaluated', synthesizability='not_established')
    assert result['provider']['reported_team_id'] is None
    assert not result['physical_stability_qualified'] and not result['provider_allocation_release_verified']
    assert not result['service_called'] and not result['solver_executed'] and not result['uploaded']
    assert result['hull']['database_snapshot_sha256'] is None
    assert 'do-not-return' not in str(result)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('field,value', [('id',uid(90)), ('route_id',uid(90)),
    ('org_id',uid(90)), ('team_id',uid(90)), ('status','error')])
def test_wrong_action_identity_rejected(evidence,field,value):
    change_action(evidence,lambda a:a.update({field:value}))
    with pytest.raises(ValueError):inspect_ouro_hull(**evidence)


@pytest.mark.parametrize('patch', [dict(complete=False),dict(missing_outputs=['required']),
    dict(request_sha256='0'*64),dict(service_status='error'),dict(artifacts={})])
def test_incomplete_collection_rejected(evidence,patch):
    path=evidence['collection']/'manifest.json';save(path,load(path)|patch)
    with pytest.raises(ValueError):inspect_ouro_hull(**evidence)


@pytest.mark.parametrize('path', ['action.json','../outside.json','alias.json'])
def test_tampered_and_escaping_collection_files_rejected(evidence,tmp_path,path):
    if path=='action.json':
        (evidence['collection']/path).write_text('{}')
    else:
        outside=tmp_path/'outside.json';outside.write_text('private')
        if path=='alias.json':(evidence['collection']/path).symlink_to(outside)
        manifest=evidence['collection']/'manifest.json';value=load(manifest)
        value['artifacts'][path]=sha256(outside);save(manifest,value)
    with pytest.raises(ValueError):inspect_ouro_hull(**evidence)


@pytest.mark.parametrize('patch', [dict(formula='Ni'),dict(e_above_hull=-1),dict(e_above_hull=True),
    dict(formation_energy_per_atom='0.1'),dict(predicted_stable=1),dict(num_reference_entries=True),
    dict(num_missing_references=-1),dict(decomposition={'Fe':-1}),dict(references_by_source={'mp':-1}),
    dict(energy_surface='pbe'),dict(input_model='another-model'),dict(relaxed_cif='changed.cif')])
def test_incompatible_scientific_payload_rejected(evidence,patch):
    change_action(evidence,lambda a:a['response'].update(patch))
    with pytest.raises((ValueError, ValidationError)):inspect_ouro_hull(**evidence)


@pytest.mark.parametrize('change', ['route','allow_unrelaxed','bypass_cif_health_screen','external_schema'])
def test_request_scope_is_not_silently_changed(evidence,change):
    def mutation(r):
        if change=='route':r['contract']['parent_id']=uid(90)
        elif change=='external_schema':r['contract']['route']['responses']['202']['content']['application/json']['schema']['$ref']='https://untrusted.invalid/schema'
        else:r['body'][change]=True
    change_request(evidence,mutation)
    with pytest.raises(ValueError):inspect_ouro_hull(**evidence)


def test_mixed_pbe_surface_is_explicit(evidence):
    change_request(evidence,lambda r:r['body'].update(energy_surface='pbe'))
    change_action(evidence,lambda a:a['response'].update(energy_surface='pbe'))
    assert inspect_ouro_hull(**evidence)['hull']['mixed_model_and_reference_energies']


def test_same_source_does_not_admit_a_different_relaxed_geometry(evidence,bundle):
    atoms=read(bundle/'orb/relaxed.extxyz');atoms.set_cell(atoms.cell.array*1.01,scale_atoms=True)
    write(bundle/'orb/relaxed.extxyz',atoms)
    request=load(bundle/'request.json');request['orb_weights_sha256']='a'*64;save(bundle/'request.json',request)
    save(bundle/'orb/result.json',dict(accepted=True,model='fixture',weights_sha256='a'*64,
        geometry_sha256=sha256(bundle/'orb/relaxed.extxyz')))
    receipt=load(bundle/'prepared.json');receipt['request_sha256']=digest(request)
    receipt['files']={k:sha256(bundle/k) for k in receipt['files']};save(bundle/'prepared.json',receipt)
    with pytest.raises(ValueError,match='CIF differs'):
        inspect_ouro_hull(**(evidence|{'geometry':dict(kind='local_orb',bundle=bundle)}))


def test_saved_orb_and_native_geometry_modes(evidence,bundle,tmp_path):
    request=load(bundle/'request.json');request['orb_weights_sha256']='a'*64;save(bundle/'request.json',request)
    save(bundle/'orb/result.json',dict(accepted=True,model='fixture',weights_sha256='a'*64,
        geometry_sha256=sha256(bundle/'orb/relaxed.extxyz')))
    receipt=load(bundle/'prepared.json');receipt['request_sha256']=digest(request)
    receipt['files']={k:sha256(bundle/k) for k in receipt['files']};save(bundle/'prepared.json',receipt)
    local=inspect_ouro_hull(**(evidence|{'geometry':dict(kind='local_orb',bundle=bundle)}))
    assert local['geometry']['weights_sha256']=='a'*64
    assert local['geometry']['source_binding']=='retrospective input geometry match'
    execute(bundle,tmp_path,[])  # synthetic backend fixture, not a solver
    native=inspect_ouro_hull(**(evidence|{'geometry':dict(kind='native_relaxed',bundle=bundle,report=tmp_path/'run/report.json')}))
    assert native['geometry']['native_report_sha256']==sha256(tmp_path/'run/report.json')
    (tmp_path/'run/dft-relax/remag.STRUCT_OUT').write_text('corrupt')
    with pytest.raises(ValueError,match='artifact changed'):
        inspect_ouro_hull(**(evidence|{'geometry':dict(kind='native_relaxed',bundle=bundle,report=tmp_path/'run/report.json')}))


def test_policy_profiles_and_withheld_source_checks(evidence,tmp_path,export):
    args=dict(attempt='attempt',collection='collected',structure='structure.json',geometry={'kind':'authored'})
    for profile in ('physics','scalar'):
        tools=ResearchTools(tmp_path,information_profile=profile)
        assert tools.call('inspect_ouro_hull',args)['source_sha256']==export['source']['sha256']
        host=ToolHost(tmp_path,information_profile=profile)
        assert host.catalog()['tools']['inspect_ouro_hull']['effect']=='inspect'
        tools=ResearchTools(tmp_path,information_profile=profile,withheld_sources=[export['source']['sha256']])
        with pytest.raises(ValueError,match='held-out'):tools.call('inspect_ouro_hull',args)
    with pytest.raises(ValueError,match='escapes'):
        ResearchTools(tmp_path).call('inspect_ouro_hull',args|{'geometry':dict(kind='local_orb',bundle='../outside')})


@pytest.mark.parametrize('change', ['cell','site','species','partial_occupancy','two_blocks','rotation_and_wrap'])
def test_cif_binding_uses_one_full_occupancy_periodic_geometry(evidence,change):
    path=evidence['attempt']/'inputs/00.cif'
    atoms=read(path)
    atoms.info.pop('occupancy', None)
    if change=='cell':atoms.set_cell(atoms.cell.array*1.01,scale_atoms=True)
    if change=='site':atoms.positions[0,0]+=.001
    if change=='species':atoms.numbers[0]=28
    if change=='rotation_and_wrap':
        atoms.rotate(31,'z',rotate_cell=True)
        atoms.positions[0]+=atoms.cell.array[0]
    write(path,atoms)
    if change=='partial_occupancy':path.write_text(path.read_text().replace('1.0000','0.5000',1))
    if change=='two_blocks':path.write_text(path.read_text()+path.read_text().replace('data_image0','data_image1'))
    def mutation(r):r['inputs']['structure'].update(sha256=sha256(path),bytes=path.stat().st_size)
    change_request(evidence,mutation)
    if change=='rotation_and_wrap':
        assert inspect_ouro_hull(**evidence)['status']=='provider_screen_validated'
    else:
        with pytest.raises(ValueError):inspect_ouro_hull(**evidence)


def test_prepared_source_binding_is_checked(evidence,bundle,tmp_path,export):
    request=load(bundle/'request.json');request['orb_weights_sha256']='a'*64;save(bundle/'request.json',request)
    save(bundle/'orb/result.json',dict(accepted=True,model='fixture',weights_sha256='a'*64,
        geometry_sha256=sha256(bundle/'orb/relaxed.extxyz')))
    receipt=load(bundle/'prepared.json');receipt['request_sha256']=digest(request)
    receipt['files']={k:sha256(bundle/k) for k in receipt['files']};save(bundle/'prepared.json',receipt)
    shutil.copyfile(bundle/'input.extxyz',tmp_path/'input.extxyz')
    binding=dict(format='REMAG-RELAXATION-SOURCE/1',source_sha256=export['source']['sha256'],
        export_sha256=export['sha256'],structure_file_sha256=sha256(tmp_path/'structure.json'),
        input_sha256=sha256(tmp_path/'input.extxyz'))
    save(tmp_path/'source-binding.json',binding)
    args=evidence|{'geometry':dict(kind='local_orb',bundle=bundle)}
    assert inspect_ouro_hull(**args)['geometry']['source_binding']=='bound before local relaxation'
    save(tmp_path/'source-binding.json',binding|{'source_sha256':'0'*64})
    with pytest.raises(ValueError,match='source binding'):inspect_ouro_hull(**args)


def test_nonfinite_provider_values_are_rejected_even_when_hash_matches(evidence):
    path=evidence['collection']/'action.json';value=load(path)
    value['response']['e_above_hull']=float('nan');path.write_text(json.dumps(value))
    path=evidence['collection']/'manifest.json';value=load(path)
    value['artifacts']['action.json']=sha256(evidence['collection']/'action.json');save(path,value)
    with pytest.raises(ValueError):inspect_ouro_hull(**evidence)


def test_declared_output_file_and_membership_are_verified(evidence):
    change_request(evidence,lambda r:r['contract']['route'].update(output_assets={'diagram':dict(asset_type='file')}))
    with pytest.raises(ValueError,match='membership'):inspect_ouro_hull(**evidence)
    asset=dict(id=uid(45),asset_type='file',org_id=uid(20),team_id=uid(21))
    change_action(evidence,lambda a:a.update(output_assets=[dict(name='diagram',asset=asset)]))
    output='output-00-'+uid(45)+'.bin';(evidence['collection']/output).write_bytes(b'fixture diagram')
    path=evidence['collection']/'manifest.json';value=load(path)
    value['artifacts'][output]=sha256(evidence['collection']/output)
    value['files']=[dict(name='diagram',asset_id=uid(45),path=output)];save(path,value)
    assert inspect_ouro_hull(**evidence)['status']=='provider_screen_validated'
    change_action(evidence,lambda a:a['output_assets'][0]['asset'].update(org_id=uid(90)))
    with pytest.raises(ValueError,match='output asset'):inspect_ouro_hull(**evidence)
