"""Read saved Ouro hull evidence without uploading, executing or promoting it.

The supported contract is Ouro's crystal-db/ehull route family. A hull screen
belongs to the submitted geometry and energy surface, not every observation of
the same authored source. Provider-side relaxation requires separate ingestion.
"""
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import numpy as np

from ..corpus import sha256
from ..discovery.common import digest, load
from ..discovery.pipeline import verify_bundle
from .ouro import _enabled, _schema_check, verify_request
from .relaxation import inspect_relaxation
from .relaxation_execution import verify_relaxation_source

HULL_ROUTE_PARENT = '9c71b718-32fa-4fc1-855f-688aa363b663'
GEOMETRY_TOLERANCE = 1e-8


def _file(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError('Missing hull artifact path')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Hull artifact missing or escapes its directory')
    return path


def _json(root, name):
    path = _file(root, name)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Oversized hull metadata')
    value = load(path)
    digest(value)  # reject non-finite JSON even in unreturned fields
    return value


def _same_geometry(actual, expected):
    occupancies = actual.info.get('occupancy', {})
    if any(len(site) != 1 or next(iter(site.values())) != 1 for site in occupancies.values()):
        raise ValueError('Partial or mixed CIF occupancy is unsupported')
    if (actual.numbers.tolist() != expected.numbers.tolist() or
            not actual.pbc.all() or not expected.pbc.all() or
            not np.allclose(actual.cell.array @ actual.cell.array.T,
                            expected.cell.array @ expected.cell.array.T,
                            atol=GEOMETRY_TOLERANCE, rtol=0)):
        raise ValueError('Submitted CIF differs from the selected geometry or site order')
    # CIF lengths/angles can rotate Cartesian axes and wrap fractional sites.
    delta = actual.get_scaled_positions() - expected.get_scaled_positions()
    delta -= np.rint(delta)
    if not np.isfinite(delta).all() or np.max(np.abs(delta)) > GEOMETRY_TOLERANCE:
        raise ValueError('Submitted CIF differs from the selected fractional sites')


def _geometry(structure, geometry):
    from ase import Atoms
    from ase.io import read
    from gpac.physical import verify_export

    source = verify_export(load(structure))
    kind = geometry.get('kind') if isinstance(geometry, dict) else None
    fields = {'authored': {'kind'}, 'local_orb': {'kind', 'bundle'},
              'native_relaxed': {'kind', 'bundle', 'report'}}
    if kind not in fields or set(geometry) != fields[kind]:
        raise ValueError('Select authored, local_orb or native_relaxed geometry with its exact fields')
    g = source['geometry']
    provenance = dict(kind=kind, source_sha256=source['source']['sha256'],
        export_sha256=source['sha256'], structure_file_sha256=sha256(structure))
    if kind == 'authored':
        atoms = Atoms(numbers=g['atomic_numbers'], cell=g['cell'],
                      scaled_positions=g['fractional'], pbc=True)
        return atoms, provenance
    bundle = Path(geometry['bundle']).resolve()
    receipt = verify_bundle(bundle)
    verify_relaxation_source(structure, bundle)
    request = _json(bundle, 'request.json')
    if sha256(_file(bundle, 'input.extxyz')) != request['input_sha256']:
        raise ValueError('Relaxation input binding differs')
    binding_path = bundle.parent / 'source-binding.json'
    provenance['source_binding'] = 'retrospective input geometry match'
    if binding_path.exists():
        binding = _json(bundle.parent, 'source-binding.json')
        if (binding.get('format') != 'REMAG-RELAXATION-SOURCE/1' or
                any(binding.get(k) != provenance[k] for k in
                    ('source_sha256', 'export_sha256', 'structure_file_sha256')) or
                binding.get('input_sha256') != request['input_sha256'] or
                sha256(_file(bundle.parent, 'input.extxyz')) != binding['input_sha256'] or
                sha256(_file(bundle.parent, 'structure.json')) != binding['structure_file_sha256']):
            raise ValueError('Prepared exact source binding differs')
        provenance.update(source_binding='bound before local relaxation',
                          source_binding_sha256=sha256(binding_path))
    provenance.update(prepared_bundle_sha256=digest(receipt),
                      prepared_file_sha256=sha256(bundle/'prepared.json'))
    if kind == 'local_orb':
        result = _json(bundle, 'orb/result.json')
        path = _file(bundle, 'orb/relaxed.extxyz')
        if (result.get('accepted') is not True or result.get('geometry_sha256') != sha256(path) or
                result.get('weights_sha256') != request['orb_weights_sha256']):
            raise ValueError('Local relaxation identity differs')
        provenance.update(geometry_file_sha256=sha256(path),
                          result_sha256=sha256(bundle/'orb/result.json'),
                          model=result['model'], weights_sha256=result['weights_sha256'])
    else:
        report = Path(geometry['report']).resolve()
        feedback = inspect_relaxation(source, bundle, report)
        path = _file(report.parent, 'dft-relaxed.extxyz')
        provenance.update(geometry_file_sha256=sha256(path), native_report_sha256=sha256(report),
                          relaxation_feedback_sha256=digest(feedback))
    atoms = read(path)
    if atoms.numbers.tolist() != g['atomic_numbers']:
        raise ValueError('Relaxation changed species or site order')
    return atoms, provenance


def _response(request, action, atoms):
    from ase.formula import Formula
    from jsonschema import Draft202012Validator

    contract = request['contract']
    route = contract['route']
    if (contract['parent_id'] != HULL_ROUTE_PARENT or route.get('path') != '/crystal-db/ehull' or
            route.get('method', '').upper() != 'POST'):
        raise ValueError('Unsupported hull route family')
    schema = route['responses']['202']['content']['application/json']['schema']
    _schema_check(schema)
    Draft202012Validator.check_schema(schema)
    response = action.get('response')
    Draft202012Validator(schema).validate(response)
    properties = route['request_body']['content']['application/json']['schema']['properties']
    option_names = ('energy_surface', 'allow_unrelaxed', 'bypass_cif_health_screen',
                    'energy_cutoff', 'include_generated_p1', 'include_phase_diagram', 'include_user_materials')
    effective = {k: request['body'].get(k, properties.get(k, {}).get('default')) for k in option_names}
    if effective.get('allow_unrelaxed') is not False or response.get('relaxed_cif') is not None:
        raise ValueError('Provider-relaxed hull geometry requires separate ingestion')
    if effective.get('bypass_cif_health_screen') is not False:
        raise ValueError('Hull input health screen was bypassed')
    surface = effective.get('energy_surface')
    models = {'orb-v3-conservative-inf-mpa': 'orb-v3-conservative-inf-mpa',
              'pbe': 'orb-v3-conservative-inf-mpa', 'prophet-oame-mbd': 'prophet-oame-mbd'}
    if (surface not in models or response.get('energy_surface') != surface or
            response.get('input_model') != models[surface]):
        raise ValueError('Hull input model or energy surface differs from the request')
    def reduced(counts):
        total = sum(counts.values())
        if total <= 0:
            raise ValueError('Empty hull formula')
        return {k: Fraction(v, total) for k, v in counts.items()}
    if reduced(Formula(response['formula']).count()) != reduced(atoms.symbols.formula.count()):
        raise ValueError('Hull formula differs from submitted composition')
    for key in ('e_above_hull', 'formation_energy_per_atom', 'correction_per_atom'):
        if type(response.get(key)) not in (int, float) or not np.isfinite(response[key]):
            raise ValueError('Hull energies must be finite eV/atom numbers')
    if response['e_above_hull'] < 0:
        raise ValueError('Negative energy above hull')
    if type(response.get('predicted_stable')) is not bool:
        raise ValueError('Missing provider stability classification')
    counts = {k: response.get(k) for k in ('num_reference_entries', 'num_missing_references',
        'num_user_contributions', 'num_entries_at_composition', 'num_relaxed_for_this_run')}
    for value in counts.values():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Invalid hull reference count')
    references = response.get('references_by_source')
    if not isinstance(references, dict) or any(type(v) is not int or v < 0 for v in references.values()):
        raise ValueError('Invalid reference source counts')
    decomposition = response.get('decomposition')
    if not isinstance(decomposition, dict) or not decomposition or any(
            type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in decomposition.values()):
        raise ValueError('Invalid hull decomposition')
    return dict(energy_above_hull_ev_per_atom=response['e_above_hull'],
        formation_energy_ev_per_atom=response['formation_energy_per_atom'],
        correction_ev_per_atom=response['correction_per_atom'], formula=response['formula'],
        energy_surface=surface, input_model=response['input_model'],
        mixed_model_and_reference_energies=surface == 'pbe',
        provider_predicted_stable=response['predicted_stable'], decomposition=decomposition,
        reference_counts=counts, references_by_source=references, request_options=effective,
        provider_weights_sha256=None, database_snapshot_sha256=None)


def _outputs(request, action, manifest):
    """Reproduce collection membership, including optional declared assets."""
    declarations = request['contract']['route'].get('output_assets') or {}
    schema = request['contract']['route']['request_body']['content']['application/json']['schema']
    context = request['execution_context']
    files, names = [], set()
    for index, item in enumerate(action.get('output_assets') or []):
        asset, name = item.get('asset') or {}, item.get('name')
        declared = declarations.get(name)
        if (not declared or name in names or not _enabled(declared, request['body'], schema) or
                asset.get('asset_type') != declared.get('asset_type') or
                asset.get('org_id') != context['org_id'] or
                (context.get('team_id') and asset.get('team_id') != context['team_id'])):
            raise ValueError('Hull output asset declaration or organization differs')
        names.add(name)
        if asset['asset_type'] == 'file':
            asset_id = str(UUID(asset['id']))
            files.append(dict(name=name, asset_id=asset_id, path=f'output-{index:02d}-{asset_id}.bin'))
    if (manifest.get('files') != files or
            set(manifest['artifacts']) != {'action.json'} | {f['path'] for f in files} or
            any(_enabled(d, request['body'], schema) and not d.get('optional') and name not in names
                for name, d in declarations.items())):
        raise ValueError('Hull output collection membership differs')


def inspect_ouro_hull(attempt, collection, structure, geometry):
    """Verify saved transport, exact source and CIF geometry; return a typed screen.

    This is local evidence validation, not independent reproduction of the
    provider's energies/database. It exposes no upload handles or raw action.
    """
    from ase.io import read

    attempt, collection, structure = Path(attempt).resolve(), Path(collection).resolve(), Path(structure).resolve()
    request = verify_request(attempt)
    state = _json(attempt, 'state.json')
    manifest = _json(collection, 'manifest.json')
    if (state.get('format') != 'REMAG-OURO-ATTEMPT/1' or
            manifest.get('format') != 'REMAG-OURO-COLLECTION/1' or
            state.get('request_sha256') != digest(request) or
            manifest.get('request_sha256') != digest(request) or
            state.get('state') != 'success' or manifest.get('service_status') != 'success' or
            manifest.get('complete') is not True or manifest.get('missing_outputs') != []):
        raise ValueError('Hull attempt or collection is unbound, incomplete or unsuccessful')
    artifacts = manifest.get('artifacts', {})
    if 'action.json' not in artifacts or len(artifacts) > 128:
        raise ValueError('Missing or oversized hull artifact manifest')
    for name, expected in artifacts.items():
        if sha256(_file(collection, name)) != expected:
            raise ValueError('Hull collection artifact changed: ' + name)
    action = _json(collection, 'action.json')
    submitted = _json(attempt, 'action-submission.json')
    for item in (action, submitted):
        if (item.get('id') != state.get('action_id') or item.get('id') != manifest.get('action_id') or
                item.get('route_id') != request['contract']['id'] or
                item.get('route_id') != manifest.get('route_id')):
            raise ValueError('Hull action or route identity differs')
    context = request.get('execution_context', {})
    if not context.get('org_id') or any(item.get('org_id') != context['org_id'] for item in (action, submitted)):
        raise ValueError('Hull organization differs from the prepared context')
    if action.get('team_id') is not None and action['team_id'] != context.get('team_id'):
        raise ValueError('Hull team differs from the prepared context')
    if action.get('status') != 'success' or set(request['inputs']) != {'structure'}:
        raise ValueError('Hull result needs one successful structure input')
    _outputs(request, action, manifest)
    atoms, provenance = _geometry(structure, geometry)
    cif = _file(attempt, request['inputs']['structure']['path'])
    submitted_geometries = read(cif, format='cif', index=':')
    if len(submitted_geometries) != 1:
        raise ValueError('Hull input must contain exactly one CIF structure')
    _same_geometry(submitted_geometries[0], atoms)
    result = _response(request, action, atoms)
    provenance.update(submitted_cif_sha256=sha256(cif), request_sha256=digest(request),
        request_file_sha256=sha256(attempt/'request.json'), contract_sha256=request['contract_sha256'],
        attempt_state_sha256=sha256(attempt/'state.json'),
        submission_sha256=sha256(attempt/'action-submission.json'),
        collection_sha256=sha256(collection/'manifest.json'), action_sha256=artifacts['action.json'])
    return dict(format='REMAG-OURO-HULL-INSPECTION/1', status='provider_screen_validated',
        source_sha256=provenance['source_sha256'], geometry=provenance,
        submitted_geometry=dict(volume_angstrom3=float(atoms.get_volume()), atoms=len(atoms),
            comparison='species/order exact; cell metric and periodic fractional sites',
            tolerance=GEOMETRY_TOLERANCE),
        provider=dict(action_id=action['id'], route_id=action['route_id'],
            organization_id=context['org_id'], requested_team_id=context.get('team_id'),
            reported_team_id=action.get('team_id'), service_status='success'),
        hull=result,
        stability=dict(thermodynamic='provider_model_screen_only', mechanical='not_evaluated',
            dynamical='not_evaluated', synthesizability='not_established'),
        physical_stability_qualified=False, provider_allocation_release_verified=False,
        metadata_quote_cents=request['price_ceiling_cents'], actual_billed_cents=None,
        scientific_scope='Submitted geometry on the named provider energy surface and reference database. '
            'No transfer to other relaxed geometries; no independent energy reproduction, '
            'dynamical stability or synthesis claim.',
        uploaded=False, solver_executed=False, service_called=False)
