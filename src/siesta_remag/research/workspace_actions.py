"""Exact-source workspace preparation and immutable saved-response views.

There is deliberately no execution adapter here. Profiles are verified prepared
experiments, not suggested numerical defaults or accuracy certifications.
"""
from uuid import uuid4

from ..corpus import sha256
from ..discovery.common import load, save, digest
from ..discovery.config import Settings
from ..discovery.pipeline import lock
from .experiments import prepare_experiment, verify_experiment
from .series import prepare_series, verify_series, validate_observables


KINDS = {'profile', 'query', 'series', 'response'}


def validate_entries(entries):
    if not isinstance(entries, list) or len(entries) > 128:
        raise ValueError('At most 128 registered workspace research items')
    ids = set()
    for e in entries:
        if (not isinstance(e, dict) or set(e) != {'id', 'label', 'kind', 'path', 'sha256'}
                or not all(isinstance(e[k], str) and 1 <= len(e[k]) <= 100 for k in ('id', 'label'))
                or e['id'] in ids or not isinstance(e['kind'], str) or e['kind'] not in KINDS
                or not isinstance(e['path'], str) or not e['path']
                or not isinstance(e['sha256'], str) or len(e['sha256']) != 64):
            raise ValueError('Invalid or duplicate workspace research registration')
        ids.add(e['id'])
    return entries


def _artifact(path, kind):
    return path/('series.json' if kind == 'series' else 'experiment.json') if kind != 'response' else path


def _read(resolver, entry):
    path = resolver.path(entry['path'])
    artifact = resolver.path(str(_artifact(path, entry['kind'])))
    if sha256(artifact) != entry['sha256']:
        raise ValueError('Registered research artifact changed: ' + entry['id'])
    if entry['kind'] in ('profile', 'query'):
        return verify_experiment(path)
    if entry['kind'] == 'series':
        return verify_series(path)
    result = load(path)
    if (result.get('format') != 'REMAG-STRUCTURAL-RESPONSE/1'
            or result.get('sha256') != digest({k: v for k, v in result.items() if k != 'sha256'})):
        raise ValueError('Saved response digest differs')
    return result


def _response_view(result, series):
    if (result.get('design') != series['design'] or result.get('baseline_id') != series['baseline_id']
            or result.get('parameter_unit') != series['parameter_unit']):
        raise ValueError('Saved response design differs from its registered series')
    definitions = validate_observables({k: v['definition'] for k, v in result['observables'].items()})
    for name, observable in result['observables'].items():
        if observable['unit'] != definitions[name]['unit'] or len(observable['points']) != len(series['points']):
            raise ValueError('Saved response observable or point inventory differs')
        for point, prepared in zip(observable['points'], series['points']):
            if any(point[k] != prepared[k] for k in ('id', 'q_exact', 'source_sha256')):
                raise ValueError('Saved response point belongs to another exact source')
    return dict(parent_source_sha256=series['parent_source_sha256'], result=result,
        verification='Saved analysis digest and registered exact series; native evidence is not recollected by viewing')


def research_view(root, manifest):
    from .tools import ResearchTools
    resolver = ResearchTools(root)
    entries = validate_entries(manifest.get('research', []))
    records = {e['id']: _read(resolver, e) for e in entries}
    series = {records[e['id']]['sha256']: records[e['id']] for e in entries if e['kind'] == 'series'}
    result = []
    for entry in entries:
        record = records[entry['id']]
        view = dict(entry)
        if entry['kind'] in ('profile', 'query'):
            view.update(source_sha256=record['source_sha256'], query=record.get('query', {'kind': 'anisotropy'}),
                settings={k: record[k] for k in ('kmesh', 'basis_size', 'basis_energy_shift_ry', 'settings', 'pseudopotentials')},
                basis_spec=record.get('basis_spec'))
        elif entry['kind'] == 'series':
            view.update(parent_source_sha256=record['parent_source_sha256'], design=record['design'],
                point_count=len(record['points']), series_sha256=record['sha256'])
        else:
            if record.get('series_sha256') not in series:
                raise ValueError('Register the exact response series before its analysis')
            view.update(_response_view(record, series[record['series_sha256']]))
        result.append(view)
    return result


def register_research(root, kind, path, identifier, label):
    """Register an existing immutable bundle or analysis; never infer a profile."""
    from .catalog import load_catalog
    from .tools import ResearchTools
    resolver = ResearchTools(root); root = resolver.root
    if kind not in KINDS:
        raise ValueError('Unknown workspace research kind')
    target = resolver.path(path)
    entry = dict(id=identifier, label=label, kind=kind, path=str(target.relative_to(root)),
        sha256=sha256(resolver.path(str(_artifact(target, kind)))))
    with lock(root/'.workspace-catalog.lock'):
        manifest = load_catalog(root)
        manifest.setdefault('research', []).append(entry)
        research_view(root, manifest)
        save(root/'workspace.json', manifest)
    return entry


def prepare_workspace_research(root, payload):
    """Prepare one explicitly selected query/family using a registered profile."""
    from .catalog import load_catalog
    from .tools import ResearchTools
    from gpac.physical import verify_export
    resolver = ResearchTools(root); root = resolver.root
    required = {'kind', 'source_id', 'expected_source_sha256', 'profile_id', 'expected_profile_sha256', 'query', 'label'}
    if (not isinstance(payload, dict) or payload.get('kind') not in ('query', 'series')
            or set(payload) != required | ({'design'} if payload['kind'] == 'series' else set())
            or not isinstance(payload['label'], str) or not 1 <= len(payload['label']) <= 100):
        raise ValueError('Select a source, saved settings, query and name')
    with lock(root/'.workspace-catalog.lock'):
        manifest = load_catalog(root)
        if len(manifest.get('research', [])) >= 128:
            raise ValueError('Workspace research limit reached')
        sources = [e for e in manifest['structures'] if e['id'] == payload['source_id']]
        profiles = [e for e in manifest.get('research', []) if e['id'] == payload['profile_id'] and e['kind'] == 'profile']
        if len(sources) != 1 or len(profiles) != 1:
            raise ValueError('Unknown registered source or calculation settings')
        selected, profile = sources[0], profiles[0]
        source = resolver.path(selected['structure']); export = verify_export(load(source))
        if len(export['geometry']['site_ids']) > 256:
            raise ValueError('Interactive structures are bounded to 256 sites')
        if export['source']['sha256'] != payload['expected_source_sha256'] or profile['sha256'] != payload['expected_profile_sha256']:
            raise ValueError('Source or calculation settings changed; refresh the workspace')
        request = _read(resolver, profile)
        options = dict(settings=Settings.from_dict(request['settings']), mesh=request['kmesh'],
            basis_size=request['basis_size'], basis_energy_shift_ry=request['basis_energy_shift_ry'],
            basis_spec=request.get('basis_spec'), query=payload['query'])
        # Optional diagnostic exports are not silently inherited from a profile.
        pseudos = resolver.path(profile['path'])/'pseudos'
        identifier = payload['kind']+'-'+uuid4().hex[:16]
        output = resolver.path('prepared/'+identifier)
        if payload['kind'] == 'query':
            prepare_experiment(export, pseudos, output,
                candidate=resolver.path(selected['candidate']) if 'candidate' in selected else None, **options)
        else:
            prepare_series(source, pseudos, output, design=payload['design'], **options)
        entry = dict(id=identifier, label=payload['label'], kind=payload['kind'],
            path=str(output.relative_to(root)), sha256=sha256(_artifact(output, payload['kind'])))
        _read(resolver, entry)
        manifest.setdefault('research', []).append(entry)
        save(root/'workspace.json', manifest)
    return dict(entry=entry, calculation_performed=False, queued=False, reservation_created=False)
