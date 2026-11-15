"""Persist existing GPAC worker results for research queries and the workspace.

Exact authoring and lineage remain GPAC operations. This module owns artifact
handoff only; it never compiles approximately, relaxes or submits a calculation.
"""
from pathlib import Path
import json

from ..corpus import sha256
from ..discovery.common import load, save
from ..discovery.pipeline import lock

EDIT_OPERATIONS = frozenset({'apply', 'physical-strain', 'physical-isotropic',
    'physical-tetragonal', 'physical-displace', 'physical-coordination-scale'})
AUTHOR_OPERATIONS = EDIT_OPERATIONS | {'inspect', 'construct', 'assemble', 'physical-export'}


def _worker(request):
    from gpac.worker import serve_bytes
    from gpac.errors import GPACError
    reply = serve_bytes(json.dumps(request, allow_nan=False).encode())
    if not reply['ok']:
        e = reply['error']
        raise GPACError(e['code'], e['detail'], path=e['path'], stage=e['stage'])
    return reply


def _program(request, reply):
    r = reply['result']; op = request['op']
    if op in EDIT_OPERATIONS - {'apply'}: return r['child']
    if op in ('apply', 'construct', 'assemble'): return r['program']
    if op == 'physical-export': return r['source']
    return r  # Explicit import/inspection of an exact program.


def inspect_structure(structure, *, candidate=None):
    from gpac.physical import verify_export, describe_dofs, coordination
    export = verify_export(load(structure))
    result = dict(structure=export, parameters=describe_dofs(export['source']['program']),
        coordination=coordination(export['geometry'], cutoff_angstrom=3.4))
    if candidate is not None:
        record = verify_candidate(candidate)
        if record['source_sha256'] != export['source']['sha256']:
            raise ValueError('Candidate lineage belongs to another exact structure')
        result['candidate'] = record
    return result


def author_candidate(request, output, *, structure=None):
    """Save one new exact candidate; no implicit canonicalization or calculation."""
    from gpac.physical import verify_export, export_geometry
    if not isinstance(request, dict) or request.get('op') not in AUTHOR_OPERATIONS:
        raise ValueError('Select a supported static GPAC authoring operation')
    output = Path(output)
    if output.exists(): raise FileExistsError('Candidate output must be new')
    request = dict(request)
    parent = None
    if structure is not None:
        if 'program' in request or request['op'] not in EDIT_OPERATIONS | {'inspect', 'physical-export'}:
            raise ValueError('A saved parent supplies the program; do not mix input sources')
        parent = verify_export(load(structure))
        if request['op'] in EDIT_OPERATIONS and request.get('expected_sha256') != parent['source']['sha256']:
            raise ValueError('Stale parent source hash')
        request['program'] = parent['source']['program']
    elif request['op'] in EDIT_OPERATIONS:
        raise ValueError('Edits require a saved parent structure and its expected hash')
    reply = _worker(request)
    if request['op'] == 'assemble' and reply['result'].get('satisfied') is not True:
        raise ValueError('Assembly assertions failed; inspect the worker result before authoring')
    program = _program(request, reply)
    export = export_geometry(program['program'], length_unit='angstrom')
    if export['source'] != program: raise ValueError('Worker program metadata differs from its exact source')
    # All GPAC admission, static checks and physical export validation precede
    # file creation. The exclusive directory prevents overwriting an attempt.
    output.mkdir(parents=True, exist_ok=False)
    save(output/'request.json', request); save(output/'response.json', reply)
    save(output/'structure.json', export)
    (output/'program.xcl').write_text(program['program'])
    if parent is not None: save(output/'parent.json', parent)
    record = dict(format='REMAG-AUTHORED-CANDIDATE/1', operation=request['op'],
        source_sha256=program['sha256'], export_sha256=export['sha256'],
        parent_sha256=parent['source']['sha256'] if parent else None,
        files={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        calculation_performed=False, physical_accuracy_qualified=False,
        scope='Exact static GPAC authoring with a NONSEMANTIC physical export; parent snapshot and original worker lineage retained')
    save(output/'candidate.json', record)
    return record


def verify_candidate(path):
    from gpac.physical import verify_export
    path = Path(path); root = path.parent; record = load(path)
    if path.name != 'candidate.json': raise ValueError('Use the original candidate.json record')
    if record.get('format') != 'REMAG-AUTHORED-CANDIDATE/1': raise ValueError('Expected an authored candidate record')
    if record.get('calculation_performed') is not False or record.get('physical_accuracy_qualified') is not False:
        raise ValueError('Authoring evidence is not a calculation or accuracy qualification')
    expected = {'request.json','response.json','structure.json','program.xcl'}
    if record.get('parent_sha256') is not None: expected.add('parent.json')
    if set(record['files']) != expected: raise ValueError('Candidate artifact inventory differs')
    for name, digest in record['files'].items():
        if not (root/name).resolve().is_relative_to(root.resolve()) or sha256(root/name) != digest:
            raise ValueError('Candidate artifact changed: '+name)
    request, reply = load(root/'request.json'), load(root/'response.json')
    if request['op'] != record['operation'] or request['op'] not in AUTHOR_OPERATIONS or not reply['ok']:
        raise ValueError('Candidate operation differs')
    export = verify_export(load(root/'structure.json'))
    if (export['source'] != _program(request, reply) or export['source']['program'] != (root/'program.xcl').read_text()
        or export['source']['sha256'] != record['source_sha256'] or export['sha256'] != record['export_sha256']):
        raise ValueError('Candidate program/export binding differs')
    if record['parent_sha256'] is not None:
        parent = verify_export(load(root/'parent.json'))
        if parent['source']['sha256'] != record['parent_sha256'] or request['program'] != parent['source']['program']:
            raise ValueError('Candidate parent binding differs')
        if request['op'] in EDIT_OPERATIONS and request['expected_sha256'] != record['parent_sha256']:
            raise ValueError('Candidate edit source differs')
    return record


def register_structure(root, structure, identifier, label, *, candidate=None):
    """Append to a workspace catalog. Existing entries/calculations are untouched."""
    from .catalog import load_catalog
    from .tools import ResearchTools
    resolver = ResearchTools(root); root = resolver.root
    if any(not isinstance(v,str) or not 1 <= len(v) <= 100 for v in (identifier,label)):
        raise ValueError('A structure identifier and label of 1..100 characters are required')
    source = resolver.path(structure)
    inspected=inspect_structure(source, candidate=resolver.path(candidate) if candidate else None)
    if len(inspected['structure']['geometry']['site_ids'])>256:
        raise ValueError('Interactive structures are bounded to 256 sites')
    with lock(root/'.workspace-catalog.lock'):
        manifest = load_catalog(root) if (root/'workspace.json').exists() else dict(
            format='REMAG-RESEARCH-WORKSPACE/1',title='Crystal research',structures=[])
        if any(e['id'] == identifier for e in manifest['structures']): raise ValueError('Workspace identifier already exists')
        if len(manifest['structures']) >= 64: raise ValueError('Workspace structure limit reached')
        entry = dict(id=identifier,label=label,structure=str(source.relative_to(root)))
        if candidate: entry['candidate'] = str(resolver.path(candidate).relative_to(root))
        manifest['structures'].append(entry); save(root/'workspace.json',manifest)
    return entry


def save_workspace_candidate(root, payload):
    """Save a static edit of one registered source and append it to the catalog."""
    from uuid import uuid4
    from .catalog import load_catalog
    from .tools import ResearchTools
    resolver=ResearchTools(root);root=resolver.root
    if (not isinstance(payload,dict) or set(payload)!={'source_id','expected_source_sha256','request','label'}
        or not isinstance(payload['label'],str) or not 1<=len(payload['label'])<=100
        or not isinstance(payload['request'],dict) or payload['request'].get('op') not in EDIT_OPERATIONS):
        raise ValueError('Provide a registered source, exact edit and candidate label')
    with lock(root/'.workspace-catalog.lock'):
        catalog=load_catalog(root)
        if len(catalog['structures'])>=64:raise ValueError('Workspace structure limit reached')
        selected=[e for e in catalog['structures'] if e['id']==payload['source_id']]
        if len(selected)!=1:raise ValueError('Unknown registered source')
        source=resolver.path(selected[0]['structure'])
        request=dict(payload['request'])
        if 'expected_sha256' in request and request['expected_sha256']!=payload['expected_source_sha256']:
            raise ValueError('Conflicting expected source hashes')
        request['expected_sha256']=payload['expected_source_sha256']
        identifier='candidate-'+uuid4().hex[:16];folder=root/'candidates'/identifier
        record=author_candidate(request,folder,structure=source)
        entry=dict(id=identifier,label=payload['label'],structure=str((folder/'structure.json').relative_to(root)),
            candidate=str((folder/'candidate.json').relative_to(root)))
        catalog['structures'].append(entry);save(root/'workspace.json',catalog)
    return dict(entry=entry,candidate=record)
