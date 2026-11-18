"""Validate authored candidate provenance supplied with REMAG experiment inputs."""
from pathlib import Path

from ..corpus import sha256
from ..discovery.common import load

EDIT_OPERATIONS = frozenset({'apply', 'physical-strain', 'physical-isotropic',
    'physical-tetragonal', 'physical-displace', 'physical-coordination-scale'})
AUTHOR_OPERATIONS = EDIT_OPERATIONS | {'inspect', 'construct', 'assemble', 'physical-export'}


def _program(request, reply):
    r = reply['result']; op = request['op']
    if op in EDIT_OPERATIONS - {'apply'}: return r['child']
    if op in ('apply', 'construct', 'assemble'): return r['program']
    if op == 'physical-export': return r['source']
    return r  # Explicit import/inspection of an exact program.


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
