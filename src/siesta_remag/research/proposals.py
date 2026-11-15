"""Persist GPAC-owned advisory assessment; no solver or feasibility verdict."""
from pathlib import Path

from ..corpus import sha256
from ..discovery.common import load, save
from .authoring import _worker


def assess_structure(structure, policy, output, *, reference=None):
    from gpac.exact import validate_proposal_policy
    from gpac.physical import verify_export
    policy = validate_proposal_policy(policy)
    output = Path(output)
    if output.exists(): raise FileExistsError('Assessment output must be new')
    source = verify_export(load(structure))
    ref = verify_export(load(reference)) if reference is not None else None
    request = dict(op='physical-assess', program=source['source']['program'], policy=policy)
    if ref is not None: request['reference_program'] = ref['source']['program']
    result = _worker(request)['result']
    output.mkdir(parents=True, exist_ok=False)
    save(output/'structure.json', source)
    save(output/'policy.json', policy)
    save(output/'request.json', request)
    save(output/'assessment.json', result)
    if ref is not None: save(output/'reference.json', ref)
    paths = ['structure.json', 'policy.json', 'request.json', 'assessment.json'] + (['reference.json'] if ref is not None else [])
    record = dict(format='REMAG-PROPOSAL-ASSESSMENT/1', source_sha256=result['source_sha256'],
        reference_source_sha256=result['reference_source_sha256'], policy_sha256=result['policy_sha256'],
        assessment_sha256=result['sha256'], files={p:sha256(output/p) for p in paths},
        calculation_performed=False, advisory_only=True)
    save(output/'record.json', record)
    return result, record


def verify_assessment(directory):
    """Reproduce a saved, small geometric assessment from its exact inputs."""
    directory = Path(directory)
    record = load(directory/'record.json')
    if record.get('format') != 'REMAG-PROPOSAL-ASSESSMENT/1': raise ValueError('Expected proposal assessment record')
    paths = {'structure.json', 'policy.json', 'request.json', 'assessment.json'}
    if record.get('reference_source_sha256') is not None: paths.add('reference.json')
    if not isinstance(record.get('files'), dict) or set(record['files']) != paths:
        raise ValueError('Assessment artifact inventory differs')
    for p in paths:
        if sha256(directory/p) != record['files'][p]: raise ValueError('Assessment artifact changed: '+p)
    from gpac.physical import verify_export
    source = verify_export(load(directory/'structure.json'))
    request = dict(op='physical-assess', program=source['source']['program'], policy=load(directory/'policy.json'))
    if 'reference.json' in paths:
        request['reference_program'] = verify_export(load(directory/'reference.json'))['source']['program']
    if load(directory/'request.json') != request: raise ValueError('Assessment request differs from source')
    result = _worker(request)['result']
    if result != load(directory/'assessment.json'): raise ValueError('Saved assessment does not reproduce')
    for k in ('source_sha256', 'reference_source_sha256', 'policy_sha256'):
        if record.get(k) != result[k]: raise ValueError('Assessment binding differs: '+k)
    if (record.get('assessment_sha256') != result['sha256'] or record.get('calculation_performed') is not False
            or record.get('advisory_only') is not True): raise ValueError('Assessment scope differs')
    return record
