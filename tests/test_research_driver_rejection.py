import json
import shutil

import pytest

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import digest, load, save
from siesta_remag.research.execution import _review_driver_rejection
from test_research_execution import prepared


@pytest.mark.parametrize('change', [
    None, 'admitted_host', 'different_source', 'different_request', 'same_grant',
    'additional_artifact', 'native_work', 'unconfirmed_stop', 'different_report',
])
@pytest.mark.parametrize('operation', ['query','relaxation'])
def test_stopped_pre_solver_host_replacement(tmp_path, prepared, change, operation):
    prior = tmp_path/'prior'
    source = {'source_sha256': 'a'*64}
    policy = dict(operation='relaxation', source_binding=source)
    if operation == 'query':
        policy = dict(experiments=['query'], spectra=[])
        request = load(prepared/'experiment.json')
        request['query'] = {'kind':'scalar'}
        save(prepared/'experiment.json', request)
        shutil.copytree(prepared, prior/'campaign/query')
    original_host = dict(cuda_runtime=12060, cuda_driver_api=13030,
                         nvidia_driver_versions=['610.57.04'])
    driver = dict(format='REMAG-SCF-RUNTIME-POLICY/1', **original_host)
    save(prior/'runtime.json', dict(driver_policy=driver))
    host = dict(original_host)
    if change != 'admitted_host':
        host['nvidia_driver_versions'] = ['580.00.00']
    reason = 'Native runtime profile is not admitted: '+json.dumps(host, sort_keys=True)
    preflight = dict(request_sha256=sha256(prepared/'experiment.json') if operation=='query' else 'b'*64)
    save(prior/'preflight.json', preflight)
    save(prior/'campaign/execution.json', policy)
    owner = dict(status='held_review', allocation_stopped=True, app_id='ap-old', attempt='prior')
    save(prior/'execution.json', dict(**owner, app_lifecycle=dict(state='stopped'),
         config_sha256='c'*64, **source, **preflight))
    save(prior/'receipt.json', dict(app_id=owner['app_id']))
    cfg = dict(authorization_id='old', operation=operation, image='im-same', runtime_sha256=sha256(prior/'runtime.json'))
    save(prior/'authorization.json', dict(config=cfg, config_sha256='c'*64))
    report = dict(status='held_error', policy=policy, invocation_sha256=digest(policy),
                  experiments=[], reason=reason, timing=dict(native_seconds=int(change=='native_work')))
    collected = prior/'collected'
    save(collected/'report.json', report)
    files = {'report.json': sha256(collected/'report.json')}
    if change == 'additional_artifact':
        save(collected/'partial.json', {})
        files['partial.json'] = sha256(collected/'partial.json')
    save(collected/'archive-manifest.json', dict(status=report['status'],
         invocation_sha256=digest(policy), files=files))
    cfg.update(authorization_id='old' if change=='same_grant' else 'new', previous_app_id='ap-old',
               driver_rejection_report_sha256='0'*64 if change=='different_report' else files['report.json'])
    if change == 'unconfirmed_stop':
        owner['allocation_stopped'] = False
    if change == 'different_source':
        if operation=='query':
            request['source_sha256']='e'*64
            save(prepared/'experiment.json',request)
        else:
            source = dict(source_sha256='e'*64)
    if change == 'different_request':
        preflight = dict(request_sha256='f'*64)
    args = (tmp_path, owner, cfg, preflight, source if operation=='relaxation' else None)
    query = dict(scalar_bundle=prepared) if operation=='query' else {}
    if change is None:
        assert _review_driver_rejection(*args, **query) == 'prior/collected/report.json'
    else:
        with pytest.raises(ValueError):
            _review_driver_rejection(*args, **query)
