"""Read-only, explicitly registered public tool receipts for the workspace."""
import hashlib
import json
from pathlib import Path


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Activity path must remain inside the workspace')
    return path


def _json(path):
    with path.open('rb') as stream:
        raw = stream.read(262145)
    if len(raw) > 262144:
        raise ValueError('Activity record exceeds 256 KiB')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def activity_state(workspace):
    """Never inspect ambient agents, model transcripts or execute a tool."""
    root = Path(workspace).resolve()
    manifest = _path(root, 'research-activity.json')
    if not manifest.exists():
        return dict(available=False)
    config, fingerprint = _json(manifest)
    if config.get('format') != 'REMAG-RESEARCH-ACTIVITY/1':
        raise ValueError('Unknown activity manifest')
    stages, policies = config.get('stages', []), config.get('policies', [])
    if not isinstance(stages, list) or len(stages) > 12 or not isinstance(policies, list) or len(policies) > 8:
        raise ValueError('Activity view exceeds stage or policy bounds')
    result = dict(available=True, manifest_sha256=fingerprint,
                  **{key: config.get(key, '') for key in ('title', 'status', 'summary', 'next', 'updated_at')})
    result['stages'] = [{key: stage.get(key, '') for key in ('label', 'status', 'detail')} for stage in stages]
    result['policies'] = []
    for policy in policies:
        calls = _path(root, policy['calls'])
        if not calls.is_dir():
            raise ValueError('Registered activity calls directory is missing')
        entries = sorted(calls.glob('*/request.json'))
        if len(entries) > 256:
            raise ValueError('Activity policy exceeds 256 calls')
        actions = []
        for entry in entries:
            entry = _path(root, entry)
            request, request_hash = _json(entry)
            if request.get('format') != 'REMAG-DIRECT-TOOL-CALL/1':
                raise ValueError('Unknown public tool receipt')
            call = request['request']
            response_path = _path(root, entry.parent / 'response.json')
            action = dict(sequence=entry.parent.name, tool=call['tool'], rationale=call.get('rationale', ''),
                          started_at=request.get('started_at', ''), status='pending', request_sha256=request_hash)
            # Only public notebook records, never arbitrary tool arguments or model context.
            if call['tool'] in ('record_hypothesis', 'record_experiment', 'propose_experiment', 'record_response_prediction', 'revise_hypothesis'):
                action['record'] = call.get('args', {})
            if response_path.exists():
                response, _ = _json(response_path)
                if (response.get('format') != 'REMAG-DIRECT-TOOL-RESULT/1'
                        or response.get('request_sha256') != request_hash):
                    raise ValueError('Activity response belongs to another request')
                action.update(status='passed' if response.get('ok') is True else 'failed',
                              finished_at=response.get('finished_at', ''), elapsed_seconds=response.get('elapsed_seconds'))
            actions.append(action)
        result['policies'].append(dict(label=policy['label'], status=policy.get('status', ''), actions=actions))
    return result
