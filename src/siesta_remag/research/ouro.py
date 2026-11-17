"""Temporary-file Ouro requests with local provenance and explicit admission.

Only preparation is exposed to policy tools. The operator API requires a fresh
route contract, a non-refreshing SDK client, and an external admission callback.
There is deliberately no generic provider/GPU admission implementation here.
"""
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import hashlib
import mimetypes
from pathlib import Path
import re
from uuid import UUID

from ..corpus import sha256
from ..discovery.common import digest, load, save

SDK_VERSION = '2.24.0'
MAX_INPUT_BYTES = 64 * 1024 * 1024
TERMINAL = {'success', 'error', 'timed-out', 'cancelled'}
CONTRACT_FIELDS = ('id', 'parent_id', 'org_id', 'team_id', 'visibility',
    'monetization', 'price_currency', 'price', 'price_usd', 'price_sats',
    'unit_cost', 'unit_cost_usd', 'unit_cost_sats', 'cost_accounting',
    'cost_unit', 'max_billable_seconds', 'route')


def _plain(value):
    return value.model_dump(mode='json') if hasattr(value, 'model_dump') else value


def contract(route):
    """Exclude display text/latency counters; bind the full executable contract."""
    raw = _plain(route)
    result = {key: raw.get(key) for key in CONTRACT_FIELDS}
    for key in ('id', 'parent_id'):
        UUID(result[key])
    if not isinstance(result['route'], dict):
        raise ValueError('Missing route contract')
    # Validate JSON, including rejection of NaN before writing any artifacts.
    digest(result)
    return result


def _nonnegative(value):
    try:
        n = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Unknown USD route price') from exc
    if not n.is_finite() or n < 0:
        raise ValueError('Invalid USD route price')
    return n


def price_ceiling_cents(route):
    """USD metadata ceiling, not an invoice, spend authorization or runtime cap."""
    if route['cost_accounting'] == 'variable':
        raise ValueError('Variable-cost routes require persistent input assets')
    # Inherited/unknown monetization must be resolved by the provider/operator.
    if route['visibility'] in ('public', 'private') and route['monetization'] in (None, 'free'):
        if any(route.get(k) not in (None, 0) for k in
               ('price', 'price_usd', 'price_sats', 'unit_cost', 'unit_cost_usd', 'unit_cost_sats')):
            raise ValueError('Ambiguous free-route pricing')
        return 0
    if route['monetization'] != 'pay-per-use' or route['cost_accounting'] not in ('fixed', 'runtime'):
        raise ValueError('Unsupported or unresolved route pricing')
    rate = route.get('unit_cost_usd')
    if rate is None and route['price_currency'] == 'usd':
        rate = route.get('unit_cost')
    rate = _nonnegative(rate)
    if route['cost_accounting'] == 'runtime':
        seconds = route.get('max_billable_seconds')
        if type(seconds) is not int or not 0 < seconds <= 86400:
            raise ValueError('Runtime route needs a finite billing ceiling')
        rate *= seconds
    return int((100 * rate).to_integral_value(rounding=ROUND_CEILING))


def _schema_check(schema):
    """Reject network schema references, even in unused branches."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == '$id' and (not isinstance(value, str) or not value.startswith('#')):
                raise ValueError('External JSON schema base identifiers are unsupported')
            if key in ('$ref', '$dynamicRef', '$recursiveRef') and (
                    not isinstance(value, str) or not value.startswith('#/')):
                raise ValueError('Only local JSON schema references are supported')
            _schema_check(value)
    elif isinstance(schema, list):
        for value in schema:
            _schema_check(value)


def _resolve(schema, root):
    seen = set()
    while '$ref' in schema:
        ref = schema['$ref']
        if ref in seen or set(schema) != {'$ref'}:
            raise ValueError('Unsupported file schema reference')
        seen.add(ref)
        schema = root
        for part in ref[2:].split('/'):
            schema = schema[part.replace('~1', '/').replace('~0', '~')]
    return schema


def _enabled(item, body, schema):
    return all(any(digest(body.get(k, schema.get('properties', {}).get(k, {}).get('default'))) == digest(x)
                   for x in (v if isinstance(v, list) else [v]))
               for k, v in (item.get('when') or {}).items())


def _validate(route, body, files, query, params):
    from jsonschema import Draft202012Validator
    spec = route['route']
    declared = spec.get('input_assets') or {}
    if not files or set(files) - set(declared) or set(files) & set(body):
        raise ValueError('Files must use declared input keys without body overrides')
    # No implicit persistent assets via unprovided input declarations.
    if set(declared) != set(files):
        raise ValueError('Every declared input must be an explicit temporary file')
    schema = spec.get('request_body', {}).get('content', {}).get('application/json', {}).get('schema')
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ValueError('Route needs an explicit JSON object request schema')
    _schema_check(schema)
    Draft202012Validator.check_schema(schema)
    service_body = dict(body)
    for name, item in files.items():
        declaration = declared[name]
        if declaration.get('asset_type') != 'file':
            raise ValueError('Only file input declarations are supported')
        extensions = declaration.get('file_extensions')
        if extensions and Path(item['file_name']).suffix.lower().lstrip('.') not in {
                x.lower().lstrip('.') for x in extensions}:
            raise ValueError('Input file extension is incompatible with route')
        field = _resolve(schema.get('properties', {}).get(name, {}), schema)
        if field.get('type') != 'object':
            raise ValueError('Temporary input requires an object field, not an asset-ID string')
        # Ouro's documented upload metadata excludes asset id/org_id/team_id.
        metadata = {'url': 'https://temporary-input.invalid/file',
            'filename': item['file_name'], 'type': mimetypes.guess_type(item['file_name'])[0] or 'application/octet-stream',
            'size': item['bytes']}
        if set(field.get('required', [])) - set(metadata):
            raise ValueError('Input schema requires metadata unavailable on temporary uploads')
        service_body[name] = metadata
    Draft202012Validator(schema).validate(service_body)
    supplied = {'query': query, 'path': params}
    allowed = {'query': set(), 'path': set()}
    for p in spec.get('parameters') or []:
        where, name = p.get('in'), p.get('name')
        if where not in supplied:
            raise ValueError('Unsupported route parameter location')
        allowed[where].add(name)
        if p.get('required') and name not in supplied[where]:
            raise ValueError('Missing required route parameter: ' + name)
        parameter_schema = p.get('schema', {})
        _schema_check(parameter_schema)
        if name in supplied[where]:
            Draft202012Validator(parameter_schema).validate(supplied[where][name])
    if any(set(supplied[k]) - allowed[k] for k in supplied):
        raise ValueError('Unknown route parameter')
    # Do not create posts/comments as an incidental consequence of a file run.
    for item in (spec.get('output_assets') or {}).values():
        if _enabled(item, body, schema) and item.get('asset_type') in ('post', 'comment'):
            raise ValueError('Disable route post/comment outputs before preparation')


def prepare_request(route, inputs, output, *, body=None, query=None, params=None):
    """Offline preparation; copies original bytes, never uploads or reserves."""
    route = contract(route)
    ceiling = price_ceiling_cents(route)
    files, contents = {}, {}
    if not isinstance(inputs, dict) or not 1 <= len(inputs) <= 16:
        raise ValueError('Inputs must map 1..16 route keys to local files')
    for index, (key, filename) in enumerate(sorted(inputs.items())):
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', key):
            raise ValueError('Invalid file input key')
        path = Path(filename)
        if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError('Missing or oversized input file')
        data = path.read_bytes()
        if not data or len(data) > MAX_INPUT_BYTES:
            raise ValueError('Empty or oversized input file')
        local = f'inputs/{index:02d}{path.suffix.lower()}'
        files[key] = {'path': local, 'file_name': path.name,
                      'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        contents[local] = data
        if sum(len(x) for x in contents.values()) > MAX_INPUT_BYTES:
            raise ValueError('Total input files exceed the local byte limit')
    body, query, params = body or {}, query or {}, params or {}
    if any(not isinstance(x, dict) for x in (body, query, params)):
        raise ValueError('Body and parameters must be objects')
    _validate(route, body, files, query, params)
    request = {'format': 'REMAG-OURO-REQUEST/1', 'sdk_version': SDK_VERSION,
        'contract': route, 'contract_sha256': digest(route), 'inputs': files,
        'body': body, 'query': query, 'params': params,
        'output': {'visibility': 'private'}, 'currency': 'usd',
        'price_ceiling_cents': ceiling, 'temporary_retention_days': 7,
        'scientific_acceptance': 'not_evaluated'}
    output = Path(output)
    digest(request)
    output.mkdir(parents=True, exist_ok=False)
    for local, data in contents.items():
        p = output / local
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
    save(output / 'request.json', request)
    return {'request': str(output / 'request.json'), 'request_sha256': digest(request),
            'price_ceiling_cents': ceiling, 'uploaded': False, 'executed': False}


def verify_request(directory):
    directory = Path(directory).resolve()
    request = load(directory / 'request.json')
    if request.get('format') != 'REMAG-OURO-REQUEST/1' or request.get('sdk_version') != SDK_VERSION:
        raise ValueError('Unsupported prepared request')
    route = contract(request['contract'])
    if digest(route) != request['contract_sha256'] or price_ceiling_cents(route) != request['price_ceiling_cents']:
        raise ValueError('Prepared route contract differs')
    if request['output'] != {'visibility': 'private'} or request['currency'] != 'usd':
        raise ValueError('Prepared output/currency policy differs')
    for item in request['inputs'].values():
        p = (directory / item['path']).resolve()
        if not p.is_relative_to(directory) or not p.is_file() or p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
            raise ValueError('Prepared file binding differs')
    _validate(route, request['body'], request['inputs'], request['query'], request['params'])
    return request


def _client_check(client):
    from importlib.metadata import version
    if version('ouro-py') != SDK_VERSION or client.can_refresh:
        raise ValueError('Use pinned Ouro SDK with an access-token client: mutation retries must be disabled')


def submit_request(directory, attempt, client, *, expected_sha256, maximum_cents, admit):
    """Operator-only submission, once. ``admit`` must durably hold resources.

    admit(request_hash, metadata_ceiling_cents) must enforce the operator's
    reviewed route/resource/time/spend authorization, shared account lock and
    ledger, then return a JSON receipt. It MUST NOT release on function return:
    the action may still be running, including on transport failures. No default
    admission is provided until a provider-specific allocation contract exists.
    """
    _client_check(client)
    request = verify_request(directory)
    if digest(request) != expected_sha256:
        raise ValueError('Request differs from operator-reviewed digest')
    if type(maximum_cents) is not int or maximum_cents < request['price_ceiling_cents']:
        raise ValueError('Route exceeds operator cost ceiling')
    if not callable(admit):
        raise ValueError('A provider-specific allocation admission controller is required')
    fresh = client.routes.retrieve(request['contract']['id'])
    if digest(contract(fresh)) != request['contract_sha256']:
        raise ValueError('Route schema or price changed; prepare and review again')
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=False)  # durable no-resubmission marker
    save(attempt / 'request.json', request)
    state = {'format': 'REMAG-OURO-ATTEMPT/1', 'request_sha256': expected_sha256,
        'state': 'admission_pending', 'upload_refs': {}, 'action_id': None,
        'allocation_release_verified': False, 'scientific_acceptance': 'not_evaluated',
        'actual_billed_cents': None}
    save(attempt / 'state.json', state)
    try:
        for item in request['inputs'].values():
            data = (Path(directory) / item['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Prepared file changed before admission')
            p = attempt / item['path']
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(data)
        state['admission'] = admit(expected_sha256, request['price_ceiling_cents'])
        if not isinstance(state['admission'], dict) or not state['admission']:
            raise ValueError('Admission must return a durable nonempty receipt')
        state['state'] = 'uploading'
        save(attempt / 'state.json', state)
        for name, item in request['inputs'].items():
            # Send the exact bytes just rehashed, not a path the SDK may reread.
            data = (attempt / item['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Prepared file changed before upload')
            ref = client.files.upload_input(data, file_name=item['file_name'])
            UUID(str(ref['upload_id']))
            state['upload_refs'][name] = {'upload_id': str(ref['upload_id']), 'file_name': item['file_name']}
            save(attempt / 'state.json', state)
        state['state'] = 'submission_uncertain'
        save(attempt / 'state.json', state)
        action = client.routes.execute(fresh, body=request['body'], query=request['query'],
            params=request['params'], input_assets=state['upload_refs'], output=request['output'],
            currency='usd', wait=False, timeout=30, notify='none', raise_on_error=False)
        raw = _plain(action)  # do not use SDK final_data: it mutates response
        # Retain any returned ID before interpretation/validation can fail.
        save(attempt / 'action-submission.json', raw)
        _bind_action(raw, request, state)
        save(attempt / 'state.json', state)
    except Exception as exc:
        state['error_type'] = type(exc).__name__  # exceptions may contain signed URLs/secrets
        save(attempt / 'state.json', state)
        raise
    return state


def _bind_action(raw, request, state):
    UUID(raw['id'])
    if raw['route_id'] != request['contract']['id'] or raw['status'] not in TERMINAL | {'queued', 'in-progress'}:
        raise ValueError('Returned action does not match the reviewed route')
    if state['action_id'] is not None and state['action_id'] != raw['id']:
        raise ValueError('Action identity changed')
    state['action_id'] = raw['id']
    state['state'] = raw['status']


@contextmanager
def _attempt_lock(attempt):
    with (attempt / '.observation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def observe_action(attempt, client):
    """One read of the existing action. Never submits, retries or frees a GPU."""
    attempt = Path(attempt)
    with _attempt_lock(attempt):
        state, request = load(attempt / 'state.json'), load(attempt / 'request.json')
        if digest(request) != state['request_sha256'] or not state['action_id']:
            raise ValueError('Unbound attempt or unknown action: reconcile without resubmission')
        raw = _plain(client.routes.retrieve_action(state['action_id']))
        _bind_action(raw, request, state)
        index = len(list(attempt.glob('action-observation-*.json')))
        save(attempt / f'action-observation-{index:04d}.json', raw)
        save(attempt / 'state.json', state)
        return state


def collect_action(attempt, client, output):
    """Save one terminal action and named file outputs; no scientific promotion.

    Failed actions and their available files are retained too. Raw response,
    usage records and asset IDs remain separate from interpreted observations.
    Output must be a fresh directory; partial downloads are never overwritten.
    """
    attempt, output = Path(attempt), Path(output)
    with _attempt_lock(attempt):
        state, request = load(attempt / 'state.json'), load(attempt / 'request.json')
        if digest(request) != state['request_sha256'] or not state['action_id']:
            raise ValueError('Unbound attempt or unknown action')
        raw = _plain(client.routes.retrieve_action(state['action_id']))
        _bind_action(raw, request, state)
        if state['state'] not in TERMINAL:
            raise ValueError('Action is still running')
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'action.json', raw)
        manifest = {'format': 'REMAG-OURO-COLLECTION/1', 'action_id': raw['id'],
            'route_id': raw['route_id'], 'request_sha256': state['request_sha256'],
            'service_status': raw['status'], 'scientific_acceptance': 'not_evaluated',
            'allocation_release_verified': False, 'artifacts': {'action.json': sha256(output / 'action.json')},
            'files': [], 'complete': False}
        save(output / 'manifest.json', manifest)
        names = set()
        declarations = request['contract']['route'].get('output_assets') or {}
        schema = request['contract']['route']['request_body']['content']['application/json']['schema']
        for index, item in enumerate(raw.get('output_assets') or []):
            asset, name = item.get('asset') or {}, item.get('name')
            declared = declarations.get(name)
            if (not declared or name in names or not _enabled(declared, request['body'], schema)
                    or asset.get('asset_type') != declared.get('asset_type')):
                raise ValueError('Undeclared or incompatible output asset')
            names.add(name)
            if asset.get('asset_type') != 'file':
                continue  # non-file output metadata is retained in action.json
            asset_id = str(UUID(asset['id']))
            filename = f'output-{index:02d}-{asset_id}.bin'
            client.assets.download(asset_id, output_path=str(output / filename), asset_type='file')
            manifest['files'].append({'name': name, 'asset_id': asset_id, 'path': filename})
            manifest['artifacts'][filename] = sha256(output / filename)
            save(output / 'manifest.json', manifest)
        missing = [name for name, item in declarations.items()
                   if _enabled(item, request['body'], schema) and not item.get('optional') and name not in names]
        manifest['missing_outputs'] = missing
        if missing and raw['status'] == 'success':
            save(output / 'manifest.json', manifest)
            raise ValueError('Successful action lacks required declared outputs')
        manifest['complete'] = True
        save(output / 'manifest.json', manifest)
        save(attempt / 'state.json', state)
        return manifest
