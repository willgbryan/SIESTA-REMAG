"""Exercise the same one-call interface used by a live chat; no model or solver."""
import json
import os
import subprocess
import sys

import pytest

from siesta_remag.corpus import sha256
from siesta_remag.discovery.common import load
from siesta_remag.research.toolhost import ToolDefinition, ToolHost, create_host


def call_cli(root, request, receipt='calls/first', *options):
    result = subprocess.run([sys.executable, '-m', 'siesta_remag.research', 'tool-call',
        '--workspace', str(root), '--request', '-', '--receipt', receipt, *options],
        input=json.dumps(request), text=True, capture_output=True)
    return result, json.loads(result.stdout)


def test_chat_roundtrip_freezes_intent_and_does_not_replay_receipt(tmp_path):
    request = dict(tool='record_hypothesis', args=dict(statement='Test a strain response',
        predictions=['Positive anisotropy'], alternatives=['Sign changes']), rationale='Retain the claim before reading results')
    process, response = call_cli(tmp_path, request)
    assert process.returncode == 0 and response['ok']
    intent = load(tmp_path/'calls/first/request.json')
    assert intent['request'] == request
    assert not intent['native_execution_enabled'] and not intent['cpu_relaxation_enabled']
    assert response['request_sha256'] == sha256(tmp_path/'calls/first/request.json')
    saved = sha256(tmp_path/'calls/first/response.json')
    process, repeated = call_cli(tmp_path, request)
    assert process.returncode == 2 and not repeated['ok']
    assert sha256(tmp_path/'calls/first/response.json') == saved
    process, history = call_cli(tmp_path, dict(tool='retrieve_research_history', args={},
        rationale='Recover the existing decision after a new process starts'), 'calls/history')
    assert process.returncode == 0 and len(history['result']) == 1
    assert history['result'][0]['id'] == response['result']


def test_default_surface_disables_compute_and_preserves_scalar_profile(tmp_path):
    host = ToolHost(tmp_path)
    assert len(host.catalog()['tools']) >= 25
    assert not {'run_prepared_query', 'prepare_relaxation'} & host.catalog()['tools'].keys()
    scalar = ToolHost(tmp_path, information_profile='scalar')
    assert 'export_electronic_matrices' not in scalar.catalog()['tools']
    for i, name in enumerate(('run_prepared_query', 'prepare_relaxation', 'export_electronic_matrices')):
        r = scalar.call(dict(tool=name, args={}, rationale='Unavailable action'), f'calls/{i}')
        assert not r['ok'] and 'unavailable' in r['error']['message']
    assert not (tmp_path/'research-memory.sqlite').exists()
    r = host.call(dict(tool='inspect_calculation', args={'report':'../outside'}, rationale='Check path boundary'), 'calls/path')
    assert not r['ok'] and 'escapes' in r['error']['message']


def test_trusted_extension_discovery_execution_and_failures(tmp_path, monkeypatch):
    module = tmp_path/'local_tools.py'
    module.write_text('''from siesta_remag.research.toolhost import ToolDefinition
def count(context, arguments):
    if not isinstance(arguments['values'], list): raise ValueError('Expected values list')
    print('Diagnostic on stderr, not the JSON transport')
    return {'count': len(arguments['values'])}
def register_tools(host):
    print('Extension registration diagnostic')
    host.register(ToolDefinition('count_samples', 'Count supplied saved samples', ('values',), (), count, 'inspect'))
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    host = create_host(tmp_path, extensions=['local_tools'])
    schema = host.catalog()['tools']['count_samples']['inputSchema']
    assert schema['required'] == ['values'] and not schema['additionalProperties']
    r = host.call(dict(tool='count_samples', args={'values':[1, 2]}, rationale='Check extension'), 'calls/extension')
    assert r['ok'] and r['result'] == {'count':2}
    assert load(tmp_path/'calls/extension/request.json')['implementation_sha256'][str(module)] == sha256(module)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    process, response = call_cli(tmp_path, dict(tool='count_samples', args={'values':[3]},
        rationale='Exercise extension through the actual chat CLI'), 'calls/plugin-cli', '--extension', 'local_tools')
    assert process.returncode == 0 and response['result'] == {'count':1}
    assert 'registration diagnostic' in process.stderr and 'JSON transport' in process.stderr
    scalar = create_host(tmp_path, extensions=['local_tools'], information_profile='scalar')
    assert 'count_samples' not in scalar.catalog()['tools']
    for i, args in enumerate(({}, {'values':[], 'unknown':True}, {'values':'bad'})):
        r = host.call(dict(tool='count_samples', args=args, rationale='Reject invalid input'), f'calls/bad{i}')
        assert not r['ok'] and not r['retry_safe']
    with pytest.raises(ValueError, match='duplicate'):
        host.register(ToolDefinition('inspect_structure', 'Cannot shadow built-ins', (), (), lambda c,a: None, 'inspect'))
    with pytest.raises(ValueError, match='reserved'):
        host.register(ToolDefinition('run_prepared_query', 'Cannot enable compute', (), (), lambda c,a: None, 'inspect'))


def test_extension_sees_intent_before_effect_and_retains_unknown_failure(tmp_path):
    host = ToolHost(tmp_path)
    def interrupted(context, arguments):
        assert (tmp_path/'calls/interrupted/request.json').is_file()
        (tmp_path/'partial.txt').write_text('partial effect')
        raise RuntimeError('Interrupted after partial work')
    host.register(ToolDefinition('partial', 'Exercise partial failure', (), (), interrupted, 'prepare'))
    r = host.call(dict(tool='partial', args={}, rationale='Retain failure evidence'), 'calls/interrupted')
    assert not r['ok'] and not r['retry_safe'] and (tmp_path/'partial.txt').exists()
    assert load(tmp_path/'calls/interrupted/response.json') == r


def test_cli_catalog_does_not_create_notebook(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'siesta_remag.research', 'tool-catalog',
        '--workspace', str(tmp_path), '--name', 'author_structure'], text=True, capture_output=True, check=True)
    catalog = json.loads(result.stdout)['result']['tools']
    assert list(catalog) == ['author_structure']
    assert catalog['author_structure']['effect'] == 'prepare'
    assert not list(tmp_path.iterdir())
