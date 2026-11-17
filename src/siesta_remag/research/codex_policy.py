"""Personal local Codex policy transport; uses Codex-managed ChatGPT login.

No token extraction, API-key fallback, shared daemon, chat resume or model tools
execute research actions. The operator's fixed broker executes returned JSON.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
from threading import Thread
import time

from ..corpus import sha256
from ..discovery.common import load, save
from . import model_policy

DISABLED = ('multi_agent', 'multi_agent_v2', 'multi_agent_v2_dynamic_tools',
    'agent_message_board', 'shell_tool', 'unified_exec', 'apps', 'plugins',
    'remote_plugin', 'hooks', 'memories', 'skill_search', 'skill_mcp_dependency_install',
    'code_mode', 'code_mode_host', 'browser_use', 'computer_use', 'image_generation',
    'view_image', 'goals', 'sleep_tool', 'workspace_dependencies',
    'unbounded_connection_retries', 'shell_snapshot', 'in_app_chat',
    'in_app_local_automation', 'tool_suggest', 'prediction_preserve_parent_tools')


def native_action(value):
    if isinstance(value, list):return any(native_action(v) for v in value)
    if not isinstance(value, dict):return False
    if value.get('type') in ('function_call','custom_tool_call','commandExecution',
            'dynamicToolCall','mcpToolCall','fileChange','webSearch','collabAgentToolCall'):
        return True
    return any(native_action(v) for v in value.values())


def config_value(value):
    required = {'format', 'binary', 'binary_sha256', 'model', 'reasoning_effort',
                'request_timeout_seconds', 'research_brief', 'disabled_mcp_servers'}
    if not isinstance(value, dict) or set(value) != required or value['format'] != 'REMAG-CODEX-POLICY/1':
        raise ValueError('Expected exact Codex policy configuration')
    if (value['model'] != 'gpt-6-astra' or value['reasoning_effort'] != 'xhigh'
            or type(value['request_timeout_seconds']) is not int
            or not 1 <= value['request_timeout_seconds'] <= 120
            or not isinstance(value['research_brief'], str) or not 1 <= len(value['research_brief']) <= 16000
            or not isinstance(value['disabled_mcp_servers'], list)
            or any(not isinstance(n, str) or not n or not n.replace('_', '').replace('-', '').isalnum()
                   for n in value['disabled_mcp_servers'])):
        raise ValueError('Invalid finite Codex policy bounds')
    binary = Path(value['binary'])
    if not binary.is_absolute() or sha256(binary) != value['binary_sha256']:
        raise ValueError('Codex binary must match the locally tested pin')
    return value


def controls(config):
    return {'agents.enabled': False, 'model_reasoning_effort': 'xhigh',
        'project_doc_max_bytes': 0, 'web_search': 'disabled',
        'analytics.enabled': False, 'check_for_update_on_startup': False,
        **{'features.'+f: False for f in DISABLED},
        'features.skip_host_skill_discovery': True,
        **{'mcp_servers.'+n+'.enabled': False for n in config['disabled_mcp_servers']}}


def decode(text, catalog):
    v = model_policy.loads(text)
    if not isinstance(v, dict) or set(v) != {'tool', 'arguments_json', 'rationale'}:
        raise ValueError('Codex must return one structured action')
    # Use the same strict catalog/argument decoder as the Responses adapter.
    response = {'status':'completed', 'output':[{'type':'function_call',
        'name':'select_research_action', 'arguments':json.dumps(v)}]}
    return model_policy.decode_decision(response, catalog)


class CodexPolicy:
    kind = 'codex-managed-chatgpt'

    def __init__(self, config_path, runtime_dir, *, local_fixture=None):
        self.config_path = Path(config_path).resolve()
        self.config = config_value(load(self.config_path))
        self.config_sha = sha256(self.config_path)
        self.root = Path(runtime_dir).resolve(); self.root.mkdir(exist_ok=False)
        self.local_fixture = local_fixture
        self.kind = 'codex-local-fixture' if local_fixture else type(self).kind
        self.events = []; self.queue = Queue(); self.counter = 0; self.held = False
        self.proc = None; self.stderr = None

    def _send(self, value):
        self.proc.stdin.write(json.dumps(value, allow_nan=False)+'\n'); self.proc.stdin.flush()

    def _next(self, deadline):
        left = deadline-time.monotonic()
        if left <= 0: raise TimeoutError('Codex deadline; no retry')
        try: raw = self.queue.get(timeout=min(left, .2))
        except Empty:
            if self.proc.poll() is not None: raise RuntimeError('Codex process exited')
            return None
        if len(raw) > 8*1024*1024: raise ValueError('Oversized Codex event')
        event = model_policy.loads(raw); self.events.append(event)
        if native_action(event):raise ValueError('Native model tool attempt; context held')
        # Native model tools are not a second path around the broker. Even a
        # rejected/aborted attempt invalidates this context; never continue it.
        method = event.get('method', '')
        params = event.get('params', {})
        item = params.get('item', {})
        if method.startswith('item/') and item.get('type') not in (None, 'userMessage', 'agentMessage', 'reasoning'):
            raise ValueError('Unexpected Codex tool/action event; context held')
        if 'id' in event and 'method' in event:
            raise ValueError('Codex requested a tool/approval/input; context held')
        if method == 'error' or event.get('error'):
            raise RuntimeError('Codex request failed; no retry')
        return event

    def _rpc(self, method, params, deadline):
        self.counter += 1; call = self.counter
        self._send({'id':call, 'method':method, 'params':params})
        while True:
            event = self._next(deadline)
            if event and event.get('id') == call: return event['result']

    def _start(self, deadline):
        c = self.config
        options = {**controls(c), 'log_dir':str(self.root/'logs'), 'sqlite_home':str(self.root/'state')}
        if self.local_fixture:
            # An explicit test seam: never derives an endpoint or auth from model output.
            from urllib.parse import urlsplit
            u = urlsplit(self.local_fixture)
            if u.scheme != 'http' or u.hostname != '127.0.0.1' or not u.port or u.username or u.password:
                raise ValueError('Fixture endpoint must be loopback HTTP')
            options.update({'model_provider':'local_fixture', 'model_providers.local_fixture':{
                'name':'Local fixture', 'base_url':self.local_fixture, 'wire_api':'responses',
                'requires_openai_auth':False, 'supports_websockets':False,
                'request_max_retries':0, 'stream_max_retries':0}})
        else:
            # Normal provider keeps Codex-managed ChatGPT credentials inside Codex.
            # Built-in providers cannot be redefined. Reported transport errors
            # hold and terminate this process; never restart the policy context.
            options.update({'model_provider':'openai'})
        cmd = [c['binary'], '--no-daemon', 'app-server', '--stdio']
        for key, value in options.items():
            encoded = ('{'+','.join(k+'='+json.dumps(v) for k,v in value.items())+'}'
                       if isinstance(value, dict) else json.dumps(value))
            cmd += ['-c', key+'='+encoded]
        env = {k:v for k,v in os.environ.items() if k in
               ('HOME','PATH','USER','LOGNAME','TMPDIR','LANG','LC_ALL','SYSTEMROOT','WINDIR')}
        self.stderr = (self.root/'stderr').open('x')
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, env=env)
        def read():
            for line in self.proc.stdout: self.queue.put(line)
        Thread(target=read, daemon=True).start()
        self._rpc('initialize', {'clientInfo':{'name':'remag_local_research_policy','version':'1'},
            'capabilities':{'experimentalApi':True}}, deadline)
        self._send({'method':'initialized','params':{}})
        if not self.local_fixture:
            auth = self._rpc('account/read', {'refreshToken':False}, deadline)
            auth_type=(auth.get('account') or {}).get('type')
            self.events.clear()
            if auth_type != 'chatgpt':
                raise ValueError('Existing Codex ChatGPT login required; no API fallback')
        # Account details are not retained or included in model context.
        self.events.clear()

    def decide(self, context, config_path, folder):
        folder = Path(folder); receipt = folder/'codex.json'
        if self.held: raise ValueError('Codex context held; no retry')
        if Path(config_path).resolve() != self.config_path or sha256(self.config_path) != self.config_sha:
            raise ValueError('Codex configuration changed')
        config_value(self.config)  # Recheck binary bytes before every invocation.
        if receipt.exists(): raise ValueError('Codex call receipt already exists')
        record = dict(format='REMAG-CODEX-CALL/1', status='intent', backend=self.kind,
            config_sha256=self.config_sha, binary_sha256=self.config['binary_sha256'],
            context_sha256=sha256(folder/'context.json'),
            requested_model=self.config['model'], reasoning_effort='xhigh',
            time=datetime.now(timezone.utc).isoformat(), automatic_context_retries=0,
            transport_policy='Terminate on reported errors; built-in provider internals are not overridden',
            output_token_ceiling_enforced=False, invoices_usd=None)
        save(receipt, record)
        deadline = time.monotonic()+min(self.config['request_timeout_seconds'], context['seconds_remaining'])
        try:
            if self.proc is None: self._start(deadline)
            self.events.clear()
            t = self._rpc('thread/start', dict(model=self.config['model'],
                modelProvider='local_fixture' if self.local_fixture else 'openai',
                allowProviderModelFallback=False, ephemeral=True, cwd=str(self.root),
                runtimeWorkspaceRoots=[str(self.root)], sandbox='read-only', approvalPolicy='never',
                baseInstructions=model_policy.INSTRUCTIONS,
                developerInstructions='Return one JSON object with tool, arguments_json and rationale. Native Codex tools are unavailable; only the external broker can execute research actions.',
                dynamicTools=[], environments=[], selectedCapabilityRoots=[],
                config=controls(self.config), experimentalRawEvents=True), deadline)
            if t['model'] != self.config['model'] or t['reasoningEffort'] != 'xhigh':
                raise ValueError('Codex changed requested model or reasoning effort')
            prompt = json.dumps(dict(research_brief=self.config['research_brief'],context=context),allow_nan=False)
            if len(prompt.encode())>model_policy.MAX_BYTES:raise ValueError('Codex context exceeds 1 MiB')
            schema = dict(type='object', properties=dict(
                tool=dict(type='string',enum=sorted(context['tools'])),
                arguments_json=dict(type='string'),rationale=dict(type='string')),
                required=['tool','arguments_json','rationale'],additionalProperties=False)
            turn = self._rpc('turn/start', dict(threadId=t['thread']['id'],effort='xhigh',
                environments=[],input=[dict(type='text',text=prompt)],outputSchema=schema), deadline)
            texts=[];usage=None
            while True:
                event=self._next(deadline)
                if not event: continue
                method=event.get('method');params=event.get('params',{})
                if method=='item/completed' and params.get('item',{}).get('type')=='agentMessage':
                    texts.append(params['item']['text'])
                if method=='thread/tokenUsage/updated':usage=params.get('tokenUsage')
                if method=='turn/completed':
                    if params['turn']['id'] != turn['turn']['id'] or params['turn']['status']!='completed':
                        raise ValueError('Codex turn did not complete')
                    break
            if len(texts)!=1: raise ValueError('Expected one Codex structured decision')
            decision=decode(texts[0],context['tools'])
            save(folder/'codex-decision.json',decision)
            record.update(status='completed',decision_sha256=sha256(folder/'codex-decision.json'),
                thread_id=t['thread']['id'],turn_id=turn['turn']['id'],token_usage=usage)
            return decision
        except BaseException as exc:
            self.held=True;record.update(status='held_error',error_type=type(exc).__name__)
            self.close();raise
        finally:
            save(folder/'codex-events.json',self.events)
            record['events_sha256']=sha256(folder/'codex-events.json');save(receipt,record)

    def audit(self, private):
        rows=[]
        for p in sorted(Path(private).glob('[0-9][0-9][0-9]/codex.json')):
            r=load(p)
            if (r['status']!='completed' or r['backend']!=self.kind or r['config_sha256']!=self.config_sha
                or sha256(p.parent/'codex-events.json')!=r['events_sha256']
                or sha256(p.parent/'context.json')!=r['context_sha256']
                or sha256(p.parent/'codex-decision.json')!=r['decision_sha256']):
                raise ValueError('Codex policy receipt changed or unresolved')
            rows.append(r)
        return dict(backend=self.kind,attempted_calls=len(rows),calls=rows,
            output_token_ceiling_enforced=False,actual_billed_usd=None,
            scope='Existing Codex quota; matched action/time limits, measured reported usage, not a claim of equal compute or API billing.')

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        if self.stderr:self.stderr.close()
