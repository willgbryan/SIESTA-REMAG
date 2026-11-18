"""Server-side client and workspace receipts for Newfound's synthesis tools."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256 as hash_bytes
from http.client import HTTPSConnection, HTTPException
from io import BytesIO
import json
from math import gcd
from functools import reduce
import os
from pathlib import Path
import re

from ..corpus import sha256
from ..discovery.common import digest, load, save
from ..discovery.pipeline import lock
from .catalog import load_catalog
from .tools import ResearchTools, _verified_report

HOST = 'mcp.newfoundmaterials.com'
ENDPOINT = 'https://' + HOST + '/mcp'
VERSION = '2025-03-26'
MAX_REPLY = 4 * 1024 * 1024


class NewfoundError(RuntimeError):
    pass


class Newfound:
    def __init__(self, key):
        if not key or any(ch.isspace() for ch in key):
            raise NewfoundError('Set NEWFOUND_API_KEY in the IDE server environment.')
        self.headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream'}
        self.sequence = 0
        result = self.rpc('initialize', {'protocolVersion': VERSION, 'capabilities': {},
            'clientInfo': {'name': 'remag-research', 'version': '1'}})
        if result.get('protocolVersion') != VERSION or 'tools' not in result.get('capabilities', {}):
            raise NewfoundError('Newfound returned an incompatible MCP handshake.')
        self.headers['MCP-Protocol-Version'] = VERSION
        self.rpc('notifications/initialized', {}, notification=True)

    def rpc(self, method, params, notification=False):
        msg = {'jsonrpc': '2.0', 'method': method, 'params': params}
        if not notification:
            self.sequence += 1
            msg['id'] = self.sequence
        conn = HTTPSConnection(HOST, timeout=210)
        try:
            conn.request('POST', '/mcp', body=json.dumps(msg).encode(), headers=self.headers)
            response = conn.getresponse()
            if response.status not in (200, 202, 204):
                raise NewfoundError(f'Newfound returned HTTP {response.status}.')
            session = response.getheader('mcp-session-id')
            if session:
                self.headers['Mcp-Session-Id'] = session
            if notification:
                return
            if response.getheader('Content-Type','').split(';')[0].strip() == 'text/event-stream':
                size = 0
                parts = []
                for line in response:
                    size += len(line)
                    if size > MAX_REPLY:
                        break
                    if line.startswith(b'data:'):
                        parts.append(line[5:].strip())
                    elif not line.strip() and parts:
                        reply = json.loads(b'\n'.join(parts))
                        if reply.get('id') == msg['id']:
                            break
                        parts.clear()
                else:
                    raise NewfoundError('Newfound closed the stream without a result.')
                if size > MAX_REPLY:
                    raise NewfoundError('Newfound reply exceeded the size limit.')
            else:
                body = response.read(MAX_REPLY + 1)
                if len(body) > MAX_REPLY:
                    raise NewfoundError('Newfound reply exceeded the size limit.')
                reply = json.loads(body)
            if reply.get('id') != msg['id'] or 'result' not in reply:
                raise NewfoundError('Newfound rejected the MCP request.')
            return reply['result']
        except (OSError, HTTPException, ValueError) as exc:
            raise NewfoundError(f'Newfound connection or response failed ({type(exc).__name__}).') from None
        finally:
            conn.close()

    def call(self, tool, arguments):
        result = self.rpc('tools/call', {'name': tool, 'arguments': arguments})
        texts = [block['text'] for block in result.get('content', []) if block.get('type') == 'text']
        if result.get('isError'):
            # Service descriptions can contain echoed inputs. Never serialize our headers.
            message = ' '.join(texts).replace(self.headers['Authorization'][7:], '[redacted]')
            raise NewfoundError(message[:600] or 'Newfound could not complete this tool.')
        value = result.get('structuredContent')
        if value is None and texts:
            value = json.loads(texts[0])
        if not isinstance(value, dict):
            raise NewfoundError('Expected a structured Newfound result.')
        return value


def candidate_input(root, identifier, source):
    """Prefer the finest verified native relaxed geometry; otherwise use the exact authoring."""
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import write
    from gpac.physical import verify_export
    resolver = ResearchTools(root)
    entry = next((e for e in load_catalog(root)['structures'] if e['id'] == identifier), None)
    if not entry:
        raise ValueError('Unknown workspace candidate.')
    export = verify_export(load(resolver.path(entry['structure'])))
    if export['source']['sha256'] != source:
        raise ValueError('The selected candidate source has changed.')
    geometry = export['geometry']
    evidence = {'origin': 'authored unit cell'}
    measured = []
    for registration in entry.get('calculations', []):
        path = resolver.path(registration['report'])
        r = load(path)
        if (r.get('status') == 'measured' and r.get('geometry_origin') == 'accepted native variable-cell relaxation'
                and r.get('source_sha256') == source and 'evaluated_geometry' in r):
            measured.append((r['request']['settings']['k_spacing_inv_ang'], str(path), registration))
    if measured:
        spacing, path, registration = min(measured, key=lambda item: item[:2])
        report = _verified_report(Path(path))
        geometry = report['evaluated_geometry']
        evidence = {'origin': 'native relaxed cell', 'report': registration['report'], 'report_sha256': sha256(Path(path)),
                    'k_spacing_inv_ang': spacing}
    numbers = geometry['atomic_numbers']
    if not 1 <= len(numbers) <= 300:
        raise ValueError('Newfound supports structures with 1–300 sites.')
    counts = Counter(numbers)
    factor = reduce(gcd, counts.values())
    formula = ''.join(chemical_symbols[z] + (str(counts[z]//factor) if counts[z]//factor!=1 else '') for z in sorted(counts))
    atoms = Atoms(numbers=numbers, cell=geometry['cell'], scaled_positions=geometry['fractional'], pbc=True)
    buf = BytesIO()
    write(buf, atoms, format='cif')
    cif = buf.getvalue()
    binding = dict(candidate=identifier, source_sha256=source, formula=formula, sites=len(atoms),
                   cif_sha256=hash_bytes(cif).hexdigest(), **evidence)
    return binding, cif.decode('latin1')


def utc():
    return datetime.now(timezone.utc).isoformat()


def receipt_path(root, tool, arguments):
    return Path(root)/'synthesis'/'newfound'/'calls'/(digest([ENDPOINT,tool,arguments])+'.json')


def cached_call(root, tool, arguments, client, *, refresh=False):
    path = receipt_path(root, tool, arguments)
    with lock(path.with_suffix('.lock')):
        if path.exists():
            record = load(path)
            if not refresh or record['status'] != 'returned':
                return record
        else:
            record = dict(provider=ENDPOINT, tool=tool, arguments=arguments, started_at=utc(), status='submitted')
            save(path,record)
        try:
            result = client.call(tool, arguments)
        except (NewfoundError, ValueError) as exc:
            record.update(status='error', error=str(exc), updated_at=utc())
        else:
            record.update(status='returned', result=result, updated_at=utc())
        save(path,record)
        return record


def selection_path(root, identifier, source):
    return Path(root)/'synthesis'/'newfound'/'candidates'/(digest([identifier,source])+'.json')


def prepare_selection(root, identifier, source):
    binding, cif = candidate_input(root, identifier, source)
    formula = binding['formula']
    calls = {'structure': ('relax_structure', {'cif': cif}),
             'accessibility': ('synthetic_accessibility', {'formula': formula, 'atmosphere':'closed','temperatures_c':[500,1000],'wait_seconds':120}),
             'recipes': ('synthesis_recipes', {'formula': formula, 'atmosphere':'closed','limit':5,'wait_seconds':120})}
    selection = dict(binding=binding, provider=ENDPOINT, calls={k:receipt_path(root,*v).name for k,v in calls.items()})
    path = selection_path(root, identifier, source)
    with lock(path.with_suffix('.lock')):
        save(path,selection)
    return selection, calls


def unfinished_jobs(result):
    rows = result.get('runs', result.get('by_temperature', []))
    return [row['job_id'] for row in rows if row.get('job_id') and row.get('status') not in
            ('completed', 'failed', 'cancelled', 'canceled', 'error')]


def view(root, identifier, source):
    if not isinstance(identifier,str) or not 1<=len(identifier)<=100 or not isinstance(source,str) or not re.fullmatch('[0-9a-f]{64}',source):
        raise ValueError('Select a registered structure with its source hash.')
    path = selection_path(root,identifier,source)
    if not path.exists():
        return {'status':'empty','candidate':identifier}
    selection = load(path)
    binding = selection['binding']
    if binding['candidate'] != identifier or binding['source_sha256'] != source:
        raise ValueError('Synthesis evidence belongs to another candidate.')
    steps = {}
    for k,filename in selection['calls'].items():
        if not re.fullmatch(r'[0-9a-f]{64}\.json',filename):
            raise ValueError('Invalid synthesis receipt.')
        p=path.parent.parent/'calls'/filename
        if p.exists():
            record=load(p)
            if receipt_path(root,record['tool'],record['arguments']) != p:
                raise ValueError('Synthesis receipt differs from its request.')
            args = record['arguments']
            if k == 'structure':
                matches = record['tool'] == 'relax_structure' and hash_bytes(args['cif'].encode('latin1')).hexdigest() == binding['cif_sha256']
            else:
                matches = record['tool'] == {'accessibility':'synthetic_accessibility', 'recipes':'synthesis_recipes'}.get(k) and args['formula'] == binding['formula']
            if not matches:
                raise ValueError('Newfound input belongs to another candidate or composition.')
            steps[k]={field:record[field] for field in ('tool','status','updated_at','error','result') if field in record}
    pending = any(step['status']=='returned' and unfinished_jobs(step.get('result', {})) for step in steps.values())
    return dict(status='saved', binding=binding, steps=steps, jobs_pending=pending, missing_steps=len(steps)<3)


def screen(root, identifier, source):
    """One explicit external assessment. Viewing a tab only reads local receipts."""
    with lock(Path(root)/'synthesis'/'newfound'/'.screen.lock'):
        _, calls = prepare_selection(root, identifier, source)
        saved = view(root, identifier, source)
        if not saved['missing_steps'] and not saved['jobs_pending']:
            return saved
        client = Newfound(os.environ.get('NEWFOUND_API_KEY'))
        for tool, arguments in calls.values():
            record = cached_call(root, tool, arguments, client)
            jobs = unfinished_jobs(record.get('result', {})) if record['status']=='returned' else []
            if jobs:
                # Re-query the identical formula request only after its existing jobs finish.
                states = [client.call('job_status', {'job_id': job}) for job in jobs]
                if all(state.get('status') == 'completed' for state in states):
                    cached_call(root, tool, arguments, client, refresh=True)
                elif any(state.get('status') in ('failed','error','cancelled','canceled') for state in states):
                    path = receipt_path(root, tool, arguments)
                    record.update(status='error', error='Newfound reported a failed or canceled recipe job.',
                                  job_statuses=states, updated_at=utc())
                    save(path, record)
        return view(root,identifier,source)
