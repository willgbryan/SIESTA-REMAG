"""Loopback viewer with optional static authoring; viewing never runs a solver."""
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
import numpy as np

from ..discovery.common import load, digest
from ..corpus import sha256
from .convergence import numerical_configuration
from .activity import activity_state


def electronic_preview(folder, *, expected_hsx_sha256=None):
    accepted = []
    for path in sorted(folder.glob('electronic-z*/electronic.json')):
        candidate = load(path)
        residual = candidate.get('native_max_abs_error_ev')
        if candidate.get('native_comparison_passed') is True and isinstance(residual, (int, float)) and 0 <= residual <= 1e-3:
            accepted.append((path, candidate))
    if not accepted: return None
    meta_path, meta = accepted[-1]
    if expected_hsx_sha256 is not None and meta.get('hsx_sha256') != expected_hsx_sha256:
        raise ValueError('Electronic Hamiltonian binding differs')
    artifact = meta_path.parent/meta['artifact']
    if not artifact.resolve().is_relative_to(meta_path.parent.resolve()) or sha256(artifact) != meta['artifact_sha256']:
        raise ValueError('Electronic artifact binding differs')
    with np.load(artifact, allow_pickle=False) as a:
        energies = a['eigenvalues_ev']-meta['fermi_ev']
        weights, populations = a['k_weights'], a['orbital_mulliken']
        grid = np.linspace(-3, 3, 161)
        sigma = .12
        kernel = np.exp(-.5*((grid[:, None, None]-energies[None, :, :])/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        groups = []
        for atom in sorted({o['atom_index'] for o in meta['orbitals']}):
            indices = [o['index'] for o in meta['orbitals'] if o['atom_index'] == atom and o['l'] == 2]
            values = np.einsum('gkn,kn,k->g', kernel, populations[:, indices, :].sum(axis=1), weights)
            groups.append(dict(atom_index=atom, l=2, density=values.tolist()))
    return dict(energy_ev=grid.tolist(), groups=groups, gaussian_sigma_ev=sigma,
                native_max_abs_error_ev=meta['native_max_abs_error_ev'], projection='Mulliken',
                sampling=meta.get('sampling', 'supplied k points'))


@lru_cache(maxsize=3)
def authored_family(kind='axial'):
    from gpac.physical import (fept_reference, strain, export_geometry, coordination,
                               isotropic_scale, volume_preserving_tetragonal)
    from fractions import Fraction
    if kind not in ('axial', 'volume', 'shape'):
        raise ValueError('Unknown structural control')
    parent = fept_reference()
    rows = []
    parameters = (['-1/50', '-1/100', '-1/200', '-1/400', '0', '1/400', '1/200', '1/100', '1/50']
                  if kind == 'axial' else ['-1/400', '0', '1/400'])
    for q in parameters:
        if kind == 'axial':
            change = strain(parent, 2, q, expected_sha256=parent.sha256)
        else:
            operation = isotropic_scale if kind == 'volume' else volume_preserving_tetragonal
            change = operation(parent, str(1 + Fraction(q)), expected_sha256=parent.sha256)
        export = export_geometry(change['child']['program'], length_unit='angstrom')
        rows.append(dict(strain=float(Fraction(q)), exact_strain=q, structure=export,
                         coordination=coordination(export['geometry'], cutoff_angstrom=3.4)))
    return rows


def comparison_key(report):
    """Physical controls and runtime match; execution ceilings are not observables."""
    config = numerical_configuration(report)
    del config['source_sha256']
    if report.get('format') == 'REMAG-DISCOVERY-OBSERVATION/1':
        config.update(geometry_origin=report['geometry_origin'], evaluated_geometry=report['evaluated_geometry'])
    return digest(config)


def dataset(workspace, *, include_sampling=True):
    workspace = Path(workspace)
    if (workspace/'workspace.json').exists():
        from .catalog import registered_dataset
        return registered_dataset(workspace,include_sampling=include_sampling)
    families = [dict(id=kind, label=label, hint=hint,
                     rows=[dict(r, result=None) for r in authored_family(kind)])
                for kind, label, hint in [('axial', 'c strain', 'Fixed a, b'),
                                         ('volume', 'Volume', 'Uniform scale'),
                                         ('shape', 'Shape', 'In-plane scale · fixed volume')]]
    rows = [row for family in families for row in family['rows']]
    # Display only reports bound to the exact source represented by each row.
    for path in sorted(Path(workspace).rglob('report.json')):
        report = load(path)
        if report.get('format') != 'REMAG-EXPERIMENT-RESULT/1': continue
        matching = [row for row in rows if report.get('source_sha256') == row['structure']['source']['sha256']]
        if not matching: continue
        calculation = dict(path=str(path.relative_to(workspace)), status=report['status'],
                    settings=report['request']['settings'], kmesh=report['request']['kmesh'],
                    basis_size=report['request']['basis_size'], anisotropy=report.get('anisotropy'),
                    basis_energy_shift_ry=report['request']['basis_energy_shift_ry'],
                    basis_spec=report['request'].get('basis_spec'),
                    pseudopotentials=report['request']['pseudopotentials'],
                    stages=report['stages'],
                    comparison_key=comparison_key(report),
                    observations={stage: {key: value for key, value in observation.items() if key != 'eigenvalues'}
                                  for stage, observation in report['observations'].items()},
                    executable_sha256=report['executable_sha256'],
                    electronic=electronic_preview(path.parent,
                        expected_hsx_sha256=report.get('artifacts', {}).get('soc-z/remag.HSX')))
        if calculation['basis_spec'] is not None:
            from .basis import orbital_count
            from ase.data import chemical_symbols
            calculation['basis_orbitals'] = orbital_count(calculation['basis_spec'],
                [chemical_symbols[z] for z in report['structure']['geometry']['atomic_numbers']])
        for row in matching:
            row.setdefault('calculations', []).append(calculation)
            row['result'] = calculation
    # Keep the original rows field for saved-state consumers; new views add families.
    from ase.data import chemical_symbols
    return dict(format='REMAG-WORKSPACE/1', material='FePt', rows=families[0]['rows'], families=families,
                element_symbols=chemical_symbols)


def browser_state(state):
    """Serialize large sampled arrays once despite retained legacy view aliases."""
    arrays={};seen={}
    def visit(value):
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                if key=='sampled' and isinstance(item,dict) and item.get('available') is True:
                    identity=id(item)
                    if identity not in seen:
                        name='sample-'+str(len(arrays));seen[identity]=name;arrays[name]=item
                    result[key]={'ref':seen[identity]}
                else:result[key]=visit(item)
            return result
        if isinstance(value,list):return [visit(v) for v in value]
        return value
    result=visit(state)
    if arrays:result['sampled_arrays']=arrays
    return result


def make_server(workspace, port=8765, *, editable=False):
    workspace = Path(workspace).resolve()
    if not workspace.is_dir(): raise ValueError('Workspace directory must exist')
    if editable and not (workspace/'workspace.json').exists():
        raise ValueError('Editable mode requires an explicitly registered workspace')
    if not (workspace/'workspace.json').exists():authored_family()
    class Handler(BaseHTTPRequestHandler):
        def local_host(self):
            if self.headers.get('Host') not in self.server.allowed_hosts:
                self.send_error(403,'Local workspace host required');return False
            return True

        def send_body(self,body,kind='application/json',status=200):
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if not self.local_host():return
            if self.path in ('/', '/ide'):
                body, kind = Path(__file__).with_name('workspace.html').read_bytes(), 'text/html; charset=utf-8'
            elif self.path == '/api/state':
                try:
                    body = json.dumps(browser_state(dataset(workspace,include_sampling=False)) | {'authoring_enabled':editable,
                        'synthesis_enabled':editable and bool(os.environ.get('NEWFOUND_API_KEY')),
                        'activity_enabled':(workspace/'research-activity.json').exists()}, allow_nan=False).encode()
                except Exception as exc:
                    self.send_error(500, type(exc).__name__); return
                kind = 'application/json'
            elif urlsplit(self.path).path == '/api/synthesis':
                from .newfound import view
                try:
                    query = parse_qs(urlsplit(self.path).query, strict_parsing=True)
                    if set(query) != {'candidate', 'source'} or any(len(v) != 1 for v in query.values()):
                        raise ValueError('Expected a candidate and source.')
                    body = json.dumps(view(workspace, query['candidate'][0], query['source'][0]), allow_nan=False).encode()
                except (ValueError, OSError) as exc:
                    self.send_error(400, type(exc).__name__); return
                kind = 'application/json'
            elif urlsplit(self.path).path == '/api/sampling':
                try:
                    from .catalog import registered_sampling
                    parsed=parse_qs(urlsplit(self.path).query,strict_parsing=True)
                    if any(len(v)!=1 for v in parsed.values()):raise ValueError('Duplicate sampling selection')
                    body=json.dumps(registered_sampling(workspace,{k:v[0] for k,v in parsed.items()}),allow_nan=False).encode()
                except Exception as exc:
                    self.send_error(400,type(exc).__name__);return
                kind='application/json'
            elif self.path == '/api/activity':
                try:
                    body = json.dumps(activity_state(workspace), allow_nan=False).encode()
                except Exception as exc:
                    self.send_error(503, type(exc).__name__); return
                kind = 'application/json'
            else:
                self.send_error(404); return
            self.send_body(body,kind)

        def do_POST(self):
            if not self.local_host():return
            if self.path not in ('/api/author','/api/prepare','/api/source/preview','/api/source/save','/api/synthesis'):self.send_error(404);return
            if not editable:self.send_error(403,'Workspace is read-only');return
            if self.headers.get('Origin')!='http://'+self.headers.get('Host',''):
                self.send_error(403,'Same-origin workspace action required');return
            length=self.headers.get('Content-Length','')
            if (self.headers.get('Content-Type','').split(';')[0].strip()!='application/json'
                or not 1<=len(length)<=5 or not length.isdecimal() or not 0<int(length)<=65536 or self.headers.get('Transfer-Encoding')):
                self.send_error(400,'Expected bounded JSON request');return
            try:
                from .model_policy import loads
                from .authoring import save_workspace_candidate, workspace_source
                from .workspace_actions import prepare_workspace_research
                self.connection.settimeout(10)
                payload=loads(self.rfile.read(int(length)))
                if self.path == '/api/synthesis':
                    from .newfound import screen
                    if not isinstance(payload, dict) or set(payload) != {'candidate', 'source'}:
                        raise ValueError('Expected a candidate and source.')
                    result = screen(workspace, payload['candidate'], payload['source'])
                elif self.path.startswith('/api/source/'):
                    result=workspace_source(workspace,payload,save_source=self.path.endswith('/save'))
                else:
                    action=save_workspace_candidate if self.path=='/api/author' else prepare_workspace_research
                    result=action(workspace,payload)
                self.send_body(json.dumps(dict(ok=True,result=result),allow_nan=False).encode(),status=201)
            except Exception as exc:
                self.send_body(json.dumps(dict(ok=False,error=dict(type=type(exc).__name__,message=str(exc))),allow_nan=False).encode(),status=400)

        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.allowed_hosts={'127.0.0.1:'+str(server.server_port),'localhost:'+str(server.server_port)}
    return server


def serve(workspace, port=8765, *, editable=False):
    server=make_server(workspace,port,editable=editable)
    print(f'Local: http://127.0.0.1:{server.server_port}', flush=True)
    try: server.serve_forever()
    finally: server.server_close()


def snapshot(workspace, output):
    """Self-contained read-only HTML, with the same source/result binding as serve."""
    state = dataset(Path(workspace).resolve())
    state['activity'] = activity_state(workspace)
    state['activity_enabled'] = state['activity']['available']
    payload = json.dumps(browser_state(state), allow_nan=False).replace('<', '\\u003c')
    html = Path(__file__).with_name('workspace.html').read_text()
    html = html.replace('<script>', '<script type="application/json" id="saved-state">'+payload+'</script>\n<script>', 1)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        stream.write(html)
    return dict(format='REMAG-WORKSPACE-SNAPSHOT/1', path=str(target), sha256=sha256(target))
