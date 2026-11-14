"""Loopback-only read-only viewer over exact authored structures and saved results."""
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import numpy as np

from ..discovery.common import load, digest
from ..corpus import sha256
from .convergence import numerical_configuration


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
    return digest(config)


def dataset(workspace):
    workspace = Path(workspace)
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
                    pseudopotentials=report['request']['pseudopotentials'],
                    stages=report['stages'],
                    comparison_key=comparison_key(report),
                    observations={stage: {key: value for key, value in observation.items() if key != 'eigenvalues'}
                                  for stage, observation in report['observations'].items()},
                    executable_sha256=report['executable_sha256'],
                    electronic=electronic_preview(path.parent,
                        expected_hsx_sha256=report.get('artifacts', {}).get('soc-z/remag.HSX')))
        for row in matching:
            row.setdefault('calculations', []).append(calculation)
            row['result'] = calculation
    # Keep the original rows field for saved-state consumers; new views add families.
    return dict(format='REMAG-WORKSPACE/1', material='FePt', rows=families[0]['rows'], families=families)


def serve(workspace, port=8765):
    workspace = Path(workspace).resolve()
    if not workspace.is_dir(): raise ValueError('Workspace directory must exist')
    authored_family()  # initialize before announcing a ready local URL
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                body, kind = Path(__file__).with_name('workspace.html').read_bytes(), 'text/html; charset=utf-8'
            elif self.path == '/api/state':
                try:
                    body = json.dumps(dataset(workspace), allow_nan=False).encode()
                except Exception as exc:
                    self.send_error(500, type(exc).__name__); return
                kind = 'application/json'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Local: http://127.0.0.1:{server.server_port}', flush=True)
    try: server.serve_forever()
    finally: server.server_close()


def snapshot(workspace, output):
    """Self-contained read-only HTML, with the same source/result binding as serve."""
    state = dataset(Path(workspace).resolve())
    payload = json.dumps(state, allow_nan=False).replace('<', '\\u003c')
    html = Path(__file__).with_name('workspace.html').read_text()
    html = html.replace('<script>', '<script type="application/json" id="saved-state">'+payload+'</script>\n<script>', 1)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        stream.write(html)
    return dict(format='REMAG-WORKSPACE-SNAPSHOT/1', path=str(target), sha256=sha256(target))
