"""Loopback-only read-only viewer over exact authored structures and saved results."""
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import numpy as np

from ..discovery.common import load
from ..corpus import sha256


def electronic_preview(folder):
    accepted = []
    for path in sorted(folder.glob('electronic-z*/electronic.json')):
        candidate = load(path)
        residual = candidate.get('native_max_abs_error_ev')
        if candidate.get('native_comparison_passed') is True and isinstance(residual, (int, float)) and 0 <= residual <= 1e-3:
            accepted.append((path, candidate))
    if not accepted: return None
    meta_path, meta = accepted[-1]
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


@lru_cache(maxsize=1)
def authored_family():
    from gpac.physical import fept_reference, strain, export_geometry, coordination
    parent = fept_reference()
    rows = []
    for q in ['-1/50', '-1/100', '-1/200', '0', '1/200', '1/100', '1/50']:
        change = strain(parent, 2, q, expected_sha256=parent.sha256)
        export = export_geometry(change['child']['program'], length_unit='angstrom')
        from fractions import Fraction
        rows.append(dict(strain=float(Fraction(q)), exact_strain=q, structure=export,
                         coordination=coordination(export['geometry'], cutoff_angstrom=3.4)))
    return rows


def dataset(workspace):
    rows = [dict(r, result=None) for r in authored_family()]
    # Display only reports bound to the exact source represented by each row.
    for path in sorted(Path(workspace).rglob('report.json')):
        report = load(path)
        if report.get('format') != 'REMAG-EXPERIMENT-RESULT/1': continue
        for row in rows:
            if report.get('source_sha256') == row['structure']['source']['sha256']:
                row.setdefault('calculations', []).append(dict(path=str(path.relative_to(workspace)), status=report['status'],
                    settings=report['request']['settings'], kmesh=report['request']['kmesh'],
                    basis_size=report['request']['basis_size'], anisotropy=report.get('anisotropy'),
                    basis_energy_shift_ry=report['request']['basis_energy_shift_ry'],
                    pseudopotentials=report['request']['pseudopotentials'],
                    stages=report['stages'], observations=report['observations'], executable_sha256=report['executable_sha256'],
                    electronic=electronic_preview(path.parent)))
                row['result'] = row['calculations'][-1]
    return dict(format='REMAG-WORKSPACE/1', material='FePt', rows=rows)


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
