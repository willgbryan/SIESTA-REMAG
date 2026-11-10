"""Load only the previously hashed, independently checked local capture set."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, eigh
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT/'benchmarks/evidence/apple-mixed-precision-20261109'
spec = importlib.util.spec_from_file_location('precision_replay', PREVIOUS/'replay.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def load():
    report = json.loads((PREVIOUS/'replay.json').read_text())
    cases = []
    for row in report['records']:
        path = ROOT/row['capture']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['metadata_sha256']:
            raise ValueError('Capture metadata changed')
        meta = json.loads(path.read_text())
        binary = path.parent/meta['matrix_file']
        if hashlib.sha256(binary.read_bytes()).hexdigest() != row['binary_sha256']:
            raise ValueError('Capture matrices changed')
        n = meta['bands']
        matrices = np.fromfile(binary, dtype='<c16').reshape(4, n*n)
        _, s, _, h = [oracle.lower(a.reshape(n, n, order='F')) for a in matrices]
        w, v = eigh(h, s, driver='gvd')
        l = cholesky(s, lower=True)
        q = l.conj().T@v
        mu, kt = row['source_fermi_ry'], row['source_temperature_ry']
        density = (q*expit((mu-w)/kt))@q.conj().T
        cases.append(dict(h=h, s=s, l=l, reference=w, density_context=(l, mu, kt, density),
                          provenance={key: row[key] for key in ('capture', 'binary_sha256', 'metadata_sha256',
                                                               'material', 'bands', 'k', 'direction', 'reference_soc_scale')}))
    return cases


def check(case, solution):
    w, v = solution
    return oracle.metrics(w, v, case['h'], case['s'], case['reference'], case['density_context'])
