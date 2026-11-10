#!/usr/bin/env python3
"""CPU precision experiment on existing selected-k complex H/S captures.

This tests arithmetic choices, not Metal performance, full SCF, or MAE.
Requires the project's dev dependencies. Run from the repository root:
  .venv/bin/python benchmarks/evidence/apple-mixed-precision-20261109/replay.py \
    results/mae-soc-free-scalar-20261026T210526Z --output <report.json>
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.special import expit

RY_EV = 13.605693122994
SPECTRUM_EV = 1e-7  # Existing independent-matrix gate, NOT a MAE tolerance.
RESIDUAL = 1e-10  # Additional component diagnostics for this experiment.
ORTHOGONALITY = 1e-10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(a):
    b = np.tril(a, -1)
    return b + b.conj().T + np.diag(a.diagonal().real)


def hermitian(a):
    return (a + a.conj().T) * 0.5


def orthogonalize(v, s):
    c = cholesky(hermitian(v.conj().T @ s @ v), lower=True)
    return solve_triangular(c, v.conj().T, lower=True).conj().T


def rayleigh(v, h, s):
    return (np.sum(v.conj() * (h @ v), axis=0).real
            / np.sum(v.conj() * (s @ v), axis=0).real)


def metrics(w, v, h, s, reference, density_context=None):
    w, v = np.asarray(w, dtype=np.float64), np.asarray(v, dtype=np.complex128)
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    residuals = np.linalg.norm(h @ v - (s @ v) * w, axis=0)
    scale = ((np.linalg.norm(h, 'fro') + abs(w)*np.linalg.norm(s, 'fro'))
             * np.linalg.norm(v, axis=0))
    error = float(np.max(abs(w-reference)) * RY_EV)
    residual = float(np.max(residuals / scale))
    orthogonality = float(np.max(abs(v.conj().T @ s @ v - np.eye(len(w)))))
    report = dict(spectrum_max_abs_error_ev=error,
                max_relative_generalized_residual=residual,
                max_abs_s_orthogonality_error=orthogonality,
                component_gates_passed=bool(error <= SPECTRUM_EV
                    and residual <= RESIDUAL and orthogonality <= ORTHOGONALITY))
    if density_context is not None:
        l, mu, kt, reference_density = density_context
        # Fixed source chemical potential, no full-mesh charge re-solve.
        # Compare density operators, not individual near-degenerate vectors.
        q = l.conj().T @ v
        f = expit((mu-w)/kt)
        density = (q*f) @ q.conj().T
        occupied = expit((mu-reference)/kt) >= 1e-8
        report['occupied_spectrum_max_abs_error_ev'] = float(np.max(abs(w[occupied]-reference[occupied]))*RY_EV)
        report['fixed_mu_density_relative_frobenius_error'] = float(
            np.linalg.norm(density-reference_density, 'fro')/np.linalg.norm(reference_density, 'fro'))
    return report


def refine(v, h, s, reference, density_context):
    """Bounded FP64 correction of an FP32 seed, with clusterwise rotation.

    Full-size Gram/projection products and Cholesky stay FP64 here. Only small
    near-degenerate diagonal blocks are re-diagonalized. This is an accuracy
    feasibility experiment, not a cheaper algorithm or a Metal implementation.
    """
    v = np.asarray(v, dtype=np.complex128)
    history = []
    largest_cluster = 1
    for step in range(5):
        v = orthogonalize(v, s)
        b = hermitian(v.conj().T @ h @ v)
        order = np.argsort(b.diagonal().real)
        v, b = v[:, order], b[np.ix_(order, order)]
        w = b.diagonal().real.copy()
        report = metrics(w, v, h, s, reference, density_context)
        history.append(dict(step=step, **report))
        # The reference spectrum is ONLY for measurement, not solver stopping.
        if (report['max_relative_generalized_residual'] <= RESIDUAL
                and report['max_abs_s_orthogonality_error'] <= ORTHOGONALITY):
            break
        if step == 4:
            break
        offdiag = b - np.diag(w)
        gap = max(1e-5, 16*float(np.max(abs(offdiag))))  # Ry, component experiment
        cuts = [0, *(np.flatnonzero(np.diff(w) > gap)+1).tolist(), len(w)]
        rotation = np.eye(len(w), dtype=np.complex128)
        labels = np.empty(len(w), dtype=int)
        for group, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            largest_cluster = max(largest_cluster, end-start)
            labels[start:end] = group
            if end-start > 1:
                _, rotation[start:end, start:end] = eigh(b[start:end, start:end])
        v = v @ rotation
        b = hermitian(v.conj().T @ h @ v)
        w = b.diagonal().real.copy()
        denominator = w[None, :] - w[:, None]
        different = labels[:, None] != labels[None, :]
        if np.any(abs(denominator[different]) < gap/2):
            raise ValueError('Cluster rotation invalidated the separation bound')
        correction = np.zeros_like(b)
        np.divide(b, denominator, out=correction, where=different)
        v = v @ (np.eye(len(w)) + correction)
    return dict(**report, correction_steps=len(history)-1,
                largest_cluster=largest_cluster, history=history)


def replay(path):
    meta = json.loads(path.read_text())
    n = meta['bands']
    if (meta['schema'] != 1 or meta['dtype'] != '<c16' or meta['order'] != 'F'
            or meta['energy_unit'] != 'Ry' or meta['reference'] != 'Final SCF generator Hamiltonian'
            or meta['matrices'] != ['generator', 'overlap', 'soc', 'rotated']
            or type(n) is not int or n not in (96, 112)
            or Path(meta['matrix_file']).name != meta['matrix_file']):
        raise ValueError('Unsupported capture')
    binary = path.parent / meta['matrix_file']
    if binary.stat().st_size != 4*n*n*16:
        raise ValueError('Truncated or oversized capture')
    matrices = np.fromfile(binary, dtype='<c16').reshape(4, n*n)
    if not np.isfinite(matrices).all():
        raise ValueError('Nonfinite capture')
    _, s, _, h = [lower(a.reshape(n, n, order='F')) for a in matrices]
    reference, vectors = eigh(h, s, driver='gvd')
    probe_path = path.parent/'remag-frozen-potential.json'
    probe = json.loads(probe_path.read_text())
    point = probe['points'][meta['direction']]
    if probe['status'] != 'complete' or not point['completed'] or point['status'] != 0:
        raise ValueError('Incomplete source calculation')
    mu, kt = point['fermi_ry'], probe['temperature_ry']
    if not np.isfinite([mu, kt]).all() or kt <= 0:
        raise ValueError('Invalid source chemical potential or temperature')
    l = cholesky(s, lower=True)
    q_reference = l.conj().T @ vectors
    reference_density = (q_reference*expit((mu-reference)/kt)) @ q_reference.conj().T
    context = (l, mu, kt, reference_density)
    result = dict(capture=str(path), metadata_sha256=digest(path), binary_sha256=digest(binary),
                  hs_sha256=hashlib.sha256(h.tobytes()+s.tobytes()).hexdigest(),
                  material='FePt' if n == 96 else 'FeCo', bands=n,
                  k=meta['k'], parent_mesh_kpoints=meta['kpoints'], direction=meta['direction'],
                  reference_soc_scale=meta['reference_soc_scale'],
                  probe_sha256=digest(probe_path), source_fermi_ry=mu, source_temperature_ry=kt,
                  overlap_condition_number=float(np.linalg.cond(s)), methods={})
    result['methods']['fp64_reference'] = metrics(reference, vectors, h, s, reference, context)
    h32, s32 = h.astype(np.complex64), s.astype(np.complex64)
    for name, a, b in [('rounded_hs_fp64_solver', h32.astype(np.complex128), s32.astype(np.complex128)),
                       ('fp32_generalized', h32, s32)]:
        try:
            w, v = eigh(a, b, driver='gvd')
            result['methods'][name] = dict(solver_output_dtype=str(v.dtype),
                                           **metrics(w, v, h, s, reference, context))
        except (ValueError, np.linalg.LinAlgError) as error:
            result['methods'][name] = dict(component_gates_passed=False, error=str(error))
    # Cacheable FP64 overlap factor; FP32 only in the standard eigensolve.
    a = solve_triangular(l, h, lower=True)
    a = hermitian(solve_triangular(l, a.conj().T, lower=True).conj().T)
    try:
        w, q = eigh(a.astype(np.complex64), driver='evd')
        v = solve_triangular(l.conj().T, q.astype(np.complex128), lower=False)
        result['methods']['fp64_overlap_fp32_standard'] = dict(
            solver_output_dtype=str(q.dtype), **metrics(w, v, h, s, reference, context))
        result['methods']['fp64_rayleigh_only'] = metrics(rayleigh(v, h, s), v, h, s, reference, context)
        result['methods']['fp64_corrected_fp32_seed'] = refine(v, h, s, reference, context)
    except (ValueError, np.linalg.LinAlgError) as error:
        result['methods']['fp64_corrected_fp32_seed'] = dict(component_gates_passed=False, error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    attempt = json.loads((args.run/'run-attempt.json').read_text())
    oracle_path = args.run/'local-independent-matrices.json'
    oracle = json.loads(oracle_path.read_text())
    paths = sorted(args.run.rglob('remag-frozen-matrices-*.json'))
    if (attempt['state'] != 'complete' or not oracle['passed'] or not paths
            or {str(p) for p in paths} != {r['capture'] for r in oracle['captures']}):
        raise ValueError('Completed, independently checked capture set required; use repository-relative run path')
    records = []
    for path in paths:
        try:
            records.append(replay(path))
        except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as error:
            records.append(dict(capture=str(path), input_error=str(error)))
    report = dict(scope='CPU selected-k arithmetic replay; no full-mesh E/F, SCF, Metal, or speedup qualification',
        source_run=str(args.run), source_attempt_id=attempt['attempt_id'],
        source_oracle_sha256=digest(oracle_path), script_sha256=digest(Path(__file__)),
        numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
        thresholds=dict(spectrum_error_ev=SPECTRUM_EV, relative_residual=RESIDUAL, s_orthogonality=ORTHOGONALITY),
        performance_seconds=None, speedup=None, mae_qualified=False, records=records, summary={})
    for material in ('FePt', 'FeCo'):
        rows = [r for r in records if r.get('material') == material]
        report['summary'][material] = dict(captures=len(rows), unique_hs_pairs=len({r['hs_sha256'] for r in rows}), methods={})
        names = sorted({name for r in rows for name in r['methods']})
        for name in names:
            samples = [r['methods'].get(name, {}) for r in rows]
            result = dict(passed=sum(bool(s.get('component_gates_passed')) for s in samples), total=len(rows))
            for key in ('spectrum_max_abs_error_ev', 'occupied_spectrum_max_abs_error_ev',
                        'fixed_mu_density_relative_frobenius_error',
                        'max_relative_generalized_residual', 'max_abs_s_orthogonality_error',
                        'correction_steps', 'largest_cluster'):
                values = [s[key] for s in samples if key in s]
                if values:
                    result[key] = max(values)
            report['summary'][material]['methods'][name] = result
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary'], indent=2))
    return int(any('input_error' in r for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
