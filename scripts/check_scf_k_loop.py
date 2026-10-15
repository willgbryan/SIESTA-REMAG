#!/usr/bin/env python3
"""Check the device k loop (REMAG_SCF_K_LOOP=1) of the device SCF cycle.

The composed-cycle check (check_scf_cycle.py) is run on sampled-k fixtures
with more k points than solver lanes, so the loop's waves include a partial
final wave. Every configuration must pass that check's CPU oracle. The cycle
is not bit-reproducible between processes (the mesh density is accumulated
with FP64 atomics), so the unrolled cycle is run twice to measure that noise;
the looped cycle must then agree with its unrolled twin (same solver settings)
in Hamiltonian, density, energy density, spectra, occupations and Fermi level
to within 100 times that noise, and never worse than 1e-11.

Usage: check_scf_k_loop.py --upstream SRC --build REFERENCE_BUILD --library
libremag_scf_cycle_check.so --output DIR [--k-points 5]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ('h', 'd', 'e', 'bank', 'q', 'fermi')
# (name, environment, unrolled twin). Twins share every solver setting but the loop.
CONFIGS = [
    ('unrolled', {}, None),
    ('unrolled-repeat', {}, None),
    ('loop-lanes1', {'REMAG_SCF_K_LOOP': '1'}, 'unrolled'),
    ('loop-lanes2', {'REMAG_SCF_K_LOOP': '1', 'REMAG_SCF_SOLVER_LANES': '2'}, 'unrolled'),
    ('loop-lanes4', {'REMAG_SCF_K_LOOP': '1', 'REMAG_SCF_SOLVER_LANES': '4'}, 'unrolled'),
    ('unrolled-cache-lanes2', {'REMAG_DEVICE_VECTOR_CACHE': '1', 'REMAG_SCF_SOLVER_LANES': '2'}, None),
    ('loop-cache-lanes2', {'REMAG_SCF_K_LOOP': '1', 'REMAG_DEVICE_VECTOR_CACHE': '1', 'REMAG_SCF_SOLVER_LANES': '2'},
     'unrolled-cache-lanes2'),
    ('unrolled-warm-lanes4', {'REMAG_DEVICE_VECTOR_CACHE': '1', 'REMAG_SCF_SOLVER_LANES': '4',
                              'REMAG_SCF_WARM_START': '1'}, None),
    ('loop-warm-lanes4', {'REMAG_SCF_K_LOOP': '1', 'REMAG_DEVICE_VECTOR_CACHE': '1', 'REMAG_SCF_SOLVER_LANES': '4',
                          'REMAG_SCF_WARM_START': '1'}, 'unrolled-warm-lanes4'),
    ('unrolled-jacobi32', {'REMAG_BLOCK_WIDTH': '32', 'REMAG_BLOCK_LOCAL_SOLVER': 'jacobi'}, None),
    ('loop-jacobi32-lanes2', {'REMAG_SCF_K_LOOP': '1', 'REMAG_SCF_SOLVER_LANES': '2', 'REMAG_BLOCK_WIDTH': '32',
                              'REMAG_BLOCK_LOCAL_SOLVER': 'jacobi'}, 'unrolled-jacobi32'),
    # A different eigensolver: checked against the CPU oracle only, with its
    # largest difference from the unrolled cycle reported.
    ('batched-lanes4', {'REMAG_SCF_K_LOOP': '1', 'REMAG_SCF_BATCHED_SOLVER': '1', 'REMAG_SCF_SOLVER_LANES': '4'}, None),
    ('batched-cache-lanes2', {'REMAG_SCF_K_LOOP': '1', 'REMAG_SCF_BATCHED_SOLVER': '1', 'REMAG_SCF_SOLVER_LANES': '2',
                              'REMAG_DEVICE_VECTOR_CACHE': '1'}, None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k-points', type=int, default=5)
    a = parser.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    check = [sys.executable, str(HERE/'check_scf_cycle.py'), '--upstream', str(a.upstream), '--build', str(a.build)]
    reference = out/'reference'
    subprocess.run([*check, '--output', str(reference/'reference.json'), '--k-points', str(a.k_points)], check=True)
    report = dict(k_points=a.k_points, configurations=[], comparisons=[])
    for name, extra, _ in CONFIGS:
        env = {**os.environ, **extra}
        for key in ('REMAG_SCF_K_LOOP', 'REMAG_SCF_SOLVER_LANES', 'REMAG_DEVICE_VECTOR_CACHE', 'REMAG_SCF_WARM_START',
                    'REMAG_BLOCK_WIDTH', 'REMAG_BLOCK_LOCAL_SOLVER', 'REMAG_SCF_BATCHED_SOLVER'):
            if key not in extra: env.pop(key, None)
        done = subprocess.run([*check, '--output', str(out/name/'report.json'), '--library', str(a.library.resolve()),
                               '--fixtures-from', str(reference)], env=env, capture_output=True, text=True)
        (out/f'{name}.log').write_text(done.stdout+done.stderr)
        result = json.loads((out/name/'report.json').read_text()) if (out/name/'report.json').exists() else {}
        nodes = [c.get('graph_nodes') for c in result.get('checks', [])]
        report['configurations'].append(dict(name=name, environment=extra, returncode=done.returncode,
                                             oracle_passed=bool(result.get('passed')), graph_nodes=nodes))
    cases = json.loads((reference/'fixtures.json').read_text())
    def differences(a, b):
        worst = {}
        for case in cases:
            x, y = (np.load(out/c/case['fixture']/'result.npz') for c in (a, b))
            for f in FIELDS: worst[f] = max(worst.get(f, 0.), float(np.abs(x[f]-y[f]).max()))
        return worst
    try:
        noise = differences('unrolled', 'unrolled-repeat')
    except FileNotFoundError:
        noise = None
    report['run_to_run_noise'] = noise
    for name, _, twin in CONFIGS:
        if not twin: continue
        try:
            worst = differences(name, twin)
        except FileNotFoundError:
            report['comparisons'].append(dict(config=name, twin=twin, passed=False, missing=True)); continue
        limits = {f: min(1.e-11, max(1.e-13, 100*noise[f])) for f in FIELDS} if noise else None
        report['comparisons'].append(dict(config=name, twin=twin, max_differences=worst, limits=limits,
                                          passed=bool(limits and all(worst[f] <= limits[f] for f in FIELDS))))
    for name in ('batched-lanes4', 'batched-cache-lanes2'):
        try:
            report.setdefault('batched_vs_unrolled', {})[name] = differences(name, 'unrolled')
        except FileNotFoundError:
            pass
    report['passed'] = (all(c['oracle_passed'] for c in report['configurations'])
                        and all(c['passed'] for c in report['comparisons']))
    text = json.dumps(report, indent=1)
    (out/'k-loop.json').write_text(text+'\n'); print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
