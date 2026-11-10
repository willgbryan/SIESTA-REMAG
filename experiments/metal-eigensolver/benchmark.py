#!/usr/bin/env python3
"""Matched CPU/Metal matrix benchmark; no SCF or physical MAE claims."""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import cholesky

from captures import ROOT, PREVIOUS, load, check
from hybrid import BatchSolver
from metal_seed import MetalSeed
from cpu_fp64 import CpuFP64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verification(cases, solutions):
    checks = [check(c, s) for c, s in zip(cases, solutions, strict=True)]
    return dict(passed=len(checks) == len(cases) and all(c['component_gates_passed'] for c in checks),
                count=len(checks), maxima={k: max(c[k] for c in checks) for k in checks[0]
                                        if k != 'component_gates_passed'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=ROOT/'build/metal-eigensolver/libmetal_seed.dylib')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--multipliers', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--cpu-backend', choices=['native', 'scipy'], default='native')
    args = parser.parse_args()
    if args.repetitions < 3 or any(m < 1 or m > 16 for m in args.multipliers) or any(w not in (1, 4) for w in args.workers):
        parser.error('Require >=3 repetitions, replay multipliers 1..16, and 1 or 4 CPU workers')
    if args.output.exists():
        parser.error('Choose a new output path; previous attempts must be preserved')
    if os.environ.get('VECLIB_MAXIMUM_THREADS') != '1':
        parser.error('Set VECLIB_MAXIMUM_THREADS=1 before process startup; parallelize across matrices explicitly')
    cases = load()
    directory = Path(__file__).parent
    report = dict(schema=1, started_at_utc=datetime.now(timezone.utc).isoformat(), state='running',
        scope='Matched local eigensolver component benchmark; selected-k captures and labelled exact replays, not full meshes or SCF',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_files={str(p.relative_to(ROOT)): digest(p) for p in sorted(directory.iterdir()) if p.suffix in ('.py', '.metal', '.mm', '.cpp')},
        library_sha256=digest(args.library), previous_report_sha256=digest(PREVIOUS/'replay.json'),
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        blas=scipy.show_config(mode='dicts')['Build Dependencies']['blas'],
        thread_environment=dict(VECLIB_MAXIMUM_THREADS=os.environ.get('VECLIB_MAXIMUM_THREADS')),
        timing_scope='Function wall time includes standardization, packing/copies, GPU submission/wait, FP64 correction, acceptance and fallback. Input I/O and independent verification excluded for both. Compile/allocation setup reported separately.',
        cpu_backend=args.cpu_backend, samples=[], inputs=[c['provenance'] for c in cases], mae_qualified=False, production_changed=False)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)

    save()
    try:
        for n in (96, 112):
            unique = [c for c in cases if c['provenance']['bands'] == n]
            for multiplier in args.multipliers:
                batch = unique*multiplier
                hs = [(c['h'], c['s']) for c in batch]
                started = perf_counter()
                factors = [cholesky(s, lower=True, check_finite=False) for _, s in hs]
                factor_setup = perf_counter()-started
                started = perf_counter()
                with MetalSeed(args.library, n, len(batch)) as seed:
                    plan_setup = perf_counter()-started
                    report['device'] = seed.device
                    for workers in args.workers:
                        solver = BatchSolver(seed, workers)
                        started = perf_counter()
                        native_cpu = CpuFP64(args.library, n, workers) if args.cpu_backend == 'native' else None
                        cpu_setup = perf_counter()-started
                        cpu_solve = native_cpu.solve if native_cpu else solver.cpu
                        try:
                            for cached in (False, True):
                                factor_arg = factors if cached else None
                                sample = dict(material=unique[0]['provenance']['material'], bands=n, batch=len(batch),
                                    unique_captures=len(unique), exact_replay_multiplier=multiplier, cpu_workers=workers,
                                    overlap_cached=cached, factor_setup_seconds=factor_setup,
                                    metal_plan_setup_seconds=plan_setup, cpu_plan_setup_seconds=cpu_setup, repetitions=[])
                                report['samples'].append(sample)
                                # One untimed warmup per path; retain its checks and fallbacks.
                                cpu = cpu_solve(hs, factor_arg)
                                mixed, details = solver.mixed(hs, factor_arg)
                                sample['warmup'] = dict(cpu=verification(batch, cpu), mixed=verification(batch, mixed), details=details)
                                save()
                                for repetition in range(args.repetitions):
                                    row = dict(index=repetition, order=['cpu', 'mixed'] if repetition%2 == 0 else ['mixed', 'cpu'])
                                    for method in row['order']:
                                        start = perf_counter()
                                        if method == 'cpu':
                                            values = cpu_solve(hs, factor_arg)
                                            duration = perf_counter()-start
                                        else:
                                            values, details = solver.mixed(hs, factor_arg)
                                            duration = perf_counter()-start
                                            row['mixed_details'] = details
                                        row[method+'_seconds'] = duration
                                        row[method+'_verification'] = verification(batch, values)
                                    sample['repetitions'].append(row)
                                    save()
                                passing = all(r[m+'_verification']['passed'] for r in sample['repetitions'] for m in ('cpu', 'mixed'))
                                no_fallback = all(r['mixed_details']['fallback_count'] == 0 for r in sample['repetitions'])
                                sample['component_qualified'] = passing and no_fallback and sample['warmup']['cpu']['passed'] and sample['warmup']['mixed']['passed'] and not sample['warmup']['details']['fallback_count']
                                sample['cpu_median_seconds'] = statistics.median(r['cpu_seconds'] for r in sample['repetitions'])
                                sample['mixed_median_seconds'] = statistics.median(r['mixed_seconds'] for r in sample['repetitions'])
                                sample['median_paired_ratio_cpu_over_mixed'] = statistics.median(r['cpu_seconds']/r['mixed_seconds'] for r in sample['repetitions']) if sample['component_qualified'] else None
                                save()
                                print(json.dumps({k: sample[k] for k in ('material', 'batch', 'cpu_workers', 'overlap_cached', 'component_qualified', 'cpu_median_seconds', 'mixed_median_seconds', 'median_paired_ratio_cpu_over_mixed')}), flush=True)
                        finally:
                            if native_cpu:
                                native_cpu.close()
                            solver.close()
        report['state'] = 'complete'
    except Exception as error:
        report['state'] = 'failed'
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
