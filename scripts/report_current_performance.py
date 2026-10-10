#!/usr/bin/env python3
"""Publish fresh repeated timings only after scientific and ownership audits."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def report(root):
    meta, ledger = read(root/'metadata.json'), read(root/'ledger.json')
    required = ('source-verification.json', 'component-evidence-verification.json',
                'scientific-recheck.json', 'device-cycle-verification.json',
                'hamiltonian-output-verification.json', 'terminal-export-verification.json',
                'integral-force-material-verification.json', 'final-density-verification.json',
                'population-material-verification.json', 'energy-density-output-verification.json')
    if meta.get('gpu_initialization'):
        required += ('initialization-verification.json',)
        if 'initial_state.py' in meta.get('harness_sha256', {}):
            required += ('initial-state-component-verification.json',
                         'initial-state-verification.json')
    if meta.get('gpu_history'):
        required += ('density-history-verification.json',)
    if meta.get('final_spectra'):
        required += ('final-spectrum-verification.json',)
    if meta.get('nonlocal_h0_shadow') or meta.get('poison_nonlocal_h0'):
        raise ValueError('H0 matrix shadows are diagnostic and cannot qualify performance')
    if meta.get('nonlocal_h0'):
        required += ('nonlocal-h0-material-verification.json',)
        h0 = read(root/'nonlocal-h0-material-verification.json')
        if h0.get('production_h0') is not True or not h0.get('material_shadow_qualification_sha256'):
            raise ValueError('Production H0 requires qualified material matrices and device leases')
    for name in required:
        if read(root/name).get('passed') is not True:
            raise ValueError('Performance prerequisite failed: '+name)
    hardware = {}
    for line in meta['gpu'].splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key.strip() in ('Product Name', 'Driver Version'):
                hardware[key.strip()] = value.strip().split(' [Deprecated;')[0]
    if set(hardware) != {'Product Name', 'Driver Version'}:
        raise ValueError('GPU model/driver provenance missing')
    if meta.get('reference_run') or meta.get('candidate_repeats', 1) < 3:
        raise ValueError('Fresh references and at least three GPU repetitions required')
    groups = {}
    for row in ledger:
        if not row['passed'] or not row['performance_comparable']:
            raise ValueError('Failed or incomparable measurement')
        folder = Path(row.get('candidate_path', row['benchmark']+'/candidate'))
        case, role = folder.parent.name, folder.name
        group=groups.setdefault(case, {})
        if role in group:
            raise ValueError('Duplicate measurement: '+str(folder))
        group[role] = row
    if set(groups) != set(meta['cases']):
        raise ValueError('Incomplete case coverage')
    wanted = {'candidate', *('candidate-repeat'+str(i) for i in range(2, meta['candidate_repeats']+1))}
    rows = []
    for case in meta['cases']:
        if set(groups[case]) != wanted:
            raise ValueError('Unexpected or missing repetitions: '+case)
        records = [groups[case][role] for role in sorted(wanted)]
        reference = read(root/case/'reference/result.json')
        for role in wanted:
            candidate = read(root/case/role/'result.json')
            if candidate['hardware'] != reference['hardware']:
                raise ValueError('Reference and candidate allocations differ')
        times = [r['candidate_seconds'] for r in records]
        median = statistics.median(times)
        rows.append(dict(case=case, cpu_seconds=records[0]['reference_seconds'],
                         gpu_seconds=times, median_gpu_seconds=median, gpu_range_seconds=[min(times), max(times)],
                         end_to_end_speedup=records[0]['reference_seconds']/median,
                         cpu_iterations=records[0]['reference_iterations'],
                         gpu_iterations=[r['candidate_iterations'] for r in records]))
    evidence = {}
    for name in (*required, 'metadata.json', 'ledger.json'):
        evidence[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    data = dict(speedup_qualified=True, incremental_speedup_qualified=False,
                strict_residency_verified=False, fresh_matched_run=root.name, measurements=rows,
                gpu_model=hardware['Product Name'], driver_version=hardware['Driver Version'],
                evidence_sha256=evidence,
                scope='Serial upstream CPU versus GPU on one allocation, one BLAS thread. GPU medians from cold process repeats; one CPU measurement per case. Includes setup, SCF, forces and requested output. Excludes cloud provisioning, builds and evidence transfer.')
    lines = [f'# Current qualified performance — {root.name}', '', data['scope'], '',
             'GPU: '+data['gpu_model']+'; driver '+data['driver_version']+'.', '',
             'The build enables NetCDF for both reference and candidate. This batch qualifies the combined implementation; it does not isolate the incremental effect of individual optimizations. An optimized multicore CPU comparison remains unmeasured.', '',
             '| Case | CPU (s) | GPU median (s) | GPU range (s) | CPU / GPU | SCF iterations: CPU → GPU |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        if '__' in r['case']:
            continue
        lo, hi = r['gpu_range_seconds']
        iterations = '/'.join(map(str, r['gpu_iterations']))
        lines.append(f"| {r['case']} | {r['cpu_seconds']:.3f} | {r['median_gpu_seconds']:.3f} | {lo:.3f}–{hi:.3f} | {r['end_to_end_speedup']:.2f}× | {r['cpu_iterations']} → {iterations} |")
    changing = [r['case'] for r in rows if any(i != r['cpu_iterations'] for i in r['gpu_iterations'])]
    if changing:
        lines.extend(['', 'Different SCF iteration counts occur in '+', '.join(changing)+'. These ratios are end-to-end solution speedups, not per-iteration or kernel speedups.'])
    remaining = ['complete memory/I/O observation', 'CPU ionic consumers']
    if not meta.get('gpu_history'):
        remaining.append('production GPU density history')
    if not meta.get('gpu_initialization'):
        remaining.append('GPU initialization')
    elif 'initial_state.py' not in meta.get('harness_sha256', {}):
        remaining.append('initial population consumers')
    elif not meta.get('gpu_history'):
        remaining.append('host density staging and terminal D/H/spectral export gating')
    else:
        remaining.append('remaining Hamiltonian consumers and exports' if meta.get('final_spectra') else
                         'remaining Hamiltonian/spectral consumers and exports')
    lines.extend(['', 'Requested TSDE/NetCDF variants are included in the machine-readable measurements and scientific qualification. Cases without TSDE/NetCDF requests omit their terminal E transfer; their other requested outputs remain. Remaining work includes '+', '.join(remaining)+'.', '',
                  '[All measurements](current-performance-scope.json) · [Scientific audit](scientific-recheck.json) · [Requested-output audit](energy-density-output-verification.json)', ''])
    (root/'current-performance-scope.json').write_text(json.dumps(data, indent=2)+'\n')
    (root/'PERFORMANCE.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report(args.run)
    print(args.run/'PERFORMANCE.md')
