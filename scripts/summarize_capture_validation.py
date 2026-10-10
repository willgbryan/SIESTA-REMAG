#!/usr/bin/env python3
"""Summarize captured component checks and their independent SIESTA regression.

Graph inventories are structural evidence, not independent transfer coverage.
Production CUDA/syscall coverage belongs only to the recorded SIESTA runs.
"""
import argparse
import json
from pathlib import Path


def summarize(root):
    read=lambda p:json.loads((root/p).read_text())
    metadata=read('metadata.json');native=read('kernel-check.json');ledger=read('ledger.json')
    instrumentation=read('instrumentation.json');source=read('source-verification.json')
    captured=[v for v in native if v.get('kind','').startswith('captured_')]
    groups={}
    for record in captured:groups.setdefault(record['kind'],[]).append(record)
    mae={p.name:json.loads(p.read_text()) for p in root.glob('mae-*.json')}
    candidates=[]
    for case in metadata['cases']:
        result=read(f'{case}/candidate/result.json')
        trace=result['telemetry']['cuda_trace'];io=result['telemetry']['io_trace']
        candidates.append(dict(case=case,valid=result['valid'],cuda_coverage_complete=trace['coverage_complete'],
            io_coverage_complete=io['coverage_complete'],steady_cuda=trace.get('steady_state'),steady_io=io.get('steady_state')))
    maximum={}
    for record in ledger:
        for name,check in record['checks'].items():
            value=check.get('max_absolute_error')
            if value is not None:maximum[name]=max(maximum.get(name,0.),value)
    return dict(scope=__doc__,run=root.name,source_hashes=source['recorded_hash_count'],
        native_checks=len(native),captured_checks=groups,upstream_pairs=len(ledger),mae=mae,
        instrumentation=instrumentation,max_scientific_errors=maximum,candidates=candidates,
        passed=bool(captured) and all(v['passed'] for v in native+ledger+list(mae.values())) and
               source['passed'] and instrumentation['passed'] and
               {v['benchmark'] for v in ledger}==set(metadata['cases']),
        full_device_scf_qualified=False,captured_component_activity_coverage_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();result=summarize(args.run)
    (args.run/'capture-validation-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Captured components and SIESTA regression: passed={result['passed']}; "
          f"native={result['native_checks']}, upstream={result['upstream_pairs']}, sources={result['source_hashes']}")
    if not result['passed']:raise SystemExit(1)
