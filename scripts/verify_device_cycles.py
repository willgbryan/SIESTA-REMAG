#!/usr/bin/env python3
"""Reproduce production device-cycle accounting and optional CUDA/filesystem traces.

The measured region covers every accepted electronic iteration, including the
first. This does not qualify final GPU observables, complete kernel activity,
managed/mapped host accesses, page faults or asynchronous writeback.
"""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
import tarfile

from siesta_remag.output import parse_output
from siesta_remag.residency import summarize_residency
from siesta_remag.cuda_trace import summarize_cuda
from siesta_remag.io_trace import summarize_io


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root,require_traces=False):
    parser_functions=dict(output=parse_output,residency=summarize_residency,cuda_trace=summarize_cuda,io_trace=summarize_io)
    parsers={}
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for name in ('output','residency','cuda_trace','io_trace'):
            path=f'src/siesta_remag/{name}.py'
            with archive.extractfile(path) as stream:recorded=hashlib.file_digest(stream,'sha256').hexdigest()
            # Hash the module actually imported. PYTHONPATH may point to a
            # safely extracted archived implementation when auditing old runs.
            if sha(Path(inspect.getsourcefile(parser_functions[name])))!=recorded:
                raise ValueError(f'Loaded parser differs from archived {path}')
            parsers[path]=recorded
    checks=[]
    for path in sorted(root.glob('*/*/result.json')):
        record=json.loads(path.read_text())
        if record.get('runtime_settings',{}).get('REMAG_DEVICE_SCF')!='1':continue
        folder=path.parent;files={}
        def read(name):
            p=folder/name;files[name]=sha(p);return json.loads(p.read_text())
        missing=[name for name in ('remag-device-scf.json','remag-residency.json') if not (folder/name).is_file()]
        if missing:
            for name in ('result.json','stdout.log','stderr.log','remag-device-scf-failure.json',
                         'remag-failed-h.bin','remag-failed-s.bin','remag-failed-solver.bin'):
                if (folder/name).is_file():files[name]=sha(folder/name)
            checks.append(dict(case=folder.parent.name,role=folder.name,passed=False,
                errors=['Required cycle evidence missing: '+', '.join(missing)],
                cycles=None,device_iterations=None,whole_cycle_explicit=None,cuda_region=None,filesystem_region=None,
                cuda_coverage=None,filesystem_coverage=None,graph_scope_events=None,
                input_sha256=record['benchmark']['input_sha256'],evidence_sha256=files))
            continue
        native=read('remag-device-scf.json');phase=read('remag-residency.json');phase['execution_mode']='device_cycle'
        files['result.json']=sha(path);files['stdout.log']=sha(folder/'stdout.log')
        output=parse_output((folder/'stdout.log').read_text(errors='replace'))
        rows=native['cycles'];intervals=[i for i in phase['intervals'] if i['phase']=='scf']
        iterations=sum(r['iterations'] for r in rows);errors=[]
        telemetry=record['telemetry'];owned=telemetry.get('gpu_device_cycle',{})
        expected=dict(native,host_to_device_bytes=sum(r['setup_bytes'] for r in rows),
                      device_to_host_bytes=sum(r['export_bytes'] for r in rows))
        if expected!=owned:errors.append('Native cycle publication differs from result telemetry')
        if not record['valid'] or not output['completed'] or not output['converged']:
            errors.append('Successful converged production completion not established')
        if not rows or iterations!=output['scf_iterations'] or iterations!=record['observables']['scf_iterations']:
            errors.append('Device iteration count differs from published SIESTA diagnostics')
        if [r['geometry'] for r in rows]!=[i['geometry'] for i in intervals] or any(i['iteration']!=1 for i in intervals):
            errors.append('Device geometries do not match complete-cycle regions')
        if any(r['status'] or not r['converged'] or r['iterations']<1 or r['forbidden_nodes'] or r['graph_nodes']<1 for r in rows):
            errors.append('Invalid cycle result or graph inventory')
        accounting=summarize_residency(phase,telemetry,iterations)
        if accounting!=telemetry.get('residency'):errors.append('Explicit accounting summary was not reproduced')
        errors.extend(accounting['accounting_errors'])
        explicit=accounting['steady_state_explicit']
        if any(explicit.values()):errors.append('Explicit copies inside complete electronic cycles')
        cuda=None;io=None
        if record['runtime_settings'].get('REMAG_TRACE_CUDA')=='1':
            raw=read('remag-activity.json')
            cuda=summarize_cuda(raw,accounting,expected_scf_iterations=len(rows),
                process_succeeded=record['returncode']==0 and not record['timed_out'],
                memory_only=True,allow_unique_graph_scope=True)
            if cuda!=telemetry.get('cuda_trace'):errors.append('Raw CUDA summary was not reproduced')
            region=cuda.get('phases',{}).get('scf',{})
            if not cuda['coverage_complete'] or not cuda['phase_attribution_verified']:
                errors.append('Incomplete CUDA copy coverage/attribution')
            if region.get('host_to_device_bytes',0) or region.get('device_to_host_bytes',0):
                errors.append('Observed host/device copies inside complete electronic cycles')
            calibration_path=root/'scf-cycle-memory-check/check.json'
            if not calibration_path.is_file():errors.append('Conditional observer calibration evidence missing')
            else:
                calibration=json.loads(calibration_path.read_text()).get('observer_calibration',[])
                if len(calibration)!=4 or not all(v.get('passed') for v in calibration):
                    errors.append('Conditional observer calibration did not pass')
        if record['runtime_settings'].get('REMAG_TRACE_IO')=='1':
            files['syscalls.log']=sha(folder/'syscalls.log')
            with (folder/'syscalls.log').open(errors='replace') as stream:
                io=summarize_io(stream,phase,process_succeeded=record['returncode']==0 and not record['timed_out'])
            if io!=telemetry.get('io_trace'):errors.append('Raw filesystem summary was not reproduced')
            if not io['coverage_complete']:errors.append('Incomplete filesystem coverage')
            if io.get('phases',{}).get('scf',{}).get('filesystem_calls',0):
                errors.append('Observed filesystem operations inside complete electronic cycles')
        if require_traces and (cuda is None or io is None):errors.append('Required independent traces are absent')
        checks.append(dict(case=folder.parent.name,role=folder.name,passed=not errors,errors=errors,
            cycles=len(rows),device_iterations=iterations,whole_cycle_explicit=explicit,
            cuda_region=cuda.get('phases',{}).get('scf',{}) if cuda else None,
            filesystem_region=io.get('phases',{}).get('scf',{}) if io else None,
            cuda_coverage=cuda.get('coverage_complete') if cuda else None,
            filesystem_coverage=io.get('coverage_complete') if io else None,
            graph_scope_events=cuda.get('graph_scope_attributed_events') if cuda else None,
            input_sha256=record['benchmark']['input_sha256'],evidence_sha256=files))
    return dict(scope=__doc__,require_traces=require_traces,checks=checks,parser_sha256=parsers,
        verifier_sha256=sha(Path(__file__)),passed=bool(checks) and all(c['passed'] for c in checks),
        strict_residency_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--require-traces',action='store_true');args=parser.parse_args()
    result=verify(args.run,args.require_traces)
    (args.run/'device-cycle-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'device-cycle-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Production cycle evidence:',len(result['checks']),'calculations; passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
