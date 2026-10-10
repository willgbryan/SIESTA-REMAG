#!/usr/bin/env python3
"""Reparse retained conditional-component activity and hash its raw evidence.

This verifies CUPTI memcpy/memset coverage and zero graph-region H2D/D2H.
It does not qualify full kernel activity, CPU work, managed/mapped memory,
filesystem I/O, production SCF integration, or performance.
"""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
import tarfile

import numpy as np
from siesta_remag.cuda_trace import summarize_cuda


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root):
    width=json.loads((root/'metadata.json').read_text()).get('block_local_width',64)
    if width not in (32,64):raise ValueError('Unknown local eigensolver width')
    parser=Path(inspect.getsourcefile(summarize_cuda))
    with tarfile.open(root/'implementation.tar.gz') as archive:
        with archive.extractfile('src/siesta_remag/cuda_trace.py') as stream:
            archived=hashlib.file_digest(stream,'sha256').hexdigest()
        # Preserve historical canonical-layout accounting. The paired layout
        # adds one inverse map to initialization; it adds no SCF host copy.
        with archive.extractfile('native/block_jacobi.cu') as stream:
            inverse_map=int(b'restore=allocate<int>' in stream.read())
    if archived!=sha(parser):raise ValueError('Current parser differs from the recorded implementation')
    components={}
    for name in ('scf-control-memory','block-jacobi-memory','scf-cycle-memory'):
        directory=root/(name+'-check');report_path=directory/'check.json'
        if not report_path.is_file():continue
        report=json.loads(report_path.read_text());fixtures={v['fixture']:v for v in report['fixtures']}
        checks=[];calibration=[]
        for record in report['observer_calibration']:
            mode=record['mode'];raw_path=directory/('observer-'+mode+'-check')/'remag-activity.json'
            actual=summarize_cuda(json.loads(raw_path.read_text()),
                dict(total_explicit=dict(host_to_device_bytes=32768,device_to_host_bytes=32776),accounting_errors=[]),
                expected_scf_iterations=1 if mode=='conditional-scope' else 2,
                process_succeeded=record['process_returncode']==0,allow_unique_graph_scope=True,memory_only=True)
            calibration.append(dict(mode=mode,passed=record['passed'] and actual==record['activity'],
                expected_coverage_complete=mode!='conditional',coverage_complete=actual['coverage_complete'],
                raw_trace=str(raw_path.relative_to(root)),raw_trace_sha256=sha(raw_path),errors=actual['errors']))
        for record in report['checks']:
            output=directory/record['output_file'];input_path=directory/fixtures[record['fixture']]['input_file']
            raw_path=output.parent/'remag-activity.json'
            with np.load(input_path) as inp,np.load(output) as out:
                if name=='scf-control-memory':
                    count=len(inp['x'])
                    explicit=dict(host_to_device_bytes=count*(112+112)+inp['x'].nbytes+inp['f'].nbytes,
                                  device_to_host_bytes=out['values'].nbytes+out['results'].nbytes+out['steps'].nbytes+8)
                elif name=='scf-cycle-memory':
                    n,kind,nh,nk,*_=np.frombuffer(inp['problem'],dtype=np.int32,count=10)
                    n,kind,nh,nk=map(int,(n,kind,nh,nk))
                    nsm=int(np.frombuffer(inp['problem'],dtype=np.int32,count=1,offset=264)[0])
                    used=n*n;size=inp['phi'].size//n;points=size//nsm**3
                    dim=n*(2 if kind>=2 else 1);padded=width*((dim+width-1)//width);rounds=padded//(width//2)-1
                    names=('h','d','e','overlap','kinetic','nonlocal_h','onsite','offsite','phi','kpoints',
                           'weights','xyz','atom','core','neutral','constants')
                    # Explicit fixture arrays + generated topology/config +
                    # blocked-plan device constants. Independent of the C ABI
                    # wrapper's reported byte counters; padding is included.
                    up=sum(inp[key].nbytes for key in names)+32*used+4*(points+1)+4*points*n+4+8*n+440
                    up+=2*(16 if kind else 8)+(rounds+inverse_map)*padded*4
                    down=2*(sum(out[key].nbytes for key in ('h','d','e','bank','q','result','history','diagnostics'))+32)
                    explicit=dict(host_to_device_bytes=up,device_to_host_bytes=down)
                    if up!=record['staged_bytes'] or down!=record['exported_bytes']:
                        raise ValueError('Cycle staging/export byte counters disagree with fixture-derived sizes')
                else:
                    h=inp['h'];cycles,n,_=h.shape;padded=width*((n+width-1)//width);rounds=padded//(width//2)-1
                    explicit=dict(host_to_device_bytes=2*h.nbytes+4+2*h.dtype.itemsize+(rounds+inverse_map)*padded*4,
                                  device_to_host_bytes=out['z'].nbytes+out['w'].nbytes+272*cycles+4)
            actual=summarize_cuda(json.loads(raw_path.read_text()),dict(total_explicit=explicit,accounting_errors=[]),
                                  expected_scf_iterations=1,process_succeeded=record['process_returncode']==0,
                                  allow_unique_graph_scope=True,memory_only=True)
            same=actual==record['cuda_trace'];region=actual.get('phases',{}).get('scf',{})
            passed=record['passed'] and same and actual['coverage_complete'] and actual['phase_attribution_verified'] and (
                region.get('host_to_device_bytes',0)==0 and region.get('device_to_host_bytes',0)==0)
            checks.append(dict(fixture=record['fixture'],nested=record.get('nested'),passed=passed,
                original_summary_reproduced=same,raw_trace=str(raw_path.relative_to(root)),raw_trace_sha256=sha(raw_path),
                output_sha256=sha(output),input_sha256=sha(input_path),region=region,
                graph_scope_attributed_events=actual['graph_scope_attributed_events'],errors=actual['errors']))
        components[name]=dict(passed=report['passed'] and report['completed_all_checks'] and bool(checks) and
                             all(v['passed'] for v in checks),report_sha256=sha(report_path),checks=checks,
                             calibration=calibration,calibration_passed=len(calibration)==4 and
                             {v['mode'] for v in calibration}=={'ordinary','graph','conditional-scope','conditional'} and
                             all(v['passed'] for v in calibration))
        components[name]['passed'] &= components[name]['calibration_passed']
    return dict(scope=__doc__,parser_sha256=archived,verifier_sha256=sha(Path(__file__)),components=components,
                passed=bool(components) and all(v['passed'] for v in components.values()),strict_residency_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();result=verify(args.run)
    (args.run/'conditional-transfer-verifier.py').write_bytes(Path(__file__).read_bytes())
    (args.run/'conditional-transfer-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Conditional transfer evidence:',result['passed'],
          {k:len(v['checks']) for k,v in result['components'].items()})
    if not result['passed']:raise SystemExit(1)
