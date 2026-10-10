#!/usr/bin/env python3
"""Replay retained failed H/S against LAPACK, varying graph nesting and warmup.

A solver failure is retained as a failed numerical check, never an accepted SCF
result. The lower triangle is authoritative, as in both production solvers.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh
from check_block_jacobi_gpu import lower_hermitian, sha


def run(source,library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    source=source.resolve();failure=json.loads((source/'remag-device-scf-failure.json').read_text())
    n=failure['dimension'];dtype=np.complex128 if failure['complex_values'] else np.float64
    h=np.fromfile(source/'remag-failed-h.bin',dtype=dtype).reshape((n,n),order='F')
    s=np.fromfile(source/'remag-failed-s.bin',dtype=dtype).reshape((n,n),order='F')
    h0=lower_hermitian(h);s0=lower_hermitian(s)
    expected=eigh(h0,s0,eigvals_only=True,driver='gvd')
    fixture=output.parent/'failed-input.npz';np.savez_compressed(fixture,h=np.repeat(h[None],8,axis=0),s=np.repeat(s[None],8,axis=0),reference_w=expected)
    checks=[]
    for nested,name in enumerate(('flat','nested','guarded','guarded-private-warmup')):
        folder=output.parent/name;folder.mkdir();config=folder/'config.json';result=folder/'result.json'
        config.write_text(json.dumps(dict(input=str(fixture),output=str(result),library=str(library.resolve()),maximum=20,nested=nested,trace=False),indent=2)+'\n')
        done=subprocess.run([sys.executable,str(Path(__file__).with_name('check_block_jacobi_gpu.py')),'--worker',str(config)],
            cwd=folder,env={**os.environ,'REMAG_REPLAY_DIAGNOSTICS':'1'},capture_output=True,text=True,timeout=180)
        (folder/'stdout.log').write_text(done.stdout+done.stderr)
        record=dict(mode=name,nested=nested,returncode=done.returncode,passed=False,checks=[])
        if result.is_file():
            record['native']=json.loads(result.read_text());data=np.load(result.with_suffix('.npz'))
            for i,status in enumerate(data['status']):
                item=dict(cycle=i,status=int(status),sweeps=int(data['sweeps'][i]),passed=False)
                if status==0 and record['native']['native_return']==0:
                    z=data['z'][i];w=data['w'][i]
                    residual=np.linalg.norm(h0@z-(s0@z)*w)/(np.linalg.norm(h0)*np.linalg.norm(z))
                    orth=np.linalg.norm(z.conj().T@s0@z-np.eye(n),ord=np.inf)
                    err=np.max(np.abs(w-expected))
                    item.update(relative_residual=float(residual),orthogonality=float(orth),eigenvalue_error=float(err),passed=bool(residual<1.e-10 and orth<1.e-9 and err<1.e-10))
                record['checks'].append(item)
            record['passed']=done.returncode==0 and record['native']['native_return']==0 and record['native']['iterations']==8 and all(c['passed'] for c in record['checks'])
        if (folder/'local-failure.json').is_file():record['local_failure']=json.loads((folder/'local-failure.json').read_text())
        record['evidence_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
        checks.append(record);print(name,'statuses',[c['status'] for c in record['checks']],'passed',record['passed'],flush=True)
    report=dict(scope=__doc__,source=str(source),source_failure=failure,block_local_width=int(os.environ.get("REMAG_BLOCK_WIDTH","64")),
        block_local_solver=os.environ.get("REMAG_BLOCK_LOCAL_SOLVER","syev"),
        source_sha256={name:sha(source/name) for name in ('remag-device-scf-failure.json','remag-failed-h.bin','remag-failed-s.bin','remag-failed-solver.bin')},
        fixture_sha256=sha(fixture),script_sha256=sha(Path(__file__)),checks=checks,passed=all(c['passed'] for c in checks))
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not run(args.source,args.library,args.output):raise SystemExit(1)
