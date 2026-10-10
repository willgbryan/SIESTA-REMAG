#!/usr/bin/env python3
"""Check initial density lease ordering, poisoned host identity and output caching."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from siesta_remag.initial_state import record_errors


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(folder, spins, requests):
    probe=json.loads((folder/'probe.json').read_text())
    native=json.loads((folder/'remag-initial-state.json').read_text())
    traffic=json.loads((folder/'remag-residency.json').read_text())
    assert probe==dict(passed=True,spins=spins,requests=requests,rejections=16+int(requests in (0,1)),device_density_preserved_after_host_poison=True)
    assert native['schema']==1
    issues=record_errors(native['cycles'],[dict(epoch=1,terminal=dict(capacity=5))],2,
                         dict(nspin=spins,nbands=3*(2 if spins>=4 else 1)),2)
    assert not issues,issues
    if (folder/'initial-owner.bin').is_file():
        raw=np.fromfile(folder/'initial-owner.bin',dtype='f8')
        np.savez_compressed(folder/'arrays.npz',staged=raw[:5*spins],retained=raw[5*spins:10*spins],population=raw[10*spins:])
    arrays=np.load(folder/'arrays.npz');expected=.125+np.arange(5*spins)*.03125
    np.testing.assert_array_equal(arrays['staged'],expected)
    np.testing.assert_array_equal(arrays['retained'],expected)
    d=expected.reshape(spins,5)[:min(spins,4)].copy()
    if spins==8:d[2:]=.5*(d[2:]+expected.reshape(8,5)[6:])
    weighted=d*np.array([.25,-.5,.125,0.,.75])
    orbital=np.stack((weighted[:,0],weighted[:,1:3].sum(axis=1),weighted[:,4]))
    atomic=np.stack((orbital[:2].sum(axis=0),orbital[2]));total=orbital.sum(axis=0)
    def view(q):
        charge=q[:2].sum();xyz=np.zeros(3)
        if len(q)>1:xyz[2]=q[0]-q[1]
        if len(q)==4:xyz[:2]=2*q[2:]
        spin=np.linalg.norm(xyz) if len(q)==4 else xyz[2]
        return np.r_[charge,spin,xyz]
    orbital_view=np.array([view(q) for q in orbital]);atom_view=np.array([np.r_[z-view(q)[0],view(q)] for z,q in zip((10.,8.),atomic)])
    species_view=np.array([view(atomic[1]),view(total)])
    vector=atom_view[:,3:].sum(axis=0)
    summary=np.r_[view(total),atom_view[:,0].sum(),np.linalg.norm(vector) if spins>=4 else vector[2],vector]
    values=[orbital.ravel(),atomic.ravel(),total,orbital_view.ravel(),atom_view.ravel(),species_view.ravel(),summary]
    kinds=list(range(7)) if requests&2 else [6] if requests else []
    expected_output=np.concatenate([values[k] for k in kinds]) if kinds else np.empty(0)
    np.testing.assert_allclose(arrays['population'],expected_output,atol=1e-12,rtol=1e-12)
    record=native['cycles'][0]
    assert record['reads']==[2*int(k in kinds) for k in range(7)]
    def counter(h=0,d=0,n=0):return dict(host_to_device_bytes=h,device_to_host_bytes=d,copy_calls=n)
    expected_copies={'initial_density_setup':counter(h=40*spins,n=1)}
    if requests:
        expected_copies['initial_population_setup']=counter(h=132,n=8)
        expected_copies['initial_population_output']=counter(d=4+8*len(expected_output),n=1+len(kinds))
    observed={}
    for interval in traffic['intervals']:
        assert interval['phase']=='initialization'
        for name,counts in interval['boundaries'].items():
            out=observed.setdefault(name,counter())
            for k,v in counts.items():out[k]+=v
    assert observed==expected_copies,(observed,expected_copies)
    return dict(spins=spins,requests=requests,rejections=probe['rejections'],passed=True,
                max_population_error=float(np.max(np.abs(arrays['population']-expected_output))) if kinds else 0.)


def run(executable, output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True);checks=[]
    for spins in (1,2,4,8):
        for requests in range(4):
            folder=output.parent/f'spin{spins}_requests{requests}'
            if executable:
                folder.mkdir()
                with (folder/'worker.log').open('w') as log:
                    subprocess.run([str(executable.resolve()),str(spins),str(requests)],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
            check=validate(folder,spins,requests)
            names=('probe.json','remag-residency.json','remag-initial-state.json','arrays.npz','worker.log')
            check['evidence_sha256']={str((folder/n).relative_to(output.parent)):sha(folder/n) for n in names}
            checks.append(check)
    report=dict(scope=__doc__,passed=True,checks=checks,checker_sha256=sha(Path(__file__)),native_exec_sha256=sha(executable) if executable else None)
    output.write_text(json.dumps(report,indent=2)+'\n');print('Initial density owner:',len(checks),'passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.executable,args.output)
