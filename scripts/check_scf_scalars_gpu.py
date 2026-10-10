#!/usr/bin/env python3
"""Check device SCF energy/charge/residual consumers against pristine routines.

Both input states are staged before execution. A captured graph selects the
next state on device, computes and packs results; only final artifacts export.
This component does not establish full SCF convergence or Hamiltonian ownership.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import check_scf_scalars as reference


class Problem(ct.Structure):
    _fields_=[(name,ct.c_int) for name in ('capacity','spins','rows','reserved')]


class Result(ct.Structure):
    _fields_=[('values',ct.c_double*13),('status',ct.c_int),('active',ct.c_int)]


class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_size_t),('graph_nodes',ct.c_int),('forbidden_nodes',ct.c_int),('iterations',ct.c_int)]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(upstream,build,output,library=None):
    output.parent.mkdir(parents=True,exist_ok=True)
    oracle=output.with_name('scf-scalars-reference.json');reference.run(upstream,build,oracle)
    ref=json.loads(oracle.read_text());array_path=oracle.parent/ref['arrays_file']
    assert sha(array_path)==ref['arrays_sha256'];arrays=np.load(array_path)
    report=dict(reference=ref['reference'],gpu_validation=None,oracle_file=oracle.name,
        oracle_sha256=sha(oracle),oracle_arrays_sha256=sha(array_path),
        fixture_script_sha256=sha(Path(__file__)),checks=[])
    candidate=None
    if library:
        candidate=ct.CDLL(str(library.resolve())).remag_check_scf_scalars
        candidate.argtypes=[ct.POINTER(Problem),ct.c_int,ct.c_int]+[ct.c_void_p]*6+[
            ct.POINTER(Result),ct.c_int,ct.POINTER(Check)]
        candidate.restype=ct.c_int;report['native_library_sha256']=sha(library)
    def execute(problem,seeds,num,ptr,overlap,source,graph,corrupt=0):
        source=np.ascontiguousarray(source);seeds=np.ascontiguousarray(seeds)
        num=np.ascontiguousarray(num);ptr=np.ascontiguousarray(ptr);overlap=np.ascontiguousarray(overlap)
        residual=np.full((len(source),problem.capacity*problem.spins),123.)
        records=(Result*len(source))();info=Check()
        code=candidate(ct.byref(problem),len(source),corrupt,seeds.ctypes.data,num.ctypes.data,ptr.ctypes.data,
            overlap.ctypes.data,source.ctypes.data,residual.ctypes.data,records,int(graph),ct.byref(info))
        assert code==0,(code,graph)
        assert info.iterations==len(source) and info.forbidden_nodes==0
        if graph: assert info.graph_nodes==8
        return np.array([list(r.values) for r in records]),residual,[r.status for r in records],info
    names=('h','hold','density','old_density','energy_density','old_energy_density')
    retained={}
    with tempfile.TemporaryDirectory(prefix='remag-scf-scalar-sequences-') as directory:
        tmp=Path(directory);exe=reference.build_reference(build,tmp)
        for record in ref['checks']:
            label=record['fixture'];n=record['capacity'];spins=record['spin_components']
            num=arrays[label+'_num'];ptr=arrays[label+'_ptr'];overlap=arrays[label+'_overlap']
            base=[arrays[label+'_'+name] for name in names]
            # Different offsets/scales change both the trace and residuals;
            # evaluating the original routines again preserves an exact oracle.
            second=[a*(.75+.05*i)+.001*(i+1) for i,a in enumerate(base)]
            energy=arrays[label+'_energy_seeds'];seeds=np.array([energy,energy*np.linspace(.8,1.2,12)])
            expected=np.array([reference.evaluate(exe,tmp,num,ptr,seeds[i],*state,overlap)[0]
                               for i,state in enumerate((base,second))])
            source=np.array([np.concatenate([a.ravel(order='F') for a in state]) for state in (base,second)])
            expected_residual=np.array([(state[0]-state[1]).ravel(order='F') for state in (base,second)])
            for name,a in (('source',source),('seeds',seeds),('expected',expected),('h_residual',expected_residual)):
                retained[label+'_'+name]=a
            p=Problem(n,spins,len(num),0)
            if not candidate:
                report['checks'].append(dict(fixture=label,cycles=2,passed=True));continue
            for graph in (False,True):
                actual,residual,statuses,info=execute(p,seeds,num,ptr,overlap,source,graph)
                assert statuses==[0,0],(label,graph,statuses)
                np.testing.assert_allclose(actual,expected,atol=1.e-10,rtol=1.e-11)
                np.testing.assert_array_equal(residual,expected_residual)
                report['checks'].append(dict(fixture=label,graph=graph,iterations=info.iterations,
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,workspace_bytes=info.workspace_bytes,
                    max_absolute_error=float(np.max(np.abs(actual-expected))),passed=True))
        if candidate:
            # Reuse the last (8-component) fixture. Mutable-input failures must
            # preserve output, then recover on the next fully device-selected state.
            matrix=n*spins;edm=n*min(spins,4)
            for invalid in ('shape','nan_seed','negative_temperature','nan_h','nan_hold','nan_density',
                            'nan_old_density','nan_energy','nan_old_energy','nan_overlap','negative_ptr',
                            'excess_num','negative_num','overflow'):
                inp=source.copy();energies=seeds.copy();ov=overlap.copy();counts=num.copy();pointers=ptr.copy()
                corrupt=int(invalid=='shape');both=False;status=2
                if invalid=='shape': status=1;both=True
                elif invalid=='nan_seed': energies[0,0]=np.nan
                elif invalid=='negative_temperature': energies[0,10]=-1.;status=1
                elif invalid in ('nan_h','nan_hold','nan_density','nan_old_density','nan_energy','nan_old_energy'):
                    offset=dict(nan_h=0,nan_hold=matrix,nan_density=2*matrix,nan_old_density=3*matrix,
                                nan_energy=4*matrix,nan_old_energy=4*matrix+edm)[invalid]
                    inp[0,offset]=np.nan
                elif invalid=='nan_overlap': ov[ptr[0]]=np.nan;both=True
                elif invalid=='negative_ptr': pointers[0]=-1;both=True
                elif invalid=='excess_num': counts[0]=n+1;both=True
                elif invalid=='negative_num': counts[0]=-1;both=True
                else: inp[0,0]=np.finfo(float).max;inp[0,2*matrix]=np.finfo(float).max
                actual,residual,statuses,info=execute(p,energies,counts,pointers,ov,inp,True,corrupt)
                assert statuses==[status,status if both else 0],(invalid,statuses)
                np.testing.assert_array_equal(actual[:2 if both else 1],0.)
                np.testing.assert_array_equal(residual[:2 if both else 1],123.)
                if not both:
                    np.testing.assert_allclose(actual[1],expected[1],atol=1.e-10,rtol=1.e-11)
                    np.testing.assert_array_equal(residual[1],expected_residual[1])
                report['checks'].append(dict(fixture=invalid,graph=True,statuses=statuses,
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            # Nonfinite unused overlap capacity must not enter sparse charge sums.
            ov=overlap.copy();used=np.zeros(n,dtype=bool)
            for start,count in zip(ptr,num): used[start:start+count]=True
            ov[~used]=np.nan
            actual,residual,statuses,info=execute(p,seeds,num,ptr,ov,source,True)
            assert statuses==[0,0]
            np.testing.assert_allclose(actual,expected,atol=1.e-10,rtol=1.e-11)
            np.testing.assert_array_equal(residual,expected_residual)
            report['checks'].append(dict(fixture='unused_overlap_capacity',graph=True,statuses=statuses,
                graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            report['gpu_validation']=True
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
    report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact),passed=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"SCF scalar scenarios: {len(report['checks'])}; passed=True; GPU validation={report['gpu_validation']}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args();run(args.upstream,args.build,args.output,args.library)
