#!/usr/bin/env python3
"""Check device Hamiltonian seeds/SOC against literal pinned upstream blocks.

Two density/grid states are staged upfront and selected on device. Captured
execution contains kernel nodes only. This is component evidence, not a full
setup_hamiltonian, ownership or SCF qualification.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import check_hamiltonian_seeds as reference


class Problem(ct.Structure):
    _fields_=[(name,ct.c_int) for name in ('capacity','spins','soc','reserved')]


class Result(ct.Structure):
    _fields_=[(name,ct.c_double) for name in ('kinetic','nonlocal_energy','soc_energy')]+[
        (name,ct.c_int) for name in ('status','active','stage','prepared_soc')]


class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_size_t),('graph_nodes',ct.c_int),('forbidden_nodes',ct.c_int),('iterations',ct.c_int)]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(upstream,build,output,library=None):
    output.parent.mkdir(parents=True,exist_ok=True)
    oracle=output.with_name('hamiltonian-seeds-reference.json');reference.run(upstream,build,oracle)
    ref=json.loads(oracle.read_text());array_path=oracle.parent/ref['arrays_file']
    assert sha(array_path)==ref['arrays_sha256'];arrays=np.load(array_path)
    report=dict(reference=ref['reference'],gpu_validation=None,oracle_file=oracle.name,
        oracle_sha256=sha(oracle),oracle_arrays_sha256=sha(array_path),
        fixture_script_sha256=sha(Path(__file__)),checks=[])
    candidate=None
    if library:
        candidate=ct.CDLL(str(library.resolve())).remag_check_hamiltonian_seeds
        candidate.argtypes=[ct.c_int]*4+[ct.POINTER(Problem),ct.c_void_p,ct.c_void_p,
            ct.POINTER(Result),ct.c_int,ct.POINTER(Check)]
        candidate.restype=ct.c_int;report['native_library_sha256']=sha(library)
    def execute(n,s,problems,source,graph,mode=0):
        source=np.ascontiguousarray(source);cycles=len(source)
        out=np.zeros((cycles,2,n*s));records=(Result*(2*cycles))();info=Check()
        code=candidate(n,s,cycles,mode,problems,source.ctypes.data,out.ctypes.data,records,int(graph),ct.byref(info))
        assert code==0,(code,mode,graph)
        assert info.iterations==cycles and info.forbidden_nodes==0
        if graph: assert info.graph_nodes==12+{1:-4,2:3,3:1}.get(mode,0)
        energies=np.array([[r.kinetic,r.nonlocal_energy,r.soc_energy] for r in records]).reshape(cycles,2,3)
        status=np.array([r.status for r in records]).reshape(cycles,2)
        stage=np.array([r.stage for r in records]).reshape(cycles,2)
        # Retain the last invocation even when a following assertion fails.
        np.savez_compressed(output.with_name('hamiltonian-seeds-last-call.npz'),source=source,
                            matrices=out,energies=energies,status=status,stage=stage)
        return out,energies,status,stage,info
    names=('kinetic','nonlocal','onsite','offsite','density','grid','h')
    retained={};fixtures={}
    with tempfile.TemporaryDirectory(prefix='remag-hamiltonian-sequences-') as directory:
        tmp=Path(directory);exe,_,_=reference.build_reference(upstream,build,tmp)
        for record in ref['checks']:
            label=record['fixture'];n=record['capacity'];s=record['spin_components'];soc=record['soc']
            base=[arrays[label+'_'+name] for name in names]
            second=[a.copy() for a in base]
            second[4]=second[4]*.75+.002
            second[5]=second[5]*1.25
            second[6]=second[6]*.5-.001
            expected=[];energy=[];source=[]
            for kin,vkb,on,off,d,grid,h in (base,second):
                old,seed,e,final,_=reference.evaluate(exe,tmp,soc,h,d,kin,vkb,on,off,grid)
                np.testing.assert_array_equal(old,h)
                expected.append([seed.ravel(order='F'),final.ravel(order='F')]);energy.append([e,e])
                source.append(np.concatenate([np.ascontiguousarray(a.ravel(order='F')).view(np.float64)
                    for a in (kin,vkb,on,off,d,grid,h)]))
            source=np.array(source);expected=np.array(expected);energy=np.array(energy)
            problems=(Problem*2)(Problem(n,s,soc,0),Problem(n,s,soc,0))
            fixtures[s,soc]=(n,s,soc,source,expected,energy)
            for name,a in (('source',source),('expected',expected),('energies',energy)):
                retained[label+'_'+name]=a
            if not candidate:
                report['checks'].append(dict(fixture=label,cycles=2,passed=True));continue
            for graph in (False,True):
                out,e,status,stage,info=execute(n,s,problems,source,graph)
                np.testing.assert_array_equal(status,0)
                np.testing.assert_array_equal(stage,[[1,2],[1,2]])
                np.testing.assert_array_equal(out,expected)
                np.testing.assert_allclose(e,energy,atol=1.e-10,rtol=1.e-11)
                report['checks'].append(dict(fixture=label,graph=graph,iterations=info.iterations,
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,workspace_bytes=info.workspace_bytes,
                    max_energy_error=float(np.max(np.abs(e-energy))),matrices_exact=True,passed=True))
        if candidate:
            for invalid in ('capacity','spins','soc','reserved','nan_kinetic','nan_nonlocal','nan_density',
                            'nan_onsite','nan_offsite','nan_grid','overflow_seed','overflow_energy'):
                n,s,soc,source,expected,energy=fixtures[(8,1 if invalid=='nan_onsite' else 2)]
                inp=source.copy();problems=(Problem*2)(Problem(n,s,soc,0),Problem(n,s,soc,0));status_expected=2
                if invalid in ('capacity','spins','soc','reserved'):
                    setattr(problems[0],invalid,dict(capacity=n+1,spins=4,soc=0,reserved=1)[invalid]);status_expected=1
                elif invalid.startswith('nan_'):
                    index=dict(nan_kinetic=0,nan_nonlocal=n,nan_onsite=2*n,nan_offsite=8*n,
                               nan_density=16*n,nan_grid=16*n+n*s)[invalid]
                    inp[0,index]=np.nan
                elif invalid=='overflow_seed': inp[0,:2*n]=np.finfo(float).max
                else: inp[0,0]=np.finfo(float).max;inp[0,16*n]=np.finfo(float).max
                out,e,status,stage,info=execute(n,s,problems,inp,True)
                np.testing.assert_array_equal(status,[[0 if invalid=='nan_grid' else status_expected,status_expected],[0,0]])
                np.testing.assert_array_equal(out[1],expected[1])
                np.testing.assert_allclose(e[1],energy[1],atol=1.e-10,rtol=1.e-11)
                if invalid=='nan_grid':
                    np.testing.assert_array_equal(out[0,0],expected[0,0])
                    np.testing.assert_array_equal(out[0,1],expected[0,0]+inp[0,16*n+n*s:16*n+2*n*s])
                    np.testing.assert_array_equal(stage[0],[1,1])
                else:
                    np.testing.assert_array_equal(out[0],np.tile(inp[0,-n*s:],(2,1)))
                    np.testing.assert_array_equal(stage[0],0)
                report['checks'].append(dict(fixture=invalid,graph=True,statuses=status.tolist(),
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            for mode,label,soc in ((1,'missing_prepare',2),(2,'duplicate_finish',2),(3,'changed_soc',2),
                                    (4,'missing_onsite',1),(5,'missing_offsite',2)):
                n,s,soc,source,expected,energy=fixtures[8,soc]
                problems=(Problem*2)(Problem(n,s,soc,0),Problem(n,s,soc,0))
                out,e,status,stage,info=execute(n,s,problems,source,True,mode)
                np.testing.assert_array_equal(status,[[int(mode in (4,5)),1]]*2)
                if mode in (1,4,5):
                    np.testing.assert_array_equal(out,np.repeat(source[:,-n*s:][:,None,:],2,axis=1))
                    np.testing.assert_array_equal(stage,0)
                elif mode==2:
                    np.testing.assert_array_equal(out,expected);np.testing.assert_array_equal(stage,[[1,2]]*2)
                else:
                    np.testing.assert_array_equal(out[:,0],expected[:,0])
                    np.testing.assert_array_equal(out[:,1],expected[:,0]+source[:,16*n+n*s:16*n+2*n*s])
                    np.testing.assert_array_equal(stage,1)
                report['checks'].append(dict(fixture=label,graph=True,statuses=status.tolist(),
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            for (s,soc),mode in (((1,0),0),((2,0),0),((4,0),0),((8,1),5),((8,2),4)):
                n,s,soc,source,expected,energy=fixtures[s,soc];inp=source.copy()
                if soc!=1: inp[:,2*n:8*n]=np.nan
                if soc!=2: inp[:,8*n:16*n]=np.nan
                problems=(Problem*2)(Problem(n,s,soc,0),Problem(n,s,soc,0))
                out,e,status,stage,info=execute(n,s,problems,inp,True,mode)
                np.testing.assert_array_equal(status,0);np.testing.assert_array_equal(out,expected)
                np.testing.assert_allclose(e,energy,atol=1.e-10,rtol=1.e-11)
                report['checks'].append(dict(fixture=f'unused_soc_inputs_spin{s}_soc{soc}',graph=True,
                    graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            report['gpu_validation']=True
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
    report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact),passed=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Hamiltonian seed scenarios: {len(report['checks'])}; passed=True; GPU validation={report['gpu_validation']}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    parser.add_argument('--build',type=Path,default=Path('build/upstream'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args();run(args.upstream,args.build,args.output,args.library)
