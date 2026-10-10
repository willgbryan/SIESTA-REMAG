#!/usr/bin/env python3
"""Check device SCF decisions against complete pristine convergence sequences."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
import check_scf_convergence as reference


class Problem(ct.Structure):
    _fields_=[(n,ct.c_int) for n in ('criteria','minimum','maximum','reserved')]+[
        ('tolerances',ct.c_double*5),('epoch',ct.c_uint64)]


class Scalars(ct.Structure):
    _fields_=[('values',ct.c_double*13),('status',ct.c_int),('active',ct.c_int)]


class Result(ct.Structure):
    _fields_=[(n,ct.c_int) for n in ('status','iteration','criteria_satisfied','converged','mix','continue_loop','exhausted','energy_flags')]


class Check(ct.Structure):
    _fields_=[('workspace_bytes',ct.c_size_t),('graph_nodes',ct.c_int),('forbidden_nodes',ct.c_int),('iterations',ct.c_int)]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(upstream,build,output,library=None):
    output.parent.mkdir(parents=True,exist_ok=True)
    oracle=output.with_name('scf-convergence-reference.json');reference.run(upstream,build,oracle)
    ref=json.loads(oracle.read_text());arrays=oracle.parent/ref['arrays_file']
    assert sha(arrays)==ref['arrays_sha256'];arrays=np.load(arrays)
    report=dict(reference=ref['reference'],gpu_validation=None,checks=[],fixture_script_sha256=sha(Path(__file__)),
        oracle_file=oracle.name,oracle_sha256=sha(oracle),oracle_arrays_sha256=ref['arrays_sha256'])
    candidate=None
    if library:
        candidate=ct.CDLL(str(library.resolve())).remag_check_scf_convergence
        candidate.argtypes=[ct.c_int,ct.POINTER(Problem),ct.POINTER(Scalars),ct.POINTER(Result),ct.c_int,ct.POINTER(Check)]
        candidate.restype=ct.c_int;report['native_library_sha256']=sha(library)
    def execute(problems,scalars,graph):
        records=(Result*len(scalars))();info=Check()
        assert candidate(len(scalars),problems,scalars,records,int(graph),ct.byref(info))==0
        assert info.iterations==len(scalars) and info.forbidden_nodes==0
        if graph: assert info.graph_nodes==3
        values=np.array([[getattr(r,name) for name,_ in Result._fields_] for r in records])
        return values,info
    def inputs(values,mask,minimum,maximum,reset):
        problems=(Problem*len(values))();scalars=(Scalars*len(values))()
        for k,v in enumerate(values,1):
            problems[k-1]=Problem(mask,minimum,maximum,0,(ct.c_double*5)(*arrays['tolerances']),1+int(reset>0 and k>=reset))
            scalars[k-1].values[6:9]=v[:3];scalars[k-1].values[3:5]=v[3:]
        return problems,scalars
    retained={}
    try:
        for check in ref['checks']:
            label=check['fixture'];values=arrays[label+'_values'];expected=arrays[label+'_decisions'];maximum=len(values)
            p,s=inputs(values,check['criteria_mask'],check['minimum_iterations'],maximum,check['reset_before'])
            flags=expected[:,2]+2*expected[:,3]
            wanted=np.column_stack((np.zeros(len(values),int),expected[:,4],expected[:,0],expected[:,1],1-expected[:,1],
                                    (1-expected[:,1])*(expected[:,4]<maximum),(1-expected[:,1])*(expected[:,4]==maximum),flags))
            retained[label+'_expected']=wanted
            if not candidate:
                report['checks'].append(dict(fixture=label,steps=len(values),passed=True));continue
            for graph in (False,True):
                report['active_fixture']=dict(fixture=label,graph=graph)
                actual,info=execute(p,s,graph)
                retained[label+f'_actual_graph{int(graph)}']=actual
                np.testing.assert_array_equal(actual,wanted)
                report['checks'].append(dict(fixture=label,graph=graph,iterations=info.iterations,graph_nodes=info.graph_nodes,
                    forbidden_nodes=info.forbidden_nodes,workspace_bytes=info.workspace_bytes,decisions_exact=True,passed=True))
        if candidate:
            for invalid in ('no_criteria','unknown_criteria','negative_minimum','zero_maximum','minimum_above_maximum',
                            'reserved','nan_tolerance','negative_tolerance','zero_epoch','scalar_error','nonfinite_scalar',
                            'past_maximum'):
                p,s=inputs(np.zeros((4,5)),3,0,4,3)
                expected=[1,1,0,0]
                if invalid=='no_criteria': p[0].criteria=0
                elif invalid=='unknown_criteria': p[0].criteria=32
                elif invalid=='negative_minimum': p[0].minimum=-1
                elif invalid=='zero_maximum': p[0].maximum=0
                elif invalid=='minimum_above_maximum': p[0].minimum=5
                elif invalid=='reserved': p[0].reserved=1
                elif invalid=='nan_tolerance': p[0].tolerances[0]=np.nan
                elif invalid=='negative_tolerance': p[0].tolerances[0]=-1.
                elif invalid=='zero_epoch': p[0].epoch=0;expected=[1,0,0,0]
                elif invalid=='scalar_error': s[0].status=2;expected=[2,2,0,0]
                elif invalid=='nonfinite_scalar': s[0].values[0]=np.nan;expected=[2,2,0,0]
                else:
                    for item in p: item.maximum=1
                    expected=[0,3,0,3]
                for graph in (False,True):
                    report['active_fixture']=dict(fixture=invalid,graph=graph)
                    actual,info=execute(p,s,graph)
                    retained[invalid+f'_actual_graph{int(graph)}']=actual
                    retained[invalid+'_expected_status']=np.array(expected)
                    np.testing.assert_array_equal(actual[:,0],expected)
                    for i,expected_status in enumerate(expected):
                        if expected_status: np.testing.assert_array_equal(actual[i,2:7],0)
                        else:
                            assert actual[i,3]==1 and actual[i,4]==0 and actual[i,5]==0 and actual[i,6]==0
                    report['checks'].append(dict(fixture=invalid,graph=graph,statuses=actual[:,0].tolist(),
                        graph_nodes=info.graph_nodes,forbidden_nodes=info.forbidden_nodes,passed=True))
            report['gpu_validation']=True
        report.update(passed=True)
        report.pop('active_fixture',None)
    except Exception as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
        report.update(arrays_file=artifact.name,arrays_sha256=sha(artifact))
        output.write_text(json.dumps(report,indent=2)+'\n')
    print(f"SCF convergence scenarios: {len(report['checks'])}; passed=True; GPU validation={report['gpu_validation']}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,default=Path('upstream/siesta'))
    p.add_argument('--build',type=Path,default=Path('build/upstream'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path)
    args=p.parse_args();run(args.upstream,args.build,args.output,args.library)
