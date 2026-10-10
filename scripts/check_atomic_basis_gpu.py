#!/usr/bin/env python3
"""GPU core/neutral radial values, gradients and compact geometry vs pristine SIESTA.

Checks outer selection independently of zero-valued results, original reord
indices, ordinary/captured replay, strided tails, sticky errors and bad inputs.
This isolated component does not remove material uploads or qualify ownership.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import tempfile
import numpy as np
from check_atomic_basis import run as build_oracle,build_reference,oracle,flatten,species_fixtures,sha

ATOL=RTOL=1e-10
FIELDS=('atomic_number','radial_index','radial_parameters','values','second','query_species','xyz')
GEOMETRY_FIELDS=('pair_atom','pair_offset','pair_cluster','atom_species','unit_atom','offset','subpoint','atom_offset')
INDEX_FAULTS=('species','sample_count','sample_offset','cutoff','delta')
VALUE_FAULTS=('xyz','values','second','range')
GEOMETRY_FAULTS=GEOMETRY_FIELDS
HOST_GEOMETRY_FAULTS=('nsm','mesh','unit_atoms','points')


class Info(ct.Structure):
    _fields_=[(name,ct.c_size_t) for name in ('allocated','staged','exported','poisoned')]+[(name,ct.c_int) for name in ('graph_nodes','forbidden_nodes','stage')]


def geometry_fixtures():
    species=list(species_fixtures())[-1][1];rng=np.random.default_rng(777291)
    atom_species=np.array([1,2,3,4,1,2],np.int32);unit_atom=np.array([2,1,3,2,1,3],np.int32)
    offset=rng.normal(size=(29,3))*1.2;offset[0]=0.;offset[1]=[3.4,0.,0.]
    atom_offset=np.array([[0.,0.,0.],[.1,-.2,.3],[0.,0.,0.],[.4,-.3,.2],[1.,-1.,2.],[-1.,1.,-2.]])
    for nsm,mesh,pairs in ((1,[5,4,3],137),(2,[8,6,4],137),(3,[9,6,3],137),(4,[8,8,4],137),(2,[8,6,4],16385)):
        nsp=nsm**3;points=int(np.prod(mesh));clusters=points//nsp
        subpoint=rng.normal(size=(nsp,3))*.07;subpoint[0]=0.
        pa=np.arange(pairs,dtype=np.int32)%len(atom_species)+1
        po=(np.arange(pairs,dtype=np.int32)*7)%len(offset)+1
        pc=(np.arange(pairs,dtype=np.int32)*11)%clusters+1
        fields=dict(flatten(species),pair_atom=pa,pair_offset=po,pair_cluster=pc,atom_species=atom_species,unit_atom=unit_atom,
            offset=offset,subpoint=subpoint,atom_offset=atom_offset)
        xyz=((offset[po-1,None,:]+subpoint[None,:,:])-atom_offset[pa-1,None,:]).reshape(-1,3)
        qs=np.repeat(atom_species[pa-1]-1,nsp)
        shape=np.array([pairs,nsp,len(atom_species),len(offset),3,points,nsm,*mesh],np.int32)
        yield f'geometry_nsm{nsm}_pairs{pairs}',species,fields,shape,qs,xyz


def run(upstream,build,library,output):
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    reference=output.with_name('reference.json');build_oracle(upstream,build,reference)
    oracle_report=json.loads(reference.read_text());data=np.load(reference.with_suffix('.npz'))
    lib=ct.CDLL(str(library.resolve()));query=lib.remag_check_atomic_queries;geometry=lib.remag_check_atomic_geometry
    query.argtypes=[ct.c_int]*8+[ct.c_void_p]*9+[ct.POINTER(ct.c_int),ct.POINTER(Info)];query.restype=ct.c_int
    geometry.argtypes=[ct.c_int]*3+[ct.c_void_p]+[ct.c_int]*3+[ct.c_void_p]*15+[ct.POINTER(ct.c_int),ct.POINTER(Info)];geometry.restype=ct.c_int
    checks=[];retained={};geometry_arrays={};geometry_rows=[]

    def record(label,kind,filter,captured,cycles,fault,native,status,info,wanted,arrays,result,integers,expected,expected_int,shape=None):
        name=f'{label}_captured{int(captured)}_cycles{cycles}'+('_'+fault if fault else '')
        staged=sum(a.nbytes for a in arrays)+4
        allocated=staged+result.nbytes+integers.nbytes+sum(a.dtype.itemsize for a in arrays if not a.size)
        valid=native==0 and status.value==wanted and info.stage==4 and info.graph_nodes==int(captured) and not info.forbidden_nodes
        valid=valid and info.staged==staged and info.allocated==allocated and info.exported==result.nbytes+integers.nbytes+4
        valid=valid and info.poisoned==cycles*(result.nbytes+integers.nbytes)
        if fault in ('kind',*HOST_GEOMETRY_FAULTS):valid=native==-1 and info.stage==3 and info.exported==0
        row=dict(scenario=name,fixture=label,kind=kind,mesh_filter=filter,captured=captured,cycles=cycles,fault=fault,
            native_return=native,status=status.value,expected_status=wanted,shape=shape.tolist() if shape is not None else None,
            expected_staged=staged,expected_allocated=allocated,expected_exported=result.nbytes+integers.nbytes+4,
            **{key:getattr(info,key) for key,_ in info._fields_})
        if fault is None or fault=='sticky':
            error=np.abs(result-expected);scaled=error/(ATOL+RTOL*np.abs(expected))
            row.update(max_absolute_error=float(error.max()),max_scaled_error=float(scaled.max()),integers_exact=bool(np.array_equal(integers,expected_int)))
            valid=valid and np.isfinite(result).all() and bool(np.all(scaled<=1.)) and row['integers_exact']
            if shape is not None:valid=valid and np.array_equal(result[4*len(integers)//3:],expected[4*len(integers)//3:])
        row['passed']=bool(valid);checks.append(row);retained[name+'_real']=result;retained[name+'_int']=integers
        if not valid:print('FAILED',row,flush=True)

    def evaluate(row,captured,cycles,fault=None):
        label=row['fixture'];arrays=[data[label+'_'+key].copy(order='C') for key in FIELDS]
        z,index,params,values,second,qs,xyz=arrays;initial=7 if fault=='sticky' else 0;wanted=initial;kind=row['kind']
        if fault in (*INDEX_FAULTS,*VALUE_FAULTS):
            qs[:]=0;xyz[:]=[.7,.8,.9];wanted=1 if fault in INDEX_FAULTS else 2
            if fault=='species':qs[:]=len(z)
            elif fault=='sample_count':index[:,0]=1
            elif fault=='sample_offset':index[:,1]=len(values)
            elif fault=='cutoff':params[:,0]=np.nan
            elif fault=='delta':params[:,1]=0.
            elif fault=='xyz':xyz[:]=np.nan
            elif fault=='values':values[:]=np.nan
            elif fault=='second':second[:]=np.nan
            elif fault=='range':params[:,1]=1.e-6
        if fault=='kind':kind=0;wanted=-1
        result=np.full((len(qs),4),np.nan);active=np.full(len(qs),-1,np.int32);status=ct.c_int(-1);info=Info()
        native=query(len(z),len(values),kind,row['mesh_filter'],len(qs),int(captured),cycles,initial,
            *[a.ctypes.data for a in arrays],result.ctypes.data,active.ctypes.data,ct.byref(status),ct.byref(info))
        record(label,kind,row['mesh_filter'],captured,cycles,fault,native,status,info,wanted,arrays,result,active,
            data[label+'_expected'],data[label+'_active'])

    for row in oracle_report['checks']:
        for captured in (False,True):
            for cycles in (1,3):evaluate(row,captured,cycles)
    for kind in (1,2):
        row=next(r for r in oracle_report['checks'] if r['source']=='smooth' and r['kind']==kind and r['mesh_filter']==1)
        for fault in (*INDEX_FAULTS,*VALUE_FAULTS,'sticky'):
            for captured in (False,True):evaluate(row,captured,3,fault)
        evaluate(row,False,1,'kind')
    with tempfile.TemporaryDirectory(prefix='remag-atomic-geometry-') as temporary:
        tmp=Path(temporary);exe=build_reference(build,tmp)
        for label,species,fields,shape,qs,xyz in geometry_fixtures():
            n=len(qs);nsp=int(shape[1]);clustered=np.repeat((fields['pair_cluster']-1)*nsp,nsp)+np.tile(np.arange(nsp),int(shape[0]))
            unit=np.repeat(fields['unit_atom'][fields['pair_atom']-1],nsp)
            for kind in (1,2):
                name=f'{label}_kind{kind}'
                value,active,inverse,input_hash=oracle(exe,tmp,species,qs,xyz,kind,1,shape[7:],int(shape[6]))
                expected=np.r_[value[:,0],value[:,1:].ravel(),xyz.ravel()]
                expected_int=np.r_[active,clustered+1 if kind==1 else inverse[clustered],unit].astype(np.int32)
                for key,array in dict(fields,shape=shape,query_species=qs,xyz=xyz,expected=expected,expected_int=expected_int,inverse=inverse).items():geometry_arrays[name+'_'+key]=array
                geometry_rows.append(dict(fixture=name,queries=n,kind=kind,shape=shape.tolist(),input_sha256=input_hash))
                scenarios=[(None,c,cycles) for c in (False,True) for cycles in (1,3)]
                if label=='geometry_nsm2_pairs137':
                    scenarios.extend((fault,c,3) for fault in (*GEOMETRY_FAULTS,'sticky') for c in (False,True))
                    scenarios.extend((fault,False,1) for fault in HOST_GEOMETRY_FAULTS)
                for fault,captured,cycles in scenarios:
                    arrays=[fields[key].copy(order='C') for key in (*FIELDS[:5],*GEOMETRY_FIELDS)];dims=shape.copy()
                    initial=7 if fault=='sticky' else 0;wanted=initial
                    if fault in GEOMETRY_FAULTS:
                        arr=arrays[5+GEOMETRY_FIELDS.index(fault)];real=arr.dtype==np.float64;arr[:]=np.nan if real else 0;wanted=2 if real else 1
                    if fault in HOST_GEOMETRY_FAULTS:
                        dims[dict(nsm=6,mesh=7,unit_atoms=4,points=5)[fault]]=0;wanted=-1
                    result=np.full(7*n,np.nan);indices=np.full(3*n,-1,np.int32);status=ct.c_int(-1);info=Info()
                    native=geometry(len(fields['atomic_number']),len(fields['values']),kind,dims.ctypes.data,int(captured),cycles,initial,
                        *[a.ctypes.data for a in arrays],result.ctypes.data,indices.ctypes.data,ct.byref(status),ct.byref(info))
                    record(name,kind,1,captured,cycles,fault,native,status,info,wanted,arrays,result,indices,expected,expected_int,dims)
    geometry_file=output.with_name('geometry-reference.npz');np.savez_compressed(geometry_file,**geometry_arrays)
    artifact=output.with_suffix('.npz');np.savez_compressed(artifact,**retained)
    report=dict(scope=__doc__,passed=all(c['passed'] for c in checks),checks=checks,atol=ATOL,rtol=RTOL,
        oracle_file=reference.name,oracle_sha256=sha(reference),oracle_arrays_sha256=sha(reference.with_suffix('.npz')),
        geometry_file=geometry_file.name,geometry_sha256=sha(geometry_file),geometry_checks=geometry_rows,
        native_library_sha256=sha(library),checker_sha256=sha(Path(__file__)),output_file=artifact.name,output_sha256=sha(artifact))
    output.write_text(json.dumps(report,indent=2)+'\n');print('Atomic basis CUDA scenarios:',len(checks),'passed=',report['passed'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','build','library','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.upstream.resolve(),a.build.resolve(),a.library,a.output)
