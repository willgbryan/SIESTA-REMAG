#!/usr/bin/env python3
"""Audit full retained atomic-radial outputs, geometry, masks and exact input provenance.

Optional --reference-build independently replays the retained input bytes with
the pristine local SIESTA library and archived Fortran wrapper. No material
ownership, production substitution or independent CUDA trace is claimed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def oracle_input(fields,qs,xyz,kind,mesh_filter,shape=(1,1,1),nsm=1):
    chunks=[np.array([len(fields['atomic_number']),len(qs),kind,mesh_filter,*shape,nsm],np.int32).tobytes()]
    for s,z in enumerate(fields['atomic_number']):
        chunks.append(np.array([z],np.int32).tobytes())
        for i in (2*s,2*s+1):
            n,offset=fields['radial_index'][i]
            chunks.extend((np.array([n],np.int32).tobytes(),fields['radial_parameters'][i].tobytes(),
                fields['values'][offset:offset+n].tobytes(),fields['second'][offset:offset+n].tobytes()))
    chunks.extend(((qs+1).tobytes(),xyz.tobytes()));return b''.join(chunks)


def replay(build,wrapper,inputs,expected,output):
    checks=[];retained={}
    with tempfile.TemporaryDirectory(prefix='remag-atomic-replay-') as temporary:
        tmp=Path(temporary);src=build.resolve()/'Src';source=tmp/'reference.f90';shutil.copyfile(wrapper,source);exe=tmp/'reference'
        link=shlex.split((src/'CMakeFiles/SIESTA.siesta.dir/link.txt').read_text());libs=link[next(i for i,v in enumerate(link) if Path(v).name=='libsiesta.a'):]
        flags=(src/'CMakeFiles/SIESTA.libsiesta.dir/flags.make').read_text()
        includes=next(v.split('=',1)[1] for v in flags.splitlines() if v.startswith('Fortran_INCLUDES ='))
        subprocess.run([link[0],'-O2',*shlex.split(includes),str(source),'-o',str(exe),*libs],cwd=src,check=True,capture_output=True)
        binary=sha(exe)
        for label,raw in inputs.items():
            inp=tmp/'input.bin';out=tmp/'output.bin';inp.write_bytes(raw)
            process=subprocess.run([str(exe),str(inp),str(out)],cwd=tmp,capture_output=True,text=True,timeout=60)
            if process.returncode:raise RuntimeError(process.stdout+process.stderr)
            wanted,wanted_int=expected[label];n=len(wanted);blob=out.read_bytes()
            values=np.frombuffer(blob,np.float64,count=4*n).reshape(n,4).copy()
            indices=np.frombuffer(blob,np.int32,offset=32*n).copy()
            delta=np.abs(values-wanted);scaled=delta/(1e-10+1e-10*np.abs(wanted))
            passed=np.isfinite(values).all() and np.all(scaled<=1.) and np.array_equal(indices,wanted_int)
            checks.append(dict(fixture=label,passed=bool(passed),max_absolute_error=float(delta.max()),max_scaled_error=float(scaled.max()),input_sha256=sha(inp)))
            retained[label+'_real']=values;retained[label+'_int']=indices
    arrays=output/'atomic-basis-cpu-replay.npz';np.savez_compressed(arrays,**retained)
    return dict(passed=all(v['passed'] for v in checks),checks=checks,reference_library_sha256=sha(build/'Src/libsiesta.a'),
        wrapper_sha256=sha(wrapper),executable_sha256=binary,arrays_file=arrays.name,arrays_sha256=sha(arrays))


def verify(root,reference_build=None):
    folder=root/'atomic-basis-check';path=folder/'check.json';report=json.loads(path.read_text())
    oracle=folder/report['oracle_file'];reference=json.loads(oracle.read_text())
    data=np.load(folder/reference['arrays_file']);geo=np.load(folder/report['geometry_file']);outputs=np.load(folder/report['output_file'])
    errors=[];checks=[];inputs={};cpu_expected={};fixtures={v['fixture']:v for v in reference['checks']};geometries={v['fixture']:v for v in report['geometry_checks']}
    basis=('atomic_number','radial_index','radial_parameters','values','second')
    fields=(*basis,'query_species','xyz')
    geometry_fields=('pair_atom','pair_offset','pair_cluster','atom_species','unit_atom','offset','subpoint','atom_offset')
    index_faults=('species','sample_count','sample_offset','cutoff','delta');value_faults=('xyz','values','second','range')
    host_faults=('kind','nsm','mesh','unit_atoms','points')
    names=('smooth','empty','no_samples','two_samples','endpoint','tiny_cutoffs','stride')
    required={f'{name}_kind{k}_mesh{m}' for name in names for k in (1,2) for m in (0,1)}
    geo_required={f'geometry_nsm{n}_pairs{p}_kind{k}' for n,p in ((1,137),(2,137),(3,137),(4,137),(2,16385)) for k in (1,2)}
    if set(fixtures)!=required or set(geometries)!=geo_required:errors.append('Required fixture coverage differs')
    if report.get('atol')!=1e-10 or report.get('rtol')!=1e-10:errors.append('Tolerance differs')
    with tarfile.open(root/'implementation.tar.gz') as archive:
        for script,digest in (('check_atomic_basis.py',reference['script_sha256']),('check_atomic_basis_gpu.py',report['checker_sha256'])):
            if hashlib.sha256(archive.extractfile('scripts/'+script).read()).hexdigest()!=digest:errors.append('Archived source differs: '+script)
    for label,row in fixtures.items():
        f={k:data[label+'_'+k] for k in fields};qs=f['query_species'];xyz=f['xyz'];n=len(qs)
        raw=oracle_input(f,qs,xyz,row['kind'],row['mesh_filter']);inputs[label]=raw
        if hashlib.sha256(raw).hexdigest()!=row['input_sha256']:errors.append('Oracle input differs: '+label)
        if n!=row['queries'] or ('stride_' in label and n!=131073):errors.append('Query coverage differs: '+label)
        expected=data[label+'_expected'];mask=data[label+'_active'];inverse=data[label+'_inverse']
        if expected.shape!=(n,4) or mask.shape!=(n,) or not np.isfinite(expected).all():errors.append('Oracle output shape/values differ: '+label)
        if row['mesh_filter']:
            cutoff=f['radial_parameters'][2*qs+row['kind']-1,0]
            radius=np.sqrt(xyz[:,0]*xyz[:,0]+xyz[:,1]*xyz[:,1]+xyz[:,2]*xyz[:,2])
            outer=(radius<cutoff)&((cutoff>1e-12) if row['kind']==1 else True)
        else:outer=np.ones(n,dtype=bool)
        if not np.array_equal(mask,outer.astype(np.int32)):errors.append('Outer mesh selection differs: '+label)
        if not np.all(expected[(mask==0)|(f['atomic_number'][qs]<0)]==0.):errors.append('Inactive/floating oracle output not zero: '+label)
        cpu_expected[label]=(expected,np.r_[mask,inverse])
    for label,row in geometries.items():
        f={k:geo[label+'_'+k] for k in (*basis,*geometry_fields)};shape=geo[label+'_shape'];pairs,nsp,na,noff,nua,points,nsm,*mesh=shape.tolist()
        atom=f['pair_atom']-1;offset=f['pair_offset']-1
        xyz=((f['offset'][offset,None,:]+f['subpoint'][None,:,:])-f['atom_offset'][atom,None,:]).reshape(-1,3)
        qs=np.repeat(f['atom_species'][atom]-1,nsp);n=len(qs)
        raw=oracle_input(f,qs,xyz,row['kind'],1,mesh,nsm);inputs[label]=raw
        if hashlib.sha256(raw).hexdigest()!=row['input_sha256']:errors.append('Geometry input differs: '+label)
        if not np.array_equal(qs,geo[label+'_query_species']) or not np.array_equal(xyz,geo[label+'_xyz']):errors.append('Geometry coordinates/species differ: '+label)
        if row['shape']!=shape.tolist() or row['queries']!=n or n!=pairs*nsp or nsp!=nsm**3 or np.prod(mesh)!=points:
            errors.append('Geometry dimensions differ: '+label)
        reference_real=geo[label+'_expected'];reference_int=geo[label+'_expected_int'];inverse=geo[label+'_inverse']
        if reference_real.shape!=(7*n,) or reference_int.shape!=(3*n,):errors.append('Geometry output shape differs: '+label)
        if not np.array_equal(reference_real[4*n:],xyz.ravel()):errors.append('Oracle displacements differ: '+label)
        if not np.array_equal(np.sort(inverse),np.arange(1,points+1)):errors.append('Original reord map is not a permutation')
        clustered=np.repeat((f['pair_cluster']-1)*nsp,nsp)+np.tile(np.arange(nsp),pairs)
        mapped=clustered+1 if row['kind']==1 else inverse[clustered]
        if not np.array_equal(reference_int[n:2*n],mapped) or not np.array_equal(reference_int[2*n:],np.repeat(f['unit_atom'][atom],nsp)):
            errors.append('Field/atom mapping differs from original reord: '+label)
        cutoff=f['radial_parameters'][2*qs+row['kind']-1,0];radius=np.sqrt(np.sum(xyz*xyz,axis=1))
        mask=(radius<cutoff)&((cutoff>1e-12) if row['kind']==1 else True)
        if not np.array_equal(reference_int[:n],mask.astype(np.int32)):errors.append('Geometry outer selection differs: '+label)
        cpu_expected[label]=(np.column_stack((reference_real[:n],reference_real[n:4*n].reshape(n,3))),np.r_[reference_int[:n],inverse])
    expected=set()
    for label in required|geo_required:
        for captured in (False,True):
            for cycles in (1,3):expected.add((label,captured,cycles,None))
    for kind in (1,2):
        label=f'smooth_kind{kind}_mesh1'
        for fault in (*index_faults,*value_faults,'sticky'):
            for captured in (False,True):expected.add((label,captured,3,fault))
        expected.add((label,False,1,'kind'))
        label=f'geometry_nsm2_pairs137_kind{kind}'
        for fault in (*geometry_fields,'sticky'):
            for captured in (False,True):expected.add((label,captured,3,fault))
        for fault in host_faults[1:]:expected.add((label,False,1,fault))
    seen=set()
    for row in report['checks']:
        label,captured,cycles,fault=(row[k] for k in ('fixture','captured','cycles','fault'));key=(label,captured,cycles,fault);issues=[]
        if key in seen:issues.append('Duplicate scenario')
        seen.add(key);is_geometry=label in geometries;arrays=geo if is_geometry else data
        reference_row=geometries[label] if is_geometry else fixtures[label];kind=reference_row['kind']
        if row['kind']!=(0 if fault=='kind' else kind) or row['mesh_filter']!=(1 if is_geometry else reference_row['mesh_filter']):issues.append('Mode differs')
        names=(*basis,*geometry_fields) if is_geometry else fields
        staged=sum(arrays[label+'_'+k].nbytes for k in names)+4
        wanted=arrays[label+'_expected'];wanted_int=arrays[label+('_expected_int' if is_geometry else '_active')]
        out=outputs[row['scenario']+'_real'];integers=outputs[row['scenario']+'_int'];output_bytes=wanted.nbytes+wanted_int.nbytes
        allocated=staged+output_bytes+sum(arrays[label+'_'+k].dtype.itemsize for k in names if not arrays[label+'_'+k].size)
        if fault in host_faults:status=-1
        elif fault=='sticky':status=7
        elif is_geometry:status=2 if fault in geometry_fields[5:] else 1 if fault in geometry_fields else 0
        else:status=1 if fault in index_faults else 2 if fault in value_faults else 0
        wanted_execution=dict(native_return=-1 if fault in host_faults else 0,status=status,expected_status=status,
            stage=3 if fault in host_faults else 4,graph_nodes=int(captured),forbidden_nodes=0,staged=staged,
            allocated=allocated,exported=0 if fault in host_faults else output_bytes+4,poisoned=cycles*output_bytes,
            expected_staged=staged,expected_allocated=allocated,expected_exported=output_bytes+4)
        if any(row.get(k)!=v for k,v in wanted_execution.items()):issues.append('Execution, graph or shape-derived traffic differs')
        name=f'{label}_captured{int(captured)}_cycles{cycles}'+('_'+fault if fault else '')
        if row['scenario']!=name:issues.append('Scenario name differs')
        error=bits=zero_signs=None
        if fault is None or fault=='sticky':
            if out.shape!=wanted.shape or integers.shape!=wanted_int.shape or not np.isfinite(out).all():issues.append('Invalid result shape/values')
            else:
                delta=np.abs(out-wanted);error=float(delta.max());scaled=float((delta/(1e-10+1e-10*np.abs(wanted))).max())
                bitmask=out.view(np.uint64)!=wanted.view(np.uint64);bits=int(bitmask.sum());zero_signs=int((bitmask&(out==0.)&(wanted==0.)).sum())
                if scaled>1. or row['max_absolute_error']!=error or row['max_scaled_error']!=scaled:issues.append('Numerical comparison differs')
                if not np.array_equal(integers,wanted_int) or row['integers_exact'] is not True:issues.append('Selection or point/atom map differs')
                if is_geometry and not np.array_equal(out[4*(len(out)//7):],wanted[4*(len(out)//7):]):issues.append('Displacements differ')
        if row['passed'] is not True:issues.append('Recorded scenario failed')
        checks.append(dict(scenario=name,passed=not issues,errors=issues,max_absolute_error=error,bit_pattern_differences=bits,zero_sign_differences=zero_signs))
    if seen!=expected or len(checks)!=238 or len(outputs.files)!=476:errors.append('Required scenario coverage differs')
    artifacts=(path,oracle,folder/reference['arrays_file'],folder/reference['wrapper_file'],folder/report['geometry_file'],folder/report['output_file'])
    for p,digest in ((oracle,report['oracle_sha256']),(folder/reference['arrays_file'],report['oracle_arrays_sha256']),
        (folder/reference['arrays_file'],reference['arrays_sha256']),(folder/reference['wrapper_file'],reference['wrapper_sha256']),
        (folder/report['geometry_file'],report['geometry_sha256']),(folder/report['output_file'],report['output_sha256'])):
        if sha(p)!=digest:errors.append('Artifact hash differs: '+p.name)
    cpu=replay(reference_build,folder/reference['wrapper_file'],inputs,cpu_expected,root) if reference_build else None
    if cpu is not None and cpu['passed'] is not True:errors.append('Independent local CPU oracle replay failed')
    return dict(scope=__doc__,passed=not errors and all(v['passed'] for v in checks),errors=errors,checks=checks,cpu_replay=cpu,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in artifacts},verifier_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--reference-build',type=Path);a=p.parse_args()
    result=verify(a.run,a.reference_build)
    (a.run/'atomic-basis-component-verifier.py').write_bytes(Path(__file__).read_bytes())
    (a.run/'atomic-basis-component-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Atomic radial component:',len(result['checks']),'scenarios; CPU replay=',result['cpu_replay']['passed'] if result['cpu_replay'] else None,'passed=',result['passed'])
    if not result['passed']:raise SystemExit(1)
